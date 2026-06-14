#!/usr/bin/env -S uv run python
"""Micro-benchmark for the SemQL compile path.

Answers one question: when SemQL compiles a batch of ``SemanticQuery``
objects to SQL, where does the CPU go — into SemQL's own pure-Python
code (which ``mypyc`` could compile), or into ``sqlglot`` / ``pydantic``
(which it can't)? That ratio decides whether mypyc is worth the
platform-wheel build matrix.

Run::

    uv run python benchmarks/bench_compile.py
    uv run python benchmarks/bench_compile.py --batch 40 --reps 50

Reports:
  - wall-clock throughput (compiles/sec, µs/compile) via ``timeit``;
  - a ``cProfile`` self-time breakdown bucketed by top-level package,
    so ``semql`` vs ``sqlglot`` vs ``pydantic`` is explicit.

Pure measurement of ``Catalog.compile`` — the parser is excluded so the
number reflects the compiler, not NL parsing.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import timeit
from io import StringIO

from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, TimeDimension
from semql.spec import BoolExpr, CompareWindow, Filter, SemanticQuery, TimeWindow

CONTEXT = {"schema": "prod"}


def _catalog() -> Catalog:
    """A realistic multi-cube catalog: a central fact (orders) with measures
    spanning the aggregate families, joined to three one-side cubes."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg"),
            Measure(name="max_amount", sql="{o}.amount", agg="max"),
            Measure(
                name="uniq_cust", sql="{o}.customer_id", agg="count_distinct", non_additive=True
            ),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
            Dimension(name="is_paid", sql="{o}.is_paid", type="bool"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at", sql="{o}.created_at", granularities=("day", "week", "month")
            )
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
            Join(to="products", relationship="many_to_one", on="{o}.product_id = {p}.id"),
        ],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        dimensions=[
            Dimension(name="name", sql="{c}.name", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )
    products = Cube(
        name="products",
        dialect=Dialect.POSTGRES,
        table="{schema}.products",
        alias="p",
        dimensions=[Dimension(name="category", sql="{p}.category", type="string")],
    )
    return Catalog([orders, customers, products])


def _queries() -> list[SemanticQuery]:
    """A spread of query shapes the compiler handles — projection, every
    filter structure, having, ordering, time breakdown, compare, and
    multi-cube joins."""
    return [
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        SemanticQuery(
            measures=["orders.revenue", "orders.count"],
            dimensions=["orders.region", "orders.status"],
        ),
        SemanticQuery(
            measures=["orders.avg_amount", "orders.max_amount"], dimensions=["orders.region"]
        ),
        SemanticQuery(measures=["orders.uniq_cust"], dimensions=["orders.region"]),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.region", op="in", values=["EMEA", "APAC", "NA"])],
        ),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.status", op="eq", values=["paid"]),
                    Filter(dimension="orders.region", op="eq", values=["EMEA"]),
                ],
            ),
        ),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            having=[Filter(dimension="orders.revenue", op="gt", values=[1000])],
            order=[("orders.revenue", "desc")],
            limit=10,
        ),
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="month",
                range=("2026-01-01", "2026-12-31"),
            ),
        ),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="month",
                range=("2026-01-01", "2026-06-30"),
            ),
            compare=CompareWindow(mode="previous_period"),
        ),
        SemanticQuery(measures=["orders.revenue"], dimensions=["customers.tier"]),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.tier", "products.category"],
            order=[("orders.revenue", "desc")],
        ),
    ]


_BUCKETS = ("semql", "sqlglot", "pydantic")


def _bucket(filename: str) -> str:
    for pkg in _BUCKETS:
        if f"/{pkg}/" in filename or filename.endswith(f"/{pkg}.py"):
            return pkg
    if "<" in filename:  # <string>, <built-in>, etc.
        return "builtin/other"
    return "stdlib/other"


def _profile_breakdown(catalog: Catalog, queries: list[SemanticQuery], reps: int) -> None:
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(reps):
        for q in queries:
            catalog.compile(q, context=CONTEXT)
    prof.disable()

    stats = pstats.Stats(prof, stream=StringIO())
    by_pkg: dict[str, float] = {}
    total_self = 0.0
    for (filename, _lineno, _func), value in stats.stats.items():  # type: ignore[attr-defined]
        tt = value[2]  # (cc, nc, tottime, cumtime, callers)
        bucket = _bucket(filename)
        by_pkg[bucket] = by_pkg.get(bucket, 0.0) + tt
        total_self += tt

    print("\nSelf-time by package (cProfile, tottime — what mypyc could touch):")
    print(f"  {'package':<16}{'self s':>10}{'share':>9}")
    for pkg, tt in sorted(by_pkg.items(), key=lambda kv: kv[1], reverse=True):
        share = (tt / total_self * 100) if total_self else 0.0
        print(f"  {pkg:<16}{tt:>10.4f}{share:>8.1f}%")
    semql_share = (by_pkg.get("semql", 0.0) / total_self * 100) if total_self else 0.0
    print(f"\n  → semql's own code is {semql_share:.1f}% of CPU self-time.")
    print("    (mypyc can only speed up that slice; sqlglot/pydantic are out of reach.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1, help="repeat the query set N times per batch")
    ap.add_argument("--reps", type=int, default=200, help="timeit/profiler repetitions")
    args = ap.parse_args()

    catalog = _catalog()
    queries = _queries() * args.batch

    # Correctness sanity + warm-up (caches, sqlglot dialect init).
    for q in queries:
        assert catalog.compile(q, context=CONTEXT).sql

    def run() -> None:
        for q in queries:
            catalog.compile(q, context=CONTEXT)

    n_per_rep = len(queries)
    secs = timeit.timeit(run, number=args.reps)
    total_compiles = args.reps * n_per_rep
    per = secs / total_compiles
    print(f"Compiled {total_compiles} queries ({n_per_rep}/rep × {args.reps} reps) in {secs:.3f}s")
    print(f"  {per * 1e6:>8.1f} µs / compile   |   {1 / per:>10,.0f} compiles / sec")

    _profile_breakdown(catalog, queries, args.reps)


if __name__ == "__main__":
    main()
