"""Resolution stage: turn free-text filter phrases into canonical keys before
the Auto-Planner runs, emitting a typed ``ResolutionOutcome`` per value.

This is the I/O-edge counterpart to ``semql.autoplan`` (which is sans-io and
assumes ``Filter.values`` are already canonical). It reuses ``lookups.resolve``
(exact -> substring -> fuzzy) and mirrors the ``resolve_lookup`` MCP tool, but
classifies the result: a single hit is ``resolved`` (splice it), several hits
are ``ambiguous`` (a structured refusal carrying candidates), none is
``unresolved`` — so the caller / LLM can re-issue instead of guessing.
"""

from __future__ import annotations

from semql import Catalog, Cube, Dialect, Dimension, Filter, Join, Lookup, Measure, SemanticQuery
from semql.lookups import (
    QueryResolution,
    resolve_outcome,
    resolve_query_filters,
)


def _catalog() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
    )
    region_lookup = Lookup(
        dimension="orders.region",
        values=("EMEA", "APAC", "NA"),
        labels={"EMEA": "Europe, Middle East & Africa", "APAC": "Asia-Pacific"},
    )
    return Catalog(cubes=[orders], lookups=[region_lookup])


# ---------------------------------------------------------------------------
# resolve_outcome — the per-phrase classifier
# ---------------------------------------------------------------------------


def test_resolve_outcome_exact_single_is_resolved() -> None:
    out = resolve_outcome(_catalog(), "orders.region", "emea")
    assert out.status == "resolved"
    assert out.resolved is True
    assert out.values == ("EMEA",)
    assert out.value == "EMEA"
    assert out.labels == {"EMEA": "Europe, Middle East & Africa"}


def test_resolve_outcome_multiple_hits_is_ambiguous() -> None:
    # "a" is a substring of EMEA, APAC, and NA -> several candidates.
    out = resolve_outcome(_catalog(), "orders.region", "a")
    assert out.status == "ambiguous"
    assert out.resolved is False
    assert set(out.values) == {"EMEA", "APAC", "NA"}


def test_resolve_outcome_no_match_is_unresolved() -> None:
    out = resolve_outcome(_catalog(), "orders.region", "zzzzz")
    assert out.status == "unresolved"
    assert out.values == ()


# ---------------------------------------------------------------------------
# resolve_query_filters — splice resolved values, surface blockers
# ---------------------------------------------------------------------------


def test_query_filter_resolves_and_splices_canonical_value() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.region", op="eq", values=["emea"])],
    )
    res = resolve_query_filters(q, _catalog())
    assert isinstance(res, QueryResolution)
    assert res.ok is True
    assert res.blocked == ()
    assert res.query.filters == [Filter(dimension="orders.region", op="eq", values=["EMEA"])]


def test_query_filter_in_resolves_each_value() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.region", op="in", values=["emea", "apac"])],
    )
    res = resolve_query_filters(q, _catalog())
    assert res.ok is True
    assert res.query.filters[0].values == ["EMEA", "APAC"]


def test_query_filter_ambiguous_blocks_and_leaves_filter_unchanged() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.region", op="eq", values=["a"])],
    )
    res = resolve_query_filters(q, _catalog())
    assert res.ok is False
    assert len(res.blocked) == 1
    assert res.blocked[0].status == "ambiguous"
    # The unresolved filter is left exactly as the caller wrote it.
    assert res.query.filters == q.filters


def test_query_filter_non_lookup_and_non_membership_ops_pass_through() -> None:
    """A dim with no Lookup, and a non-membership op (contains/gt/...) on a
    lookup dim, are left verbatim — resolution only canonicalizes membership
    predicates on lookup-backed dimensions."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[
            Filter(dimension="orders.id", op="gt", values=[10]),
            Filter(dimension="orders.region", op="contains", values=["em"]),
        ],
    )
    res = resolve_query_filters(q, _catalog())
    assert res.ok is True
    assert res.outcomes == ()  # nothing was attempted
    assert res.query.filters == q.filters


# ---------------------------------------------------------------------------
# End-to-end: resolution stage -> Auto-Planner compose (the "Nikhil" shape)
# ---------------------------------------------------------------------------


def _federated_catalog() -> Catalog:
    """orders (Postgres fact) bridged to customers (BigQuery metadata) on
    customer_id = id; customers.region is Lookup-backed."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )
    region_lookup = Lookup(dimension="customers.region", values=("EMEA", "APAC", "NA"))
    return Catalog(cubes=[orders, customers], lookups=[region_lookup])


def test_resolution_then_autoplan_injects_semi_join_with_canonical_value() -> None:
    from semql.autoplan import autoplan

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="customers.region", op="eq", values=["emea"])],
    )

    # Stage 1 (I/O edge): canonicalize the free-text phrase.
    resolved = resolve_query_filters(q, catalog)
    assert resolved.ok is True

    # Stage 2 (sans-io): plan pushes the filter-only foreign cube as a semi-join.
    plan = autoplan(resolved.query, catalog.as_dict(), lookups=catalog.lookups)
    assert [d.strategy for d in plan.decisions] == ["semi_join"]
    assert len(plan.query.semi_joins) == 1
    sj = plan.query.semi_joins[0]
    assert sj.dimension == "orders.customer_id"
    assert sj.select == "customers.id"
    # The canonical value (not the free-text "emea") rides into the inner source.
    assert sj.source.filters == [Filter(dimension="customers.region", op="eq", values=["EMEA"])]
