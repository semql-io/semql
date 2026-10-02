#!/usr/bin/env python3
"""Offline installed-package recipes; no LLM, credentials, or network calls.

Install semql, semql-engine, and duckdb in a clean environment, then run:
    python demos/integration_recipes.py
"""

from __future__ import annotations

import duckdb
from semql import (
    AuthContext,
    Catalog,
    CatalogContext,
    ComparisonResult,
    ComparisonScope,
    Cube,
    Dialect,
    Dimension,
    FederatedPlan,
    Lookup,
    Measure,
    ResolutionContext,
    ScopePredicate,
    SemanticQuery,
    compare_analysis,
    compile_federated_query,
    enrich_all,
)
from semql_engine import DuckDBAdapter, DuckDBMergeEngine, Engine


class RegionEnricher:
    """Tiny local lookup standing in for a batched reference-table query."""

    def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
        del ctx
        labels = {"emea": "Europe", "amer": "Americas"}
        return {key: labels[key] for key in ids if key in labels}


def team_scope(cube: Cube, viewer: AuthContext) -> ScopePredicate:
    del cube, viewer
    return ScopePredicate(sql="{s}.team = {ctx.team}", ctx_keys=["ctx.team"])


def query(*, alias: bool = False) -> SemanticQuery:
    return SemanticQuery(
        dimensions=["sales.region"],
        measures=["sales.revenue"],
        aliases={"sales_region": "sales.region"} if alias else {},
    )


def build_catalog(*, scoped: bool = False) -> Catalog:
    sales = Cube(
        name="sales",
        dialect=Dialect.DUCKDB,
        table="sales",
        alias="s",
        measures=[Measure(name="revenue", sql="{s}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{s}.region", type="string"),
            Dimension(name="team", sql="{s}.team", type="string"),
        ],
        scope="team_scope" if scoped else None,
    )
    return Catalog(
        [sales],
        scope_fns={"team_scope": team_scope} if scoped else None,
        lookups=[Lookup(dimension="sales.region", enricher=RegionEnricher())],
    )


def compile_plan(catalog: Catalog, *, viewer: AuthContext | None = None) -> FederatedPlan:
    return compile_federated_query(
        query(alias=True),
        catalog.as_dict(),
        viewer=viewer,
        context={"ctx.team": "blue"} if viewer is not None else None,
        scope_fns={"team_scope": team_scope} if viewer is not None else None,
        catalog_context=CatalogContext(namespace="demo", semantic_revision="fixture-v1"),
        mode="distributive",
    )


def main() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE sales(region VARCHAR, team VARCHAR, amount INTEGER)")
    con.execute(
        "INSERT INTO sales VALUES ('emea', 'blue', 10), ('emea', 'blue', 5), ('amer', 'red', 7)"
    )
    engine = Engine(merge_engine=DuckDBMergeEngine())
    engine.register(Dialect.DUCKDB, DuckDBAdapter(con))
    catalog = build_catalog()
    try:
        print("ordinary compile + execute")
        ordinary = compile_plan(catalog)
        result = engine.run(ordinary)
        assert sorted(result.rows) == [("amer", 7), ("emea", 15)]
        assert result.columns == ["sales_region", "revenue"]
        print(f"  {result.columns}: {sorted(result.rows)}")

        print("trusted host identity and scoped context")
        # The host authenticates the caller elsewhere and supplies its verified
        # identity. Never construct these fields by trusting client claims.
        viewer = AuthContext(viewer_id="fixture-user")
        scoped_plan = compile_plan(build_catalog(scoped=True), viewer=viewer)
        scoped = engine.run(scoped_plan)
        assert all(region != "amer" for region, _ in scoped.rows)
        print(f"  verified identity fixture-user; scoped rows: {scoped.rows}")

        print("semantic comparison: not_established then evidence")
        scoped_peer = compile_plan(build_catalog(scoped=True), viewer=viewer)
        unknown = compare_analysis(
            scoped_plan.analysis,
            scoped_peer.analysis,
            scope=ComparisonScope.RESULT,
            shared_catalog_snapshot=True,
        )
        assert unknown.outcome == ComparisonResult.NOT_ESTABLISHED
        established = compare_analysis(
            scoped_plan.analysis,
            scoped_peer.analysis,
            scope=ComparisonScope.RESULT,
            shared_catalog_snapshot=True,
            bound_context_equal=True,
        )
        assert established.outcome == ComparisonResult.EQUIVALENT
        print(f"  without proof: {unknown.outcome}; with host attestation: {established.outcome}")

        print("alias-aware enrichment preserves source keys")
        enriched = enrich_all(
            [dict(zip(result.columns, row, strict=True)) for row in result.rows],
            catalog,
            ResolutionContext(viewer=viewer),
            analysis=result.analysis,
        )
        amer_row = next(row for row in enriched.rows if row["sales_region"] == "amer")
        assert amer_row["sales_region__label"] == "Americas"
        assert enriched.analysis.result == result.analysis.result
        print(f"  {amer_row}")

        print("explicit federation strategy")
        print("  compile_federated_query(..., mode='distributive')")

        print("cache partitioning by trust boundary")
        cached = Engine(merge_engine=DuckDBMergeEngine(), cache_size=4)
        cached.register(Dialect.DUCKDB, DuckDBAdapter(con))
        cached.run(ordinary, cache_namespace="tenant-a")
        cached.run(ordinary, cache_namespace="tenant-b")
        assert cached.cache_misses == 2
        cached.run(ordinary, cache_namespace="tenant-a")
        assert cached.cache_hits == 1
        print("  distinct namespaces miss independently; same namespace hits")

        print("stream cleanup responsibility")
        stream = engine.iter_rows(ordinary)
        try:
            first = next(stream)
            assert first["revenue"] in {7, 15}
        finally:
            stream.close()
        print("  early-exit iterator closed in finally")
        print("all integration recipes passed")
    finally:
        con.close()


if __name__ == "__main__":
    main()
