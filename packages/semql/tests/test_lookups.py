"""Tests for dimension-value catalogs — ``Lookup`` model + Catalog wiring.

A ``Lookup`` declares the finite set of valid values for a string
dimension. Static lookups inline values in the catalog; dynamic
lookups resolve via a ``loader(ResolutionContext)`` at prompt-render
time. The compiler never fires loaders — they're a presentation-layer
concern.

A ``Lookup`` also carries an orthogonal ``enricher`` — a post-query
slot that attaches reference columns to result rows and never feeds the
prompt (so a large/sensitive keyspace is labelled silently).

Coverage here:
- Model validators (mutual exclusion of ``values`` / ``loader``,
  enricher orthogonal, qualified dimension required, max_inline bounds).
- Catalog wiring: dimension reference must resolve to a real
  string-typed dimension; duplicates rejected.
- Enricher decoupling: enricher-only lookups render no prompt line.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Lookup,
    Measure,
    ResolutionContext,
)
from semql.model import AuthContext, LookupValues
from semql_prompt import planner_prompt

# ---------------------------------------------------------------------------
# Lookup model validation
# ---------------------------------------------------------------------------


def test_lookup_requires_values_loader_or_enricher() -> None:
    with pytest.raises(ValidationError, match=r"(?i)must declare a vocabulary"):
        Lookup(dimension="orders.region")


def test_lookup_rejects_values_and_loader_together() -> None:
    with pytest.raises(ValidationError, match=r"(?i)mutually exclusive"):
        Lookup(
            dimension="orders.region",
            values=("EMEA",),
            loader=lambda _ctx: ["EMEA"],
        )


def test_lookup_requires_qualified_dimension() -> None:
    with pytest.raises(ValidationError, match=r"(?i)must be qualified"):
        Lookup(dimension="region", values=("EMEA",))


def test_lookup_rejects_overqualified_dimension() -> None:
    with pytest.raises(ValidationError, match=r"(?i)must be qualified"):
        Lookup(dimension="db.schema.region", values=("EMEA",))


def test_lookup_rejects_negative_max_inline() -> None:
    with pytest.raises(ValidationError, match=r"(?i)max_inline"):
        Lookup(dimension="orders.region", values=("EMEA",), max_inline=-1)


def test_lookup_accepts_static_values_with_labels() -> None:
    lk = Lookup(
        dimension="orders.region",
        values=("EMEA", "APAC"),
        labels={"EMEA": "Europe, Middle East & Africa"},
    )
    assert lk.values == ("EMEA", "APAC")
    assert lk.labels is not None
    assert lk.labels["EMEA"].startswith("Europe")


def test_lookup_accepts_dynamic_loader() -> None:
    def _load(ctx: ResolutionContext) -> LookupValues:
        return ["A", "B", "C"]

    lk = Lookup(dimension="orders.region", loader=_load)
    assert lk.loader is _load
    assert lk.values is None


# ---------------------------------------------------------------------------
# Catalog wiring: Lookup.dimension must resolve to a real string Dimension
# ---------------------------------------------------------------------------


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
    )


def test_catalog_accepts_valid_lookup() -> None:
    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", values=("EMEA", "APAC", "NA"))],
    )
    assert "orders.region" in cat.lookups
    assert cat.lookups["orders.region"].values == ("EMEA", "APAC", "NA")


def test_catalog_rejects_lookup_for_unknown_cube() -> None:
    with pytest.raises(ValueError, match=r"(?i)not in the catalog"):
        Catalog(
            [_orders_cube()],
            lookups=[Lookup(dimension="ghosts.region", values=("X",))],
        )


def test_catalog_rejects_lookup_for_unknown_dimension() -> None:
    with pytest.raises(ValueError, match=r"(?i)no dimension named"):
        Catalog(
            [_orders_cube()],
            lookups=[Lookup(dimension="orders.unicorn", values=("X",))],
        )


def test_catalog_rejects_lookup_on_non_string_dimension() -> None:
    """``amount`` is type=number — lookups only make sense for string."""
    with pytest.raises(ValueError, match=r"(?i)only string-typed"):
        Catalog(
            [_orders_cube()],
            lookups=[Lookup(dimension="orders.amount", values=("100",))],
        )


def test_catalog_rejects_duplicate_lookups() -> None:
    with pytest.raises(ValueError, match=r"(?i)duplicate Lookup"):
        Catalog(
            [_orders_cube()],
            lookups=[
                Lookup(dimension="orders.region", values=("EMEA",)),
                Lookup(dimension="orders.region", values=("APAC",)),
            ],
        )


def test_resolution_context_carries_viewer_and_context() -> None:
    rc = ResolutionContext(
        viewer=AuthContext(viewer_id="u1", roles=["admin"]),
        context={"tenant_schema": "t1"},
    )
    assert rc.viewer is not None
    assert rc.viewer.viewer_id == "u1"
    assert rc.context["tenant_schema"] == "t1"


# ---------------------------------------------------------------------------
# Prompt rendering: lookups appear inline beneath their dimensions
# ---------------------------------------------------------------------------


def test_static_lookup_inlines_values_in_prompt() -> None:
    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", values=("EMEA", "APAC", "NA"))],
    )
    out = planner_prompt(
        cat,
    )
    # DB-sourced lookup values are fenced as untrusted data (S9).
    assert "Lookup (3 values): <untrusted-data>`EMEA`, `APAC`, `NA`</untrusted-data>" in out


def test_static_lookup_with_labels_renders_pairs() -> None:
    cat = Catalog(
        [_orders_cube()],
        lookups=[
            Lookup(
                dimension="orders.region",
                values=("EMEA", "APAC"),
                labels={"EMEA": "Europe", "APAC": "Asia Pacific"},
            )
        ],
    )
    out = planner_prompt(
        cat,
    )
    assert "`EMEA` (Europe)" in out
    assert "`APAC` (Asia Pacific)" in out


def test_oversized_lookup_renders_tool_hint() -> None:
    """Beyond max_inline the prompt surfaces a sample + tool hint, not the full list."""
    big_values = tuple(f"region_{i}" for i in range(100))
    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", values=big_values, max_inline=10)],
    )
    out = planner_prompt(
        cat,
    )
    assert "100 values" in out
    assert "resolve_orders_region(query)" in out
    # Sample is rendered but not the full list.
    assert "`region_0`" in out
    assert "`region_99`" not in out


def test_dynamic_lookup_fires_loader_with_ctx() -> None:
    seen_ctxs: list[ResolutionContext] = []

    def _load(c: ResolutionContext) -> list[str]:
        seen_ctxs.append(c)
        return ["dyn_a", "dyn_b"]

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=_load)],
    )
    rc = ResolutionContext(context={"tenant_schema": "tenant42"})
    out = planner_prompt(cat, ctx=rc)
    assert "Lookup (2 values): <untrusted-data>`dyn_a`, `dyn_b`</untrusted-data>" in out
    assert len(seen_ctxs) == 1
    assert seen_ctxs[0].context["tenant_schema"] == "tenant42"


def test_dynamic_lookup_without_ctx_renders_tool_hint() -> None:
    def _load(_ctx: ResolutionContext) -> list[str]:
        raise AssertionError("loader must not fire when ctx is None")

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=_load)],
    )
    out = planner_prompt(
        cat,
    )  # no ctx
    assert "resolved at runtime" in out
    assert "resolve_orders_region(query)" in out


def test_loader_returning_mapping_uses_labels() -> None:
    def _load(_ctx: ResolutionContext) -> dict[str, str]:
        return {"NA": "North America", "EMEA": "Europe"}

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=_load)],
    )
    out = planner_prompt(cat, ctx=ResolutionContext())
    assert "`NA` (North America)" in out
    assert "`EMEA` (Europe)" in out


def test_no_lookup_means_no_extra_line() -> None:
    cat = Catalog([_orders_cube()])
    out = planner_prompt(
        cat,
    )
    assert "Lookup" not in out


# ---------------------------------------------------------------------------
# semql.lookups.resolve — substring + fuzzy matching
# ---------------------------------------------------------------------------


def _regions_catalog() -> Catalog:
    return Catalog(
        [_orders_cube()],
        lookups=[
            Lookup(
                dimension="orders.region",
                values=("EMEA", "APAC", "NA", "LATAM"),
                labels={
                    "EMEA": "Europe, Middle East & Africa",
                    "APAC": "Asia Pacific",
                    "NA": "North America",
                    "LATAM": "Latin America",
                },
            )
        ],
    )


def test_resolve_exact_match_by_canonical_value() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    assert resolve_lookup(cat, "orders.region", "EMEA") == ["EMEA"]
    # Case-insensitive.
    assert resolve_lookup(cat, "orders.region", "emea") == ["EMEA"]


def test_resolve_exact_match_by_label() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    assert resolve_lookup(cat, "orders.region", "Asia Pacific") == ["APAC"]


def test_resolve_substring_match() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    # "europe" appears in EMEA's label only.
    assert resolve_lookup(cat, "orders.region", "europe") == ["EMEA"]
    # "america" appears in NA + LATAM labels.
    hits = resolve_lookup(cat, "orders.region", "america")
    assert set(hits) == {"NA", "LATAM"}


def test_resolve_fuzzy_match_for_typos() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    # "apec" → "APAC" via fuzzy match.
    hits = resolve_lookup(cat, "orders.region", "apec")
    assert "APAC" in hits


def test_resolve_returns_empty_for_no_match() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    assert resolve_lookup(cat, "orders.region", "zzz_unknown_zzz") == []


def test_resolve_unknown_dimension_raises() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    with pytest.raises(KeyError, match=r"(?i)no lookup registered"):
        resolve_lookup(cat, "orders.unknown", "anything")


def test_resolve_dynamic_lookup_with_ctx() -> None:
    from semql import resolve_lookup

    def _load(c: ResolutionContext) -> list[str]:
        tenant = c.context.get("tenant", "")
        return [f"{tenant}_alpha", f"{tenant}_beta"]

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=_load)],
    )
    rc = ResolutionContext(context={"tenant": "t42"})
    assert resolve_lookup(cat, "orders.region", "t42_alpha", ctx=rc) == ["t42_alpha"]


def test_resolve_dynamic_lookup_without_ctx_returns_empty() -> None:
    from semql import resolve_lookup

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=lambda _c: ["a"])],
    )
    assert resolve_lookup(cat, "orders.region", "a") == []


def test_resolve_empty_query_returns_empty() -> None:
    from semql import resolve_lookup

    cat = _regions_catalog()
    assert resolve_lookup(cat, "orders.region", "") == []
    assert resolve_lookup(cat, "orders.region", "   ") == []


# ---------------------------------------------------------------------------
# materialize_lookup — same I/O surface used by prompt rendering
# ---------------------------------------------------------------------------


def test_materialize_static_lookup() -> None:
    from semql import materialize_lookup

    lk = Lookup(dimension="orders.region", values=("EMEA", "APAC"))
    materialized = materialize_lookup(lk, None)
    assert materialized is not None
    values, labels = materialized
    assert values == ["EMEA", "APAC"]
    assert labels is None


def test_materialize_dynamic_lookup_with_mapping_loader() -> None:
    from semql import materialize_lookup

    lk = Lookup(
        dimension="orders.region",
        loader=lambda _c: {"NA": "North America"},
    )
    materialized = materialize_lookup(lk, ResolutionContext())
    assert materialized is not None
    values, labels = materialized
    assert values == ["NA"]
    assert labels == {"NA": "North America"}


def test_materialize_dynamic_lookup_without_ctx_returns_none() -> None:
    from semql import materialize_lookup

    lk = Lookup(dimension="orders.region", loader=lambda _c: ["x"])
    assert materialize_lookup(lk, None) is None


# ---------------------------------------------------------------------------
# LookupEnricher Protocol + enrich_result helper
# ---------------------------------------------------------------------------


def test_plain_callable_is_not_lookup_enricher() -> None:
    """A plain LookupLoader callable doesn't satisfy LookupEnricher."""
    from semql.model import LookupEnricher

    def plain(ctx: object) -> list[str]:
        return ["a", "b"]

    assert not isinstance(plain, LookupEnricher)


def test_class_implementing_enrich_satisfies_protocol() -> None:
    """A class with an enrich() method satisfies the runtime-checkable Protocol."""
    from semql.model import LookupEnricher

    class MyEnricher:
        def __call__(self, ctx: object) -> list[str]:
            return ["x"]

        def enrich(self, ids: list[str], ctx: object) -> dict[str, str]:
            return {i: f"Label:{i}" for i in ids}

    assert isinstance(MyEnricher(), LookupEnricher)


def test_enrich_result_adds_label_column_for_known_ids() -> None:
    """enrich_result adds __label column for known IDs via LookupEnricher."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, LookupEnricher, ResolutionContext

    class EnrichingLoader:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return ["u1", "u2"]

        def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
            return {i: f"Name:{i}" for i in ids}

    enricher = EnrichingLoader()
    assert isinstance(enricher, LookupEnricher)
    lk = Lookup(dimension="orders.customer_id", enricher=enricher)
    rows: list[dict[str, object]] = [
        {"customer_id": "u1", "revenue": 100},
        {"customer_id": "u2", "revenue": 200},
    ]
    out = enrich_result(rows, "customer_id", lk, ResolutionContext())
    assert out[0]["customer_id__label"] == "Name:u1"
    assert out[1]["customer_id__label"] == "Name:u2"


def test_enrich_result_echoes_raw_id_for_unknown_ids() -> None:
    """Unknown IDs (not in enricher mapping) get the raw ID echoed as the label."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    class EnrichingLoader:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return []

        def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
            return {}  # nothing known

    lk = Lookup(dimension="orders.customer_id", enricher=EnrichingLoader())
    rows: list[dict[str, object]] = [{"customer_id": "u99", "revenue": 5}]
    out = enrich_result(rows, "customer_id", lk, ResolutionContext())
    assert out[0]["customer_id__label"] == "u99"


def test_enrich_result_skips_none_ids() -> None:
    """Rows with None dimension value are skipped; label is None."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    enriched: list[list[str]] = []

    class EnrichingLoader:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return []

        def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
            enriched.append(ids)
            return {}

    lk = Lookup(dimension="orders.customer_id", enricher=EnrichingLoader())
    rows: list[dict[str, object]] = [{"customer_id": None, "revenue": 5}]
    out = enrich_result(rows, "customer_id", lk, ResolutionContext())
    # None rows should not contribute an id to the enricher call.
    assert None not in enriched[0] if enriched else True
    assert out[0].get("customer_id__label") is None


def test_enrich_result_plain_callable_returns_rows_unchanged() -> None:
    """When the loader is not a LookupEnricher, rows pass through unchanged."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    lk = Lookup(dimension="orders.region", values=("EMEA",))
    rows: list[dict[str, object]] = [{"region": "EMEA", "cnt": 1}]
    out = enrich_result(rows, "region", lk, ResolutionContext())
    assert out == rows
    assert "region__label" not in out[0]


def test_enrich_result_only_calls_enricher_with_unique_non_null_ids() -> None:
    """enrich is called with deduplicated non-null IDs, not duplicates."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    seen: list[list[str]] = []

    class EnrichingLoader:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return []

        def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
            seen.append(list(ids))
            return {i: f"L:{i}" for i in ids}

    lk = Lookup(dimension="orders.customer_id", enricher=EnrichingLoader())
    rows: list[dict[str, object]] = [
        {"customer_id": "u1"},
        {"customer_id": "u1"},  # duplicate
        {"customer_id": "u2"},
        {"customer_id": None},
    ]
    enrich_result(rows, "customer_id", lk, ResolutionContext())
    assert len(seen) == 1
    assert set(seen[0]) == {"u1", "u2"}  # no duplicates, no None


# ---------------------------------------------------------------------------
# MultiFieldEnricher — several reference fields per id
# ---------------------------------------------------------------------------


def test_class_implementing_enrich_fields_satisfies_multi_field_protocol() -> None:
    """A class with enrich_fields() satisfies the runtime-checkable Protocol."""
    from semql.model import MultiFieldEnricher

    class MultiEnricher:
        def __call__(self, ctx: object) -> list[str]:
            return ["x"]

        def enrich_fields(self, ids: list[str], ctx: object) -> dict[str, dict[str, str]]:
            return {i: {"name": i} for i in ids}

    assert isinstance(MultiEnricher(), MultiFieldEnricher)


def test_enrich_result_adds_one_column_per_field() -> None:
    """A MultiFieldEnricher attaches a ``<dim>__<field>`` column per field."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    class RegionEnricher:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return ["r1", "r2"]

        def enrich_fields(
            self, ids: list[str], ctx: ResolutionContext
        ) -> dict[str, dict[str, str]]:
            data = {
                "r1": {"name": "EMEA", "manager": "Alice", "currency": "EUR"},
                "r2": {"name": "APAC", "manager": "Bob", "currency": "JPY"},
            }
            return {i: data[i] for i in ids if i in data}

    lk = Lookup(dimension="orders.region_id", enricher=RegionEnricher())
    rows: list[dict[str, object]] = [
        {"region_id": "r1", "revenue": 100},
        {"region_id": "r2", "revenue": 200},
    ]
    out = enrich_result(rows, "region_id", lk, ResolutionContext())
    assert out[0]["region_id__name"] == "EMEA"
    assert out[0]["region_id__manager"] == "Alice"
    assert out[0]["region_id__currency"] == "EUR"
    assert out[1]["region_id__name"] == "APAC"
    # The single-label column is NOT added on the multi-field path.
    assert "region_id__label" not in out[0]


def test_enrich_result_multi_field_omits_unknown_ids_and_missing_fields() -> None:
    """An unknown id adds no columns; a field absent for a known id is omitted."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    class PartialEnricher:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return []

        def enrich_fields(
            self, ids: list[str], ctx: ResolutionContext
        ) -> dict[str, dict[str, str]]:
            # r1 known (only "name"); r9 unknown.
            return {"r1": {"name": "EMEA"}} if "r1" in ids else {}

    lk = Lookup(dimension="orders.region_id", enricher=PartialEnricher())
    rows: list[dict[str, object]] = [
        {"region_id": "r1"},
        {"region_id": "r9"},
    ]
    out = enrich_result(rows, "region_id", lk, ResolutionContext())
    assert out[0]["region_id__name"] == "EMEA"
    assert "region_id__manager" not in out[0]  # field absent → omitted
    assert "region_id__name" not in out[1]  # unknown id → no columns


def test_enrich_result_prefers_multi_field_when_both_protocols_present() -> None:
    """A loader implementing both enrich and enrich_fields uses multi-field."""
    from semql.lookups import enrich_result
    from semql.model import Lookup, ResolutionContext

    class BothEnricher:
        def __call__(self, ctx: ResolutionContext) -> list[str]:
            return []

        def enrich(self, ids: list[str], ctx: ResolutionContext) -> dict[str, str]:
            return {i: f"label:{i}" for i in ids}

        def enrich_fields(
            self, ids: list[str], ctx: ResolutionContext
        ) -> dict[str, dict[str, str]]:
            return {i: {"name": f"name:{i}"} for i in ids}

    lk = Lookup(dimension="orders.region_id", enricher=BothEnricher())
    rows: list[dict[str, object]] = [{"region_id": "r1"}]
    out = enrich_result(rows, "region_id", lk, ResolutionContext())
    assert out[0]["region_id__name"] == "name:r1"
    assert "region_id__label" not in out[0]


# ---------------------------------------------------------------------------
# sql_enricher — declarative "SELECT from reference table" enricher
# ---------------------------------------------------------------------------


def test_sql_enricher_builds_parameterized_in_query() -> None:
    """enrich_fields issues SELECT ... WHERE key IN (?, ?) with the ids bound
    as params (never interpolated) and maps fields to <dim>__<field>."""
    from semql.lookups import enrich_result, sql_enricher
    from semql.model import Lookup, ResolutionContext

    calls: list[tuple[str, list[object]]] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        calls.append((sql, params))
        return [
            {"id": "r1", "name": "EMEA", "manager": "Alice"},
            {"id": "r2", "name": "APAC", "manager": "Bob"},
        ]

    lk = Lookup(
        dimension="orders.region_id",
        enricher=sql_enricher(
            table="regions", key="id", fields=["name", "manager"], execute=execute
        ),
    )
    rows: list[dict[str, object]] = [{"region_id": "r1"}, {"region_id": "r2"}]
    out = enrich_result(rows, "region_id", lk, ResolutionContext())

    sql, params = calls[-1]
    assert "FROM regions" in sql
    assert "WHERE id IN (?, ?)" in sql
    assert {str(p) for p in params} == {"r1", "r2"}  # ids bound, not literal
    assert out[0]["region_id__name"] == "EMEA"
    assert out[0]["region_id__manager"] == "Alice"
    assert out[1]["region_id__name"] == "APAC"


def test_sql_enricher_substitutes_table_template_from_ctx() -> None:
    """A {placeholder} in `table` is filled from ctx.context (multi-tenant)."""
    from semql.lookups import sql_enricher
    from semql.model import ResolutionContext

    seen: list[str] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        seen.append(sql)
        return []

    enr = sql_enricher(table="{schema}.regions", key="id", fields=["name"], execute=execute)
    enr.enrich_fields(["r1"], ResolutionContext(context={"schema": "acme"}))
    assert "FROM acme.regions" in seen[-1]


def test_sql_enricher_format_paramstyle() -> None:
    """paramstyle='format' uses %s placeholders (psycopg/mysql)."""
    from semql.lookups import sql_enricher
    from semql.model import ResolutionContext

    seen: list[str] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        seen.append(sql)
        return []

    enr = sql_enricher(
        table="regions", key="id", fields=["name"], execute=execute, paramstyle="format"
    )
    enr.enrich_fields(["r1", "r2"], ResolutionContext())
    assert "IN (%s, %s)" in seen[-1]


def test_sql_enricher_default_key_cast_reproduces_identical_sql() -> None:
    """Regression: ``key_cast`` unset must emit byte-for-byte the same SQL as
    before the parameter existed — no no-op CAST wrapper for existing
    text-keyed callers."""
    from semql.lookups import sql_enricher
    from semql.model import ResolutionContext

    seen: list[str] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        seen.append(sql)
        return []

    enr = sql_enricher(table="regions", key="id", fields=["name", "manager"], execute=execute)
    enr.enrich_fields(["r1", "r2"], ResolutionContext())
    assert seen[-1] == "SELECT id, name, manager FROM regions WHERE id IN (?, ?)"
    assert "CAST" not in seen[-1]


def test_sql_enricher_key_cast_wraps_where_side_only() -> None:
    """``key_cast`` casts the key column in the WHERE clause (comparison
    against string-coerced ids), for e.g. a uuid-typed reference table PK.
    Ids remain bound as parameters, never inlined as literals."""
    from semql.lookups import sql_enricher
    from semql.model import ResolutionContext

    calls: list[tuple[str, list[object]]] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        calls.append((sql, params))
        return [{"id": "11111111-1111-1111-1111-111111111111", "name": "EMEA"}]

    enr = sql_enricher(
        table="regions",
        key="id",
        fields=["name"],
        execute=execute,
        paramstyle="format",
        key_cast="text",
    )
    ids = ["11111111-1111-1111-1111-111111111111"]
    out = enr.enrich_fields(ids, ResolutionContext())

    sql, params = calls[-1]
    assert sql == "SELECT id, name FROM regions WHERE CAST(id AS text) IN (%s)"
    # SELECT projection is not cast — only the WHERE-side comparison.
    assert "SELECT CAST" not in sql
    assert params == ids  # ids bound as parameters, not string-interpolated
    assert out[ids[0]]["name"] == "EMEA"


def test_sql_enricher_is_not_a_vocabulary_loader() -> None:
    """The enricher is post-query only — it is not callable as a vocabulary
    loader, so its reference table can never be dumped into the prompt."""
    from semql.lookups import sql_enricher

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        raise AssertionError("enricher must not fire at vocabulary/prompt time")

    enr = sql_enricher(table="regions", key="id", fields=["name"], execute=execute)
    assert not callable(enr)


# ---------------------------------------------------------------------------
# Enricher / vocabulary decoupling — the enricher never feeds the prompt
# ---------------------------------------------------------------------------


def test_lookup_accepts_enricher_only() -> None:
    """A lookup may carry only an enricher (no values, no loader) — the
    canonical id→label case where the planner must never see the keyspace."""
    from semql.lookups import sql_enricher

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        return []

    lk = Lookup(
        dimension="orders.customer_id",
        enricher=sql_enricher(table="users", key="id", fields=["name"], execute=execute),
    )
    assert lk.values is None
    assert lk.loader is None
    assert lk.enricher is not None


def test_lookup_accepts_vocabulary_and_enricher_together() -> None:
    """``values`` (vocabulary) and ``enricher`` (post-query) are orthogonal."""
    from semql.lookups import sql_enricher

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        return []

    lk = Lookup(
        dimension="orders.region",
        values=("EMEA", "APAC"),
        enricher=sql_enricher(table="regions", key="id", fields=["name"], execute=execute),
    )
    assert lk.values == ("EMEA", "APAC")
    assert lk.enricher is not None


def test_enricher_only_lookup_renders_no_prompt_line() -> None:
    """The headline guarantee: an enricher-only lookup contributes NOTHING
    to the planner prompt — no ids, no labels, not even a tool hint."""
    from semql.lookups import sql_enricher

    fired = False

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        nonlocal fired
        fired = True
        return [{"id": "u1", "name": "Nikhil"}]

    cat = Catalog(
        [
            Cube(
                name="orders",
                dialect=Dialect.POSTGRES,
                table="orders",
                alias="o",
                measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
                dimensions=[Dimension(name="customer_id", sql="{o}.customer_id", type="string")],
            )
        ],
        lookups=[
            Lookup(
                dimension="orders.customer_id",
                enricher=sql_enricher(table="users", key="id", fields=["name"], execute=execute),
            )
        ],
    )
    out = planner_prompt(cat)
    assert "Lookup" not in out
    assert "customer_id" in out  # the dimension still appears, just no Lookup line
    assert not fired  # the enricher never ran at prompt-render time


def test_materialize_ignores_enricher() -> None:
    """``materialize`` reads only the vocabulary (values/loader); an
    enricher-only lookup materialises to ``None``."""
    from semql.lookups import materialize, sql_enricher

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        raise AssertionError("materialize must not touch the enricher")

    lk = Lookup(
        dimension="orders.customer_id",
        enricher=sql_enricher(table="users", key="id", fields=["name"], execute=execute),
    )
    assert materialize(lk, ResolutionContext()) is None


# ---------------------------------------------------------------------------
# enrich_all — apply every catalog lookup in one call
# ---------------------------------------------------------------------------


def test_enrich_all_applies_matching_lookups() -> None:
    """enrich_all walks catalog.lookups and enriches columns present in rows."""
    from semql import Catalog, Cube, Dialect, Dimension, Measure
    from semql.lookups import enrich_all, sql_enricher
    from semql.model import Lookup, ResolutionContext

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        return [{"id": "r1", "name": "EMEA"}]

    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region_id", sql="{o}.region_id", type="string")],
    )
    catalog = Catalog(
        [cube],
        lookups=[
            Lookup(
                dimension="orders.region_id",
                enricher=sql_enricher(table="regions", key="id", fields=["name"], execute=execute),
            )
        ],
    )
    rows: list[dict[str, object]] = [{"region_id": "r1", "revenue": 100}]
    out = enrich_all(rows, catalog, ResolutionContext())
    assert out[0]["region_id__name"] == "EMEA"


def test_enrich_all_skips_lookups_not_in_result() -> None:
    """A lookup whose dimension column isn't in the rows is a no-op."""
    from semql import Catalog, Cube, Dialect, Dimension, Measure
    from semql.lookups import enrich_all, sql_enricher
    from semql.model import Lookup, ResolutionContext

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        raise AssertionError("should not be called — column absent")

    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region_id", sql="{o}.region_id", type="string")],
    )
    catalog = Catalog(
        [cube],
        lookups=[
            Lookup(
                dimension="orders.region_id",
                enricher=sql_enricher(table="regions", key="id", fields=["name"], execute=execute),
            )
        ],
    )
    rows: list[dict[str, object]] = [{"revenue": 100}]  # no region_id column
    out = enrich_all(rows, catalog, ResolutionContext())
    assert out == rows
