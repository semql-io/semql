"""Tests for dimension-value catalogs — ``Lookup`` model + Catalog wiring.

A ``Lookup`` declares the finite set of valid values for a string
dimension. Static lookups inline values in the catalog; dynamic
lookups resolve via a ``loader(ResolutionContext)`` at prompt-render
time. The compiler never fires loaders — they're a presentation-layer
concern.

Coverage here:
- Model validators (mutual exclusion of ``values`` / ``loader``,
  qualified dimension required, max_inline bounds).
- Catalog wiring: dimension reference must resolve to a real
  string-typed dimension; duplicates rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Backend,
    Catalog,
    Cube,
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


def test_lookup_requires_values_or_loader() -> None:
    with pytest.raises(ValidationError, match=r"(?i)must declare either"):
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
        backend=Backend.POSTGRES,
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
    assert "Lookup (3 values): `EMEA`, `APAC`, `NA`" in out


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
    assert "Lookup (2 values): `dyn_a`, `dyn_b`" in out
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
# E1: LookupEnricher Protocol + enrich_result helper
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

    loader = EnrichingLoader()
    assert isinstance(loader, LookupEnricher)
    lk = Lookup(dimension="orders.customer_id", loader=loader)
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

    lk = Lookup(dimension="orders.customer_id", loader=EnrichingLoader())
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

    lk = Lookup(dimension="orders.customer_id", loader=EnrichingLoader())
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

    lk = Lookup(dimension="orders.customer_id", loader=EnrichingLoader())
    rows: list[dict[str, object]] = [
        {"customer_id": "u1"},
        {"customer_id": "u1"},  # duplicate
        {"customer_id": "u2"},
        {"customer_id": None},
    ]
    enrich_result(rows, "customer_id", lk, ResolutionContext())
    assert len(seen) == 1
    assert set(seen[0]) == {"u1", "u2"}  # no duplicates, no None
