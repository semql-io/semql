"""Unit tests for the semantic-layer compiler.

Pin compile contracts that callers (planners, runners, presenters) depend
on. All tests are deterministic and free of I/O — `compile_query` is pure.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql.compile import MAX_UNGROUPED_ROWS, CompiledQuery, CompileError, compile_query
from semql.introspect import quote_literal
from semql.model import Backend, Cube, Measure
from semql.spec import CompareWindow, Filter, SemanticQuery, TimeWindow
from semql.visualize import (
    BAR_MAX_BARS,
    PIE_MAX_SLICES,
    VizColumn,
    VizDecision,
    decide_visualization,
)
from semql_prompt import (
    build_planner_prompt_fragment,
    build_router_prompt_fragment,
    render_catalog_block,
)

from .conftest import CONTEXT

# ---------------------------------------------------------------------------
# Catalog invariants
# ---------------------------------------------------------------------------


def test_catalog_has_expected_cubes(catalog: dict[str, Cube]) -> None:
    assert {"orders", "customers", "products", "sessions", "restricted"}.issubset(catalog.keys())


def test_some_cubes_hidden_from_prompt(catalog: dict[str, Cube]) -> None:
    hidden = [c for c in catalog.values() if not c.expose_in_prompt]
    assert hidden
    hidden_names = {c.name for c in hidden}
    assert "customers" in hidden_names
    assert "products" in hidden_names


def test_render_catalog_block_shows_exposed_only(catalog: dict[str, Cube]) -> None:
    rendered = render_catalog_block(catalog)
    assert "### orders" in rendered
    assert "### sessions" in rendered
    assert "### customers" not in rendered
    assert "### products" not in rendered


def test_render_catalog_block_full_includes_all(catalog: dict[str, Cube]) -> None:
    rendered = render_catalog_block(catalog, only_exposed=False)
    assert "### customers" in rendered
    assert "### products" in rendered
    assert "### restricted" in rendered


# ---------------------------------------------------------------------------
# Happy-path compile — single backend, single cube
# ---------------------------------------------------------------------------


def test_compile_pg_measure_with_dimension(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.POSTGRES
    assert out.columns == ["region", "revenue"]
    assert "AS revenue" in out.sql
    assert "test_schema" in out.sql
    assert "GROUP BY region" in out.sql
    assert "o.deleted_at IS NULL" in out.sql  # base_predicate lifted to WHERE


def test_compile_pg_time_dimension_with_granularity(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.columns == ["region", "created_at_day", "revenue"]
    assert "date_trunc('day'" in out.sql
    assert "%(p0)s" in out.sql
    assert "%(p1)s" in out.sql
    assert out.params == {"p0": "2026-01-01", "p1": "2026-02-01"}
    assert "GROUP BY" in out.sql


def test_compile_pg_time_dimension_no_granularity_is_filter_only(
    catalog: dict[str, Cube],
) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.columns == ["region", "count"]
    assert "created_at_" not in out.sql  # not a SELECT/GROUP BY column
    assert "o.created_at >=" in out.sql  # but it IS in WHERE


def test_compile_ch_measure(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["sessions.duration"],
        dimensions=["sessions.app_name"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.CLICKHOUSE
    assert out.columns == ["app_name", "duration"]
    assert "s.event_type = 'active'" in out.sql  # base_predicate


def test_compile_ch_time_granularity_uses_clickhouse_trunc(
    catalog: dict[str, Cube],
) -> None:
    q = SemanticQuery(
        measures=["sessions.count"],
        time_dimension=TimeWindow(
            dimension="sessions.started_at",
            granularity="hour",
            range=("2026-01-01", "2026-01-02"),
        ),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "toStartOfHour" in out.sql


def test_compile_ungrouped_lookup(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["customers.email"],
        filters=[Filter(dimension="customers.is_active", op="eq", values=[True])],
        ungrouped=True,
        limit=20,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "GROUP BY" not in out.sql
    assert "LIMIT 20" in out.sql
    assert out.params == {"p0": True}


def test_compile_distinct_when_no_measures(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(dimensions=["orders.region"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.sql.startswith("SELECT DISTINCT")


# ---------------------------------------------------------------------------
# Multi-cube PG joins
# ---------------------------------------------------------------------------


def test_compile_join_emits_inner_join(catalog: dict[str, Cube]) -> None:
    """A default multi-cube join is an INNER JOIN (D9): ``Join.kind`` is
    now honoured at emission, and a plain join with no ``left_joins`` is
    stamped ``kind="inner"``. The LEFT-JOIN spine is opt-in via
    ``left_joins`` (see test_left_joins)."""
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["customers.name"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.POSTGRES
    assert "INNER JOIN" in out.sql.upper()
    assert "LEFT JOIN" not in out.sql.upper()
    assert "c.id" in out.sql


def test_column_collision_prefixes_with_cube_name(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["customers.name", "products.name"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "customers_name" in out.columns
    assert "products_name" in out.columns
    assert "name" not in out.columns  # bare form must be absent


def test_compile_having_on_measure(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="revenue", op="gt", values=[100])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "HAVING" in out.sql
    assert "SUM" in out.sql  # HAVING repeats the aggregate


# ---------------------------------------------------------------------------
# Parameter binding & filter ops
# ---------------------------------------------------------------------------


def test_filter_in_op_emits_one_param_per_value(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.status", op="in", values=["paid", "pending"])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "IN (%(p0)s, %(p1)s)" in out.sql
    assert out.params == {"p0": "paid", "p1": "pending"}


def test_filter_is_null_no_param(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["customers.email"],
        filters=[Filter(dimension="customers.name", op="is_null")],
        ungrouped=True,
        limit=10,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "IS NULL" in out.sql
    assert out.params == {}


def test_filter_contains_pg_uses_ilike(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["customers.email"],
        filters=[Filter(dimension="customers.email", op="contains", values=["@acme.com"])],
        ungrouped=True,
        limit=10,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "ILIKE" in out.sql
    assert out.params == {"p0": "%@acme.com%"}


def test_filter_contains_ch_uses_position(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["sessions.count"],
        filters=[Filter(dimension="sessions.app_name", op="contains", values=["chrome"])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "positionCaseInsensitive" in out.sql


def test_filter_numeric_rejects_string(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.amount", op="eq", values=["not-a-number"])],
    )
    with pytest.raises(CompileError, match="non-numeric"):
        compile_query(q, catalog, context=CONTEXT)


def test_filter_bool_rejects_string(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["customers.email"],
        filters=[Filter(dimension="customers.is_active", op="eq", values=["true"])],
        ungrouped=True,
        limit=10,
    )
    with pytest.raises(CompileError, match="non-bool"):
        compile_query(q, catalog, context=CONTEXT)


def test_filter_bool_rejects_int(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=[1])],
    )
    with pytest.raises(CompileError, match="non-bool"):
        compile_query(q, catalog, context=CONTEXT)


def test_filter_time_rejects_non_iso(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.created_at", op="gt", values=["last tuesday"])],
    )
    with pytest.raises(CompileError, match="non-ISO-8601"):
        compile_query(q, catalog, context=CONTEXT)


def test_filter_time_accepts_iso(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.created_at", op="gt", values=["2026-01-01T00:00:00"])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "o.created_at" in out.sql


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_cube_names_it(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["nope.thing"])
    with pytest.raises(CompileError, match="Unknown cube: 'nope'"):
        compile_query(q, catalog, context=CONTEXT)


def test_unknown_field_names_it(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.no_such_metric"])
    with pytest.raises(CompileError, match="Unknown field 'no_such_metric'"):
        compile_query(q, catalog, context=CONTEXT)


def test_empty_query_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery()
    with pytest.raises(CompileError, match="empty"):
        compile_query(q, catalog, context=CONTEXT)


def test_cross_backend_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["sessions.app_name"],
    )
    with pytest.raises(CompileError, match="Cross-backend"):
        compile_query(q, catalog, context=CONTEXT)


def test_compare_window_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"], compare=CompareWindow())
    with pytest.raises(CompileError, match="compare"):
        compile_query(q, catalog, context=CONTEXT)


def test_ungrouped_without_limit_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(dimensions=["customers.email"], ungrouped=True)
    with pytest.raises(CompileError, match="Ungrouped"):
        compile_query(q, catalog, context=CONTEXT)


def test_ungrouped_over_cap_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["customers.email"],
        ungrouped=True,
        limit=MAX_UNGROUPED_ROWS + 1,
    )
    with pytest.raises(CompileError, match="Ungrouped"):
        compile_query(q, catalog, context=CONTEXT)


def test_having_on_non_measure_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        having=[Filter(dimension="something_else", op="gt", values=[1])],
    )
    with pytest.raises(CompileError, match="HAVING"):
        compile_query(q, catalog, context=CONTEXT)


def test_join_path_not_found_names_endpoints(catalog: dict[str, Cube]) -> None:
    # restricted (PG) has no join to orders (PG) — same backend, no path
    q = SemanticQuery(
        measures=["restricted.count"],
        dimensions=["orders.region"],
        filters=[
            Filter(dimension="restricted.flag_type", op="eq", values=["x"]),
        ],
    )
    with pytest.raises(CompileError, match="No join path"):
        compile_query(q, catalog, context=CONTEXT)


def test_granularity_not_in_allowed_list_rejected(catalog: dict[str, Cube]) -> None:
    # orders.created_at only allows day/week/month — not hour
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="hour",
            range=("2026-01-01", "2026-01-02"),
        ),
    )
    with pytest.raises(CompileError, match="Granularity"):
        compile_query(q, catalog, context=CONTEXT)


def test_required_filter_missing_rejected(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["restricted.count"])
    with pytest.raises(CompileError, match="flag_type"):
        compile_query(q, catalog, context=CONTEXT)


def test_required_filter_present_compiles(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["restricted.count"],
        filters=[Filter(dimension="restricted.flag_type", op="eq", values=["fraud"])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.POSTGRES


def test_ungrouped_with_measures_rejected_at_spec_level(catalog: dict[str, Cube]) -> None:
    with pytest.raises(ValidationError, match="ungrouped=True is incompatible"):
        SemanticQuery(measures=["orders.count"], ungrouped=True, limit=10)


def test_unknown_placeholder_raises(catalog: dict[str, Cube]) -> None:
    bad_cube = catalog["orders"].model_copy(
        update={
            "measures": [
                *catalog["orders"].measures,
                Measure(name="bogus", sql="{nope}.id", agg="count", unit="count"),
            ]
        }
    )
    patched = {**catalog, "orders": bad_cube}
    q = SemanticQuery(measures=["orders.bogus"])
    with pytest.raises(CompileError, match="Unknown placeholder.*nope"):
        compile_query(q, patched, context=CONTEXT)


# ---------------------------------------------------------------------------
# Context substitution
# ---------------------------------------------------------------------------


def test_context_substituted_in_table_name(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"])
    out = compile_query(q, catalog, context={"schema": "acme_corp"})
    assert "acme_corp.orders" in out.sql


def test_alias_placeholders_resolve_in_emitted_sql(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert "{o}" not in out.sql
    assert "o.amount" in out.sql
    assert "o.region" in out.sql


# ---------------------------------------------------------------------------
# group_by_alias / having_alias kwargs
# ---------------------------------------------------------------------------


def test_group_by_alias_default(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "GROUP BY region" in out.sql


def test_group_by_alias_false_repeats_expression(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region"],
    )
    out = compile_query(q, catalog, context=CONTEXT, group_by_alias=False)
    assert "GROUP BY o.region" in out.sql


def test_having_alias_false_repeats_aggregate(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="revenue", op="gt", values=[100])],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "HAVING SUM(" in out.sql


def test_having_alias_true_references_name(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="revenue", op="gt", values=[100])],
    )
    out = compile_query(q, catalog, context=CONTEXT, having_alias=True)
    assert "HAVING revenue" in out.sql
    assert "HAVING SUM" not in out.sql


# ---------------------------------------------------------------------------
# Introspection (Backend.META) cubes
# ---------------------------------------------------------------------------


def test_meta_cubes_registered_but_hidden(catalog: dict[str, Cube]) -> None:
    for name in ("catalog_cubes", "catalog_measures", "catalog_dimensions"):
        assert name in catalog
        assert catalog[name].backend is Backend.META
        assert catalog[name].expose_in_prompt is False


def test_compile_meta_cubes_lists_self(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["catalog_cubes.name", "catalog_cubes.backend"],
        ungrouped=True,
        limit=100,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.META
    assert "'orders'" in out.sql
    assert "'catalog_cubes'" in out.sql
    assert "test_schema" not in out.sql  # META emits no tenant tables


def test_compile_meta_measures_includes_known_measures(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["catalog_measures.cube", "catalog_measures.name"],
        ungrouped=True,
        limit=1000,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "'revenue'" in out.sql
    assert "'duration'" in out.sql


def test_meta_backend_uses_pg_param_style(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        dimensions=["catalog_measures.cube", "catalog_measures.name"],
        filters=[Filter(dimension="catalog_measures.unit", op="eq", values=["count"])],
        ungrouped=True,
        limit=10,
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.META
    assert "%(p0)s" in out.sql
    assert "{p0:" not in out.sql


def test_meta_value_escapes_apostrophes() -> None:
    assert quote_literal("it's") == "'it''s'"
    assert quote_literal(None) == "NULL"
    assert quote_literal("plain") == "'plain'"


def test_meta_cube_count_aggregates(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["catalog_cubes.count"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.backend is Backend.META
    assert "COUNT(*)" in out.sql


# ---------------------------------------------------------------------------
# Prompt fragment builders
# ---------------------------------------------------------------------------


def test_planner_fragment_contains_spec_and_catalog(catalog: dict[str, Cube]) -> None:
    fragment = build_planner_prompt_fragment(catalog)
    assert "SemanticQuery" in fragment
    assert "### orders" in fragment
    assert "raw SQL" in fragment
    assert "### customers" not in fragment  # hidden by default


def test_planner_fragment_with_introspection(catalog: dict[str, Cube]) -> None:
    fragment = build_planner_prompt_fragment(catalog, include_introspection=True)
    assert "catalog_cubes" in fragment
    assert "catalog_measures" in fragment


def test_planner_fragment_surfaces_required_filters(catalog: dict[str, Cube]) -> None:
    fragment = build_planner_prompt_fragment(catalog, only_exposed=False)
    assert "Required filters" in fragment
    assert "`restricted.flag_type`" in fragment


def test_router_fragment_lists_topics(catalog: dict[str, Cube]) -> None:
    fragment = build_router_prompt_fragment(catalog)
    assert "`orders`" in fragment
    assert "`sessions`" in fragment
    assert "`customers`" not in fragment  # hidden
    assert "`catalog_cubes`" not in fragment


def test_router_fragment_can_omit_topics(catalog: dict[str, Cube]) -> None:
    fragment = build_router_prompt_fragment(catalog, include_topic_summary=False)
    assert "Catalog topics" not in fragment
    assert "`orders`" not in fragment
    assert "semantic" in fragment  # routing criteria still present


# ---------------------------------------------------------------------------
# Visualization decisions
# ---------------------------------------------------------------------------


def _compile_and_decide(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    n_rows: int = 1,
) -> tuple[CompiledQuery, VizDecision]:
    out = compile_query(q, catalog, context=CONTEXT)
    viz = decide_visualization(q, out, n_rows=n_rows, catalog=catalog)
    return out, viz


def test_decide_single_value_is_text_only(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue"])
    _, viz = _compile_and_decide(q, catalog)
    assert viz.chart_type == "text_only"


def test_decide_time_series_picks_line_chart(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    _, viz = _compile_and_decide(q, catalog)
    assert viz.chart_type == "line_chart"
    assert viz.x_axis == "Created At"
    assert viz.y_axes == ["Revenue"]


def test_decide_time_dim_without_granularity_is_text_only(
    catalog: dict[str, Cube],
) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    _, viz = _compile_and_decide(q, catalog)
    assert viz.chart_type == "text_only"


def test_decide_small_breakdown_picks_pie(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"], dimensions=["orders.status"])
    _, viz = _compile_and_decide(q, catalog, n_rows=3)
    assert viz.chart_type == "pie_chart"


def test_decide_medium_breakdown_picks_bar(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"], dimensions=["orders.region"])
    _, viz = _compile_and_decide(q, catalog, n_rows=PIE_MAX_SLICES + 5)
    assert viz.chart_type == "bar_chart"


def test_decide_many_rows_falls_back_to_table(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"], dimensions=["orders.region"])
    _, viz = _compile_and_decide(q, catalog, n_rows=BAR_MAX_BARS + 100)
    assert viz.chart_type == "data_table"


def test_decide_ungrouped_is_always_table(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(dimensions=["customers.email"], ungrouped=True, limit=50)
    _, viz = _compile_and_decide(q, catalog)
    assert viz.chart_type == "data_table"


def test_decide_respects_cube_default_chart_type(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["catalog_cubes.count"])
    _, viz = _compile_and_decide(q, catalog)
    assert viz.chart_type == "data_table"


def test_decide_row_count_changes_chart_type(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.count"], dimensions=["orders.status"])
    _, viz_small = _compile_and_decide(q, catalog, n_rows=3)
    assert viz_small.chart_type == "pie_chart"
    _, viz_huge = _compile_and_decide(q, catalog, n_rows=BAR_MAX_BARS + 1)
    assert viz_huge.chart_type == "data_table"


def test_decide_columns_aligned_with_compiled_output(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    out, viz = _compile_and_decide(q, catalog)
    assert [c.name for c in viz.columns] == out.columns


def test_decide_measure_column_is_measure_flagged(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    _, viz = _compile_and_decide(q, catalog)
    measure_cols = [c for c in viz.columns if c.is_measure]
    assert len(measure_cols) == 1
    assert measure_cols[0].name == "revenue"


def test_decide_rejects_measure_in_dimensions_list(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(dimensions=["orders.revenue"])  # revenue is a measure
    # The rejection now happens in compile_query — decide_visualization
    # only sees a fully-compiled bundle, so the type check lives upstream.
    with pytest.raises(CompileError, match="not a dimension"):
        compile_query(q, catalog, context=CONTEXT)


def test_decide_multi_measure_y_axes(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
    )
    _, viz = _compile_and_decide(q, catalog, n_rows=5)
    assert viz.chart_type == "bar_chart"
    assert len(viz.y_axes) == 2


def test_decide_unknown_column_gets_humanised_label(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="week",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out, viz = _compile_and_decide(q, catalog)
    assert "created_at_week" in out.columns
    time_col: VizColumn = next(c for c in viz.columns if c.is_time)
    assert time_col.name == "created_at_week"
    assert time_col.display_name == "Created At"


# ---------------------------------------------------------------------------
# ORDER BY referencing a non-SELECT cube.field — intentional fallthrough.
# ---------------------------------------------------------------------------


def test_order_by_unprojected_cube_field_emits_resolved_expression(
    catalog: dict[str, Cube],
) -> None:
    """ORDER BY can reference any known ``cube.field``, not just the
    output columns. SQL allows ordering by an expression you didn't
    SELECT (when there's no GROUP BY conflict), and that's useful for
    things like ``ORDER BY orders.created_at DESC`` while selecting
    only the count. The compiler resolves the reference and emits the
    SQL expression directly."""
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region"],
        order=[("orders.amount", "desc")],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    assert "ORDER BY" in out.sql.upper()
    # Resolved expression, not an output-column alias.
    assert "o.amount" in out.sql


def test_order_by_output_column_alias_still_works(
    catalog: dict[str, Cube],
) -> None:
    """The other branch — ORDER BY references an output column alias —
    keeps working unchanged."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        order=[("revenue", "desc"), ("region", "asc")],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    # sqlglot's PG renderer may add ``NULLS LAST`` to the DESC clause
    # — match the structure, not the verbatim string.
    assert "ORDER BY revenue DESC" in out.sql
    assert "region ASC" in out.sql


def test_order_by_unknown_cube_field_rejected(catalog: dict[str, Cube]) -> None:
    """An unknown identifier still raises — the fallthrough goes
    through the resolver, which surfaces a precise error."""
    q = SemanticQuery(
        measures=["orders.count"],
        order=[("orders.no_such_field", "asc")],
    )
    with pytest.raises(CompileError, match=r"(?i)order by"):
        compile_query(q, catalog, context=CONTEXT)


# ---------------------------------------------------------------------------
# CompiledQuery.column_meta — per-output-column type + presentation metadata
# ---------------------------------------------------------------------------


def test_column_meta_populated_for_each_output_column(catalog: dict[str, Cube]) -> None:
    """``column_meta`` lines up 1:1 with ``columns``; same length, same
    order. Downstream renderers iterate them together."""
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert [m.name for m in out.column_meta] == out.columns
    assert len(out.column_meta) == 2


def test_column_meta_classifies_measure_vs_dimension(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    out = compile_query(q, catalog, context=CONTEXT)
    by_name = {m.name: m for m in out.column_meta}
    assert by_name["revenue"].kind == "measure"
    assert by_name["region"].kind == "dimension"


def test_column_meta_propagates_measure_unit(catalog: dict[str, Cube]) -> None:
    """Whatever ``unit`` the catalog's Measure declared rides through
    to ``column_meta`` — no re-resolution against the catalog needed."""
    q = SemanticQuery(measures=["sessions.duration"], dimensions=["sessions.app_name"])
    out = compile_query(q, catalog, context=CONTEXT)
    dur = next(m for m in out.column_meta if m.name == "duration")
    assert dur.kind == "measure"
    assert dur.unit == "duration"


def test_column_meta_marks_time_dimension(catalog: dict[str, Cube]) -> None:
    """Time columns get ``kind="time"`` so a renderer can pick an
    x-axis without re-deriving the granularity."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    time_col = next(m for m in out.column_meta if m.kind == "time")
    assert time_col.name == "created_at_day"


def test_column_meta_compare_mode_marks_pct_change_as_percent(
    catalog: dict[str, Cube],
) -> None:
    """In compare mode, ``foo_current``/``foo_prior``/``foo_delta``
    keep the measure's unit; ``foo_pct_change`` is dimensionless and
    always reads as ``"percent"``."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-02-01", "2026-03-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    by_name = {m.name: m for m in out.column_meta}
    # All four derived columns are present.
    for suffix in ("_current", "_prior", "_delta", "_pct_change"):
        assert f"revenue{suffix}" in by_name
    # Current/prior/delta inherit the measure's unit (currency).
    assert by_name["revenue_current"].unit == "currency"
    assert by_name["revenue_prior"].unit == "currency"
    assert by_name["revenue_delta"].unit == "currency"
    # pct_change is dimensionless.
    assert by_name["revenue_pct_change"].format == "percent"
    assert by_name["revenue_pct_change"].unit is None


# ---------------------------------------------------------------------------
# storage_type — tightest resolved type per output column
# ---------------------------------------------------------------------------


def test_column_meta_storage_type_for_dimension(catalog: dict[str, Cube]) -> None:
    """Dimensions pass their declared ``type`` straight through. Lets a
    downstream caller (visualiser, MCP server, type-aware UI) know what
    cell-level rendering each column wants."""
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region", "orders.amount", "orders.is_paid"],
    )
    out = compile_query(q, catalog, context=CONTEXT)
    by_name = {m.name: m for m in out.column_meta}
    assert by_name["region"].storage_type == "string"
    assert by_name["amount"].storage_type == "number"
    assert by_name["is_paid"].storage_type == "bool"


def test_column_meta_storage_type_for_measure_count(catalog: dict[str, Cube]) -> None:
    """``count`` always produces a non-negative integer regardless of the
    underlying column — tighten the type so the LLM doesn't quote it."""
    q = SemanticQuery(measures=["orders.count"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.column_meta[0].storage_type == "integer"


def test_column_meta_storage_type_for_measure_sum(catalog: dict[str, Cube]) -> None:
    """``sum`` is int-or-float depending on the source — we don't parse
    the SQL to tell, so it stays the generic ``"number"`` literal."""
    q = SemanticQuery(measures=["orders.revenue"])
    out = compile_query(q, catalog, context=CONTEXT)
    assert out.column_meta[0].storage_type == "number"


def test_column_meta_storage_type_for_time(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    time_col = next(m for m in out.column_meta if m.kind == "time")
    assert time_col.storage_type == "time"


def test_column_meta_storage_type_compare_pct_change_is_float(
    catalog: dict[str, Cube],
) -> None:
    """pct_change is always a ratio — float, regardless of the measure's
    underlying integer/numeric storage."""
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-02-01", "2026-03-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    pct = next(m for m in out.column_meta if m.name.endswith("_pct_change"))
    assert pct.storage_type == "float"


def test_column_meta_storage_type_compare_delta_inherits_measure(
    catalog: dict[str, Cube],
) -> None:
    """current / prior / delta all carry the underlying measure's
    storage_type (here ``"integer"`` from a count)."""
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-02-01", "2026-03-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    out = compile_query(q, catalog, context=CONTEXT)
    by_name = {m.name: m for m in out.column_meta}
    for suffix in ("_current", "_prior", "_delta"):
        assert by_name[f"count{suffix}"].storage_type == "integer"


# ---------------------------------------------------------------------------
# FilterTypeError — now surfaces during field resolution, not at SQL emit
# ---------------------------------------------------------------------------


def test_filter_type_error_fires_during_resolution(catalog: dict[str, Cube]) -> None:
    """A type-mismatched filter (string dim, int value) raises before
    SQL composition begins — same error class as before, but it now
    surfaces alongside resolution failures so an LLM planner can fix
    both classes of problem in one round-trip."""
    from semql import FilterTypeError
    from semql.spec import Filter

    q = SemanticQuery(
        measures=["orders.count"],
        # 123 is an int; ``region`` is a string dim.
        filters=[Filter(dimension="orders.region", op="eq", values=[123])],
    )
    with pytest.raises(FilterTypeError):
        compile_query(q, catalog, context=CONTEXT)


def test_filter_type_error_in_where_tree(catalog: dict[str, Cube]) -> None:
    """The relocation covers ``where``-tree leaves the same way it
    covers the flat ``filters`` list."""
    from semql import FilterTypeError
    from semql.spec import BoolExpr, Filter

    q = SemanticQuery(
        measures=["orders.count"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.amount", op="eq", values=["not-a-number"]),
            ],
        ),
    )
    with pytest.raises(FilterTypeError):
        compile_query(q, catalog, context=CONTEXT)


# ---------------------------------------------------------------------------
# Resolution-error accumulation — surface every wrong reference in one shot
# ---------------------------------------------------------------------------


def test_resolve_accumulates_multiple_unknown_fields(catalog: dict[str, Cube]) -> None:
    """Three wrong references → one CompileError mentioning all three.
    Saves the LLM planner two retry round-trips versus a serial
    fail-fast contract."""
    q = SemanticQuery(
        measures=["orders.revvenue", "orders.kount"],  # both typos
        dimensions=["orders.regionn"],
    )
    with pytest.raises(CompileError) as exc_info:
        compile_query(q, catalog, context=CONTEXT)
    msg = str(exc_info.value)
    assert "revvenue" in msg
    assert "kount" in msg
    assert "regionn" in msg


def test_resolve_accumulates_unknown_with_filter_type_mismatch(
    catalog: dict[str, Cube],
) -> None:
    """Mixed bag — one unknown dim plus one type-mismatched filter both
    surface in the same exception. Combined-mode message lists each."""
    from semql.spec import Filter

    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.bogus_dim"],
        filters=[Filter(dimension="orders.region", op="eq", values=[42])],  # int into string dim
    )
    with pytest.raises(CompileError) as exc_info:
        compile_query(q, catalog, context=CONTEXT)
    msg = str(exc_info.value)
    assert "bogus_dim" in msg
    # The type-mismatch message says "non-string value 42" — both must surface.
    assert "42" in msg


def test_resolve_single_error_re_raises_original_class(
    catalog: dict[str, Cube],
) -> None:
    """When only one resolution error fires, the original exception
    class propagates verbatim — so callers branching on ``FilterTypeError``
    (UIs that highlight a specific row) still see the typed instance,
    not a generic ``CompileError`` wrapper."""
    from semql import FilterTypeError
    from semql.spec import Filter

    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="eq", values=[99])],
    )
    with pytest.raises(FilterTypeError):
        compile_query(q, catalog, context=CONTEXT)
