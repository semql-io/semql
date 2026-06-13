"""Full public-API-surface coverage for the ``semql`` package.

``test_package_surface.py`` pins the *core* quick-start contract; this
module pins the *whole* ``__all__`` (every exported symbol resolves, is
unique, is public) and drives a single end-to-end smoke pipeline that
actually touches the major exported entry points — so an export that
imports but is wired up wrong is caught, not just a missing name.
"""

from __future__ import annotations

import semql
from semql import (
    Backend,
    Catalog,
    CompiledQuery,
    Cube,
    Dimension,
    Measure,
    Segment,
    SemanticQuery,
    TimeDimension,
    compile_query,
    decide_visualization,
    diff_catalogs,
    estimate_cost,
    is_safe_select,
    iter_cubes,
    iter_fields,
    iter_joins,
    lint_catalog,
    parse_sql_statement,
    render_catalog_markdown,
    resolve_query,
    to_logical_plan,
    validate,
    validate_and_resolve,
)
from semql.compile import compile_plan

# ---------------------------------------------------------------------------
# __all__ hygiene
# ---------------------------------------------------------------------------


def test_every_export_resolves_and_is_non_none() -> None:
    """Every name in ``__all__`` is a real, non-None attribute."""
    missing = [n for n in semql.__all__ if not hasattr(semql, n)]
    assert not missing, f"__all__ lists names that don't resolve: {missing}"
    none_valued = [n for n in semql.__all__ if getattr(semql, n) is None]
    assert not none_valued, f"__all__ names resolve to None: {none_valued}"


def test_all_entries_unique() -> None:
    dupes = sorted({n for n in semql.__all__ if semql.__all__.count(n) > 1})
    assert not dupes, f"__all__ has duplicate entries: {dupes}"


def test_no_private_names_exported() -> None:
    private = [n for n in semql.__all__ if n.startswith("_")]
    assert not private, f"__all__ exports private names: {private}"


def test_documented_function_exports_are_callable() -> None:
    """The verb-named exports are functions, not accidentally shadowed
    by a value of another kind."""
    fn_exports = [
        "compile_query",
        "compile_federated_query",
        "to_logical_plan",
        "apply_rollup_to_plan",
        "validate",
        "validate_and_resolve",
        "resolve_query",
        "resolve_field",
        "resolve_lookup",
        "diff_catalogs",
        "estimate_cost",
        "decide_visualization",
        "lint_catalog",
        "rewrite",
        "parse_sql_statement",
        "render_catalog_markdown",
        "is_safe_select",
        "iter_cubes",
        "iter_fields",
        "iter_joins",
        "materialize_lookup",
        "enrich_result",
        "partition_scans",
        "build_default_retriever",
    ]
    not_callable = [n for n in fn_exports if not callable(getattr(semql, n))]
    assert not not_callable, f"expected callables, got non-callables: {not_callable}"
    # And each is genuinely in the public surface.
    missing = [n for n in fn_exports if n not in semql.__all__]
    assert not missing, f"function exports missing from __all__: {missing}"


# ---------------------------------------------------------------------------
# End-to-end smoke pipeline touching the major entry points
# ---------------------------------------------------------------------------


def _smoke_catalog() -> Catalog:
    orders = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        segments=[Segment(name="big", sql="{o}.amount > 1000")],
    )
    return Catalog([orders])


def test_smoke_compile_and_introspect_surface() -> None:
    cat = _smoke_catalog()
    by_name = cat.as_dict()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])

    # compile (two entry points) + the LogicalPlan path
    compiled = compile_query(q, by_name)
    assert isinstance(compiled, CompiledQuery)
    assert is_safe_select(compiled.sql)
    plan = to_logical_plan(q, by_name)
    assert compile_plan(plan, by_name).sql == compiled.sql

    # validation / resolution surface
    assert validate(q, cat) == []
    assert validate_and_resolve(q, by_name) is not None
    assert resolve_query(q, by_name) is not None

    # introspection iterators
    assert any(c.name == "orders" for c in iter_cubes(by_name))
    assert list(iter_fields(by_name["orders"]))
    list(iter_joins(by_name))  # no joins here, but must not raise


def test_smoke_downstream_surface() -> None:
    cat = _smoke_catalog()
    by_name = cat.as_dict()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    compiled = compile_query(q, by_name)

    # cost, visualization, docs
    assert estimate_cost(q, by_name) is not None
    viz = decide_visualization(q, compiled, n_rows=3, catalog=by_name)
    assert viz is not None
    assert "orders" in render_catalog_markdown(cat).lower()

    # lint + diff over the introspection layer
    assert lint_catalog(by_name) is not None
    no_change = diff_catalogs(by_name, by_name)
    assert not no_change.changes  # identical snapshots → empty diff

    # SQL parser front door
    decision = parse_sql_statement("SELECT region FROM orders", by_name)
    assert decision is not None
