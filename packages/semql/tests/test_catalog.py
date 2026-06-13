"""Unit tests for the Catalog wrapper class."""

from __future__ import annotations

import pytest
from semql.catalog import Catalog
from semql.introspect import META_CUBES
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import SemanticQuery
from semql_prompt import planner_prompt


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )


def test_catalog_auto_appends_meta_cubes() -> None:
    cat = Catalog([_orders()])
    cube_names = set(cat.as_dict().keys())
    for meta in META_CUBES:
        assert meta.name in cube_names


def test_catalog_does_not_duplicate_meta_cubes_when_user_provides_them() -> None:
    user_cubes = [_orders(), *META_CUBES]
    cat = Catalog(user_cubes)
    cube_names = list(cat.as_dict().keys())
    # Each META cube appears exactly once.
    for meta in META_CUBES:
        assert cube_names.count(meta.name) == 1


def test_catalog_rejects_duplicate_cube_names() -> None:
    a = _orders()
    b = _orders()
    with pytest.raises(ValueError, match="duplicate"):
        Catalog([a, b])


def test_compile_refuses_query_joining_alias_colliding_cubes() -> None:
    """Two cubes sharing a SQL alias is a wrong-SQL bug only when they're
    joined in one query: the compiler emits ``FROM a AS t JOIN b AS t ON
    ...`` and every ``{t}.col`` is ambiguous. The refusal is per query
    (not catalog-wide) so variant cubes that share an alias but are never
    co-queried stay legal."""
    from semql.errors import CompileError

    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="t",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{t}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{t}.id", type="number"),
            Dimension(name="cust", sql="{t}.cust", type="number", foreign_key="customers"),
        ],
    )
    customers = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="t",  # collides with orders
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{t}.id", type="number"),
            Dimension(name="region", sql="{t}.region", type="string"),
        ],
    )
    cat = Catalog([orders, customers])  # construction stays legal
    with pytest.raises(CompileError, match="share SQL alias"):
        cat.compile(SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"]))


def test_compile_allows_alias_colliding_cubes_not_co_queried() -> None:
    """A catalog may hold variant cubes that share an alias (e.g.
    role-gated alternatives); a query touching only one of them compiles
    fine — the collision check is scoped to a query's joined cubes."""
    a = _orders()  # alias "o"
    b = _customers().model_copy(update={"alias": "o"})  # shares "o", never joined here
    cat = Catalog([a, b])
    out = cat.compile(SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]))
    assert "SUM" in out.sql


def test_catalog_rejects_unknown_join_target() -> None:
    orphan = Cube(
        name="orphan",
        backend=Dialect.POSTGRES,
        table="orphan",
        alias="x",
        joins=[Join(to="ghost", relationship="many_to_one", on="{x}.id = {g}.id")],
        dimensions=[Dimension(name="id", sql="{x}.id", type="string")],
    )
    with pytest.raises(ValueError, match="ghost"):
        Catalog([orphan])


def test_catalog_accepts_resolvable_joins() -> None:
    orders = _orders().model_copy(
        update={
            "joins": [
                Join(to="customers", relationship="many_to_one", on="{o}.cust_id = {c}.id"),
            ]
        }
    )
    cat = Catalog([orders, _customers()])
    assert "orders" in cat.as_dict()
    assert "customers" in cat.as_dict()


def test_catalog_compile_delegates() -> None:
    cat = Catalog([_orders()])
    out = cat.compile(SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]))
    assert "SUM" in out.sql
    assert "orders" in out.sql or "{o}" not in out.sql  # alias resolved


def test_catalog_prompt_returns_fragment() -> None:
    cat = Catalog([_orders()])
    prompt = planner_prompt(
        cat,
    )
    # Catalog block must include the cube and its measure/dimension.
    assert "orders" in prompt
    assert "revenue" in prompt
    assert "region" in prompt


def test_catalog_prompt_only_exposed_default() -> None:
    hidden = Cube(
        name="hidden",
        backend=Dialect.POSTGRES,
        table="hidden",
        alias="h",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{h}.x", type="string")],
    )
    cat = Catalog([_orders(), hidden])
    prompt = planner_prompt(
        cat,
    )
    assert "orders" in prompt
    # Hidden cubes don't appear by default.
    assert "### hidden" not in prompt


def test_catalog_compile_threads_context() -> None:
    schema_orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([schema_orders])
    out = cat.compile(
        SemanticQuery(measures=["orders.count"]),
        context={"schema": "prod"},
    )
    assert "prod.orders" in out.sql


def test_catalog_iter_and_len() -> None:
    """Convenience: a Catalog should report how many cubes it holds and
    let you iterate over them."""
    cat = Catalog([_orders()])
    # 1 user cube + len(META_CUBES) meta cubes.
    assert len(cat) == 1 + len(META_CUBES)
    cube_names = {c.name for c in cat}
    assert "orders" in cube_names


# ---------------------------------------------------------------------------
# Unit / display_unit validation at construction
# ---------------------------------------------------------------------------


def test_catalog_rejects_display_unit_without_unit() -> None:
    """display_unit only makes sense when paired with unit — the
    renderer has nothing to convert from otherwise."""
    bad = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="watch_time",
                sql="{o}.x",
                agg="sum",
                display_unit="hours",  # no unit set
            ),
        ],
    )
    with pytest.raises(ValueError, match="requires unit to also be set"):
        Catalog([bad])


def test_catalog_rejects_unknown_unit_pair() -> None:
    """Typo ``hour`` vs ``hours`` passes Pydantic but should fail at
    construction so the bug doesn't only surface at render time."""
    bad = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="watch_time",
                sql="{o}.duration",
                agg="sum",
                unit="seconds",
                display_unit="hour",  # typo — should be "hours"
            ),
        ],
    )
    with pytest.raises(ValueError, match="cannot convert"):
        Catalog([bad])


def test_catalog_accepts_known_unit_pair() -> None:
    """Properly registered (unit, display_unit) pair must not raise."""
    good = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="watch_time",
                sql="{o}.duration",
                agg="sum",
                unit="seconds",
                display_unit="hours",
            ),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    Catalog([good])  # no exception


def test_catalog_validation_also_runs_on_dimensions() -> None:
    """Dimensions carry the same unit fields; the same checks apply."""
    bad = Cube(
        name="events",
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        dimensions=[
            Dimension(
                name="duration",
                sql="{e}.dur",
                type="number",
                unit="seconds",
                display_unit="fortnights",  # not in registry
            ),
        ],
    )
    with pytest.raises(ValueError, match="cannot convert"):
        Catalog([bad])


def test_catalog_validation_uses_custom_registry() -> None:
    """A catalog with a custom registry should validate against THAT
    registry, not the global default."""
    from semql.units import Registry

    r = Registry()
    r.register("widgets", "megawidgets", 1.0 / 1_000_000.0)

    cube = Cube(
        name="parts",
        backend=Dialect.POSTGRES,
        table="parts",
        alias="p",
        measures=[
            Measure(
                name="produced",
                sql="{p}.count",
                agg="sum",
                unit="widgets",
                display_unit="megawidgets",
            ),
        ],
        dimensions=[Dimension(name="factory", sql="{p}.factory", type="string")],
    )
    # Without the custom registry this would fail (widgets/megawidgets
    # aren't in the default). With it, construction succeeds.
    Catalog([cube], unit_registry=r)
