"""W2 stage 2 — one output-alias convention, not two.

Review B1: the output-column collision logic existed twice —
``logical._col_alias`` (used to set ``ColumnRef.alias`` in the plan's
Project) and ``_CompileEnv._col_name`` (used by the emitter) — and both
were load-bearing in the same query. They computed the collision set on
*different* bases: the emitter counts resolved field *names*, while
``_col_alias`` counted the query's ref *locals*. They agree for ordinary
queries, but diverge under I7 input-aliases: referencing
``orders.territory`` (an alias for the ``region`` dimension) makes
``_col_alias`` count ``territory`` while the emitter counts ``region`` —
so the plan path failed to prefix a genuine collision and produced two
columns both named ``region``.

Stage 2 collapses both onto one shared helper on the resolved-field-name
basis. These tests pin: (1) the plan's project aliases are always unique
and (2) they equal ``compile_query(...).columns`` — the invariant that
makes a single source of truth observable.
"""

from __future__ import annotations

from semql.compile import compile_query
from semql.logical import to_logical_plan
from semql.model import Backend, Cube, Dimension, Measure
from semql.model import Join as ModelJoin
from semql.spec import SemanticQuery

from .conftest import CONTEXT


def _catalog(*, region_alias: list[str] | None = None) -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Backend.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="customer_id", sql="{o}.customer_id", type="number"),
            Dimension(
                name="region",
                sql="{o}.region",
                type="string",
                aliases=region_alias or [],
            ),
        ],
        joins=[
            ModelJoin(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")
        ],
    )
    customers = Cube(
        name="customers",
        alias="c",
        table="prod.customers",
        backend=Backend.POSTGRES,
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )
    return {"orders": orders, "customers": customers}


def _plan_aliases(q: SemanticQuery, catalog: dict[str, Cube]) -> list[str]:
    return [c.alias for c in to_logical_plan(q, catalog).project.columns]


def test_plan_aliases_match_compiled_columns_on_collision() -> None:
    """Two cubes' same-named dimension → both prefixed, and the plan's
    aliases equal the compiled output columns."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "customers.region"],
    )
    aliases = _plan_aliases(q, catalog)
    assert "orders_region" in aliases
    assert "customers_region" in aliases
    compiled = compile_query(q, catalog, context=CONTEXT)
    assert aliases == compiled.columns


def test_plan_aliases_unique_under_input_alias() -> None:
    """Referencing a dimension by its I7 alias must not defeat collision
    detection: the plan path must still prefix, so the two region columns
    don't both render as ``region``."""
    catalog = _catalog(region_alias=["territory"])
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.territory", "customers.region"],
    )
    aliases = _plan_aliases(q, catalog)
    assert len(set(aliases)) == len(aliases), f"duplicate output aliases: {aliases}"
    # And the plan must agree with the emitter (which already prefixes).
    compiled = compile_query(q, catalog, context=CONTEXT)
    assert aliases == compiled.columns
