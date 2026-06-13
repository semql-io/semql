"""I10 — refuse silent cross-cube type coercion on federated joins.

Architecture review (2026-06) R6 promoted I10 to the correctness tier:
a cross-backend bridge join whose two keys have *different* declared
``Dimension.type`` (e.g. a ``uuid`` order key equated with a ``string``
customer id) was emitted as a plain ``a.k = b.k`` predicate, letting the
merge engine silently coerce — a refusal-over-omission violation that
can return wrong rows.

The fix refuses at compile time with a ``FederationError`` (reason
``cross_cube_type_coercion``) unless the catalog author opts in via
``Dimension.coerce_to`` — declaring the type a dimension is willing to
be compared as. A comparison is allowed when the two keys share at least
one acceptable type (own type ∪ ``coerce_to``).

Scope: this guards the *federated* bridge path, where SemQL has the join
keys as structured, typed dimensions. Same-backend joins use a raw-SQL
``on`` clause (the raw-SQL escape hatch, review B2) whose key types SemQL
can't see, so they're out of scope here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.errors import FederationError
from semql.model import Join
from semql_prompt import render_catalog_block


def _orders(*, customer_id_type: str = "uuid", coerce_to: str | None = None) -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type=customer_id_type,  # type: ignore[arg-type]
                foreign_key="customers",
                coerce_to=coerce_to,  # type: ignore[arg-type]
            ),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(*, id_type: str = "string") -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type=id_type),  # type: ignore[arg-type]
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


def _query() -> SemanticQuery:
    # Touches both cubes -> forces the bridge join on customer_id = id.
    return SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


def test_mismatched_bridge_key_types_refused() -> None:
    """uuid order key vs string customer id -> FederationError, not a
    silently-coercing join."""
    catalog = {
        "orders": _orders(customer_id_type="uuid"),
        "customers": _customers(id_type="string"),
    }
    with pytest.raises(FederationError) as ei:
        compile_federated_query(_query(), catalog)
    assert ei.value.reason == "cross_cube_type_coercion"
    # The message names both sides and points at the escape hatch.
    msg = str(ei.value)
    assert "orders.customer_id" in msg
    assert "customers.id" in msg
    assert "coerce_to" in msg


def test_number_vs_string_bridge_refused() -> None:
    catalog = {
        "orders": _orders(customer_id_type="number"),
        "customers": _customers(id_type="string"),
    }
    with pytest.raises(FederationError) as ei:
        compile_federated_query(_query(), catalog)
    assert ei.value.reason == "cross_cube_type_coercion"


# ---------------------------------------------------------------------------
# Matching types — no refusal
# ---------------------------------------------------------------------------


def test_matching_bridge_key_types_allowed() -> None:
    """Equal types need no opt-in and must compile."""
    catalog = {
        "orders": _orders(customer_id_type="string"),
        "customers": _customers(id_type="string"),
    }
    plan = compile_federated_query(_query(), catalog)
    assert len(plan.fragments) == 2


# ---------------------------------------------------------------------------
# Escape hatch — Dimension.coerce_to
# ---------------------------------------------------------------------------


def test_coerce_to_on_one_side_allows_join() -> None:
    """A uuid key that declares coerce_to='string' may join a string key."""
    catalog = {
        "orders": _orders(customer_id_type="uuid", coerce_to="string"),
        "customers": _customers(id_type="string"),
    }
    plan = compile_federated_query(_query(), catalog)
    assert len(plan.fragments) == 2


def test_coerce_to_must_differ_from_type() -> None:
    """Declaring coerce_to equal to the dimension's own type is a
    configuration error (it coerces nothing)."""
    with pytest.raises(ValidationError, match=r"(?i)coerce_to"):
        Dimension(name="x", sql="{o}.x", type="string", coerce_to="string")


def test_coerce_to_unset_by_default() -> None:
    assert Dimension(name="x", sql="{o}.x", type="string").coerce_to is None


# ---------------------------------------------------------------------------
# Planner prompt surfacing (R6: "surfaced in the planner prompt")
# ---------------------------------------------------------------------------


def test_coerce_to_rendered_in_catalog_prompt() -> None:
    catalog = {
        "orders": _orders(customer_id_type="uuid", coerce_to="string"),
        "customers": _customers(id_type="string"),
    }
    block = render_catalog_block(catalog, only_exposed=False)
    assert "coerce_to=string" in block


def test_no_coerce_to_not_rendered() -> None:
    catalog = {"orders": _orders(customer_id_type="uuid"), "customers": _customers()}
    block = render_catalog_block(catalog, only_exposed=False)
    assert "coerce_to" not in block
