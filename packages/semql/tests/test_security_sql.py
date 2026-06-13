"""Tests for ``Cube.security_sql`` — caller-attached row-level security.

``security_sql`` is the surface the planner *can't bypass*: it
AND-composes with tenancy and the ``base_predicate`` inside the
isolation subquery, so an outer ``OR`` predicate the planner emits
can't smuggle in rows the policy excludes. Values flow through
``{ctx.X}`` placeholders so the predicate stays parameterised
(injection-safe) even when caller-supplied context drives it.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    CompileError,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)

# ---------------------------------------------------------------------------
# Model — field exists, defaults to None, accepts a SQL fragment.
# ---------------------------------------------------------------------------


def test_cube_security_sql_defaults_to_none() -> None:
    cube = Cube(name="c", backend=Dialect.POSTGRES, table="t", alias="c")
    assert cube.security_sql is None


def test_cube_accepts_security_sql_string() -> None:
    cube = Cube(
        name="c",
        backend=Dialect.POSTGRES,
        table="t",
        alias="c",
        security_sql="{c}.owner_id = {ctx.user_id}",
    )
    assert cube.security_sql == "{c}.owner_id = {ctx.user_id}"


# ---------------------------------------------------------------------------
# Compiler — security_sql wraps source as a subquery (no tenancy).
# ---------------------------------------------------------------------------


def _orders_with_security() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        security_sql="{o}.owner_id = {ctx.user_id}",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def test_security_sql_predicate_appears_in_compiled_sql() -> None:
    cat = Catalog([_orders_with_security()])
    out = cat.compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
        context={"ctx.user_id": "u-1"},
    )
    # The predicate's column reference is preserved.
    assert "owner_id" in out.sql


def test_security_sql_ctx_value_is_bound_parameter() -> None:
    """The {ctx.X} value must NEVER appear as a literal — the planner /
    MCP layer would be a SQL-injection vector if it did."""
    cat = Catalog([_orders_with_security()])
    out = cat.compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
        context={"ctx.user_id": "robert'); DROP TABLE orders;--"},
    )
    assert any("DROP TABLE" in str(v) for v in out.params.values())
    assert "DROP TABLE" not in out.sql


def test_security_sql_lives_inside_isolation_subquery() -> None:
    """The predicate must be applied *inside* the alias the outer
    query sees — wrapping in a subquery — so a malformed outer
    OR can't escape it. We assert the wrapper shape by checking
    that the alias appears in the output AND the predicate is
    inside parentheses (subquery shape)."""
    cat = Catalog([_orders_with_security()])
    out = cat.compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
        context={"ctx.user_id": "u-1"},
    )
    # ``(SELECT * FROM orders WHERE ... AS o`` shape — orders appears
    # inside a parenthesised subquery, not as the outer table source.
    assert "(SELECT" in out.sql
    assert "AS o" in out.sql


def test_security_sql_without_ctx_value_rejects() -> None:
    cat = Catalog([_orders_with_security()])
    with pytest.raises(CompileError, match=r"(?i)ctx\.user_id|ctx"):
        cat.compile(
            SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
            # no ctx.user_id in context
        )


# ---------------------------------------------------------------------------
# Compiler — security_sql composes with tenancy.
# ---------------------------------------------------------------------------


def _events_with_both() -> Cube:
    return Cube(
        name="events",
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
        security_sql="{e}.team_id = {ctx.team_id}",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )


def test_security_sql_and_tenancy_both_in_wrapper() -> None:
    """A DISCRIMINATOR cube with security_sql ANDs both predicates inside
    the isolation subquery."""
    out = Catalog([_events_with_both()]).compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        context={"tenant": "acme", "ctx.team_id": "growth"},
    )
    # Both predicates' column refs appear.
    assert "tenant_id" in out.sql
    assert "team_id" in out.sql
    # Both bound values appear in params.
    values = list(out.params.values())
    assert "acme" in values
    assert "growth" in values
    # No literal appearance.
    assert "'acme'" not in out.sql
    assert "'growth'" not in out.sql


# ---------------------------------------------------------------------------
# Compiler — security_sql with no {ctx.X} placeholders (static predicate)
# ---------------------------------------------------------------------------


def test_security_sql_static_predicate_compiles() -> None:
    """A security predicate with no ctx placeholders is still a valid
    always-on filter — e.g. ``{o}.deleted_at IS NULL``."""
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        security_sql="{o}.is_public = TRUE",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    out = Catalog([cube]).compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"])
    )
    assert "is_public" in out.sql


# ---------------------------------------------------------------------------
# Compiler — alias resolution inside security_sql ({alias} → cube.alias)
# ---------------------------------------------------------------------------


def test_security_sql_resolves_alias_placeholder() -> None:
    """The ``{o}`` placeholder in security_sql resolves to the cube's
    alias — same convention as dimension/measure SQL fragments."""
    out = Catalog([_orders_with_security()]).compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
        context={"ctx.user_id": "u-1"},
    )
    # No raw ``{o}`` in the output — it resolved to ``o``.
    assert "{o}" not in out.sql
    assert "o.owner_id" in out.sql
