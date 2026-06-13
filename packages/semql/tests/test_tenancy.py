"""Tests for per-cube tenancy modes — SCHEMA / DISCRIMINATOR / NONE.

Tenancy is the load-bearing data-isolation primitive. ``SCHEMA`` is
the default (each tenant has its own schema; ``{tenant_schema}`` in
the cube's ``table`` is substituted via the compile-time ``context``).
``DISCRIMINATOR`` mode shares one physical table across tenants and
filters by a column the compiler wraps in a subquery so a malformed
outer ``OR`` predicate can't cross tenants. ``NONE`` is for cubes
with no tenant boundary (META reflection cubes, public lookup tables).

The compiler's invariant: the tenant value NEVER appears as a SQL
literal — it always flows through the bind closure as a parameter.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
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
# Model: TenancyMode and Cube field defaults
# ---------------------------------------------------------------------------


def test_cube_defaults_to_schema_tenancy() -> None:
    """Back-compat: existing catalogs without an explicit tenancy field
    must behave exactly as before (SCHEMA via ``{tenant_schema}``)."""
    cube = Cube(name="c", backend=Dialect.POSTGRES, table="{tenant_schema}.t", alias="c")
    assert cube.tenancy == "schema"
    assert cube.tenancy_column is None


def test_cube_accepts_explicit_discriminator_mode() -> None:
    cube = Cube(
        name="events",
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
    )
    assert cube.tenancy == "discriminator"
    assert cube.tenancy_column == "tenant_id"


def test_cube_accepts_none_for_meta_backend() -> None:
    cube = Cube(name="m", backend=Dialect.META, table="m", alias="m", tenancy="none")
    assert cube.tenancy == "none"


def test_cube_rejects_unknown_tenancy_value() -> None:
    with pytest.raises(ValidationError):
        Cube(
            name="c",
            backend=Dialect.POSTGRES,
            table="t",
            alias="c",
            tenancy="rowlevel",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Cube root validator — cross-field consistency
# ---------------------------------------------------------------------------


def test_discriminator_requires_tenancy_column() -> None:
    with pytest.raises(ValidationError, match=r"(?i)tenancy_column"):
        Cube(
            name="events",
            backend=Dialect.POSTGRES,
            table="events",
            alias="e",
            tenancy="discriminator",
        )


def test_discriminator_table_must_not_contain_tenant_schema_placeholder() -> None:
    """DISCRIMINATOR shares a physical table — including
    ``{tenant_schema}`` would be self-contradictory (schema isolation
    AND row-level filtering on the same table)."""
    with pytest.raises(ValidationError, match=r"(?i)tenant_schema"):
        Cube(
            name="events",
            backend=Dialect.POSTGRES,
            table="{tenant_schema}.events",
            alias="e",
            tenancy="discriminator",
            tenancy_column="tenant_id",
        )


def test_schema_mode_does_not_require_tenancy_column() -> None:
    """SCHEMA mode encodes isolation in the table name; no column needed."""
    cube = Cube(
        name="c",
        backend=Dialect.POSTGRES,
        table="{tenant_schema}.t",
        alias="c",
        tenancy="schema",
    )
    assert cube.tenancy_column is None


# ---------------------------------------------------------------------------
# Compiler — DISCRIMINATOR cubes get a WHERE predicate on the inner subquery
# ---------------------------------------------------------------------------


def _disc_catalog() -> Catalog:
    events = Cube(
        name="events",
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )
    return Catalog([events])


def test_discriminator_compile_wraps_source_with_tenant_predicate() -> None:
    """The compiled SQL must scope the cube to the current tenant
    inside a subquery — outer predicates can't accidentally leak
    cross-tenant rows."""
    out = _disc_catalog().compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        context={"tenant": "acme"},
    )
    # The cube's table is wrapped in a subquery (with alias preserved)
    # that carries the tenancy predicate.
    assert "tenant_id" in out.sql
    # The tenant value rides as a bound parameter, never a literal.
    assert "acme" in out.params.values()
    assert "'acme'" not in out.sql


def test_discriminator_without_tenant_context_rejects() -> None:
    """Missing ``context['tenant']`` (or equivalent) for a DISCRIMINATOR
    cube must fail at compile time — not produce a tenant-less query."""
    with pytest.raises(CompileError, match=r"(?i)tenant"):
        _disc_catalog().compile(
            SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
            # no context — tenant not supplied
        )


def test_discriminator_tenant_value_is_bound_parameter() -> None:
    """The tenant value must NEVER appear as a SQL literal — the planner /
    MCP layer would be a SQL-injection vector if it did."""
    out = _disc_catalog().compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        context={"tenant": "robert'); DROP TABLE events;--"},
    )
    # The exploitation string lives in params, not in SQL.
    assert any("DROP TABLE" in str(v) for v in out.params.values())
    assert "DROP TABLE" not in out.sql


# ---------------------------------------------------------------------------
# Compiler — SCHEMA mode unchanged (back-compat)
# ---------------------------------------------------------------------------


def test_schema_mode_substitutes_tenant_schema_via_context() -> None:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{tenant_schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    out = Catalog([cube]).compile(
        SemanticQuery(measures=["orders.count"], dimensions=["orders.region"]),
        context={"tenant_schema": "tenant_acme"},
    )
    assert "tenant_acme.orders" in out.sql
    # No discriminator predicate — SCHEMA doesn't need one.
    assert "tenant_id" not in out.sql


# ---------------------------------------------------------------------------
# Compiler — NONE mode for META cubes (no tenancy wrapping)
# ---------------------------------------------------------------------------


def test_none_mode_meta_cube_unchanged() -> None:
    """META cubes default to NONE tenancy; their VALUES literal must
    not carry any tenancy predicate."""
    cube = Cube(
        name="public_lookup",
        backend=Dialect.POSTGRES,
        table="lookup_table",
        alias="l",
        tenancy="none",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="key", sql="{l}.key", type="string")],
    )
    out = Catalog([cube]).compile(
        SemanticQuery(measures=["public_lookup.count"], dimensions=["public_lookup.key"]),
    )
    assert "tenant_id" not in out.sql


# ---------------------------------------------------------------------------
# Cross-mode join: SCHEMA + DISCRIMINATOR in the same query
# ---------------------------------------------------------------------------


def test_cross_mode_join_carries_both_predicates() -> None:
    """A query that touches a SCHEMA cube and a DISCRIMINATOR cube must
    apply tenant isolation independently to each."""
    schema_cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{tenant_schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="cust_id", sql="{o}.cust_id", type="string")],
        joins=[
            __import__("semql").Join(
                to="events",
                relationship="many_to_one",
                on="{o}.cust_id = {e}.cust_id",
            )
        ],
    )
    disc_cube = Cube(
        name="events",
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="cust_id", sql="{e}.cust_id", type="string")],
    )
    cat = Catalog([schema_cube, disc_cube])
    out = cat.compile(
        SemanticQuery(measures=["orders.count"], dimensions=["events.cust_id"]),
        context={"tenant_schema": "tenant_acme", "tenant": "acme"},
    )
    assert "tenant_acme.orders" in out.sql
    assert "tenant_id" in out.sql  # discriminator predicate on events
