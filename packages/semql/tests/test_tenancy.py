"""Tests for per-cube tenancy modes — SCHEMA / DISCRIMINATOR / NONE.

Tenancy is the load-bearing data-isolation primitive. ``NONE`` is the
default — a cube that says nothing about tenancy is honestly unscoped
rather than silently inert. ``SCHEMA`` gives each tenant its own schema
(``{tenant_schema}`` in the cube's source is substituted from the
identity's ``tenant`` / the compile-time ``context``) and now *requires*
the placeholder. ``DISCRIMINATOR`` shares one physical table across
tenants and filters by one or more ``tenancy_columns`` the compiler
wraps in a subquery so a malformed outer ``OR`` can't cross tenants.

The compiler's invariant: the tenant value NEVER appears as a SQL
literal — it always flows through the bind closure as a parameter.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    AuthContext,
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


def test_cube_defaults_to_none_tenancy() -> None:
    """A cube that says nothing about tenancy now defaults to ``none`` —
    honestly unscoped — rather than the old silently-inert ``schema``
    default that emitted no isolation unless the table happened to
    contain ``{tenant_schema}``."""
    cube = Cube(name="c", dialect=Dialect.POSTGRES, table="t", alias="c")
    assert cube.tenancy == "none"
    assert cube.tenancy_columns == []


def test_cube_accepts_explicit_discriminator_mode() -> None:
    cube = Cube(
        name="events",
        dialect=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["tenant_id"],
    )
    assert cube.tenancy == "discriminator"
    assert cube.tenancy_columns == ["tenant_id"]


def test_cube_accepts_none_for_meta_backend() -> None:
    cube = Cube(name="m", dialect=Dialect.META, table="m", alias="m", tenancy="none")
    assert cube.tenancy == "none"


def test_cube_rejects_unknown_tenancy_value() -> None:
    with pytest.raises(ValidationError):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="t",
            alias="c",
            tenancy="rowlevel",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Cube root validator — cross-field consistency
# ---------------------------------------------------------------------------


def test_discriminator_requires_tenancy_columns() -> None:
    with pytest.raises(ValidationError, match=r"(?i)tenancy_columns"):
        Cube(
            name="events",
            dialect=Dialect.POSTGRES,
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
            dialect=Dialect.POSTGRES,
            table="{tenant_schema}.events",
            alias="e",
            tenancy="discriminator",
            tenancy_columns=["tenant_id"],
        )


def test_schema_mode_does_not_require_tenancy_columns() -> None:
    """SCHEMA mode encodes isolation in the table name; no column needed."""
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="{tenant_schema}.t",
        alias="c",
        tenancy="schema",
    )
    assert cube.tenancy_columns == []


def test_schema_mode_requires_tenant_schema_placeholder() -> None:
    """Explicit SCHEMA mode whose source never mentions ``{tenant_schema}``
    would emit zero isolation — the old silent-inert default. It is now a
    construction error (the mirror of the discriminator placeholder check)."""
    with pytest.raises(ValidationError, match=r"(?i)tenant_schema"):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="orders",  # no {tenant_schema}
            alias="c",
            tenancy="schema",
        )


# ---------------------------------------------------------------------------
# Compiler — DISCRIMINATOR cubes get a WHERE predicate on the inner subquery
# ---------------------------------------------------------------------------


def _disc_catalog() -> Catalog:
    events = Cube(
        name="events",
        dialect=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["tenant_id"],
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
        dialect=Dialect.POSTGRES,
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
        dialect=Dialect.POSTGRES,
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
        dialect=Dialect.POSTGRES,
        table="{tenant_schema}.orders",
        alias="o",
        tenancy="schema",
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
        dialect=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["tenant_id"],
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


# ---------------------------------------------------------------------------
# Schema-name injection — `{tenant_schema}` interpolates into an identifier
# position (no bind form), so the context value must be a safe identifier.
# ---------------------------------------------------------------------------


def _schema_cube() -> Cube:
    return Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="{tenant_schema}.t",
        alias="c",
        measures=[Measure(name="n", sql="*", agg="count", unit="count")],
    )


def test_schema_tenancy_refuses_injection_in_context_value() -> None:
    """A request-controlled ``tenant_schema`` that isn't a plain identifier
    (here, a stacked ``DROP TABLE``) is refused rather than spliced raw —
    closing the schema-name injection vector."""
    cat = Catalog([_schema_cube()])
    q = SemanticQuery(measures=["c.n"])
    with pytest.raises(CompileError, match="safe SQL identifier"):
        cat.compile(q, context={"tenant_schema": "public; DROP TABLE users; --"})


def test_schema_tenancy_refuses_quoted_or_spaced_context_value() -> None:
    cat = Catalog([_schema_cube()])
    q = SemanticQuery(measures=["c.n"])
    for bad in ('"evil"', "a b", "schema'); --", "1schema"):
        with pytest.raises(CompileError, match="safe SQL identifier"):
            cat.compile(q, context={"tenant_schema": bad})


def test_schema_tenancy_accepts_plain_and_qualified_identifier() -> None:
    cat = Catalog([_schema_cube()])
    q = SemanticQuery(measures=["c.n"])
    assert "tenant_acme.t" in cat.compile(q, context={"tenant_schema": "tenant_acme"}).sql
    # Dot-qualified (catalog.schema) is allowed — still pure identifier chars.
    assert "db.tenant_acme.t" in cat.compile(q, context={"tenant_schema": "db.tenant_acme"}).sql


# ---------------------------------------------------------------------------
# Composite discriminator keys (S18) — tenancy_columns AND-composes
# ---------------------------------------------------------------------------


def _composite_catalog() -> Catalog:
    events = Cube(
        name="events",
        dialect=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["org_id", "region"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="kind", sql="{e}.kind", type="string")],
    )
    return Catalog([events])


def test_composite_discriminator_binds_each_column() -> None:
    out = _composite_catalog().compile(
        SemanticQuery(measures=["events.count"]),
        context={"org_id": "o1", "region": "us-east"},
    )
    assert "org_id" in out.sql
    assert "region" in out.sql
    # Both values ride as bound parameters, never literals.
    assert "o1" in out.params.values()
    assert "us-east" in out.params.values()
    assert "'o1'" not in out.sql and "'us-east'" not in out.sql


def test_composite_discriminator_refuses_missing_column_value() -> None:
    with pytest.raises(CompileError, match=r"(?i)region"):
        _composite_catalog().compile(
            SemanticQuery(measures=["events.count"]),
            context={"org_id": "o1"},  # region missing
        )


def test_composite_discriminator_value_is_inert_against_injection() -> None:
    out = _composite_catalog().compile(
        SemanticQuery(measures=["events.count"]),
        context={"org_id": "o1", "region": "x'); DROP TABLE events;--"},
    )
    assert any("DROP TABLE" in str(v) for v in out.params.values())
    assert "DROP TABLE" not in out.sql


# ---------------------------------------------------------------------------
# Tenant first-class on AuthContext (S19) — viewer.tenant threads through
# ---------------------------------------------------------------------------


def _schema_orders_catalog() -> Catalog:
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="{tenant_schema}.orders",
        alias="o",
        tenancy="schema",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    return Catalog([cube])


def test_viewer_tenant_drives_schema_substitution() -> None:
    """viewer.tenant is the canonical tenant — schema substitution uses
    it with no context['tenant_schema'] needed."""
    out = _schema_orders_catalog().compile(
        SemanticQuery(measures=["orders.count"]),
        viewer=AuthContext(viewer_id="u1", tenant="acme"),
    )
    assert "acme.orders" in out.sql


def test_viewer_tenant_drives_single_column_discriminator() -> None:
    out = _disc_catalog().compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        viewer=AuthContext(viewer_id="u1", tenant="acme"),
    )
    assert "tenant_id" in out.sql
    assert "acme" in out.params.values()


def test_explicit_context_overrides_viewer_tenant() -> None:
    out = _schema_orders_catalog().compile(
        SemanticQuery(measures=["orders.count"]),
        viewer=AuthContext(viewer_id="u1", tenant="acme"),
        context={"tenant_schema": "beta"},
    )
    assert "beta.orders" in out.sql
    assert "acme" not in out.sql


def test_discriminator_refuses_when_identity_carries_no_tenant() -> None:
    """No viewer.tenant and no context value for a tenancy-required cube
    is a hard refusal, not a tenant-less query."""
    with pytest.raises(CompileError, match=r"(?i)tenant"):
        _disc_catalog().compile(
            SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
            viewer=AuthContext(viewer_id="u1"),  # tenant=None
        )


def test_composite_discriminator_resolves_from_viewer_attrs() -> None:
    out = _composite_catalog().compile(
        SemanticQuery(measures=["events.count"]),
        viewer=AuthContext(viewer_id="u1", attrs={"org_id": "o9", "region": "eu"}),
    )
    assert "o9" in out.params.values()
    assert "eu" in out.params.values()


# ---------------------------------------------------------------------------
# Default-deny lint (S17) — Catalog(strict_tenancy=True)
# ---------------------------------------------------------------------------


def test_strict_tenancy_rejects_unguarded_cube() -> None:
    cube = Cube(
        name="secrets",
        dialect=Dialect.POSTGRES,
        table="secrets",
        alias="s",
        tenancy="none",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
    )
    with pytest.raises(ValueError, match="secrets"):
        Catalog([cube], strict_tenancy=True)


def test_strict_tenancy_allows_cube_guarded_by_required_roles() -> None:
    cube = Cube(
        name="ok",
        dialect=Dialect.POSTGRES,
        table="ok",
        alias="o",
        tenancy="none",
        required_roles=["admin"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
    )
    # Does not raise.
    assert Catalog([cube], strict_tenancy=True) is not None


def test_strict_tenancy_default_off_allows_open_cube() -> None:
    cube = Cube(name="open", dialect=Dialect.POSTGRES, table="open", alias="o")
    assert Catalog([cube]) is not None  # no strict_tenancy → no error
