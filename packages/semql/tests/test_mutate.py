"""Tests for compiled mutations (M4): the four gates, value validation,
pinned-value injection, scope injection, the DML-shape guard, and the
bind-never-inline invariant.
"""

from __future__ import annotations

import pytest
import sqlglot
from semql import (
    AuthContext,
    Catalog,
    CtxRef,
    Cube,
    Dimension,
    Measure,
    MutableEntity,
    MutableField,
    Op,
    SemanticMutation,
)
from semql.errors import AuthError, CompileError
from semql.model import Dialect


def _cube(**kw: object) -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="n", sql="{o}.id", agg="count", unit="count")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        primary_key="id",
        **kw,  # type: ignore[arg-type]
    )


def _entity(**kw: object) -> MutableEntity:
    base: dict[str, object] = dict(
        name="order",
        cubes=["orders"],
        key="orders.id",
        target_cube="orders",
        operations=frozenset({Op.INSERT, Op.UPDATE, Op.DELETE}),
        mutable_fields={
            "region": MutableField(type="string"),
            "amount": MutableField(type="number", required=True),
        },
    )
    base.update(kw)
    return MutableEntity(**base)  # type: ignore[arg-type]


def _catalog(
    entity: MutableEntity | None = None, *, allow: bool = True, **cube_kw: object
) -> Catalog:
    return Catalog(
        [_cube(**cube_kw)],
        entities=[entity or _entity()],
        allow_mutations=allow,
    )


def _where_sql(sql: str) -> str:
    node = sqlglot.parse_one(sql, dialect="postgres")
    where = node.args.get("where")
    return where.sql(dialect="postgres") if where is not None else ""


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_gate1_mutations_disabled() -> None:
    cat = _catalog(allow=False)
    with pytest.raises(AuthError, match=r"(?i)disabled|allow_mutations"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
            )
        )


def test_gate2_read_only_entity_refused() -> None:
    from semql import Entity

    cat = Catalog(
        [_cube()],
        entities=[Entity(name="order", cubes=["orders"], key="orders.id")],
        allow_mutations=True,
    )
    with pytest.raises(AuthError, match=r"(?i)read-only|MutableEntity"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
            )
        )


def test_gate2_operation_not_permitted() -> None:
    cat = _catalog(_entity(operations=frozenset({Op.UPDATE})))
    with pytest.raises(AuthError, match=r"(?i)not permitted|operation"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 1}))


def test_gate3_role_policy_denied() -> None:
    cat = _catalog(required_roles=["admin"])
    viewer = AuthContext(viewer_id="u1", roles=["viewer"])
    with pytest.raises(AuthError, match=r"(?i)may not mutate|role"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
            ),
            viewer=viewer,
        )


def test_gate4_predicate_targeting_disabled() -> None:
    cat = _catalog()  # predicate_targeting=False by default
    with pytest.raises(AuthError, match=r"(?i)predicate|where"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, where={"region": "eu"}
            )
        )


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------


def test_insert_builds_parameterised_dml() -> None:
    cat = _catalog()
    out = cat.mutate(
        SemanticMutation(
            entity="order", operation=Op.INSERT, values={"region": "us", "amount": 100}
        )
    )
    assert out.sql.upper().startswith("INSERT INTO")
    assert "us" not in out.sql and "100" not in out.sql  # bind-never-inline
    assert set(out.params.values()) == {"us", 100}
    assert out.affects == ["orders"]


def test_insert_missing_required_field() -> None:
    cat = _catalog()
    with pytest.raises(CompileError, match=r"(?i)required|amount"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.INSERT, values={"region": "us"}))


def test_insert_unknown_field_refused() -> None:
    cat = _catalog()
    with pytest.raises(CompileError, match=r"(?i)not mutable|ghost"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.INSERT, values={"amount": 1, "ghost": "x"}
            )
        )


def test_insert_type_mismatch() -> None:
    cat = _catalog()
    with pytest.raises(CompileError, match=r"(?i)number|amount"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.INSERT, values={"amount": "lots"}))


# ---------------------------------------------------------------------------
# Pinned values
# ---------------------------------------------------------------------------


def _pinned_entity() -> MutableEntity:
    return _entity(pinned_values={"org_id": CtxRef(attr="tenant")})


def test_pinned_value_injected_from_ctx() -> None:
    cat = _catalog(_pinned_entity())
    viewer = AuthContext(viewer_id="u1", tenant="acme")
    out = cat.mutate(
        SemanticMutation(entity="order", operation=Op.INSERT, values={"amount": 1}),
        viewer=viewer,
    )
    assert "org_id" in out.sql
    assert "acme" in out.params.values()
    assert "acme" not in out.sql  # bound, not inlined


def test_pinned_column_supplied_in_values_fails_loudly() -> None:
    cat = _catalog(_pinned_entity())
    viewer = AuthContext(viewer_id="u1", tenant="acme")
    with pytest.raises(CompileError, match=r"(?i)pinned|override"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.INSERT, values={"amount": 1, "org_id": "globex"}
            ),
            viewer=viewer,
        )


def test_insert_scope_force_pin_refusal() -> None:
    """A discriminator-scoped cube must pin its tenancy column on insert."""
    cat = _catalog(_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    viewer = AuthContext(viewer_id="u1", tenant="acme")
    with pytest.raises(CompileError, match=r"(?i)discriminator|pinned|org_id"):
        cat.mutate(
            SemanticMutation(entity="order", operation=Op.INSERT, values={"amount": 1}),
            viewer=viewer,
        )


def test_insert_scope_force_pin_satisfied() -> None:
    cat = _catalog(_pinned_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    viewer = AuthContext(viewer_id="u1", tenant="acme")
    out = cat.mutate(
        SemanticMutation(entity="order", operation=Op.INSERT, values={"amount": 1}),
        viewer=viewer,
    )
    assert "org_id" in out.sql


# ---------------------------------------------------------------------------
# UPDATE / DELETE targeting + scope
# ---------------------------------------------------------------------------


def test_update_by_pk() -> None:
    cat = _catalog()
    out = cat.mutate(
        SemanticMutation(
            entity="order", operation=Op.UPDATE, values={"region": "emea"}, pk={"id": 5}
        )
    )
    assert out.sql.upper().startswith("UPDATE")
    assert "emea" not in out.sql and "5" not in out.sql
    assert set(out.params.values()) == {"emea", 5}


def test_update_immutable_field_refused() -> None:
    e = _entity(
        mutable_fields={
            "region": MutableField(type="string"),
            "amount": MutableField(type="number", immutable=True),
        }
    )
    cat = _catalog(e)
    with pytest.raises(CompileError, match=r"(?i)immutable|amount"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"amount": 9}, pk={"id": 1}
            )
        )


def test_delete_by_pk() -> None:
    cat = _catalog()
    out = cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 7}))
    assert out.sql.upper().startswith("DELETE")
    assert "7" not in out.sql and out.params == {"t0": 7}


def test_update_without_target_refused() -> None:
    cat = _catalog()
    with pytest.raises(CompileError, match=r"(?i)pk|where|target"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.UPDATE, values={"region": "us"}))


def test_update_both_pk_and_where_refused() -> None:
    cat = _catalog(_entity(predicate_targeting=True))
    with pytest.raises(CompileError, match=r"(?i)exactly one|both"):
        cat.mutate(
            SemanticMutation(
                entity="order",
                operation=Op.UPDATE,
                values={"region": "us"},
                pk={"id": 1},
                where={"region": "eu"},
            )
        )


def test_predicate_targeting_allowed_when_opted_in() -> None:
    cat = _catalog(_entity(predicate_targeting=True))
    out = cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, where={"region": "eu"}))
    assert out.sql.upper().startswith("DELETE")
    assert "eu" in out.params.values()


def test_scope_injected_into_update_where() -> None:
    cat = _catalog(_pinned_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    viewer = AuthContext(viewer_id="u1", tenant="acme")
    out = cat.mutate(
        SemanticMutation(
            entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
        ),
        viewer=viewer,
    )
    assert "org_id" in out.sql  # scope predicate present in WHERE
    assert "acme" in out.params.values()


# ---------------------------------------------------------------------------
# Preview == DML WHERE (A.4.1) + DML-shape guard
# ---------------------------------------------------------------------------


def test_preview_where_matches_dml_where() -> None:
    cat = _catalog()
    out = cat.mutate(
        SemanticMutation(entity="order", operation=Op.UPDATE, values={"region": "x"}, pk={"id": 3})
    )
    assert _where_sql(out.sql) == _where_sql(out.preview_sql)
    assert out.preview_sql.upper().startswith("SELECT")


def test_delete_preview_where_matches() -> None:
    cat = _catalog()
    out = cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 3}))
    assert _where_sql(out.sql) == _where_sql(out.preview_sql)


def test_max_affected_rows_carried() -> None:
    cat = Catalog([_cube()], entities=[_entity()], allow_mutations=True, max_mutation_rows=50)
    out = cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 1}))
    assert out.max_affected_rows == 50


# ---------------------------------------------------------------------------
# Fail-closed on raw-SQL scope
# ---------------------------------------------------------------------------


def test_raw_security_sql_scope_refused() -> None:
    cat = _catalog(_entity(), security_sql="{o}.region = 'x'")
    with pytest.raises(CompileError, match=r"(?i)raw-SQL scope|security_sql"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 1}))


# ---------------------------------------------------------------------------
# Discriminator fail-closed (SEMQL-MUTATION-DISCRIMINATOR-FAILOPEN):
# update/delete on a discriminator-scoped cube without a tenant must
# raise rather than emit a tenantless WHERE clause.
# ---------------------------------------------------------------------------


def test_discriminator_update_without_viewer_raises() -> None:
    """No viewer → CompileError, not a silently tenantless UPDATE."""
    cat = _catalog(_pinned_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    with pytest.raises(CompileError, match=r"(?i)discriminator|tenant"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
            )
        )


def test_discriminator_update_without_tenant_raises() -> None:
    """Viewer present but tenant is None → CompileError."""
    cat = _catalog(_pinned_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    viewer = AuthContext(viewer_id="u1")  # no tenant
    with pytest.raises(CompileError, match=r"(?i)discriminator|tenant"):
        cat.mutate(
            SemanticMutation(
                entity="order", operation=Op.UPDATE, values={"region": "us"}, pk={"id": 1}
            ),
            viewer=viewer,
        )


def test_discriminator_delete_without_viewer_raises() -> None:
    """Same fail-closed behaviour for DELETE."""
    cat = _catalog(_pinned_entity(), tenancy="discriminator", tenancy_columns=["org_id"])
    with pytest.raises(CompileError, match=r"(?i)discriminator|tenant"):
        cat.mutate(SemanticMutation(entity="order", operation=Op.DELETE, pk={"id": 1}))
