"""Regression tests for security-audit findings fixed in the semql core.

Each test pins a specific vulnerability so it can't silently regress:

- SEMQL-LOOKUP-ENRICHER-IDENT (#1): the SQL enricher must validate
  context-substituted ``table`` identifiers, like the compiler does.
- SEMQL-COMPILEPLAN-PROJECTION-AUTH (#3): ``compile_plan`` must
  re-authorize every projected field, so a forged ``LogicalPlan`` whose
  projection diverges from its aggregate can't emit a role-gated field.
"""

from __future__ import annotations

from dataclasses import replace

import duckdb
import pytest
from semql.compile import compile_plan
from semql.errors import UnknownIdentifierError
from semql.logical import ColumnRef, LogicalPlan, to_logical_plan
from semql.lookups import sql_enricher
from semql.model import AuthContext, Cube, Dialect, Dimension, Measure, ResolutionContext
from semql.spec import SemanticQuery

from .conftest import CONTEXT

# ---------------------------------------------------------------------------
# #1 — SQL enricher identifier validation
# ---------------------------------------------------------------------------


def test_sql_enricher_rejects_unsafe_context_identifier() -> None:
    """A templated ``table`` filled from request-influenced ``ctx.context``
    must refuse a value that isn't a safe SQL identifier rather than splice
    it raw into the FROM clause."""
    seen: list[str] = []

    def execute(sql: str, params: list[object]) -> list[dict[str, object]]:
        seen.append(sql)
        return []

    enr = sql_enricher(table="{schema}.regions", key="id", fields=["name"], execute=execute)

    # A benign identifier still works.
    enr.enrich_fields(["r1"], ResolutionContext(context={"schema": "acme"}))
    assert "FROM acme.regions" in seen[-1]

    # An injection payload is refused before any SQL is built.
    with pytest.raises(ValueError, match="not a safe SQL identifier"):
        enr.enrich_fields(
            ["r1"],
            ResolutionContext(context={"schema": "x; DROP TABLE audit_log; --"}),
        )
    # No SQL was executed for the malicious call.
    assert all("DROP TABLE" not in s for s in seen)


# ---------------------------------------------------------------------------
# #3 — compile_plan projection re-authorization
# ---------------------------------------------------------------------------


def _orders_with_secret() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        dialect=Dialect.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="secret", sql="{o}.secret", type="string", required_roles=["admin"]),
        ],
    )
    return {"orders": orders}


def _forged_plan(catalog: dict[str, Cube]) -> LogicalPlan:
    """A grouped plan whose ``Project.columns`` smuggles the role-gated
    ``orders.secret`` dimension while ``Aggregate.group_by`` lists only the
    authorized ``orders.region`` — the exact projection/aggregate divergence
    the field-hide gate (which reads the aggregate) would otherwise miss."""
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    plan = to_logical_plan(q, catalog)
    secret = next(d for d in catalog["orders"].dimensions if d.name == "secret")
    forged_col = ColumnRef(
        cube=catalog["orders"],
        field_name="secret",
        alias="secret",
        kind="dimension",
        field=secret,
    )
    return replace(plan, project=replace(plan.project, columns=[*plan.project.columns, forged_col]))


def test_compile_plan_refuses_role_gated_field_smuggled_via_projection() -> None:
    catalog = _orders_with_secret()
    forged = _forged_plan(catalog)

    low = AuthContext(viewer_id="u", roles=["viewer"])
    with pytest.raises(UnknownIdentifierError) as exc:
        compile_plan(forged, catalog, context=CONTEXT, viewer=low)
    # Indistinguishable from "field doesn't exist": no role leakage.
    assert exc.value.name == "secret"


def test_compile_plan_allows_projection_field_for_authorized_viewer() -> None:
    """The guard is role-specific, not a blanket structural rejection. A
    well-formed plan that legitimately projects the gated field compiles for
    a viewer holding the role and for the unauthed path (``viewer=None``,
    which bypasses the gate, mirroring ``_sees``), but is refused for a
    low-role viewer."""
    catalog = _orders_with_secret()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region", "orders.secret"])
    plan = to_logical_plan(q, catalog)

    admin = AuthContext(viewer_id="a", roles=["admin"])
    compiled = compile_plan(plan, catalog, context=CONTEXT, viewer=admin)
    assert "secret" in compiled.sql

    # Unauthed path is open, same as the field-hide gate.
    compile_plan(plan, catalog, context=CONTEXT, viewer=None)

    # A low-role viewer is refused even on this well-formed plan.
    low = AuthContext(viewer_id="u", roles=["viewer"])
    with pytest.raises(UnknownIdentifierError):
        compile_plan(plan, catalog, context=CONTEXT, viewer=low)


def _masked_catalog() -> dict[str, Cube]:
    cube = Cube(
        name="users",
        alias="u",
        table="users",
        dialect=Dialect.DUCKDB,
        measures=[
            Measure(
                name="income",
                sql="{u}.income",
                agg="sum",
                required_roles=["analyst"],
                mask_roles=["analyst"],
                mask_value="'REDACTED'",
            )
        ],
        dimensions=[
            Dimension(
                name="email",
                sql="{u}.email",
                type="string",
                required_roles=["analyst"],
                mask_roles=["analyst"],
                mask_value="'REDACTED'",
            ),
            Dimension(name="region", sql="{u}.region", type="string"),
        ],
    )
    return {"users": cube}


def _execute_users(sql: str, params: dict[str, object]) -> list[tuple[object, ...]]:
    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE TABLE users(email VARCHAR, region VARCHAR, income DOUBLE)")
        connection.execute(
            "INSERT INTO users VALUES "
            "('secret@example.test', 'west', 900.0), "
            "('other@example.test', 'east', 100.0)"
        )
        return connection.execute(sql, params).fetchall()


def test_compile_plan_restores_forged_dimension_and_measure_masks() -> None:
    catalog = _masked_catalog()
    query = SemanticQuery(
        measures=["users.income"],
        dimensions=["users.email"],
        aliases={"person": "users.email"},
    )
    plan = to_logical_plan(query, catalog)
    forged_columns = [
        replace(
            col,
            field=col.field.model_copy(update={"mask_roles": [], "mask_value": None}),
        )
        if col.field is not None and col.field.name in {"email", "income"}
        else col
        for col in plan.project.columns
    ]
    forged = replace(plan, project=replace(plan.project, columns=forged_columns))
    compiled = compile_plan(
        forged,
        catalog,
        context=CONTEXT,
        viewer=AuthContext(viewer_id="a", roles=["analyst"]),
    )

    rows = _execute_users(compiled.sql, compiled.params)
    assert len(rows) == 2
    assert all(row == ("REDACTED", "REDACTED") for row in rows)
    assert compiled.columns == ["person", "income"]
    assert all(meta.masked for meta in compiled.column_meta)
    assert {
        node.catalog_ref: node.masked
        for node in compiled.analysis.nodes
        if node.catalog_ref in {"users.email", "users.income"}
    } == {"users.email": True, "users.income": True}
    assert "'REDACTED'" in compiled.sql


def test_compile_plan_restores_required_roles_before_projection_authorization() -> None:
    catalog = _masked_catalog()
    plan = to_logical_plan(SemanticQuery(dimensions=["users.region"]), catalog)
    email = catalog["users"].dimensions[0]
    forged_email = ColumnRef(
        cube=catalog["users"],
        field_name="email",
        alias="email",
        kind="dimension",
        field=email.model_copy(update={"required_roles": [], "mask_roles": [], "mask_value": None}),
    )
    forged = replace(
        plan,
        project=replace(
            plan.project,
            columns=[*plan.project.columns, forged_email],
        ),
    )

    with pytest.raises(UnknownIdentifierError) as exc:
        compile_plan(
            forged,
            catalog,
            context=CONTEXT,
            viewer=AuthContext(viewer_id="u", roles=["viewer"]),
        )
    assert exc.value.name == "email"


def test_compile_plan_preserves_physical_field_rewrite_and_alias() -> None:
    catalog = _masked_catalog()
    users = catalog["users"].model_copy(
        update={
            "dimensions": [
                catalog["users"]
                .dimensions[0]
                .model_copy(update={"required_roles": ["analyst", "admin"]}),
                *catalog["users"].dimensions[1:],
            ]
        }
    )
    catalog["users"] = users
    query = SemanticQuery(
        dimensions=["users.email"],
        aliases={"person": "users.email"},
        ungrouped=True,
        limit=10,
    )
    plan = to_logical_plan(query, catalog)
    col = plan.project.columns[0]
    assert col.field is not None
    rewritten_field = col.field.model_copy(update={"sql": "{u}.contact"})
    rewritten_cube = catalog["users"].model_copy(update={"table": "users_rollup"})
    transformed = replace(
        plan,
        scans=[replace(scan, cube=rewritten_cube) for scan in plan.scans],
        project=replace(
            plan.project,
            columns=[replace(col, field=rewritten_field)],
        ),
    )
    compiled = compile_plan(
        transformed,
        catalog,
        context=CONTEXT,
        viewer=AuthContext(viewer_id="a", roles=["admin"]),
    )

    assert compiled.columns == ["person"]
    assert compiled.column_meta[0].masked is False
    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE TABLE users_rollup(contact VARCHAR)")
        connection.execute("INSERT INTO users_rollup VALUES ('raw@example.test')")
        assert connection.execute(compiled.sql, compiled.params).fetchall() == [
            ("raw@example.test",)
        ]


def test_compile_plan_rejects_projection_missing_trusted_field() -> None:
    catalog = _masked_catalog()
    query = SemanticQuery(dimensions=["users.region"])
    plan = to_logical_plan(query, catalog)
    col = plan.project.columns[0]
    forged = replace(
        plan,
        project=replace(plan.project, columns=[replace(col, field=None)]),
    )

    with pytest.raises(UnknownIdentifierError):
        compile_plan(forged, catalog, context=CONTEXT)


def test_compile_plan_rejects_projection_without_trusted_catalog_field() -> None:
    catalog = _masked_catalog()
    query = SemanticQuery(dimensions=["users.region"])
    plan = to_logical_plan(query, catalog)
    col = plan.project.columns[0]
    assert col.field is not None
    forged_field = col.field.model_copy(update={"name": "unlisted"})
    renamed = replace(col, field_name="unlisted", field=forged_field)
    forged = replace(plan, project=replace(plan.project, columns=[renamed]))

    with pytest.raises(UnknownIdentifierError):
        compile_plan(forged, catalog, context=CONTEXT)
