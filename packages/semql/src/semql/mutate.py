# pyright: reportPrivateImportUsage=false
# sqlglot re-exports its node classes (exp.Expression, exp.Insert, ...)
# through a star import; pyright flags them as private. They are the
# public API in practice — same pragma the rest of the codebase uses.
"""Compiled mutations: turn a validated ``SemanticMutation`` into a single
parameterised DML statement plus a preview SELECT (entities spec M4).

The LLM may *construct* a ``SemanticMutation`` (which entity, which
operation, which values / target), but never SQL — ``compile_mutation``
template-generates the DML from the validated spec, binding every value as
a parameter (the bind-never-inline invariant, A.4.2). Four independent
gates must all pass (§5):

1. ``Catalog.allow_mutations`` — global hard gate (off by default).
2. The entity is a :class:`~semql.model.MutableEntity` and the operation
   is in its ``operations``.
3. The viewer passes the target cube's role policy (same door as cube
   visibility).
4. Predicate (``where``) targeting only when ``predicate_targeting=True``.

Scope is fail-closed: v1 injects only *structured* discriminator tenancy
into every UPDATE/DELETE WHERE (bound, bypass-proof). A target cube
carrying raw ``security_sql`` / a ``scope`` ScopeFn is refused — silently
not enforcing raw scope on a write would be worse than refusing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp

from semql.backend import dialect_for, render
from semql.dialect import dialect_for as sqlglot_dialect_for
from semql.errors import AuthError, CompileError
from semql.introspect import viewer_sees
from semql.model import AuthContext, Cube, MutableEntity, Op

if TYPE_CHECKING:
    from semql.catalog import Catalog

Value = str | int | float | bool

__all__ = [
    "CompiledMutation",
    "SemanticMutation",
    "compile_mutation",
]

# A writable dimension's sql must be a bare ``{alias}.column`` — anything
# richer (an expression) has no single physical column to write to.
_SIMPLE_COLUMN = re.compile(r"^\s*\{[^}]+\}\.([A-Za-z_]\w*)\s*$")


class SemanticMutation(BaseModel):
    """An LLM-constructed, compiled mutation request (never raw SQL)."""

    model_config = ConfigDict(frozen=True)
    entity: str
    operation: Op
    values: dict[str, Value] = Field(default_factory=lambda: dict[str, Value]())
    pk: dict[str, Value] | None = None
    where: dict[str, Value | list[Value]] | None = None


class CompiledMutation(BaseModel):
    """A compiled mutation: one DML statement + a preview SELECT.

    ``sql`` / ``params`` are the parameterised DML. ``preview_sql`` /
    ``preview_params`` select the rows the DML will affect (same WHERE,
    scope included) so a caller can show-then-confirm (§5). ``affects``
    names the tables touched. ``max_affected_rows`` is the confirm-time cap
    the executor enforces against the preview count (A.4.5)."""

    model_config = ConfigDict(frozen=True)
    operation: Op
    sql: str
    params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    preview_sql: str
    preview_params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    affects: list[str] = Field(default_factory=lambda: list[str]())
    max_affected_rows: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _physical_column(cube: Cube, field_name: str) -> str:
    """Map a semantic dimension name to its physical column, refusing any
    dimension whose sql isn't a bare ``{alias}.column``."""
    dim = next((d for d in cube.dimensions if d.name == field_name), None)
    if dim is None:
        raise CompileError(f"Cube {cube.name!r} has no dimension named {field_name!r} to write to.")
    m = _SIMPLE_COLUMN.match(str(dim.sql))
    if m is None:
        raise CompileError(
            f"Dimension {field_name!r} on cube {cube.name!r} maps to a SQL "
            f"expression ({dim.sql!r}), not a single column — it is not writable."
        )
    return m.group(1)


def _check_value_type(field_name: str, value: Value, ftype: str) -> None:
    if ftype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompileError(f"Field {field_name!r} expects a number, got {value!r}.")
    elif ftype == "bool":
        if not isinstance(value, bool):
            raise CompileError(f"Field {field_name!r} expects a bool, got {value!r}.")
    elif not isinstance(value, str):
        raise CompileError(f"Field {field_name!r} expects a string, got {value!r}.")


def _resolve_attr(viewer: AuthContext | None, attr: str) -> Value:
    """Read a pinned column's value from the viewer's AuthContext."""
    if viewer is None:
        raise CompileError(
            f"Pinned column needs ctx attribute {attr!r} but no viewer was supplied."
        )
    for source in (getattr(viewer, attr, None), viewer.attrs.get(attr), viewer.metadata.get(attr)):
        if source is not None:
            return source
    raise CompileError(f"Pinned column needs ctx attribute {attr!r}, absent on the viewer.")


def _resolve_table(cube: Cube, context: dict[str, str] | None, viewer: AuthContext | None) -> str:
    table = cube.table
    subs = dict(context or {})
    if viewer is not None and viewer.tenant is not None:
        subs.setdefault("tenant_schema", viewer.tenant)
        subs.setdefault("tenant", viewer.tenant)
    for k, v in subs.items():
        table = table.replace("{" + k + "}", v)
    return table


def _and(conds: list[exp.Expression]) -> exp.Expression:
    cond = conds[0]
    for extra in conds[1:]:
        cond = exp.And(this=cond, expression=extra)
    return cond


def _assert_single_dml(sql: str, dialect_name: str, operation: Op) -> None:
    """Belt-and-braces (A.4.1): the rendered statement must be exactly one
    INSERT / UPDATE / DELETE, and update/delete must carry a WHERE."""
    statements = sqlglot.parse(sql, dialect=dialect_name)
    if len(statements) != 1 or statements[0] is None:
        raise CompileError(f"Compiled mutation produced {len(statements)} statements, expected 1.")
    node = statements[0]
    expected = {
        Op.INSERT: exp.Insert,
        Op.UPSERT: exp.Insert,
        Op.UPDATE: exp.Update,
        Op.DELETE: exp.Delete,
    }[operation]
    if not isinstance(node, expected):
        got = type(node).__name__
        raise CompileError(
            f"Compiled {operation} did not render as {expected.__name__} (got {got})."
        )
    if operation in (Op.UPDATE, Op.DELETE) and node.args.get("where") is None:
        raise CompileError(f"Compiled {operation} has no WHERE — refusing an unbounded mutation.")


# ---------------------------------------------------------------------------
# compile_mutation
# ---------------------------------------------------------------------------


def compile_mutation(
    mutation: SemanticMutation,
    catalog: Catalog,
    *,
    viewer: AuthContext | None = None,
    context: dict[str, str] | None = None,
) -> CompiledMutation:
    """Compile a :class:`SemanticMutation` to DML + preview. See module docstring."""
    # Gate 1: global hard gate.
    if not catalog.allow_mutations:
        raise AuthError(
            "Mutations are disabled on this catalog. Construct it with "
            "Catalog(allow_mutations=True) to enable the write surface.",
            reason="mutations_disabled",
        )

    entity = catalog.entities.get(mutation.entity)
    if entity is None:
        raise CompileError(f"Unknown entity {mutation.entity!r}.")
    # Gate 2: must be a MutableEntity and the op must be permitted.
    if not isinstance(entity, MutableEntity):
        raise AuthError(
            f"Entity {entity.name!r} is read-only (not a MutableEntity).",
            reason="not_mutable",
        )
    if mutation.operation not in entity.operations:
        raise AuthError(
            f"Operation {mutation.operation} is not permitted on entity "
            f"{entity.name!r}; allowed: {sorted(entity.operations)}.",
            reason="operation_not_allowed",
        )

    cube = catalog.as_dict()[entity.target_cube]

    # Gate 3: viewer role policy (same door as cube visibility).
    if not viewer_sees(cube, viewer, catalog.policy):
        raise AuthError(
            f"Viewer may not mutate entity {entity.name!r} (cube {cube.name!r} "
            "role policy denied).",
            reason="forbidden",
        )

    # Gate 4: predicate targeting is opt-in.
    if mutation.where is not None and not entity.predicate_targeting:
        raise AuthError(
            f"Entity {entity.name!r} does not allow predicate (where) targeting; "
            "set predicate_targeting=True to enable it.",
            reason="predicate_targeting_disabled",
        )

    # Fail closed on raw-SQL scope (v1): we can only inject structured
    # tenancy into the DML WHERE.
    if cube.security_sql or cube.scope is not None:
        raise CompileError(
            f"Entity {entity.name!r}: target cube {cube.name!r} uses raw-SQL scope "
            "(security_sql / scope). Compiled mutations only enforce structured "
            "(discriminator) tenancy in v1 — refusing rather than silently "
            "skipping row-level security on a write."
        )

    op = mutation.operation
    strategy = dialect_for(cube.dialect)
    sql_dialect = sqlglot_dialect_for(cube.dialect)
    table = _resolve_table(cube, context, viewer)
    table_node = exp.to_table(table)

    if op in (Op.UPDATE, Op.DELETE):
        return _compile_update_delete(
            mutation, entity, cube, strategy, sql_dialect, table, table_node, viewer, catalog
        )
    return _compile_insert(
        mutation, entity, cube, strategy, sql_dialect, table, table_node, viewer, catalog
    )


def _pinned_columns(
    entity: MutableEntity, viewer: AuthContext | None
) -> tuple[dict[str, Value], dict[str, str]]:
    """Resolve ``{physical_col: value}`` and ``{physical_col: param_name}``
    for every pinned column, reading values from the viewer's ctx."""
    values: dict[str, Value] = {}
    params: dict[str, str] = {}
    for i, (col, ref) in enumerate(entity.pinned_values.items()):
        values[col] = _resolve_attr(viewer, ref.attr)
        params[col] = f"pin_{i}"
    return values, params


def _validate_values(
    mutation: SemanticMutation, entity: MutableEntity, *, for_insert: bool
) -> None:
    pinned = set(entity.pinned_values)
    for name in mutation.values:
        if name in pinned:
            raise CompileError(
                f"Column {name!r} is pinned (ctx-derived) and cannot be supplied "
                "in values — refusing the override attempt."
            )
        if name not in entity.mutable_fields:
            raise CompileError(
                f"Field {name!r} is not mutable on entity {entity.name!r}; "
                f"mutable: {sorted(entity.mutable_fields)}."
            )
        field = entity.mutable_fields[name]
        _check_value_type(name, mutation.values[name], field.type)
        if not for_insert and field.immutable:
            raise CompileError(f"Field {name!r} is immutable — it cannot be set on update.")
    if for_insert:
        missing = [
            n for n, f in entity.mutable_fields.items() if f.required and n not in mutation.values
        ]
        if missing:
            raise CompileError(
                f"Insert on {entity.name!r} is missing required field(s): {missing}."
            )


def _compile_insert(
    mutation: SemanticMutation,
    entity: MutableEntity,
    cube: Cube,
    strategy: Any,  # noqa: ANN401 — DialectStrategy Protocol
    sql_dialect: str,
    table: str,
    table_node: exp.Expression,
    viewer: AuthContext | None,
    catalog: Catalog,
) -> CompiledMutation:
    if mutation.pk is not None or mutation.where is not None:
        raise CompileError("Insert/upsert take neither pk nor where.")
    _validate_values(mutation, entity, for_insert=True)

    pin_values, pin_params = _pinned_columns(entity, viewer)

    # A.4.3: a discriminator-scoped cube must force-pin its tenancy
    # column(s) on insert, or a row could be written that the viewer
    # can't read back.
    if cube.tenancy == "discriminator":
        unpinned = [c for c in cube.tenancy_columns if c not in pin_values]
        if unpinned:
            raise CompileError(
                f"Entity {entity.name!r}: target cube {cube.name!r} is "
                f"discriminator-scoped but tenancy column(s) {unpinned} are not "
                "pinned. Add them to pinned_values so inserts can't escape tenancy."
            )

    columns: list[str] = []
    placeholders: list[exp.Expression] = []
    params: dict[str, Any] = {}
    select_aliases: list[exp.Expression] = []

    for i, (name, value) in enumerate(mutation.values.items()):
        col = _physical_column(cube, name)
        pname = f"v{i}"
        ph = strategy.placeholder(pname, entity.mutable_fields[name].type)
        columns.append(col)
        placeholders.append(ph)
        params[pname] = value
        select_aliases.append(exp.Alias(this=ph.copy(), alias=exp.to_identifier(col)))
    for col, value in pin_values.items():
        pname = pin_params[col]
        ph = strategy.placeholder(pname, "string")
        columns.append(col)
        placeholders.append(ph)
        params[pname] = value
        select_aliases.append(exp.Alias(this=ph.copy(), alias=exp.to_identifier(col)))

    insert = exp.Insert(
        this=exp.Schema(this=table_node, expressions=[exp.column(c) for c in columns]),
        expression=exp.Values(expressions=[exp.Tuple(expressions=placeholders)]),
    )
    sql = render(insert, cube.dialect)
    _assert_single_dml(sql, sql_dialect, mutation.operation)

    # Preview: the row that would be written, as a single SELECT.
    preview = exp.Select(expressions=select_aliases)
    preview_sql = render(preview, cube.dialect)

    return CompiledMutation(
        operation=mutation.operation,
        sql=sql,
        params=params,
        preview_sql=preview_sql,
        preview_params=dict(params),
        affects=[table],
        max_affected_rows=catalog.max_mutation_rows,
    )


def _target_predicate(
    mutation: SemanticMutation,
    cube: Cube,
    strategy: Any,  # noqa: ANN401 — DialectStrategy Protocol
    viewer: AuthContext | None,
) -> tuple[exp.Expression, dict[str, Any]]:
    """Build the WHERE condition (pk or predicate target) plus structured
    scope predicates, and the bound params for both."""
    conds: list[exp.Expression] = []
    params: dict[str, Any] = {}

    if mutation.pk is not None and mutation.where is not None:
        raise CompileError("Provide exactly one of pk / where for update/delete, not both.")
    if mutation.pk is None and mutation.where is None:
        raise CompileError("Update/delete need exactly one of pk / where to target rows.")

    target = mutation.pk if mutation.pk is not None else mutation.where
    assert target is not None
    for i, (name, value) in enumerate(target.items()):
        col = _physical_column(cube, name)
        if isinstance(value, list):
            phs: list[exp.Expression] = []
            for j, item in enumerate(value):
                pname = f"t{i}_{j}"
                phs.append(strategy.placeholder(pname, "string"))
                params[pname] = item
            conds.append(exp.In(this=exp.column(col), expressions=phs))
        else:
            pname = f"t{i}"
            conds.append(
                exp.EQ(this=exp.column(col), expression=strategy.placeholder(pname, "string"))
            )
            params[pname] = value

    # Structured discriminator-tenancy scope, injected and bypass-proof.
    # Fail closed: if the cube is discriminator-scoped and no tenant is
    # present, refuse rather than silently emit a tenantless DML.
    if cube.tenancy == "discriminator":
        if viewer is None or viewer.tenant is None:
            raise CompileError(
                f"Cube {cube.name!r} is discriminator-scoped but no tenant "
                "value is present — refusing update/delete without tenancy predicate."
            )
        for i, col in enumerate(cube.tenancy_columns):
            pname = f"scope_{i}"
            conds.append(
                exp.EQ(this=exp.column(col), expression=strategy.placeholder(pname, "string"))
            )
            params[pname] = viewer.tenant

    return _and(conds), params


def _compile_update_delete(
    mutation: SemanticMutation,
    entity: MutableEntity,
    cube: Cube,
    strategy: Any,  # noqa: ANN401 — DialectStrategy Protocol
    sql_dialect: str,
    table: str,
    table_node: exp.Expression,
    viewer: AuthContext | None,
    catalog: Catalog,
) -> CompiledMutation:
    cond, params = _target_predicate(mutation, cube, strategy, viewer)
    where = exp.Where(this=cond)

    if mutation.operation == Op.DELETE:
        if mutation.values:
            raise CompileError("Delete takes no values.")
        dml: exp.Expression = exp.Delete(this=table_node, where=where)
    else:
        _validate_values(mutation, entity, for_insert=False)
        if not mutation.values:
            raise CompileError("Update needs at least one value to set.")
        set_exprs: list[exp.Expression] = []
        for i, (name, value) in enumerate(mutation.values.items()):
            col = _physical_column(cube, name)
            pname = f"v{i}"
            set_exprs.append(
                exp.EQ(
                    this=exp.column(col),
                    expression=strategy.placeholder(pname, entity.mutable_fields[name].type),
                )
            )
            params[pname] = value
        dml = exp.Update(this=table_node, expressions=set_exprs, where=where)

    sql = render(dml, cube.dialect)
    _assert_single_dml(sql, sql_dialect, mutation.operation)

    # Preview shares the *exact* WHERE (A.4.1 structural invariant).
    preview = exp.Select(expressions=[exp.Star()]).from_(table_node).where(cond.copy())
    preview_sql = render(preview, cube.dialect)

    return CompiledMutation(
        operation=mutation.operation,
        sql=sql,
        params=params,
        preview_sql=preview_sql,
        preview_params=dict(params),
        affects=[table],
        max_affected_rows=catalog.max_mutation_rows,
    )
