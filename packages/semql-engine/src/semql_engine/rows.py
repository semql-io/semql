"""Row-mode execution: run an entity ``fetch`` / ``list`` against a backend.

A :class:`~semql.rows.CompiledEntityQuery` carries two interchangeable
representations of the same read (M2):

- ``sql`` + ``params`` for SQL backends — executed by an ordinary
  :class:`~semql_engine.adapter.Adapter`.
- ``plan`` (a :class:`~semql.rows.RowPlan`) for custom, non-SQL backends —
  executed by a :class:`RowCapableAdapter` that interprets the restricted
  predicate vocabulary directly.

:func:`execute_entity` dispatches on which one is present. The
:class:`InMemoryRowAdapter` is the reference custom backend — a tiny
table-shaped store that proves a non-SQL adapter needs to implement only
``execute_rows`` (D1). The golden test asserts the SQL path and the plan
path return identical records for the same request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from semql.bindings import final_binding_requirements, validate_bindings
from semql.errors import ContractError
from semql.model import Dialect
from semql.rows import CompiledEntityQuery, RowPlan, RowPred
from semql.safe import is_read_only_statement

from semql_engine.adapter import Adapter, AdapterResult
from semql_engine.engine import ExecutionContractError

__all__ = [
    "InMemoryRowAdapter",
    "RowCapableAdapter",
    "execute_entity",
]


class RowCapableAdapter(Protocol):
    """A backend that can interpret a :class:`~semql.rows.RowPlan` directly.

    The entire non-SQL extension story (D1): implement one method, map the
    plan's structured predicates to the store's own query surface (REST
    query params, KV lookups, a dataframe filter). Register it under the
    cube's backend name."""

    def execute_rows(self, plan: RowPlan) -> AdapterResult: ...


def _materialize_result(result: AdapterResult) -> AdapterResult:
    return AdapterResult(
        columns=list(result.columns),
        rows=[tuple(row) for row in result.rows],
        column_types=(list(result.column_types) if result.column_types is not None else None),
    )


def execute_entity(
    compiled: CompiledEntityQuery,
    adapter: Adapter | RowCapableAdapter,
) -> AdapterResult:
    """Run a compiled entity read on the right kind of adapter.

    SQL backends (``compiled.sql is not None``) run via an
    :class:`~semql_engine.adapter.Adapter`; custom backends run via a
    :class:`RowCapableAdapter`. Raises ``TypeError`` if the adapter doesn't
    match the compiled shape."""
    if compiled.plan.version != 1:
        raise ExecutionContractError(
            "Entity row plan uses an unsupported version.",
            reason="artifact_version_unsupported",
            artifact="entity_plan",
            operation="execute",
            stage="preflight",
        )
    if compiled.sql is not None:
        if not hasattr(adapter, "execute"):
            raise TypeError(
                "compiled query has SQL but the adapter is not a SQL Adapter (no execute method)."
            )
        sql_adapter = cast(Adapter, adapter)
        dialect = Dialect(compiled.plan.source.backend)
        try:
            requirements = final_binding_requirements(compiled.sql, dialect, compiled.params)
            validate_bindings(
                compiled.sql,
                dialect,
                compiled.params,
                requirements,
                artifact="entity_query",
            )
        except ContractError as error:
            raise ExecutionContractError.from_contract_error(error) from error
        if not is_read_only_statement(compiled.sql, dialect=dialect.value):
            raise ExecutionContractError(
                "Entity SQL is not a read-only SELECT.",
                reason="artifact_not_read_only",
                artifact="entity_query",
                operation="execute",
                stage="preflight",
            )
        result = _materialize_result(sql_adapter.execute(compiled.sql, compiled.params))
    else:
        if not hasattr(adapter, "execute_rows"):
            raise TypeError(
                "compiled query has no SQL (custom backend) but the adapter is "
                "not a RowCapableAdapter (no execute_rows method)."
            )
        row_adapter: RowCapableAdapter = adapter  # type: ignore[assignment]
        result = _materialize_result(row_adapter.execute_rows(compiled.plan))
    if list(result.columns) != list(compiled.columns):
        raise ExecutionContractError(
            "Entity adapter columns do not match the compiled output projection.",
            reason="output_columns_mismatch",
            artifact="entity_query",
            names=tuple(compiled.columns),
            operation="execute",
            stage="output_validation",
        )
    return result


# ---------------------------------------------------------------------------
# Reference custom backend: an in-memory, table-shaped store.
# ---------------------------------------------------------------------------


class InMemoryRowAdapter:
    """Reference :class:`RowCapableAdapter` over in-memory tables.

    ``tables`` maps a source table name (``RowPlan.source.table``) to a
    list of row dicts keyed by column name. ``execute_rows`` applies the
    plan's scope predicates and user predicates, orders, limits and
    projects — the same logic a real custom adapter would map onto its
    store. Proves the plan is a complete, SQL-free execution contract."""

    def __init__(self, tables: Mapping[str, list[dict[str, Any]]]) -> None:
        self._tables = {name: list(rows) for name, rows in tables.items()}

    def execute_rows(self, plan: RowPlan) -> AdapterResult:
        rows = list(self._tables.get(plan.source.table, []))

        # Scope predicates are injected first and are non-negotiable
        # (bypass-proof, as on the SQL path).
        for pred in (*plan.scope_predicates, *plan.predicates):
            rows = [r for r in rows if _matches(r, pred, plan.params)]

        for col, direction in reversed(plan.order):
            rows.sort(key=_order_key(col), reverse=direction == "desc")

        rows = rows[: plan.limit]
        projected: list[Sequence[Any]] = [tuple(r.get(c) for c in plan.columns) for r in rows]
        return AdapterResult(columns=list(plan.columns), rows=projected)


def _matches(row: dict[str, Any], pred: RowPred, params: Mapping[str, Any]) -> bool:
    value = row.get(pred.column)
    if pred.op == "eq":
        return bool(value == params[pred.params[0]])
    if pred.op == "in":
        return value in params[pred.params[0]]
    if pred.op == "time_range":
        start = params[pred.params[0]]
        end = params[pred.params[1]]
        # Half-open [start, end), matching TimeWindow semantics. ISO-8601
        # strings sort lexicographically in instant order.
        return value is not None and start <= value < end
    raise ValueError(f"InMemoryRowAdapter: unsupported predicate op {pred.op!r}.")


def _order_key(col: str) -> Callable[[dict[str, Any]], tuple[int, Any]]:
    """Build a stable sort key for ``col`` (binds ``col`` via the closure
    argument, so it's safe to use inside the order loop)."""
    return lambda r: _sort_key(r.get(col))


def _sort_key(value: Any) -> tuple[int, Any]:  # noqa: ANN401 — row values are arbitrary store types
    """Sort key that keeps NULLs first and never compares None to a value
    (which would raise). Returns a (null-flag, value) pair."""
    return (0, "") if value is None else (1, value)
