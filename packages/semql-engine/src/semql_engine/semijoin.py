"""Executor for a staged cross-backend semi-join.

:func:`semql.compile_semi_join_query` produces a :class:`SemiJoinPlan`: a set
of compiled inner plans plus a recipe for folding their results into an outer
query. This module drives the stages — run each inner, project its key column
to a de-duplicated value list, splice the lists into the outer query as
``in`` / ``not_in`` filters, then compile and run the outer.

Empty value lists are handled here, not at compile time, because the list is
only known after the inner runs:

- ``in`` over an empty list matches nothing — the whole query short-circuits
  to zero rows (a literal ``IN ()`` filter is invalid to compile anyway).
- ``not_in`` over an empty list restricts nothing — the filter is dropped.

NULL keys are dropped from every list: ``x IN (NULL)`` never matches and
would only bloat the list.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from semql.semijoin import SEMI_JOIN_PLAN_VERSION
from semql.spec import Filter

from semql_engine.engine import ExecutionContractError, ExecutionResult

if TYPE_CHECKING:
    from semql.semijoin import SemiJoinPlan

    from semql_engine.engine import Engine


def _project_values(result: ExecutionResult, column: str) -> list[Any]:
    """De-duplicated, NULL-free, order-preserving values of ``column``."""
    idx = result.columns.index(column)
    seen: set[Any] = set()
    out: list[Any] = []
    for row in result.rows:
        v = row[idx]
        if v is None or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _empty_result(plan: SemiJoinPlan) -> ExecutionResult:
    """Zero-row result carrying the plan's declared output shape."""
    return ExecutionResult(
        columns=list(plan.columns),
        column_meta=[replace(m) for m in plan.column_meta],
        rows=[],
    )


def run_semi_join(
    plan: SemiJoinPlan,
    engine: Engine,
    *,
    cache_namespace: str | None = None,
) -> ExecutionResult:
    """Execute a :class:`SemiJoinPlan` end-to-end on ``engine``.

    Each inner plan runs first; its ``select_column`` is projected to a value
    list and bound as a literal filter on the outer query. ``cache_namespace``
    is forwarded to every stage's :meth:`Engine.run`."""
    with engine.execution_scope() as scope:
        result = _run_semi_join(plan, engine, cache_namespace=cache_namespace)
        scope.check_deadline()
        return result


def _run_semi_join(
    plan: SemiJoinPlan,
    engine: Engine,
    *,
    cache_namespace: str | None,
) -> ExecutionResult:
    if plan.version != SEMI_JOIN_PLAN_VERSION or len(plan.inner_plans) != len(plan.bindings):
        raise ExecutionContractError(
            "Semi-join plan violates its declared contract.",
            reason="semi_join_plan_invalid",
            artifact="semi_join_plan",
            operation="execute",
            stage="preflight",
        )
    engine.preflight(plan.outer_plan)
    for inner_plan, binding in zip(plan.inner_plans, plan.bindings, strict=True):
        engine.preflight(inner_plan)
        if binding.select_column not in inner_plan.columns:
            raise ExecutionContractError(
                "Semi-join selection is absent from its inner projection.",
                reason="semi_join_projection_mismatch",
                artifact="semi_join_plan",
                operation="execute",
                stage="preflight",
            )
    filters: list[Filter] = []
    for inner_plan, binding in zip(plan.inner_plans, plan.bindings, strict=True):
        inner_result = engine.run(inner_plan, cache_namespace=cache_namespace)
        values = _project_values(inner_result, binding.select_column)
        if not values:
            if binding.op == "in":
                # in (empty) matches nothing -> the conjunction is empty.
                return _empty_result(plan)
            # not_in (empty) restricts nothing -> drop the filter.
            continue
        filters.append(Filter(dimension=binding.bind_dimension, op=binding.op, values=values))

    outer_plan = plan.bind_outer(filters)
    return engine.run(outer_plan, cache_namespace=cache_namespace)


__all__ = ["run_semi_join"]
