"""Auto-Planner P1: rewrite cross-source *filter-only* foreign cubes into
injected ``SemiJoin`` nodes.

These are pure ``SemanticQuery -> SemanticQuery`` rewrite tests (no I/O, no
execution). The end-to-end proof that the rewrite preserves semantics — a
semi-join plan and the equivalent bridge-merge plan return identical rows —
lives in ``semql-engine/tests/test_autoplan_equivalence.py``.

Fixtures mirror the two-backend cubes used across the engine tests:
``activity`` (BigQuery) carries the measure; ``employees`` (Postgres) is the
metadata cube bridged on ``activity.employee_id = employees.id``.
"""

from __future__ import annotations

from semql import (
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
)
from semql.autoplan import AutoPlan, CrossSourceDecision, autoplan


def _activity_cube() -> Cube:
    return Cube(
        name="activity",
        dialect=Dialect.BIGQUERY,
        table="activity",
        alias="a",
        primary_key="id",
        measures=[Measure(name="active_secs", sql="{a}.secs", agg="sum", unit="duration")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(
                name="employee_id", sql="{a}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{a}.employee_id = {e}.id")],
    )


def _employees_cube() -> Cube:
    return Cube(
        name="employees",
        dialect=Dialect.POSTGRES,
        table="employees",
        alias="e",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{e}.id", type="number"),
            Dimension(name="dept", sql="{e}.dept", type="string"),
        ],
    )


def _catalog(*extra: Cube) -> dict[str, Cube]:
    return {c.name: c for c in (_activity_cube(), _employees_cube(), *extra)}


# ---------------------------------------------------------------------------
# The motivating rewrite: filter-only foreign cube -> injected semi-join
# ---------------------------------------------------------------------------


def test_filter_only_foreign_cube_becomes_semi_join() -> None:
    """`count(activity) where employees.dept = Sales` — employees is touched
    only by a filter and lives on another backend, so the planner pushes it
    across as a value-list semi-join instead of a bridge-merge."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )

    result = autoplan(q, _catalog())

    assert isinstance(result, AutoPlan)
    # One semi-join injected, oriented primary-key IN inner-key.
    assert len(result.query.semi_joins) == 1
    sj = result.query.semi_joins[0]
    assert sj.dimension == "activity.employee_id"
    assert sj.op == "in"
    assert sj.select == "employees.id"
    # The foreign filter moved into the inner source; the inner projects the
    # bridge key.
    assert sj.source.dimensions == ["employees.id"]
    assert sj.source.filters == [Filter(dimension="employees.dept", op="eq", values=["Sales"])]
    # The outer query keeps measures + dimensions and loses the foreign filter.
    assert result.query.measures == ["activity.active_secs"]
    assert result.query.dimensions == ["activity.employee_id"]
    assert result.query.filters == []
    # The decision is recorded as a semi_join strategy on the foreign cube.
    assert result.decisions == (
        CrossSourceDecision(
            foreign_cube="employees",
            strategy="semi_join",
            reason=result.decisions[0].reason,
        ),
    )
    assert "employees" in result.decisions[0].reason


def test_same_backend_filter_is_left_untouched() -> None:
    """A filter on a cube that shares the measure's backend is a normal join,
    not a federation concern — the planner must not rewrite it."""
    same_backend_employees = _employees_cube().model_copy(update={"dialect": Dialect.BIGQUERY})
    catalog = {c.name: c for c in (_activity_cube(), same_backend_employees)}
    q = SemanticQuery(
        measures=["activity.active_secs"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )

    result = autoplan(q, catalog)

    assert result.query == q
    assert result.decisions == ()


def test_foreign_output_dimension_is_not_semi_joined() -> None:
    """When the foreign cube contributes an *output* dimension it is not
    filter-only — a semi-join can only filter, not return attributes — so the
    planner leaves it for the bridge-merge path."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["employees.dept"],  # foreign cube projected, not just filtered
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )

    result = autoplan(q, _catalog())

    assert result.query == q
    assert result.decisions == ()


def test_no_foreign_filter_is_a_noop() -> None:
    """A single-backend query (no cross-source filter) passes through."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        filters=[Filter(dimension="activity.id", op="gt", values=[0])],
    )

    result = autoplan(q, _catalog())

    assert result.query == q
    assert result.decisions == ()


def test_caller_supplied_semi_join_is_respected() -> None:
    """If the caller already authored semi_joins, the planner does not
    re-plan (Option C override)."""
    inner = SemanticQuery(
        dimensions=["employees.id"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Ops"])],
    )
    from semql import SemiJoin

    q = SemanticQuery(
        measures=["activity.active_secs"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
        semi_joins=[
            SemiJoin(dimension="activity.employee_id", op="in", select="employees.id", source=inner)
        ],
    )

    result = autoplan(q, _catalog())

    assert result.query == q
    assert result.decisions == ()
