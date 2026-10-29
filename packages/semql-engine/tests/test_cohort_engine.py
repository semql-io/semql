"""Behavior regressions for staged cohort execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import duckdb
import pytest
from semql import (
    Cohort,
    CohortComparison,
    CohortMeasurement,
    CohortMetric,
    CohortPlan,
    CohortQuery,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    compile_cohort_query,
)
from semql_engine import (
    AdapterResult,
    DuckDBAdapter,
    Engine,
    ExecutionContractError,
    ExecutionLimits,
    run_cohort,
)


class _Adapter:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.inner = DuckDBAdapter(connection)

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        return self.inner.execute(sql, params)


@pytest.fixture()
def db() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE activity (id INTEGER, period VARCHAR, region VARCHAR, amount DECIMAL(12,2))"
    )
    connection.execute(
        "INSERT INTO activity VALUES (1, 'prior', 'north', 5), "
        "(2, 'prior', 'south', 7), (1, 'current', 'new', 9)"
    )
    return connection


def _plan(*, source_limit: int | None = None, max_members: int = 4) -> CohortPlan:
    cube = Cube(
        name="activity",
        dialect=Dialect.DUCKDB,
        table="activity",
        alias="a",
        measures=[Measure(name="amount", sql="{a}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(name="period", sql="{a}.period", type="string"),
            Dimension(name="region", sql="{a}.region", type="string"),
        ],
    )
    query = CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(
                dimensions=["activity.id", "activity.region"],
                filters=[Filter(dimension="activity.period", op="eq", values=["prior"])],
                order=[("activity.id", "asc")],
                limit=source_limit,
            ),
            member_keys=["activity.id"],
            attributes=["activity.region"],
            max_members=max_members,
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["activity.id", "activity.region"],
                    measures=["activity.amount"],
                    filters=[Filter(dimension="activity.period", op="eq", values=["current"])],
                ),
                key_mapping={"activity.id": "activity.id"},
                attributes=["activity.region"],
            )
        ],
        comparisons=[
            CohortComparison(
                name="delta",
                left=CohortMetric(measurement="current", column="amount"),
                right=CohortMetric(measurement="current", column="amount"),
                op="difference",
            )
        ],
    )
    return compile_cohort_query(query, {cube.name: cube})


def test_source_explicit_limit_is_preserved(db: duckdb.DuckDBPyConnection) -> None:
    result = run_cohort(_plan(source_limit=1), Engine(adapters={Dialect.DUCKDB: _Adapter(db)}))
    assert tuple(row[0] for row in result.rows) == (1,)
    assert result.provenance[0].query.limit == 1


def test_member_overflow_fails_before_measurement_io(db: duckdb.DuckDBPyConnection) -> None:
    calls: list[str] = []

    class CountingAdapter(_Adapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(sql)
            return super().execute(sql, params)

    with pytest.raises(ExecutionContractError) as error:
        run_cohort(_plan(max_members=1), Engine(adapters={Dialect.DUCKDB: CountingAdapter(db)}))
    assert error.value.reason == "cohort_member_limit_exceeded"
    assert len(calls) == 1


def test_keeps_selected_attribution_and_missing_member(db: duckdb.DuckDBPyConnection) -> None:
    engine = Engine(adapters={Dialect.DUCKDB: _Adapter(db)})
    result = run_cohort(_plan(), engine)
    assert result.columns == (
        "selection.id",
        "selection.region",
        "current.region",
        "current.amount",
        "comparison.delta",
    )
    assert result.rows == (
        (1, "north", "new", 9, 0),
        (2, "south", None, None, None),
    )
    assert result.coverage[0].matched_members == 1
    assert result.provenance[0].identity_validated


def test_stale_cohort_plan_is_rejected_before_io(db: duckdb.DuckDBPyConnection) -> None:
    calls: list[str] = []

    class CountingAdapter(_Adapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(sql)
            return super().execute(sql, params)

    engine = Engine(adapters={Dialect.DUCKDB: CountingAdapter(db)})
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(replace(_plan(), version=99), engine)
    assert error.value.reason == "artifact_version_unsupported"
    assert calls == []


def test_duplicate_selection_identity_fails_before_measurement_io(
    db: duckdb.DuckDBPyConnection,
) -> None:
    db.execute("INSERT INTO activity VALUES (1, 'prior', 'other', 11)")
    calls: list[str] = []

    class CountingAdapter(_Adapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(sql)
            return super().execute(sql, params)

    engine = Engine(adapters={Dialect.DUCKDB: CountingAdapter(db)})
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(_plan(), engine)
    assert error.value.reason == "cohort_duplicate_member_key"
    assert len(calls) == 1


def test_composite_keys_follow_declared_order_not_mapping_order(
    db: duckdb.DuckDBPyConnection,
) -> None:
    db.execute(
        "INSERT INTO activity VALUES "
        "(1, 'current', 'north', 10), (2, 'current', 'south', 12), "
        "(1, 'current', 'south', 20), (2, 'current', 'north', 30)"
    )
    cube = Cube(
        name="activity",
        dialect=Dialect.DUCKDB,
        table="activity",
        alias="a",
        measures=[Measure(name="amount", sql="{a}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(name="period", sql="{a}.period", type="string"),
            Dimension(name="region", sql="{a}.region", type="string"),
        ],
    )
    query = CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(
                dimensions=["activity.id", "activity.region"],
                filters=[Filter(dimension="activity.period", op="eq", values=["prior"])],
                order=[("activity.id", "asc")],
            ),
            member_keys=["activity.id", "activity.region"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["activity.id", "activity.region"],
                    measures=["activity.amount"],
                    filters=[Filter(dimension="activity.period", op="eq", values=["current"])],
                ),
                key_mapping={
                    "activity.region": "activity.region",
                    "activity.id": "activity.id",
                },
            )
        ],
    )
    plan = compile_cohort_query(query, {cube.name: cube})
    result = run_cohort(plan, Engine(adapters={Dialect.DUCKDB: _Adapter(db)}))
    assert result.rows == ((1, "north", 10), (2, "south", 12))
    assert result.presence["current"] == (True, True)


def test_empty_cohort_skips_measurement_io(db: duckdb.DuckDBPyConnection) -> None:
    query = _plan().cohort.model_copy(
        update={
            "cohort": _plan().cohort.cohort.model_copy(
                update={
                    "source": SemanticQuery(
                        dimensions=["activity.id", "activity.region"],
                        filters=[Filter(dimension="activity.period", op="eq", values=["absent"])],
                    )
                }
            )
        }
    )
    cube = Cube(
        name="activity",
        dialect=Dialect.DUCKDB,
        table="activity",
        alias="a",
        measures=[Measure(name="amount", sql="{a}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(name="period", sql="{a}.period", type="string"),
            Dimension(name="region", sql="{a}.region", type="string"),
        ],
    )
    plan = compile_cohort_query(query, {cube.name: cube})
    result = run_cohort(plan, Engine(adapters={Dialect.DUCKDB: _Adapter(db)}))
    assert result.rows == ()
    assert result.presence["current"] == ()


def test_nested_engine_runs_reuse_single_admission_slot(db: duckdb.DuckDBPyConnection) -> None:
    engine = Engine(
        adapters={Dialect.DUCKDB: _Adapter(db)},
        limits=ExecutionLimits(max_in_flight=1, admission_timeout=0.1),
    )
    result = run_cohort(_plan(), engine)
    assert len(result.rows) == 2


def test_cohort_execution_timeout_is_shared_across_stages(
    db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr("semql_engine.execution.time", SimpleNamespace(monotonic=lambda: clock[0]))
    calls = 0

    class AdvancingAdapter(_Adapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            nonlocal calls
            calls += 1
            result = super().execute(sql, params)
            clock[0] += 0.6
            return result

    engine = Engine(
        adapters={Dialect.DUCKDB: AdvancingAdapter(db)},
        limits=ExecutionLimits(execution_timeout=1.0),
    )
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(_plan(), engine)
    assert error.value.reason == "execution_deadline_exceeded"
    assert calls == 2


def test_cohort_assembled_output_obeys_value_byte_cap(
    db: duckdb.DuckDBPyConnection,
) -> None:
    plan = _plan(source_limit=1)
    engine = Engine(
        adapters={Dialect.DUCKDB: _Adapter(db)},
        limits=ExecutionLimits(max_value_bytes=10),
    )
    assert len(engine.run(plan.source.plan).rows) == 1
    bind_members = plan.measurements[0].bind_members
    assert bind_members is not None
    bound = bind_members([(1,)])
    assert len(engine.run(bound).rows) == 1
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(plan, engine)
    assert error.value.reason == "execution_value_byte_limit_exceeded"


def test_later_unregistered_measurement_backend_fails_before_selection() -> None:
    prior = Cube(
        name="prior",
        dialect=Dialect.POSTGRES,
        table="prior_activity",
        alias="p",
        dimensions=[Dimension(name="id", sql="{p}.id", type="number")],
    )
    current = Cube(
        name="current",
        dialect=Dialect.BIGQUERY,
        table="current_activity",
        alias="c",
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
        measures=[Measure(name="amount", sql="{c}.amount", agg="sum")],
    )
    request = CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(dimensions=["prior.id"]),
            member_keys=["prior.id"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(dimensions=["current.id"], measures=["current.amount"]),
                key_mapping={"prior.id": "current.id"},
            )
        ],
    )
    plan = compile_cohort_query(request, {"prior": prior, "current": current})
    calls: list[str] = []

    class CountingAdapter:
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(sql)
            return AdapterResult(list(plan.source.columns), [(1,)])

    engine = Engine(adapters={Dialect.POSTGRES: CountingAdapter()})
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(plan, engine)
    assert error.value.reason == "adapter_missing"
    assert calls == []
