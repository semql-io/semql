"""Integrity, authorization and dependent-failure boundaries for cohorts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any

import duckdb
import pytest
from semql import (
    AuthContext,
    Cohort,
    CohortComparison,
    CohortMeasurement,
    CohortMetric,
    CohortQuery,
    Cube,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Measure,
    SemanticQuery,
    compile_cohort_query,
)
from semql.errors import CompileError, FederationError
from semql_engine import AdapterResult, DuckDBAdapter, Engine, ExecutionContractError, run_cohort


@pytest.fixture()
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE facts(id INTEGER, region VARCHAR, period VARCHAR, owner VARCHAR, "
            "amount DECIMAL(38, 2))"
        )
        connection.execute(
            "INSERT INTO facts VALUES (1, 'West', 'prior', 'reader', 10), "
            "(2, 'East', 'prior', 'reader', 20), "
            "(1, 'West', 'current', 'reader', 7), "
            "(1, 'West', 'current', 'other', 1000)"
        )
        yield connection


def _cube() -> Cube:
    return Cube(
        name="facts",
        table="facts",
        alias="f",
        dialect=Dialect.DUCKDB,
        security_sql="{f}.owner = {ctx.viewer_id}",
        dimensions=[
            Dimension(name="id", sql="{f}.id", type="number"),
            Dimension(name="region", sql="{f}.region", type="string"),
            Dimension(name="period", sql="{f}.period", type="string"),
        ],
        measures=[Measure(name="amount", sql="{f}.amount", agg="sum")],
    )


def _query() -> CohortQuery:
    return CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(
                dimensions=["facts.id", "facts.region"],
                filters=[Filter(dimension="facts.period", op="eq", values=["prior"])],
                order=[("facts.id", "asc")],
            ),
            member_keys=["facts.id"],
            attributes=["facts.region"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["facts.id", "facts.region"],
                    measures=["facts.amount"],
                    filters=[Filter(dimension="facts.period", op="eq", values=["current"])],
                ),
                key_mapping={"facts.id": "facts.id"},
                attributes=["facts.region"],
            )
        ],
    )


class _CountingAdapter:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.inner = DuckDBAdapter(connection)
        self.calls: list[str] = []

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        self.calls.append(sql)
        return self.inner.execute(sql, params)


def test_null_selection_key_aborts_before_dependent_io(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute("INSERT INTO facts VALUES (NULL, 'West', 'prior', 'reader', 10)")
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))
    adapter = _CountingAdapter(connection)
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(plan, Engine(adapters={Dialect.DUCKDB: adapter}))
    assert error.value.reason == "cohort_null_member_key"
    assert len(adapter.calls) == 1


def test_multivalued_current_attribution_aborts(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("INSERT INTO facts VALUES (1, 'East', 'current', 'reader', 3)")
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(plan, Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}))
    assert error.value.reason == "cohort_duplicate_measurement_key"


@pytest.mark.parametrize("column", ["id", "region"])
def test_masked_identity_or_attribution_rejects_before_execution(column: str) -> None:
    cube = _cube()
    cube = cube.model_copy(
        update={
            "dimensions": [
                field.model_copy(update={"required_roles": ["analyst"], "mask_roles": ["analyst"]})
                if field.name == column
                else field
                for field in cube.dimensions
            ]
        }
    )
    with pytest.raises(FederationError) as error:
        compile_cohort_query(
            _query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader", roles=["analyst"])
        )
    assert error.value.reason == "cohort_masked_identity"


def test_denied_measurement_fails_static_compilation() -> None:
    source = _cube()
    denied = source.model_copy(update={"name": "denied", "required_roles": ["admin"]})
    query = _query()
    measurement = query.measurements[0]
    query = query.model_copy(
        update={
            "measurements": [
                measurement.model_copy(
                    update={
                        "query": SemanticQuery(
                            dimensions=["denied.id"], measures=["denied.amount"]
                        ),
                        "key_mapping": {"facts.id": "denied.id"},
                        "attributes": [],
                    }
                )
            ]
        }
    )
    with pytest.raises(CompileError):
        compile_cohort_query(
            query,
            {source.name: source, denied.name: denied},
            viewer=AuthContext(viewer_id="reader"),
        )


def test_bound_stage_reapplies_dynamic_policy(connection: duckdb.DuckDBPyConnection) -> None:
    allowed = True
    cube = _cube()
    plan = compile_cohort_query(
        _query(),
        {cube.name: cube},
        viewer=AuthContext(viewer_id="reader"),
        policy=lambda _cube, _viewer: allowed,
    )
    allowed = False
    adapter = _CountingAdapter(connection)
    with pytest.raises(CompileError):
        run_cohort(plan, Engine(adapters={Dialect.DUCKDB: adapter}))
    # Selection used the compiled request snapshot; dependent binding must not
    # retain an old policy verdict or execute under a widened identity.
    assert len(adapter.calls) == 1


def test_dependent_adapter_failure_returns_no_partial_result(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))

    class FailingAdapter(_CountingAdapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            if self.calls:
                raise RuntimeError("dependent adapter unavailable")
            return super().execute(sql, params)

    with pytest.raises(RuntimeError, match="dependent adapter unavailable"):
        run_cohort(plan, Engine(adapters={Dialect.DUCKDB: FailingAdapter(connection)}))


def test_actual_tenant_predicates_exclude_same_named_member(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))
    result = run_cohort(plan, Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}))
    assert result.rows == ((1, "West", "West", Decimal("7.00")), (2, "East", None, None))
    assert result.presence["current"] == (True, False)


def test_missing_binder_is_rejected_before_source_io(connection: duckdb.DuckDBPyConnection) -> None:
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))
    plan = replace(plan, measurements=(replace(plan.measurements[0], bind_members=None),))
    adapter = _CountingAdapter(connection)
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(plan, Engine(adapters={Dialect.DUCKDB: adapter}))
    assert error.value.reason == "cohort_binding_missing"
    assert adapter.calls == []


def test_decimal_comparison_preserves_full_driver_precision(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute(
        "UPDATE facts SET amount = 123456789012345678901234567890.12 "
        "WHERE id = 1 AND period = 'current' AND owner = 'reader'"
    )
    query = _query()
    current = query.measurements[0]
    prior = current.model_copy(
        update={
            "name": "prior",
            "query": current.query.model_copy(
                update={"filters": [Filter(dimension="facts.period", op="eq", values=["prior"])]}
            ),
        }
    )
    query = query.model_copy(
        update={
            "measurements": [prior, current],
            "comparisons": [
                CohortComparison(
                    name="delta",
                    left=CohortMetric(measurement="current", column="amount"),
                    right=CohortMetric(measurement="prior", column="amount"),
                )
            ],
        }
    )
    cube = _cube()
    result = run_cohort(
        compile_cohort_query(query, {cube.name: cube}, viewer=AuthContext(viewer_id="reader")),
        Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}),
    )
    assert result.rows[0][-1] == Decimal("123456789012345678901234567880.12")


@pytest.mark.parametrize("projection", ["truncated", "reversed"])
def test_corrupt_composed_projection_rejects_before_io(
    connection: duckdb.DuckDBPyConnection, projection: str
) -> None:
    cube = _cube()
    plan = compile_cohort_query(_query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader"))
    columns = plan.columns[:-1] if projection == "truncated" else tuple(reversed(plan.columns))
    adapter = _CountingAdapter(connection)
    with pytest.raises(ExecutionContractError) as error:
        run_cohort(replace(plan, columns=columns), Engine(adapters={Dialect.DUCKDB: adapter}))
    assert error.value.reason == "cohort_projection_mismatch"
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("logical_type", "sql_type", "first", "second"),
    [
        ("date", "DATE", "2026-10-23", "2026-10-24"),
        ("time", "TIMESTAMP", "2026-10-23T01:02:03", "2026-10-24T01:02:03"),
        ("time", "TIMESTAMPTZ", "2026-10-23T01:02:03+00:00", "2026-10-24T01:02:03+00:00"),
        (
            "uuid",
            "UUID",
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
        ),
        (
            "number",
            "DECIMAL(38,30)",
            "0.123456789012345678901234567890",
            "0.123456789012345678901234567891",
        ),
    ],
)
def test_native_member_keys_bind_losslessly(
    logical_type: str, sql_type: str, first: str, second: str
) -> None:
    if sql_type == "TIMESTAMPTZ":
        # DuckDB's Python driver requires this optional module to fetch aware timestamps.
        pytest.importorskip("pytz")
    cube = Cube(
        name="typed",
        table="typed",
        alias="t",
        dialect=Dialect.DUCKDB,
        dimensions=[
            Dimension.model_validate({"name": "id", "sql": "{t}.id", "type": logical_type}),
            Dimension(name="pick", sql="{t}.pick", type="bool"),
        ],
        measures=[Measure(name="amount", sql="{t}.amount", agg="sum")],
    )
    query = CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(
                dimensions=["typed.id"],
                filters=[Filter(dimension="typed.pick", op="eq", values=[True])],
            ),
            member_keys=["typed.id"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(dimensions=["typed.id"], measures=["typed.amount"]),
                key_mapping={"typed.id": "typed.id"},
            )
        ],
    )
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TABLE typed(id {sql_type}, pick BOOLEAN, amount INTEGER)")
        connection.execute(
            f"INSERT INTO typed VALUES (CAST(? AS {sql_type}), true, 7), "
            f"(CAST(? AS {sql_type}), false, 999)",
            [first, second],
        )
        expected_key = connection.execute("SELECT id FROM typed WHERE pick").fetchone()
        assert expected_key is not None
        result = run_cohort(
            compile_cohort_query(query, {cube.name: cube}),
            Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}),
        )
    assert result.rows == ((expected_key[0], 7),)
    assert result.presence["current"] == (True,)


def test_composite_membership_avoids_cross_product_and_exponential_expansion() -> None:
    def cube(name: str) -> Cube:
        return Cube(
            name=name,
            table=name,
            alias=name[0],
            dialect=Dialect.DUCKDB,
            dimensions=[
                Dimension(name="id", sql=f"{{{name[0]}}}.id", type="number"),
                Dimension(name="region", sql=f"{{{name[0]}}}.region", type="string"),
            ],
            measures=[Measure(name="amount", sql=f"{{{name[0]}}}.amount", agg="sum")],
        )

    source, target = cube("selected"), cube("observed")
    query = CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(dimensions=["selected.id", "selected.region"]),
            member_keys=["selected.id", "selected.region"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["observed.id", "observed.region"], measures=["observed.amount"]
                ),
                key_mapping={"selected.id": "observed.id", "selected.region": "observed.region"},
            )
        ],
    )
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE selected AS SELECT i AS id, 'r' || i AS region, 0 AS amount "
            "FROM range(32) t(i)"
        )
        connection.execute(
            "CREATE TABLE observed AS SELECT i AS id, 'r' || j AS region, 1 AS amount "
            "FROM range(32) t(i) CROSS JOIN range(32) u(j)"
        )
        result = run_cohort(
            compile_cohort_query(query, {source.name: source, target.name: target}),
            Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}),
        )
    assert set(result.rows) == {(i, f"r{i}", 1) for i in range(32)}


def test_masked_measurement_retains_mask_metadata(connection: duckdb.DuckDBPyConnection) -> None:
    cube = _cube()
    cube = cube.model_copy(
        update={
            "measures": [
                cube.measures[0].model_copy(
                    update={"required_roles": ["analyst"], "mask_roles": ["analyst"]}
                )
            ]
        }
    )
    result = run_cohort(
        compile_cohort_query(
            _query(), {cube.name: cube}, viewer=AuthContext(viewer_id="reader", roles=["analyst"])
        ),
        Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}),
    )
    metadata = {column.name: column for column in result.column_meta}
    assert result.rows[0][-1] is None
    assert result.presence["current"] == (True, False)
    assert metadata["current.amount"].masked
    assert not metadata["selection.id"].masked


def test_measurement_arithmetic_does_not_require_comparison_units(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    cube = _cube()
    cube = cube.model_copy(
        update={
            "measures": [
                *cube.measures,
                Measure(name="fee", sql="{f}.amount", agg="sum"),
            ]
        }
    )
    query = _query()
    measurement = query.measurements[0]
    query = query.model_copy(
        update={
            "measurements": [
                measurement.model_copy(
                    update={
                        "query": measurement.query.model_copy(
                            update={
                                "measures": [],
                                "derived_measures": [
                                    InlineDerived(
                                        name="total",
                                        op="sum",
                                        operands=["facts.amount", "facts.fee"],
                                    )
                                ],
                            }
                        )
                    }
                )
            ]
        }
    )
    result = run_cohort(
        compile_cohort_query(query, {cube.name: cube}, viewer=AuthContext(viewer_id="reader")),
        Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)}),
    )
    assert result.rows[0][-1] == Decimal("14")
    assert result.rows[1][-1] is None
