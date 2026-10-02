"""End-to-end artifact checks that must dominate cache and adapter calls."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from semql import Cube, Dialect, Dimension, Measure, SemanticQuery, compile_federated_query
from semql.errors import ContractError
from semql.federate import MergeKeyRequirement
from semql.spec import Filter
from semql_engine import AdapterResult, AsyncEngine, Engine, EngineError


def _plan(*, filtered: bool = False) -> Any:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )
    return compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.status"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])]
            if filtered
            else [],
        ),
        {"orders": cube},
    )


class _CountingAdapter:
    def __init__(self, columns: list[str] | None = None) -> None:
        self.calls = 0
        self.columns = columns

    def execute(self, sql: str, params: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(columns=list(self.columns or []), rows=[])


class _AsyncCountingAdapter:
    def __init__(self, columns: list[str] | None = None) -> None:
        self.calls = 0
        self.columns = columns

    async def execute(self, sql: str, params: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(columns=list(self.columns or []), rows=[])


def test_missing_final_binding_fails_before_sync_adapter() -> None:
    plan = _plan(filtered=True)
    plan.fragments[0].params.clear()
    adapter = _CountingAdapter(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter})

    with pytest.raises(EngineError) as raised:
        engine.run(plan)

    assert isinstance(raised.value, ContractError)
    assert raised.value.reason == "binding_values_mismatch"
    assert adapter.calls == 0


def test_renamed_final_binding_fails_before_sync_adapter() -> None:
    plan = _plan(filtered=True)
    params = plan.fragments[0].params
    name, value = next(iter(params.items()))
    del params[name]
    params["renamed"] = value
    adapter = _CountingAdapter(list(plan.fragments[0].columns))

    with pytest.raises(EngineError) as raised:
        Engine(adapters={Dialect.DUCKDB: adapter}).run(plan)

    assert isinstance(raised.value, ContractError)
    assert raised.value.reason == "binding_values_mismatch"
    assert adapter.calls == 0


def test_renamed_final_binding_fails_before_async_adapter() -> None:
    plan = _plan(filtered=True)
    params = plan.fragments[0].params
    name, value = next(iter(params.items()))
    del params[name]
    params["renamed"] = value
    adapter = _AsyncCountingAdapter(list(plan.fragments[0].columns))

    async def run() -> None:
        with pytest.raises(EngineError) as raised:
            await AsyncEngine(adapters={Dialect.DUCKDB: adapter}).run(plan)
        assert isinstance(raised.value, ContractError)
        assert raised.value.reason == "binding_values_mismatch"
        assert adapter.calls == 0

    asyncio.run(run())


def test_missing_final_binding_fails_before_async_adapter() -> None:
    plan = _plan(filtered=True)
    plan.fragments[0].params.clear()
    adapter = _AsyncCountingAdapter(list(plan.fragments[0].columns))
    engine = AsyncEngine(adapters={Dialect.DUCKDB: adapter})

    async def run() -> None:
        with pytest.raises(EngineError) as raised:
            await engine.run(plan)
        assert isinstance(raised.value, ContractError)
        assert raised.value.reason == "binding_values_mismatch"
        assert adapter.calls == 0

    asyncio.run(run())


def test_unsupported_fragment_version_rejects_even_when_cached() -> None:
    plan = _plan()
    adapter = _CountingAdapter(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, cache_size=4)
    engine.run(plan)
    plan.fragments[0].version = 999

    with pytest.raises(EngineError) as raised:
        engine.run(plan)

    assert isinstance(raised.value, ContractError)
    assert raised.value.reason == "artifact_version_unsupported"
    assert adapter.calls == 1


def test_sync_column_order_mismatch_is_a_typed_contract_error() -> None:
    plan = _plan()
    declared = list(plan.fragments[0].columns)
    adapter = _CountingAdapter(list(reversed(declared)))
    engine = Engine(adapters={Dialect.DUCKDB: adapter})

    with pytest.raises(EngineError) as raised:
        engine.run(plan)
    assert isinstance(raised.value, ContractError)

    assert raised.value.reason == "output_columns_mismatch"


def test_invalid_materialized_merge_key_manifest_fails_before_adapter() -> None:
    plan = _plan()
    requirement = MergeKeyRequirement(0, ("not_projected",))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _CountingAdapter(list(plan.fragments[0].columns))

    with pytest.raises(EngineError) as raised:
        Engine(adapters={Dialect.DUCKDB: adapter}).run(plan)

    assert isinstance(raised.value, ContractError)
    assert raised.value.reason == "merge_key_requirement_invalid"
    assert adapter.calls == 0


def test_cache_hit_rebinds_current_semantic_metadata() -> None:
    plan = _plan()
    adapter = _CountingAdapter(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, cache_size=4)
    engine.run(plan)

    marker = object()
    current_meta = [
        replace(meta, display_name=f"current:{meta.display_name}") for meta in plan.column_meta
    ]
    current_plan = replace(plan, analysis=marker, column_meta=current_meta)
    cached = engine.run(current_plan)

    assert adapter.calls == 1
    assert cached.analysis is marker
    assert cached.column_meta == current_meta


def test_iterator_metadata_is_owned_by_its_stream() -> None:
    plan = _plan()
    adapter = _CountingAdapter(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter})
    marker = object()
    plan = replace(plan, analysis=marker)

    stream = engine.iter_rows(plan)
    assert stream.analysis is marker
    assert list(stream) == []
    assert stream.validation_evidence == ()


def test_async_run_and_iterator_carry_current_analysis() -> None:
    plan = _plan()
    adapter = _AsyncCountingAdapter(list(plan.fragments[0].columns))
    engine = AsyncEngine(adapters={Dialect.DUCKDB: adapter})
    marker = object()
    current_plan = replace(plan, analysis=marker)

    async def run() -> None:
        result = await engine.run(current_plan)
        stream = engine.iter_run(current_plan)
        chunks = [chunk async for chunk in stream]
        assert result.analysis is marker
        assert stream.analysis is marker
        assert chunks == []
        assert adapter.calls == 2

    asyncio.run(run())
