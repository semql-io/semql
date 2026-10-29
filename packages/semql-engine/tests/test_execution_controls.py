"""Consumer-visible opt-in execution control regressions."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Generator, Mapping
from typing import Any

import pytest
from semql import Cube, Dialect, Dimension, Join, Measure, SemanticQuery, compile_federated_query
from semql.federate import FederatedPlan, MergeSpec
from semql_engine import (
    AdapterResult,
    AsyncEngine,
    Engine,
    ExecutionContractError,
    ExecutionLimits,
    to_async_adapter,
    to_async_merge_engine,
)


def _plan() -> FederatedPlan:
    cube = Cube(
        name="items",
        dialect=Dialect.DUCKDB,
        table="items",
        alias="i",
        primary_key="id",
        measures=[Measure(name="total", sql="{i}.amount", agg="sum")],
        dimensions=[Dimension(name="id", sql="{i}.id", type="number")],
    )
    return compile_federated_query(SemanticQuery(dimensions=["items.id"]), {"items": cube})


class _Rows:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.calls = 0

    def execute(self, sql: str, params: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(self.columns, [(1.0,), (2.0,)], ["DOUBLE"])


def test_fragment_row_cap_accepts_exact_limit_and_rejects_next_row() -> None:
    plan = _plan()
    adapter = _Rows(list(plan.fragments[0].columns))
    Engine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_fragment_rows=2),
    ).run(plan)
    assert adapter.calls == 1

    adapter.calls = 0
    limited = Engine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_fragment_rows=1),
    )
    with pytest.raises(ExecutionContractError) as raised:
        limited.run(plan)
    assert raised.value.reason == "execution_row_limit_exceeded"
    assert adapter.calls == 1


def test_final_row_cap_rejects_complete_result_overflow() -> None:
    plan = _plan()
    adapter = _Rows(list(plan.fragments[0].columns))
    engine = Engine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_rows=0),
    )
    with pytest.raises(ExecutionContractError) as raised:
        engine.run(plan)
    assert raised.value.reason == "execution_row_limit_exceeded"


def test_fragment_value_byte_cap_is_exact_and_incremental() -> None:
    plan = _plan()
    adapter = _Rows(list(plan.fragments[0].columns))
    Engine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_fragment_value_bytes=16),
    ).run(plan)

    adapter.calls = 0
    engine = Engine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_fragment_value_bytes=15),
    )
    with pytest.raises(ExecutionContractError) as raised:
        engine.run(plan)
    assert raised.value.reason == "execution_value_byte_limit_exceeded"
    assert adapter.calls == 1


def _two_backend_plan() -> FederatedPlan:
    items = Cube(
        name="items",
        table="items",
        alias="i",
        dialect=Dialect.DUCKDB,
        measures=[Measure(name="total", sql="{i}.amount", agg="sum")],
        dimensions=[Dimension(name="person_id", sql="{i}.person_id", type="number")],
        joins=[Join(to="people", relationship="many_to_one", on="{i}.person_id = {p}.id")],
    )
    people = Cube(
        name="people",
        table="people",
        alias="p",
        dialect=Dialect.POSTGRES,
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{p}.id", type="number"),
            Dimension(name="label", sql="{p}.label", type="string"),
        ],
    )
    return compile_federated_query(
        SemanticQuery(measures=["items.total"], dimensions=["people.label"]),
        {"items": items, "people": people},
    )


def test_missing_later_adapter_preflight_occurs_before_any_source_call() -> None:
    plan = _two_backend_plan()
    adapter = _Rows(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter})
    with pytest.raises(ExecutionContractError) as raised:
        engine.run(plan)
    assert raised.value.reason == "adapter_missing"
    assert adapter.calls == 0


def test_limited_async_fast_stream_raises_after_yielded_rows() -> None:
    plan = _plan()

    class AsyncRows:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns
            self.calls = 0

        async def execute(self, sql: str, params: Any) -> AdapterResult:
            self.calls += 1
            return AdapterResult(self.columns, [(1.0,), (2.0,)], ["DOUBLE"])

    adapter = AsyncRows(list(plan.fragments[0].columns))
    engine = AsyncEngine(
        adapters={Dialect.DUCKDB: adapter},
        limits=ExecutionLimits(max_rows=1),
    )

    async def consume() -> None:
        stream = engine.iter_run(plan, chunk_rows=1)
        assert await stream.__anext__() == [(1.0,)]
        with pytest.raises(ExecutionContractError) as raised:
            await stream.__anext__()
        assert raised.value.reason == "execution_row_limit_exceeded"
        await stream.aclose()

    asyncio.run(consume())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": True},
        {"max_rows": 1.5},
        {"max_in_flight": 0},
        {"execution_timeout": float("nan")},
    ],
)
def test_invalid_limit_values_reject(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExecutionLimits(**kwargs)


def test_opaque_custom_output_is_accepted_unless_byte_accounting_is_requested() -> None:
    opaque = object()

    class Merge:
        def merge(self, fragment_results: list[AdapterResult], spec: MergeSpec) -> AdapterResult:
            return AdapterResult(["id"], [(opaque,)])

    plan = _plan()
    adapter = _Rows(list(plan.fragments[0].columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=Merge())
    assert engine.run(plan).rows == [(opaque,)]
    limited = Engine(
        adapters={Dialect.DUCKDB: adapter},
        merge_engine=Merge(),
        limits=ExecutionLimits(max_value_bytes=10),
    )
    with pytest.raises(ExecutionContractError) as raised:
        limited.run(plan)
    assert raised.value.reason == "execution_value_unaccountable"


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
def test_timed_out_execution_keeps_admission_until_all_blocking_workers_finish(
    streaming: bool,
) -> None:
    async def exercise() -> None:
        plan = _plan()
        loop = asyncio.get_running_loop()
        started = [asyncio.Event(), asyncio.Event()]
        closed = [asyncio.Event(), asyncio.Event()]
        release = [threading.Event(), threading.Event()]

        class LateRows:
            def __init__(self, index: int) -> None:
                self.index = index

            def __iter__(self) -> LateRows:
                return self

            def __next__(self) -> tuple[int]:
                raise StopIteration

            def close(self) -> None:
                loop.call_soon_threadsafe(closed[self.index].set)

        class BlockingAdapter:
            def __init__(self, index: int) -> None:
                self.index = index

            def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
                loop.call_soon_threadsafe(started[self.index].set)
                if not release[self.index].wait(5):
                    raise AssertionError("Blocking worker was not released")
                return AdapterResult(["id"], LateRows(self.index), ["INTEGER"])

        adapters = [to_async_adapter(BlockingAdapter(index)) for index in range(2)]

        class Fanout:
            async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
                results = await asyncio.gather(
                    *(adapter.execute(sql, params) for adapter in adapters)
                )
                return results[0]

        class Ready:
            async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
                return AdapterResult(["id"], [(1,), (2,)], ["INTEGER"])

        engine = AsyncEngine(
            adapters={Dialect.DUCKDB: Fanout()},
            limits=ExecutionLimits(
                max_in_flight=1,
                admission_timeout=0.02,
                execution_timeout=0.05,
            ),
        )

        async def consume() -> None:
            if streaming:
                stream = engine.iter_run(plan, chunk_rows=1)
                try:
                    async for _chunk in stream:
                        raise AssertionError("Blocked workers cannot produce a completed chunk")
                finally:
                    await stream.aclose()
            else:
                await engine.run(plan)

        operation = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
            with pytest.raises(ExecutionContractError) as timeout:
                await operation
            assert timeout.value.reason == "execution_deadline_exceeded"
            release[0].set()
            await asyncio.wait_for(closed[0].wait(), 2)
            with pytest.raises(ExecutionContractError) as busy:
                await engine.run(plan)
            assert busy.value.reason == "execution_admission_timeout"
            release[1].set()
            await asyncio.wait_for(closed[1].wait(), 2)
            engine.register(Dialect.DUCKDB, Ready())
            assert (await engine.run(plan)).rows == [(1,), (2,)]
        finally:
            for gate in release:
                gate.set()
            if not operation.done():
                operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(exercise())


def test_async_stream_can_advance_and_close_in_other_tasks_with_immediate_cleanup() -> None:
    plan = _plan()
    closed: list[threading.Event] = []

    class Rows:
        def __init__(self) -> None:
            self.values = iter([(1.0,), (2.0,)])
            self.closed = threading.Event()
            closed.append(self.closed)

        def __iter__(self) -> Rows:
            return self

        def __next__(self) -> tuple[float]:
            return next(self.values)

        def close(self) -> None:
            self.closed.set()

    class AsyncRows:
        async def execute(self, sql: str, params: Any) -> AdapterResult:
            return AdapterResult(list(plan.fragments[0].columns), Rows(), ["DOUBLE"])

    async def consume() -> None:
        engine = AsyncEngine(adapters={Dialect.DUCKDB: AsyncRows()})
        stream = engine.iter_run(plan, chunk_rows=1)
        advance = asyncio.create_task(asyncio.wait_for(stream.__anext__(), timeout=1))
        assert await advance == [(1.0,)]
        second_advance = asyncio.create_task(asyncio.wait_for(stream.__anext__(), timeout=1))
        assert await second_advance == [(2.0,)]
        close = asyncio.create_task(stream.aclose())
        await close
        assert closed[0].is_set()

        exhausted = engine.iter_run(plan, chunk_rows=1)
        assert await asyncio.wait_for(exhausted.__anext__(), timeout=1) == [(1.0,)]
        assert await asyncio.wait_for(exhausted.__anext__(), timeout=1) == [(2.0,)]
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(exhausted.__anext__(), timeout=1)
        assert closed[1].is_set()

    asyncio.run(consume())


def test_async_merge_timeout_retains_admission_and_closes_late_result() -> None:
    plan = _plan()
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Rows:
        def __iter__(self) -> Rows:
            return self

        def __next__(self) -> tuple[int]:
            raise StopIteration

        def close(self) -> None:
            closed.set()

    class BlockingMerge:
        def merge(self, fragment_results: list[AdapterResult], spec: MergeSpec) -> AdapterResult:
            started.set()
            if not release.wait(5):
                raise AssertionError("merge worker was not released")
            return AdapterResult(list(plan.columns), Rows())

    async def exercise() -> None:
        engine = AsyncEngine(
            adapters={Dialect.DUCKDB: to_async_adapter(_Rows(list(plan.fragments[0].columns)))},
            merge_engine=to_async_merge_engine(BlockingMerge()),
            limits=ExecutionLimits(max_in_flight=1, admission_timeout=0.02, execution_timeout=0.05),
        )
        operation = asyncio.create_task(engine.run(plan))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            with pytest.raises(ExecutionContractError) as timeout:
                await operation
            assert timeout.value.reason == "execution_deadline_exceeded"
            with pytest.raises(ExecutionContractError) as busy:
                await engine.run(plan)
            assert busy.value.reason == "execution_admission_timeout"
            assert not closed.is_set()
            release.set()
            assert await asyncio.to_thread(closed.wait, 2)
        finally:
            release.set()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(exercise())


def test_async_engine_preserves_driver_timeouterror_with_long_deadline() -> None:
    plan = _plan()

    class FailingAdapter:
        async def execute(self, sql: str, params: Any) -> AdapterResult:
            raise TimeoutError("driver socket timed out")

    async def exercise() -> None:
        engine = AsyncEngine(
            adapters={Dialect.DUCKDB: FailingAdapter()},
            limits=ExecutionLimits(execution_timeout=60),
        )
        with pytest.raises(TimeoutError, match="driver socket timed out") as raised:
            await engine.run(plan)
        assert not isinstance(raised.value, ExecutionContractError)

    asyncio.run(exercise())


def test_empty_sync_adapter_result_checks_deadline_before_next_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _two_backend_plan()
    calls = [0, 0]
    clock = [0.0]
    monkeypatch.setattr("semql_engine.execution.time.monotonic", lambda: clock[0])

    class EmptyAdapter:
        def __init__(self, index: int) -> None:
            self.index = index

        def execute(self, sql: str, params: Any) -> AdapterResult:
            calls[self.index] += 1
            if self.index == 0:
                clock[0] = 20.0
            return AdapterResult(list(plan.fragments[self.index].columns), [])

    engine = Engine(
        adapters={
            Dialect.DUCKDB: EmptyAdapter(0),
            Dialect.POSTGRES: EmptyAdapter(1),
        },
        limits=ExecutionLimits(execution_timeout=10),
    )
    with pytest.raises(ExecutionContractError) as raised:
        engine.run(plan)
    assert raised.value.reason == "execution_deadline_exceeded"
    assert calls == [1, 0]


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "stream"])
def test_inline_fetch_keeps_connection_and_admission_until_worker_finishes(
    streaming: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _two_backend_plan()
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Cursor:
        description = [(name,) for name in plan.columns]

        def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
            started.set()
            if not release.wait(5):
                raise AssertionError("fetch worker was not released")
            return []

    class Connection:
        def execute(self, sql: str, params: Any = None) -> Cursor:
            return Cursor()

        def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
            return None

        def close(self) -> None:
            closed.set()

    connection = Connection()

    @contextlib.contextmanager
    def merge_connection() -> Generator[Connection]:
        try:
            yield connection
        finally:
            connection.close()

    class Ready:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns

        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            return AdapterResult(self.columns, [], ["INTEGER"] * len(self.columns))

    async def exercise() -> None:
        engine = AsyncEngine(
            adapters={
                fragment.dialect: Ready(list(fragment.columns)) for fragment in plan.fragments
            },
            limits=ExecutionLimits(max_in_flight=1, admission_timeout=0.02, execution_timeout=0.05),
        )

        # The fake connection isolates the lifetime boundary from DuckDB scheduling.
        def connection_factory(_n_fragments: int) -> contextlib.AbstractContextManager[Connection]:
            return merge_connection()

        monkeypatch.setattr(engine, "_merge_con", connection_factory)
        stream: Any = None
        if streaming:
            stream = engine.iter_run(plan, chunk_rows=1)
            operation = asyncio.create_task(stream.__anext__())
        else:
            operation = asyncio.create_task(engine.run(plan))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            with pytest.raises(ExecutionContractError) as timeout:
                await operation
            assert timeout.value.reason == "execution_deadline_exceeded"
            if streaming:
                await stream.aclose()
            assert not closed.is_set()
            with pytest.raises(ExecutionContractError) as busy:
                await engine.run(plan)
            assert busy.value.reason == "execution_admission_timeout"
            release.set()
            assert await asyncio.to_thread(closed.wait, 2)
        finally:
            release.set()
            await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(exercise())


def test_empty_sync_merge_result_checks_deadline_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    clock = [0.0]
    monkeypatch.setattr("semql_engine.execution.time.monotonic", lambda: clock[0])

    class EmptyAdapter:
        def execute(self, sql: str, params: Any) -> AdapterResult:
            return AdapterResult(list(plan.fragments[0].columns), [])

    class ExpiredMerge:
        def merge(self, fragment_results: list[AdapterResult], spec: MergeSpec) -> AdapterResult:
            clock[0] = 20.0
            return AdapterResult(list(plan.columns), [])

    engine = Engine(
        adapters={Dialect.DUCKDB: EmptyAdapter()},
        merge_engine=ExpiredMerge(),
        limits=ExecutionLimits(execution_timeout=10),
    )
    with pytest.raises(ExecutionContractError) as raised:
        engine.run(plan)
    assert raised.value.reason == "execution_deadline_exceeded"


@pytest.mark.parametrize("multiple_fragments", [False, True], ids=["single", "multi"])
def test_late_stream_adapter_results_are_closed_before_deadline_rejection(
    monkeypatch: pytest.MonkeyPatch, multiple_fragments: bool
) -> None:
    plan = _two_backend_plan() if multiple_fragments else _plan()
    clock = [0.0]
    monkeypatch.setattr("semql_engine.execution.time.monotonic", lambda: clock[0])
    closed: list[bool] = []

    class Rows:
        def __init__(self) -> None:
            self.index = len(closed)
            closed.append(False)

        def __iter__(self) -> Rows:
            return self

        def __next__(self) -> tuple[int]:
            raise StopIteration

        def close(self) -> None:
            closed[self.index] = True

    class LateAdapter:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns

        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            # Model a native coroutine that blocks the loop past its deadline.
            clock[0] += 20
            return AdapterResult(self.columns, Rows())

    async def exercise() -> None:
        engine = AsyncEngine(
            adapters={
                fragment.dialect: LateAdapter(list(fragment.columns)) for fragment in plan.fragments
            },
            limits=ExecutionLimits(execution_timeout=10),
        )
        stream = engine.iter_run(plan)
        with pytest.raises(ExecutionContractError) as raised:
            await anext(stream)
        assert raised.value.reason == "execution_deadline_exceeded"
        assert closed == [True] * len(plan.fragments)
        await stream.aclose()

    asyncio.run(exercise())
