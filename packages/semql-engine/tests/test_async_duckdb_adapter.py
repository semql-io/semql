"""Lifecycle tests for the synchronous DuckDB adapter bridge."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Iterator
from typing import Any

import pytest
from semql_engine.adapter import Adapter, AdapterResult, AsyncDuckDBAdapter, to_async_adapter


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _CloseableIterator:
    def __init__(self, closed: threading.Event) -> None:
        self._closed = closed
        self._yielded = False

    def __iter__(self) -> _CloseableIterator:
        return self

    def __next__(self) -> tuple[int]:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return (1,)

    def close(self) -> None:
        self._closed.set()


class _CloseableRows:
    def __init__(self, closed: threading.Event) -> None:
        self._closed = closed

    def __iter__(self) -> Iterator[tuple[int]]:
        return _CloseableIterator(self._closed)


class _BlockingCursor:
    description = [("value", int)]

    def __init__(self, rows: _CloseableRows) -> None:
        self._rows = rows

    def fetchall(self) -> _CloseableRows:
        return self._rows


class _BlockingConnection:
    def __init__(
        self,
        started: threading.Event,
        release: threading.Event,
        rows: _CloseableRows,
    ) -> None:
        self._started = started
        self._release = release
        self._rows = rows

    def execute(self, sql: str, params: Any = None) -> _BlockingCursor:
        _ = sql, params
        self._started.set()
        self._release.wait()
        return _BlockingCursor(self._rows)


async def _cancel_and_wait_for_late_close(
    adapter: Any,
    started: threading.Event,
    release: threading.Event,
    closed: threading.Event,
) -> None:
    task = asyncio.create_task(adapter.execute("SELECT 1", {}))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(closed.wait, 2)


def test_async_duckdb_adapter_closes_rows_from_late_result() -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    rows = _CloseableRows(closed)
    adapter = AsyncDuckDBAdapter(_BlockingConnection(started, release, rows))

    try:
        _run(_cancel_and_wait_for_late_close(adapter, started, release, closed))
    finally:
        release.set()


def test_to_async_adapter_closes_rows_from_late_result() -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class BlockingAdapter(Adapter):
        def execute(self, sql: str, params: Any) -> AdapterResult:
            _ = sql, params
            started.set()
            release.wait()
            return AdapterResult(columns=["value"], rows=_CloseableRows(closed))

    adapter = to_async_adapter(BlockingAdapter())
    try:
        _run(_cancel_and_wait_for_late_close(adapter, started, release, closed))
    finally:
        release.set()


def test_to_async_adapter_consumes_worker_error_after_waiter_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors: list[dict[str, object]] = []

    class FailingAdapter(Adapter):
        def execute(self, sql: str, params: Any) -> AdapterResult:
            _ = sql, params
            started.set()
            release.wait()
            finished.set()
            raise RuntimeError("late driver failure")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        task = asyncio.create_task(to_async_adapter(FailingAdapter()).execute("SELECT 1", {}))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await asyncio.sleep(0)
        assert errors == []

    try:
        _run(scenario())
    finally:
        release.set()
