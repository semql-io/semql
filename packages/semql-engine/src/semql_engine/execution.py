"""Optional execution limits and accounting primitives."""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

_blocking_completion: ContextVar[Callable[[], Callable[[], None]] | None] = ContextVar(
    "semql_blocking_completion", default=None
)


def _error(message: str, reason: str, stage: str) -> Exception:
    from semql_engine.engine import ExecutionContractError

    return ExecutionContractError(
        message, reason=reason, artifact="execution", operation="execute", stage=stage
    )


@dataclass(frozen=True)
class ExecutionLimits:
    """Unset limits preserve unrestricted defaults. Byte bounds cover deterministic
    row value payloads only, not Python overhead, RSS, driver buffering, or transport.
    """

    max_in_flight: int | None = None
    admission_timeout: float | None = None
    execution_timeout: float | None = None
    max_fragment_rows: int | None = None
    max_fragment_value_bytes: int | None = None
    max_rows: int | None = None
    max_value_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_in_flight",
            "max_fragment_rows",
            "max_fragment_value_bytes",
            "max_rows",
            "max_value_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "max_in_flight" else 0)
            ):
                raise ValueError(f"invalid {name}")
        for name in ("admission_timeout", "execution_timeout"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid {name}")


def value_payload_bytes(value: object, active: set[int] | None = None) -> int:
    """Count supported values, excluding container and object overhead."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, memoryview):
        return value.nbytes
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, (value.bit_length() + 8) // 8)
    if isinstance(value, float):
        return 8
    if isinstance(value, Decimal):
        return len(str(value).encode("ascii"))
    if isinstance(value, uuid.UUID):
        return 16
    if isinstance(value, (dt.date, dt.time, dt.datetime, dt.timedelta)):
        rendered = str(value) if isinstance(value, dt.timedelta) else value.isoformat()
        return len(rendered.encode("ascii"))
    if isinstance(value, (list, tuple, dict)):
        seen: set[int] = active if active is not None else set()
        identity = id(cast(object, value))
        if identity in seen:
            raise _error(
                "Cyclic result value cannot be byte-accounted.",
                "execution_value_unaccountable",
                "output",
            )
        seen.add(identity)
        try:
            if isinstance(value, dict):
                mapping = cast(dict[object, object], value)
                return sum(
                    value_payload_bytes(k, seen) + value_payload_bytes(v, seen)
                    for k, v in mapping.items()
                )
            sequence = cast(list[object] | tuple[object, ...], value)
            return sum(value_payload_bytes(item, seen) for item in sequence)
        finally:
            seen.remove(identity)
    raise _error(
        "Result value type cannot be byte-accounted.", "execution_value_unaccountable", "output"
    )


class ExecutionScope:
    def __init__(self, limits: ExecutionLimits | None) -> None:
        self.limits = limits or ExecutionLimits()
        timeout = self.limits.execution_timeout
        self.deadline = time.monotonic() + timeout if timeout is not None else None

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _error("Execution deadline exceeded.", "execution_deadline_exceeded", "execution")

    def output_budget(self) -> ExecutionBudget:
        return ExecutionBudget(self, materialized=False)

    def fragment_budget(self) -> ExecutionBudget:
        return ExecutionBudget(self, materialized=True)


class ExecutionBudget:
    def __init__(self, scope: ExecutionScope, *, materialized: bool) -> None:
        self.scope = scope
        self.materialized = materialized
        self.rows = self.value_bytes = 0

    def check_row(self, row: object) -> None:
        limits = self.scope.limits
        row_cap = limits.max_fragment_rows if self.materialized else limits.max_rows
        byte_cap = limits.max_fragment_value_bytes if self.materialized else limits.max_value_bytes
        if self.scope.deadline is None and row_cap is None and byte_cap is None:
            return
        self.scope.check_deadline()
        count = self.rows + 1
        size = self.value_bytes
        if byte_cap is not None:
            size += value_payload_bytes(row)
        stage = "materialization" if self.materialized else "output"
        if row_cap is not None and count > row_cap:
            raise _error("Execution row limit exceeded.", "execution_row_limit_exceeded", stage)
        if byte_cap is not None and size > byte_cap:
            raise _error(
                "Execution value-byte limit exceeded.", "execution_value_byte_limit_exceeded", stage
            )
        self.rows, self.value_bytes = count, size


class Admission:
    def __init__(self, limits: ExecutionLimits | None) -> None:
        self._semaphore = (
            threading.BoundedSemaphore(limits.max_in_flight)
            if limits and limits.max_in_flight
            else None
        )
        self._timeout = limits.admission_timeout if limits else None

    def acquire(self) -> None:
        if self._semaphore is not None and not self._semaphore.acquire(timeout=self._timeout):
            raise _error(
                "Execution admission wait limit exceeded.",
                "execution_admission_timeout",
                "admission",
            )

    def release(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()


@contextmanager
def execution_scope(
    limits: ExecutionLimits | None, admission: Admission | None = None
) -> Generator[ExecutionScope]:
    if admission is not None:
        admission.acquire()
    try:
        yield ExecutionScope(limits)
    finally:
        if admission is not None:
            admission.release()


@contextmanager
def blocking_work_owner(factory: Callable[[], Callable[[], None]]) -> Generator[None]:
    token = _blocking_completion.set(factory)
    try:
        yield
    finally:
        _blocking_completion.reset(token)


async def run_blocking_owned[T](
    function: Callable[..., T],
    *args: object,
    on_complete: Callable[[], None] | None = None,
) -> T:
    """Shield and own sync worker work through late completion and exception consumption."""
    factory = _blocking_completion.get()
    callbacks = tuple(
        callback
        for callback in (on_complete, factory() if factory is not None else None)
        if callback is not None
    )
    task = asyncio.create_task(asyncio.to_thread(function, *args))

    def completed(finished: asyncio.Task[T]) -> None:
        try:
            if not finished.cancelled():
                finished.exception()
        finally:
            for callback in callbacks:
                callback()

    task.add_done_callback(completed)
    return await asyncio.shield(task)


async def await_with_deadline[T](
    awaitable: Awaitable[T],
    scope: ExecutionScope,
    *,
    dispose_late_result: Callable[[T], None] | None = None,
) -> T:
    if scope.deadline is None:
        return await awaitable
    timeout = asyncio.timeout(max(0.0, scope.deadline - time.monotonic()))
    try:
        async with timeout:
            result = await awaitable
    except TimeoutError:
        if not timeout.expired():
            raise
        raise _error(
            "Execution deadline exceeded.",
            "execution_deadline_exceeded",
            "execution",
        ) from None
    try:
        scope.check_deadline()
    except BaseException:
        if dispose_late_result is not None:
            dispose_late_result(result)
        raise
    return result
