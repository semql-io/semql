"""Public surface of semql-engine."""

from __future__ import annotations

from semql_engine.adapter import (
    Adapter,
    AdapterResult,
    AsyncAdapter,
    AsyncDuckDBAdapter,
    DBAPIAdapter,
    DuckDBAdapter,
    to_async_adapter,
)
from semql_engine.engine import AsyncEngine, Engine, EngineError, ExecutionResult

__all__ = [
    "Adapter",
    "AdapterResult",
    "AsyncAdapter",
    "AsyncDuckDBAdapter",
    "AsyncEngine",
    "DBAPIAdapter",
    "DuckDBAdapter",
    "Engine",
    "EngineError",
    "ExecutionResult",
    "to_async_adapter",
]
