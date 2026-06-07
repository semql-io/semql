"""Public surface of semql-engine."""

from __future__ import annotations

from semql_engine.adapter import Adapter, AdapterResult, DBAPIAdapter, DuckDBAdapter
from semql_engine.engine import Engine, EngineError, ExecutionResult

__all__ = [
    "Adapter",
    "AdapterResult",
    "DBAPIAdapter",
    "DuckDBAdapter",
    "Engine",
    "EngineError",
    "ExecutionResult",
]
