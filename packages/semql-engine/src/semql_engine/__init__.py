"""Public surface of semql-engine."""

from __future__ import annotations

from semql_engine.adapter import (
    Adapter,
    AdapterResult,
    AsyncAdapter,
    AsyncBigQueryAdapter,
    AsyncClickHouseAdapter,
    AsyncDBAPIAdapter,
    AsyncDuckDBAdapter,
    DBAPIAdapter,
    DuckDBAdapter,
    to_async_adapter,
)
from semql_engine.engine import (
    AsyncEngine,
    AsyncExecutionIterator,
    AsyncMergeEngine,
    DuckDBMergeEngine,
    Engine,
    EngineError,
    ExecutionContractError,
    ExecutionResult,
    ExecutionRowIterator,
    MergeEngine,
    MergeKeyValidationEvidence,
    to_async_merge_engine,
)
from semql_engine.rows import (
    InMemoryRowAdapter,
    RowCapableAdapter,
    execute_entity,
)
from semql_engine.semijoin import run_semi_join

__all__ = [
    "Adapter",
    "AdapterResult",
    "AsyncAdapter",
    "AsyncBigQueryAdapter",
    "AsyncClickHouseAdapter",
    "AsyncDBAPIAdapter",
    "AsyncDuckDBAdapter",
    "AsyncEngine",
    "AsyncExecutionIterator",
    "AsyncMergeEngine",
    "DBAPIAdapter",
    "DuckDBMergeEngine",
    "DuckDBAdapter",
    "Engine",
    "EngineError",
    "ExecutionContractError",
    "ExecutionResult",
    "ExecutionRowIterator",
    "MergeKeyValidationEvidence",
    "InMemoryRowAdapter",
    "MergeEngine",
    "RowCapableAdapter",
    "execute_entity",
    "run_semi_join",
    "to_async_adapter",
    "to_async_merge_engine",
]
