"""In-process executor for :class:`semql.FederatedPlan`.

The :class:`Engine` runs each per-backend fragment via a registered
:class:`Adapter`, materialises the resulting rows into in-memory DuckDB
under the tables ``frag_0``, ``frag_1``, … expected by the plan's
``merge.sql``, and finally executes the merge to produce the final
shape.

Single-fragment plans (returned by :func:`semql.compile_federated_query`
when the query touches one backend) are handled identically — the merge
SQL is a trivial ``SELECT * FROM frag_0`` in that case.

The engine keeps a private DuckDB connection. Adapters that are
themselves DuckDB-backed run against their own connections; results
still flow through the engine's connection via the materialisation
step, so isolation is preserved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import duckdb
from semql.compile import ColumnMeta
from semql.federate import FederatedPlan
from semql.model import Backend

from semql_engine.adapter import Adapter, AdapterResult, AsyncAdapter


class EngineError(RuntimeError):
    """Raised by the engine when a plan can't be executed.

    Distinct from ``FederationError`` (compile-time refusals): this
    surfaces runtime issues such as a missing adapter for a backend the
    plan references, or an adapter returning rows whose columns don't
    match the fragment's declared output."""


@dataclass
class ExecutionResult:
    """Final result of running a :class:`FederatedPlan`.

    ``columns`` and ``column_meta`` are pass-throughs from the plan so a
    consumer that wants formatted output (units, percent, etc.) has
    everything it needs without re-resolving against the catalogue.
    """

    columns: list[str]
    column_meta: list[ColumnMeta]
    rows: list[tuple[Any, ...]]


class Engine:
    """Runs federated plans by materialising fragments into DuckDB.

    Register one adapter per backend you intend to query against, then
    call :meth:`run`. The engine isn't tied to a specific catalog;
    register adapters once and execute many plans.
    """

    def __init__(self, duckdb_connection: Any | None = None) -> None:  # noqa: ANN401
        self._con: Any = duckdb_connection or duckdb.connect(":memory:")
        self._adapters: dict[Backend, Adapter] = {}

    def register(self, backend: Backend, adapter: Adapter) -> None:
        """Bind an adapter to a backend. Replacing an existing
        registration is allowed (so callers can swap adapters mid-flight
        in tests)."""
        self._adapters[backend] = adapter

    def run(self, plan: FederatedPlan) -> ExecutionResult:
        """Execute a :class:`FederatedPlan` end-to-end.

        For each fragment, runs the SQL via the matching adapter and
        materialises the rows into a DuckDB temp table. Then runs the
        plan's merge SQL and returns the final rows + metadata.

        Raises :class:`EngineError` for missing adapters or column
        mismatches between adapter output and the fragment's declared
        columns.
        """
        self._reset_frag_tables(len(plan.fragments))
        for i, fragment in enumerate(plan.fragments):
            adapter = self._adapters.get(fragment.backend)
            if adapter is None:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{fragment.backend.value!r}. Call Engine.register("
                    f"Backend.{fragment.backend.name}, your_adapter) "
                    f"before running this plan."
                )
            result = adapter.execute(fragment.sql, fragment.params)
            if set(result.columns) != set(fragment.columns):
                raise EngineError(
                    f"Fragment {i} (backend {fragment.backend.value!r}) "
                    f"adapter returned columns {result.columns!r} but the "
                    f"fragment declares {fragment.columns!r}. Adapter "
                    f"must preserve the SELECT-list aliases."
                )
            materialised: list[tuple[Any, ...]] = [tuple(r) for r in result.rows]
            self._load_fragment(i, result.columns, materialised)

        merge_cursor = self._con.execute(plan.merge.sql, dict(plan.merge.params))
        rows = merge_cursor.fetchall()
        return ExecutionResult(
            columns=plan.columns,
            column_meta=plan.column_meta,
            rows=rows,
        )

    def iter_rows(self, plan: FederatedPlan) -> Iterator[dict[str, Any]]:
        """Convenience: run the plan and yield each row as a
        ``{column: value}`` dict. Useful for callers wiring the result
        into a templating layer / JSON envelope."""
        result = self.run(plan)
        for row in result.rows:
            yield dict(zip(result.columns, row, strict=True))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_frag_tables(self, n: int) -> None:
        """Drop any frag_* tables left over from a previous run so we
        don't accidentally join against stale data. n is conservatively
        larger than needed in case a previous plan had more fragments."""
        # We drop generously to also clean up old runs with more frags.
        # A failed query won't recurse into Python-level state.
        for i in range(max(n, 32)):
            self._con.execute(f"DROP TABLE IF EXISTS frag_{i}")

    def _load_fragment(
        self,
        index: int,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> None:
        """Materialise a fragment's rows into ``frag_<index>``.

        Strategy: infer a DuckDB type per column from the first non-NULL
        value in each column, CREATE TABLE with those types, then
        ``executemany`` the rows. Adapters that return empty result
        sets get a VARCHAR-typed table (we have no per-column type
        info in the adapter contract) — that's fine for merge joins
        that produce an empty result themselves."""
        col_idents = ", ".join(_quote(c) for c in columns)
        types = _infer_column_types(columns, rows)
        type_decls = ", ".join(f"{_quote(c)} {t}" for c, t in zip(columns, types, strict=True))
        self._con.execute(f"CREATE TABLE frag_{index} ({type_decls})")
        if not rows:
            return
        placeholders = ", ".join("?" for _ in columns)
        self._con.executemany(
            f"INSERT INTO frag_{index} ({col_idents}) VALUES ({placeholders})",
            rows,
        )


def _infer_column_types(columns: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
    """Pick a DuckDB type per column from the first non-NULL value.

    Falls back to ``VARCHAR`` for fully-NULL columns and unknown types
    — DuckDB will widen on insert if the data is heterogeneous, and
    callers wanting strict types should cast on the source side."""
    types: list[str] = []
    for col_idx in range(len(columns)):
        chosen = "VARCHAR"
        for row in rows:
            v = row[col_idx]
            if v is None:
                continue
            chosen = _duckdb_type_for(v)
            break
        types.append(chosen)
    return types


def _duckdb_type_for(value: Any) -> str:  # noqa: ANN401 — any row value
    """Map a Python value to a DuckDB type literal.

    Order matters: ``bool`` is a subclass of ``int`` in Python, check
    it first."""
    import datetime as _dt

    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "DOUBLE"
    if isinstance(value, str):
        return "VARCHAR"
    if isinstance(value, _dt.datetime):
        return "TIMESTAMP"
    if isinstance(value, _dt.date):
        return "DATE"
    if isinstance(value, _dt.time):
        return "TIME"
    if isinstance(value, bytes):
        return "BLOB"
    return "VARCHAR"


def _quote(name: str) -> str:
    """DuckDB identifier quoting; matches semql.federate."""
    return f'"{name}"'


class AsyncEngine:
    """Async counterpart to :class:`Engine`.

    Runs federated plans by awaiting per-fragment adapters in parallel
    via :func:`asyncio.gather`, then merging the results in DuckDB.
    Fragments of a single ``FederatedPlan`` are always independent
    (they're per-backend sub-queries; the join lives in the merge SQL),
    so the parallelism is safe for any plan the federation layer
    produces.

    :meth:`iter_run` adds chunked streaming: the merge cursor's rows
    are fetched in batches of ``chunk_rows`` so a result set with
    millions of rows doesn't have to land in memory all at once. The
    merge itself still runs in DuckDB; that's the right shape for >1
    fragment because DuckDB owns the join. (A single-fragment streaming
    fast path that skips DuckDB entirely is documented as a follow-up
    in the gap-analysis proposal.)
    """

    def __init__(self, duckdb_connection: Any | None = None) -> None:  # noqa: ANN401
        self._con: Any = duckdb_connection or duckdb.connect(":memory:")
        self._adapters: dict[Backend, AsyncAdapter] = {}

    def register(self, backend: Backend, adapter: AsyncAdapter) -> None:
        """Bind an async adapter to a backend. Replacing an existing
        registration is allowed."""
        self._adapters[backend] = adapter

    async def run(self, plan: FederatedPlan) -> ExecutionResult:
        """Execute a :class:`FederatedPlan` end-to-end on an event loop.

        Fragments are launched concurrently via :func:`asyncio.gather`;
        a single slow adapter doesn't block the others. Once every
        fragment has returned, results are materialised into DuckDB and
        the merge SQL runs to produce the final shape.

        Raises :class:`EngineError` for missing adapters or column
        mismatches.
        """
        self._adapters_present(plan)
        self._reset_frag_tables(len(plan.fragments))

        results = await asyncio.gather(
            *(
                self._adapters[frag.backend].execute(frag.sql, frag.params)
                for frag in plan.fragments
            )
        )

        for i, (fragment, result) in enumerate(zip(plan.fragments, results, strict=True)):
            self._load_result(i, fragment, result)

        merge_cursor = self._con.execute(plan.merge.sql, dict(plan.merge.params))
        rows = merge_cursor.fetchall()
        return ExecutionResult(
            columns=plan.columns,
            column_meta=plan.column_meta,
            rows=rows,
        )

    async def iter_run(
        self,
        plan: FederatedPlan,
        *,
        chunk_rows: int = 10_000,
    ) -> AsyncIterator[list[tuple[Any, ...]]]:
        """Run ``plan`` and yield merge-cursor rows in chunks of
        ``chunk_rows``.

        Fragments still materialise into DuckDB up front — the merge
        join needs them resident — but the final cursor is pulled
        lazily via ``fetchmany``, so a 10M-row merge doesn't have to
        land in Python memory all at once.

        Yields a list of row tuples per iteration; an empty list is not
        emitted (the iterator terminates instead).
        """
        if chunk_rows <= 0:
            raise EngineError(f"iter_run: chunk_rows must be positive, got {chunk_rows!r}.")
        self._adapters_present(plan)
        self._reset_frag_tables(len(plan.fragments))

        results = await asyncio.gather(
            *(
                self._adapters[frag.backend].execute(frag.sql, frag.params)
                for frag in plan.fragments
            )
        )
        for i, (fragment, result) in enumerate(zip(plan.fragments, results, strict=True)):
            self._load_result(i, fragment, result)

        cursor = self._con.execute(plan.merge.sql, dict(plan.merge.params))
        while True:
            chunk = await asyncio.to_thread(cursor.fetchmany, chunk_rows)
            if not chunk:
                return
            yield [tuple(row) for row in chunk]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _adapters_present(self, plan: FederatedPlan) -> None:
        for frag in plan.fragments:
            if frag.backend not in self._adapters:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{frag.backend.value!r}. Call AsyncEngine.register("
                    f"Backend.{frag.backend.name}, your_adapter) before "
                    f"running this plan."
                )

    def _load_result(self, index: int, fragment: Any, result: AdapterResult) -> None:  # noqa: ANN401
        if set(result.columns) != set(fragment.columns):
            raise EngineError(
                f"Fragment {index} (backend {fragment.backend.value!r}) "
                f"adapter returned columns {result.columns!r} but the "
                f"fragment declares {fragment.columns!r}. Adapter "
                f"must preserve the SELECT-list aliases."
            )
        materialised: list[tuple[Any, ...]] = [tuple(r) for r in result.rows]
        # Reuse Engine's loader; signature matches.
        Engine._load_fragment(self, index, result.columns, materialised)  # type: ignore[arg-type]

    def _reset_frag_tables(self, n: int) -> None:
        Engine._reset_frag_tables(self, n)  # type: ignore[arg-type]


__all__ = ["AsyncEngine", "Engine", "EngineError", "ExecutionResult"]
