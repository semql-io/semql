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

:class:`AsyncEngine.iter_run` has a single-fragment fast path that
uses ``FederatedPlan.merge_spec`` to stream one-backend passthrough
plans directly from the adapter, skipping DuckDB's CREATE TABLE +
INSERT roundtrip plus the full second pass over materialised rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, runtime_checkable

import duckdb
from semql.compile import ColumnMeta
from semql.federate import FederatedPlan
from semql.model import Dialect

from semql_engine.adapter import Adapter, AdapterResult, AsyncAdapter


class EngineError(RuntimeError):
    """Raised by the engine when a plan can't be executed.

    Distinct from ``FederationError`` (compile-time refusals): this
    surfaces runtime issues such as a missing adapter for a backend the
    plan references, or an adapter returning rows whose columns don't
    match the fragment's declared output."""


OnExecuteHook = Callable[..., Any]
"""P7 observability hook fired after every ``Engine.run``.

Signature: ``(plan, elapsed_ms, *, cache_hit) -> Any``. The hook
is best-effort: if it raises, the engine still returns the result.
The return value is ignored.

A common use is to ship timing + hit/miss info to a metrics
backend (Prometheus, OpenTelemetry) without coupling the engine
to any one of them. Implemented as ``Callable[..., Any]`` rather
than a strict signature because the keyword-only ``cache_hit``
arg doesn't compose well with the ``Callable[...]`` syntax in
older Pythons; the engine's call site enforces the contract."""


@dataclass
class ExecutionResult:
    """Final result of running a :class:`FederatedPlan`.

    ``columns`` and ``column_meta`` are pass-throughs from the plan so a
    consumer that wants formatted output (units, percent, etc.) has
    everything it needs without re-resolving against the catalog.
    """

    columns: list[str]
    column_meta: list[ColumnMeta]
    rows: list[tuple[Any, ...]]


@dataclass
class _CacheEntry:
    """One result-cache slot: the stored result plus an optional
    monotonic expiry deadline (``None`` = never expires)."""

    result: ExecutionResult
    expires_at: float | None


def _isolate(result: ExecutionResult) -> ExecutionResult:
    """Return a copy that shares no mutable container (or mutable
    element) with ``result``.

    The cache hands a fresh ``ExecutionResult`` to every caller so that
    ``result.rows.sort()`` / ``.append(...)`` / ``columns.pop()`` on one
    consumer can't corrupt the stored entry or any other consumer.
    ``rows`` elements are tuples and ``columns`` elements are strings —
    both immutable, so new lists suffice — but ``column_meta`` holds
    mutable :class:`ColumnMeta` dataclasses, so each is duplicated."""
    return ExecutionResult(
        columns=list(result.columns),
        column_meta=[replace(m) for m in result.column_meta],
        rows=list(result.rows),
    )


def _freeze_param(value: object) -> object:
    """Turn a param value into a hashable, order-preserving key part.

    Adapters may bind container-valued params — a BigQuery
    ``ArrayQueryParameter`` arrives as a Python ``list`` for an
    ``IN (...)`` filter — which makes the raw value unhashable and broke
    the ``key in cache`` lookup. Lists/tuples become tuples (order
    significant: ``[1,2]`` ≠ ``[2,1]`` as SQL params), dicts become
    key-sorted tuples, sets become frozensets; scalars pass through
    unchanged."""
    if isinstance(value, (list, tuple)):
        seq = cast("list[object] | tuple[object, ...]", value)
        return tuple(_freeze_param(v) for v in seq)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        items = [(k, _freeze_param(v)) for k, v in mapping.items()]
        items.sort(key=lambda kv: repr(kv[0]))
        return tuple(items)
    if isinstance(value, (set, frozenset)):
        members = cast("set[object] | frozenset[object]", value)
        return frozenset(_freeze_param(v) for v in members)
    return value


@runtime_checkable
class MergeEngine(Protocol):
    def merge(
        self,
        fragment_results: list[AdapterResult],
        spec: Any,  # noqa: ANN401 — semql.federate.MergeSpec travels across package versions
    ) -> AdapterResult: ...


@runtime_checkable
class AsyncMergeEngine(Protocol):
    async def merge(
        self,
        fragment_results: list[AdapterResult],
        spec: Any,  # noqa: ANN401 — semql.federate.MergeSpec travels across package versions
    ) -> AdapterResult: ...


class _SyncAsAsyncMergeEngine:
    def __init__(self, inner: MergeEngine) -> None:
        self._inner = inner

    async def merge(self, fragment_results: list[AdapterResult], spec: Any) -> AdapterResult:  # noqa: ANN401
        return await asyncio.to_thread(self._inner.merge, fragment_results, spec)


def to_async_merge_engine(engine: MergeEngine) -> AsyncMergeEngine:
    return _SyncAsAsyncMergeEngine(engine)


class DuckDBMergeEngine:
    """Default merge engine marker for the built-in DuckDB merge path."""


class Engine:
    """Runs federated plans by materialising fragments into DuckDB.

    Register one adapter per backend you intend to query against, then
    call :meth:`run`. The engine isn't tied to a specific catalog;
    register adapters once and execute many plans.

    Optional P7 features:

    - ``cache_size``: a positive int enables an LRU result cache. The
      cache key is the plan's emitted shape (merge SQL + params + per-
      fragment SQL + per-fragment params + column list). A cache hit
      skips the per-fragment adapter calls and the DuckDB
      materialise-and-merge. The cache is *in-process*; the engine
      does not invalidate it on catalog mutation. Callers that
      mutate the catalog must call :meth:`clear_cache` themselves.
      Each read and write hands out an isolated copy, so a caller that
      mutates a returned result can't corrupt later hits.

    - ``cache_ttl``: optional time-to-live in seconds (must be > 0). An
      entry older than its TTL is treated as a miss and re-executed —
      useful for "cache for N seconds, then re-check the source". With
      no TTL (the default) entries live until LRU eviction or
      :meth:`clear_cache`.

    - ``on_execute``: a callback fired after every run with timing
      and hit/miss info. The hook is best-effort: if it raises, the
      engine still returns the result.
    """

    def __init__(
        self,
        duckdb_connection: Any | None = None,  # noqa: ANN401
        *,
        adapters: dict[Dialect, Adapter] | None = None,
        merge_engine: MergeEngine | None = None,
        cache_size: int = 0,
        cache_ttl: float | None = None,
        on_execute: OnExecuteHook | None = None,
    ) -> None:
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")
        if cache_ttl is not None and cache_ttl <= 0:
            raise ValueError(f"cache_ttl must be positive when set, got {cache_ttl}")
        self._con: Any = duckdb_connection or duckdb.connect(":memory:")
        self._adapters: dict[Dialect, Adapter] = dict(adapters or {})
        self._merge_engine = merge_engine
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        # OrderedDict gives us insertion-order iteration; popitem(last=False)
        # evicts the oldest entry on overflow (LRU semantics).
        self._cache: OrderedDict[tuple[Any, ...], _CacheEntry] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._on_execute = on_execute
        # Monotonic clock used for TTL deadlines. An attribute (not a
        # hard-coded call) so tests can drive expiry deterministically.
        self._clock: Callable[[], float] = time.monotonic

    def register(self, backend: Dialect, adapter: Adapter) -> None:
        """Bind an adapter to a backend. Replacing an existing
        registration is allowed (so callers can swap adapters mid-flight
        in tests)."""
        self._adapters[backend] = adapter

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def clear_cache(self) -> None:
        """Drop all cached results. The hit/miss counters are *not*
        reset — they're a cumulative view, useful for /metrics."""
        self._cache.clear()

    def _cache_key(self, plan: FederatedPlan) -> tuple[Any, ...]:
        """Build a hashable key from the plan's emitted shape.

        Includes merge SQL + params, per-fragment SQL + params, and the
        output column list. Excludes column metadata (which is
        presentation-layer) and any rendering-only fields."""
        return (
            plan.merge.sql,
            _freeze_param(plan.merge.params),
            tuple((f.backend.value, f.sql, _freeze_param(f.params)) for f in plan.fragments),
            tuple(plan.columns),
        )

    def run(self, plan: FederatedPlan) -> ExecutionResult:
        """Execute a :class:`FederatedPlan` end-to-end.

        For each fragment, runs the SQL via the matching adapter and
        materialises the rows into a DuckDB temp table. Then runs the
        plan's merge SQL and returns the final rows + metadata.

        Raises :class:`EngineError` for missing adapters or column
        mismatches between adapter output and the fragment's declared
        columns.

        If the engine has a cache (P7) and the plan's emitted shape
        matches a prior run, the cached result is returned without
        touching the adapter or DuckDB. The on_execute hook fires
        either way.

        ``cache_misses`` increments on every plan execution, even when
        caching is disabled (``cache_size=0``): a miss is "the engine
        actually ran the plan" — a hit is "the engine returned from
        cache". The two counters are independent and useful for
        /metrics emission."""
        cache_enabled = self._cache_size > 0
        cache_key: tuple[Any, ...] | None = self._cache_key(plan) if cache_enabled else None
        if cache_enabled and cache_key is not None:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if entry.expires_at is not None and self._clock() >= entry.expires_at:
                    # Past its TTL: drop it and fall through to re-execute.
                    del self._cache[cache_key]
                else:
                    # Mark as recently used: pop + reinsert moves to the end.
                    self._cache.move_to_end(cache_key)
                    self._cache_hits += 1
                    self._fire_hook(plan, 0.0, cache_hit=True)
                    # Hand back a private copy so caller mutation can't
                    # poison the stored entry or other consumers.
                    return _isolate(entry.result)

        start = time.perf_counter()
        result = self._execute_uncached(plan)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if cache_enabled and cache_key is not None:
            expires_at = self._clock() + self._cache_ttl if self._cache_ttl is not None else None
            # Store an isolated copy so the result we return to the
            # caller stays independent of the cached one.
            self._cache[cache_key] = _CacheEntry(_isolate(result), expires_at)
            self._cache.move_to_end(cache_key)
            # Evict oldest entry if over capacity.
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self._cache_misses += 1
        self._fire_hook(plan, elapsed_ms, cache_hit=False)
        return result

    def _fire_hook(self, plan: FederatedPlan, elapsed_ms: float, *, cache_hit: bool) -> None:
        if self._on_execute is None:
            return
        with contextlib.suppress(Exception):
            # The hook's exception must not break the engine. We
            # intentionally swallow; callers who want a louder
            # failure mode can wrap their own hook to log + re-raise.
            self._on_execute(plan, elapsed_ms, cache_hit=cache_hit)

    def _execute_uncached(self, plan: FederatedPlan) -> ExecutionResult:
        fragment_results: list[AdapterResult] = []
        for i, fragment in enumerate(plan.fragments):
            adapter = self._adapters.get(fragment.backend)
            if adapter is None:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{fragment.backend.value!r}. Call Engine.register("
                    f"Dialect.{fragment.backend.name}, your_adapter) "
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
            fragment_results.append(result)

        if self._merge_engine is not None:
            merged = self._merge_engine.merge(fragment_results, plan.merge_spec)
            if list(merged.columns) != plan.columns:
                raise EngineError(
                    f"Merge engine returned columns {merged.columns!r} but plan declares "
                    f"{plan.columns!r}."
                )
            return ExecutionResult(
                columns=list(plan.columns),
                column_meta=[replace(m) for m in plan.column_meta],
                rows=[tuple(r) for r in merged.rows],
            )

        self._reset_frag_tables(len(plan.fragments))
        for i, result in enumerate(fragment_results):
            materialised: list[tuple[Any, ...]] = [tuple(r) for r in result.rows]
            self._load_fragment(i, result.columns, materialised)

        merge_cursor = self._con.execute(plan.merge.sql, dict(plan.merge.params))
        rows = merge_cursor.fetchall()
        return ExecutionResult(
            columns=list(plan.columns),
            column_meta=[replace(m) for m in plan.column_meta],
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


_FRAG_TABLE_RE = re.compile(r"\bfrag_(\d+)\b")


def _can_stream_single_fragment(plan: FederatedPlan) -> bool:
    if len(plan.fragments) != 1:
        return False
    spec = plan.merge_spec
    if spec.primary_index != 0 or spec.bridges:
        return False
    if any(measure.merge_agg != "passthrough" for measure in spec.measures):
        return False
    return plan.fragments[0].columns == plan.columns


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
    millions of rows doesn't have to land in memory all at once. For
    single-fragment passthrough plans, the adapter rows stream directly
    from the structural ``merge_spec`` shape. Multi-fragment plans
    continue to merge in DuckDB because that's where the join belongs.

    ``last_iter_run_used_fast_path`` records which path the most-recent
    ``iter_run`` call took. Useful for tests + observability; not part
    of the wire protocol.
    """

    def __init__(
        self,
        duckdb_connection: Any | None = None,  # noqa: ANN401
        *,
        adapters: dict[Dialect, AsyncAdapter] | None = None,
        merge_engine: AsyncMergeEngine | None = None,
    ) -> None:
        self._con: Any = duckdb_connection or duckdb.connect(":memory:")
        self._adapters: dict[Dialect, AsyncAdapter] = dict(adapters or {})
        self._merge_engine = merge_engine
        self.last_iter_run_used_fast_path: bool = False

    def register(self, backend: Dialect, adapter: AsyncAdapter) -> None:
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
            self._validate_result(i, fragment, result)

        if self._merge_engine is not None:
            merged = await self._merge_engine.merge(list(results), plan.merge_spec)
            if list(merged.columns) != plan.columns:
                raise EngineError(
                    f"Merge engine returned columns {merged.columns!r} but plan declares "
                    f"{plan.columns!r}."
                )
            return ExecutionResult(
                columns=plan.columns,
                column_meta=plan.column_meta,
                rows=[tuple(r) for r in merged.rows],
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
        """Run ``plan`` and yield merge result rows in chunks.

        Two paths:

        - **Single-fragment fast path** — when ``merge_spec`` says the
          plan has one fragment and all measures are passthrough, rows
          stream directly from the adapter without DuckDB.
          ``last_iter_run_used_fast_path`` is set to ``True``.
        - **DuckDB merge** — multi-fragment plans, or shapes the fast
          path doesn't recognise (HAVING etc.). Fragments materialise
          into DuckDB temp tables and the merge cursor is fetched via
          ``fetchmany`` for memory-bounded streaming.

        Yields a list of row tuples per iteration; an empty list is
        never emitted — the iterator terminates instead.
        """
        if chunk_rows <= 0:
            raise EngineError(f"iter_run: chunk_rows must be positive, got {chunk_rows!r}.")
        self._adapters_present(plan)
        self.last_iter_run_used_fast_path = False

        if _can_stream_single_fragment(plan):
            fragment = plan.fragments[0]
            self.last_iter_run_used_fast_path = True
            adapter = self._adapters[fragment.backend]
            result = await adapter.execute(fragment.sql, fragment.params)
            self._validate_result(0, fragment, result)
            rows = [tuple(row) for row in result.rows]
            for start in range(0, len(rows), chunk_rows):
                yield rows[start : start + chunk_rows]
            return

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
                    f"Dialect.{frag.backend.name}, your_adapter) before "
                    f"running this plan."
                )

    def _load_result(self, index: int, fragment: Any, result: AdapterResult) -> None:  # noqa: ANN401
        self._validate_result(index, fragment, result)
        materialised: list[tuple[Any, ...]] = [tuple(r) for r in result.rows]
        # Reuse Engine's loader; signature matches.
        Engine._load_fragment(self, index, result.columns, materialised)  # type: ignore[arg-type]

    def _validate_result(self, index: int, fragment: Any, result: AdapterResult) -> None:  # noqa: ANN401
        if set(result.columns) != set(fragment.columns):
            raise EngineError(
                f"Fragment {index} (backend {fragment.backend.value!r}) "
                f"adapter returned columns {result.columns!r} but the "
                f"fragment declares {fragment.columns!r}. Adapter "
                f"must preserve the SELECT-list aliases."
            )

    def _reset_frag_tables(self, n: int) -> None:
        Engine._reset_frag_tables(self, n)  # type: ignore[arg-type]


__all__ = [
    "AsyncEngine",
    "AsyncMergeEngine",
    "DuckDBMergeEngine",
    "Engine",
    "EngineError",
    "ExecutionResult",
    "MergeEngine",
    "to_async_merge_engine",
]
