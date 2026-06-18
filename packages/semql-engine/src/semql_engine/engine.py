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
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, runtime_checkable

import duckdb
from semql.compile import ColumnMeta
from semql.federate import FederatedPlan
from semql.model import Dialect
from semql.safe import is_read_only_statement

from semql_engine.adapter import Adapter, AdapterResult, AsyncAdapter
from semql_engine.merge import render_merge_sql

# Inline-merge deprecation: the engine's built-in DuckDB merge (render
# the spec and run it on the engine's own connection) is superseded by
# routing through a MergeEngine. ``DuckDBMergeEngine`` is the drop-in
# replacement and will become the default; the inline path warns once
# per engine instance until then.
_INLINE_MERGE_DEPRECATION = (
    "The engine's built-in inline DuckDB merge is deprecated and will be removed in "
    "a future release. Pass merge_engine=DuckDBMergeEngine() (which will become the "
    "default) to route the merge through the MergeEngine protocol."
)


class EngineError(RuntimeError):
    """Raised by the engine when a plan can't be executed.

    Distinct from ``FederationError`` (compile-time refusals): this
    surfaces runtime issues such as a missing adapter for a backend the
    plan references, or an adapter returning rows whose columns don't
    match the fragment's declared output."""


def _assert_fragments_read_only(plan: FederatedPlan) -> None:
    """Defense-in-depth: refuse to execute any fragment that isn't a
    read-only SELECT.

    The compiler emits SELECT by construction, but RawSQL escape hatches
    (``DerivedTable.sql``, ``with_ctes``, ``security_sql``,
    ``ScopePredicate.sql``) splice author-controlled strings into the
    emitted fragments. Checked at the execution choke point — before any
    fragment SQL (or its ``derived_sources``) reaches a driver — per
    PHILOSOPHY, "the defensive guarantee is implemented in the recipe."
    """
    for i, frag in enumerate(plan.fragments):
        for sql in (frag.sql, *frag.derived_sources):
            if not is_read_only_statement(sql, dialect=frag.dialect.value):
                raise EngineError(
                    f"Fragment {i} (backend {frag.dialect.value!r}) is not a "
                    f"read-only SELECT; refusing to execute."
                )


def _assert_merge_read_only(merge_sql: str) -> None:
    """Refuse to run merge SQL that isn't a read-only SELECT. Checked
    immediately before the DuckDB merge executes — paths that never run
    the merge SQL (the single-fragment fast path, a custom merge engine)
    skip this by construction."""
    if not is_read_only_statement(merge_sql, dialect="duckdb"):
        raise EngineError("Merge SQL is not a read-only SELECT; refusing to execute.")


OnExecuteHook = Callable[..., Any]
"""Observability hook fired after every ``Engine.run``.

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


def _merge_spec_key(spec: Any) -> tuple[Any, ...]:  # noqa: ANN401 — MergeSpec travels across versions
    """Stable, hashable key for a ``MergeSpec``.

    Captures everything the DuckDB renderer reads — the same content that
    used to be keyed via the rendered merge SQL + params. Nested frozen
    dataclasses / pydantic filters have deterministic ``repr``, so a
    repr of the list-valued fields is a faithful, collision-free digest;
    ``cross_partition_clauses`` is already a hashable tuple of values."""
    return (
        spec.primary_index,
        spec.mode,
        spec.limit,
        spec.offset,
        repr(spec.bridges),
        repr(spec.dimensions),
        repr(spec.measures),
        repr(spec.having),
        tuple(spec.order_by),
        spec.cross_partition_clauses,
    )


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
    """Merge engine that renders a ``MergeSpec`` to DuckDB SQL and runs it.

    The structural counterpart to the engine's built-in inline merge:
    each fragment result is materialised into a private in-memory DuckDB
    connection, the spec is rendered via :func:`render_merge_sql`,
    executed, and the merged rows returned. Pass it as ``merge_engine=``
    to route the merge through the :class:`MergeEngine` protocol instead
    of the (now deprecated) inline path — it will become the default.

    Holds no state and opens a fresh connection per ``merge`` call, so a
    single instance is safe to share across threads and concurrent runs.
    """

    def merge(
        self,
        fragment_results: list[AdapterResult],
        spec: Any,  # noqa: ANN401 — semql.federate.MergeSpec travels across package versions
    ) -> AdapterResult:
        sql, params = render_merge_sql(spec)
        _assert_merge_read_only(sql)
        con = duckdb.connect(":memory:")
        try:
            for i, result in enumerate(fragment_results):
                _load_fragment_into(con, i, result.columns, [tuple(r) for r in result.rows])
            cursor = con.execute(sql, params)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        finally:
            con.close()
        return AdapterResult(columns=columns, rows=rows)


class Engine:
    """Runs federated plans by materialising fragments into DuckDB.

    Register one adapter per backend you intend to query against, then
    call :meth:`run`. The engine isn't tied to a specific catalog;
    register adapters once and execute many plans.

    Optional features:

    - ``cache_size``: a positive int enables an LRU result cache. The
      cache key is the plan's emitted shape (merge SQL + params + per-
      fragment SQL + per-fragment params + column list). A cache hit
      skips the per-fragment adapter calls and the DuckDB
      materialise-and-merge. The cache is *in-process*; the engine
      does not invalidate it on catalog mutation. Callers that
      mutate the catalog must call :meth:`clear_cache` themselves.
      Each read and write hands out an isolated copy, so a caller that
      mutates a returned result can't corrupt later hits.

      **The cache key is viewer-blind.** It is the *compiled plan*, not
      the identity that produced it. This is safe for the scoping the
      compiler applies, because every viewer-dependent decision lands in
      the plan: schema-tenancy puts the tenant in the SQL, discriminator
      tenancy and ``{ctx.X}`` / ``{ctx.viewer_id}`` predicates bind it as
      a parameter — both are part of the key, so two viewers who *should*
      see different rows compile to different keys and never collide.
      The hazard is scoping the engine can't see: if you apply row
      filtering *outside* the compiled plan (post-filter the rows, or
      reuse one Engine across trust boundaries), identical plans will
      share a slot across viewers. Pass ``cache_namespace`` (e.g. the
      viewer id or tenant) to :meth:`run` to partition the cache along
      that boundary, or give each trust boundary its own Engine.

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
        # Inline-merge deprecation warning fires once per instance.
        self._warned_inline = False

    def _warn_inline_once(self) -> None:
        if not self._warned_inline:
            self._warned_inline = True
            warnings.warn(_INLINE_MERGE_DEPRECATION, DeprecationWarning, stacklevel=3)

    def register(self, dialect: Dialect, adapter: Adapter) -> None:
        """Bind an adapter to a backend. Replacing an existing
        registration is allowed (so callers can swap adapters mid-flight
        in tests)."""
        self._adapters[dialect] = adapter

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

    def _cache_key(self, plan: FederatedPlan, namespace: str | None) -> tuple[Any, ...]:
        """Build a hashable key from the plan's emitted shape.

        Includes the merge spec (its structural ``repr`` — capturing the
        merge SQL the renderer would produce, including bound values),
        per-fragment SQL + params, and the output column list. Excludes
        column metadata (presentation-layer). ``namespace`` — the
        caller's optional partition key (viewer / tenant) — leads the
        tuple so distinct namespaces never share a slot."""
        return (
            namespace,
            _merge_spec_key(plan.merge_spec),
            tuple((f.dialect.value, f.sql, _freeze_param(f.params)) for f in plan.fragments),
            tuple(plan.columns),
        )

    def run(self, plan: FederatedPlan, *, cache_namespace: str | None = None) -> ExecutionResult:
        """Execute a :class:`FederatedPlan` end-to-end.

        For each fragment, runs the SQL via the matching adapter and
        materialises the rows into a DuckDB temp table. Then runs the
        plan's merge SQL and returns the final rows + metadata.

        Raises :class:`EngineError` for missing adapters or column
        mismatches between adapter output and the fragment's declared
        columns.

        If the engine has a cache and the plan's emitted shape
        matches a prior run, the cached result is returned without
        touching the adapter or DuckDB. The on_execute hook fires
        either way.

        ``cache_misses`` increments on every plan execution, even when
        caching is disabled (``cache_size=0``): a miss is "the engine
        actually ran the plan" — a hit is "the engine returned from
        cache". The two counters are independent and useful for
        /metrics emission."""
        cache_enabled = self._cache_size > 0
        cache_key: tuple[Any, ...] | None = (
            self._cache_key(plan, cache_namespace) if cache_enabled else None
        )
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
        _assert_fragments_read_only(plan)
        fragment_results: list[AdapterResult] = []
        for i, fragment in enumerate(plan.fragments):
            adapter = self._adapters.get(fragment.dialect)
            if adapter is None:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{fragment.dialect.value!r}. Call Engine.register("
                    f"Dialect.{fragment.dialect.name}, your_adapter) "
                    f"before running this plan."
                )
            result = adapter.execute(fragment.sql, fragment.params)
            if set(result.columns) != set(fragment.columns):
                raise EngineError(
                    f"Fragment {i} (backend {fragment.dialect.value!r}) "
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

        self._warn_inline_once()
        merge_sql, merge_params = render_merge_sql(plan.merge_spec)
        _assert_merge_read_only(merge_sql)
        merge_cursor = self._con.execute(merge_sql, dict(merge_params))
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
        don't accidentally join against stale data. The sync engine
        runs calls sequentially on one connection, so reuse is safe."""
        _reset_frag_tables_on(self._con, n)

    def _load_fragment(
        self,
        index: int,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> None:
        """Materialise a fragment's rows into ``frag_<index>`` on the
        engine's connection."""
        _load_fragment_into(self._con, index, columns, rows)


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
    return '"' + name.replace('"', '""') + '"'


def _reset_frag_tables_on(con: Any, n: int) -> None:  # noqa: ANN401 — duckdb conn
    """Drop any ``frag_*`` tables on ``con`` so a merge can't join stale
    data. ``n`` is over-dropped (max(n, 32)) to clean up larger prior
    plans on a reused connection."""
    for i in range(max(n, 32)):
        con.execute(f"DROP TABLE IF EXISTS frag_{i}")


def _load_fragment_into(
    con: Any,  # noqa: ANN401 — duckdb conn
    index: int,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> None:
    """Materialise a fragment's rows into ``frag_<index>`` on ``con``.

    Infers a DuckDB type per column from the first non-NULL value, then
    ``executemany``s the rows. Empty result sets get a VARCHAR-typed
    table (no per-column type info in the adapter contract)."""
    col_idents = ", ".join(_quote(c) for c in columns)
    types = _infer_column_types(columns, rows)
    type_decls = ", ".join(f"{_quote(c)} {t}" for c, t in zip(columns, types, strict=True))
    con.execute(f"CREATE TABLE frag_{index} ({type_decls})")
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO frag_{index} ({col_idents}) VALUES ({placeholders})",
        rows,
    )


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
        # The merge step is per-call scratch over fixed ``frag_<i>`` table
        # names. A single shared connection would let two in-flight run()
        # coroutines (the normal FastAPI fan-out) race on the same tables
        # and return wrong results. So each call gets its OWN isolated
        # in-memory connection by default. A caller-supplied connection is
        # honoured for backwards-compat but is NOT safe under concurrent
        # run()/iter_run() on one instance — omit it to get isolation.
        self._user_con: Any = duckdb_connection
        self._adapters: dict[Dialect, AsyncAdapter] = dict(adapters or {})
        self._merge_engine = merge_engine
        self.last_iter_run_used_fast_path: bool = False
        self._warned_inline = False

    def _warn_inline_once(self) -> None:
        if not self._warned_inline:
            self._warned_inline = True
            warnings.warn(_INLINE_MERGE_DEPRECATION, DeprecationWarning, stacklevel=3)

    @contextlib.contextmanager
    def _merge_con(self, n_fragments: int) -> Generator[Any]:
        """Yield the DuckDB connection to materialise + merge into.

        Default: a fresh ``:memory:`` connection per call (own catalog →
        ``frag_<i>`` tables can't collide across concurrent calls), closed
        on exit. If the caller supplied a connection, reuse it (resetting
        stale frag tables first) and leave it open — that path trades
        isolation for the caller's control and isn't concurrency-safe."""
        if self._user_con is not None:
            _reset_frag_tables_on(self._user_con, n_fragments)
            yield self._user_con
            return
        con = duckdb.connect(":memory:")
        try:
            yield con
        finally:
            con.close()

    def register(self, dialect: Dialect, adapter: AsyncAdapter) -> None:
        """Bind an async adapter to a backend. Replacing an existing
        registration is allowed."""
        self._adapters[dialect] = adapter

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
        _assert_fragments_read_only(plan)

        results = await asyncio.gather(
            *(
                self._adapters[frag.dialect].execute(frag.sql, frag.params)
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

        with self._merge_con(len(plan.fragments)) as con:
            for i, result in enumerate(results):
                _load_fragment_into(con, i, result.columns, [tuple(r) for r in result.rows])
            self._warn_inline_once()
            merge_sql, merge_params = render_merge_sql(plan.merge_spec)
            _assert_merge_read_only(merge_sql)
            rows = con.execute(merge_sql, dict(merge_params)).fetchall()
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
        _assert_fragments_read_only(plan)
        self.last_iter_run_used_fast_path = False

        if _can_stream_single_fragment(plan):
            fragment = plan.fragments[0]
            self.last_iter_run_used_fast_path = True
            adapter = self._adapters[fragment.dialect]
            result = await adapter.execute(fragment.sql, fragment.params)
            self._validate_result(0, fragment, result)
            rows = [tuple(row) for row in result.rows]
            for start in range(0, len(rows), chunk_rows):
                yield rows[start : start + chunk_rows]
            return

        results = await asyncio.gather(
            *(
                self._adapters[frag.dialect].execute(frag.sql, frag.params)
                for frag in plan.fragments
            )
        )
        for i, (fragment, result) in enumerate(zip(plan.fragments, results, strict=True)):
            self._validate_result(i, fragment, result)

        # The per-call connection stays open for the whole streaming loop;
        # the contextmanager closes (or releases) it when the generator
        # finishes or is closed early.
        with self._merge_con(len(plan.fragments)) as con:
            for i, result in enumerate(results):
                _load_fragment_into(con, i, result.columns, [tuple(r) for r in result.rows])
            self._warn_inline_once()
            merge_sql, merge_params = render_merge_sql(plan.merge_spec)
            _assert_merge_read_only(merge_sql)
            cursor = con.execute(merge_sql, dict(merge_params))
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
            if frag.dialect not in self._adapters:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{frag.dialect.value!r}. Call AsyncEngine.register("
                    f"Dialect.{frag.dialect.name}, your_adapter) before "
                    f"running this plan."
                )

    def _validate_result(self, index: int, fragment: Any, result: AdapterResult) -> None:  # noqa: ANN401
        if set(result.columns) != set(fragment.columns):
            raise EngineError(
                f"Fragment {index} (backend {fragment.dialect.value!r}) "
                f"adapter returned columns {result.columns!r} but the "
                f"fragment declares {fragment.columns!r}. Adapter "
                f"must preserve the SELECT-list aliases."
            )


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
