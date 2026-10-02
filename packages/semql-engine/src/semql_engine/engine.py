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
import time
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

import duckdb
from semql.analysis import SemanticAnalysis
from semql.bindings import (
    BindingRequirement,
    final_binding_requirements,
    validate_bindings,
)
from semql.compile import COMPILED_QUERY_VERSION, ColumnMeta, CompiledQuery
from semql.errors import ContractError
from semql.federate import FEDERATED_PLAN_VERSION, FederatedPlan, MergeSpec
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


class ExecutionContractError(ContractError, EngineError):
    """Typed execution failure that remains catchable as EngineError."""

    _payload_code = "ContractError"

    @classmethod
    def from_contract_error(cls, error: ContractError) -> ExecutionContractError:
        return cls(
            str(error),
            reason=error.reason,
            artifact=error.artifact,
            names=error.names,
            references=error.references,
            operation=error.operation,
            stage=error.stage,
        )


def _execution_final_binding_requirements(
    sql: str, dialect: Dialect, params: Mapping[str, Any]
) -> tuple[BindingRequirement, ...]:
    try:
        return final_binding_requirements(sql, dialect, params)
    except ContractError as error:
        raise ExecutionContractError.from_contract_error(error) from error


def _execution_validate_bindings(
    sql: str,
    dialect: Dialect,
    params: Mapping[str, Any],
    requirements: Sequence[BindingRequirement],
    *,
    artifact: str,
) -> None:
    try:
        validate_bindings(sql, dialect, params, requirements, artifact=artifact)
    except ContractError as error:
        raise ExecutionContractError.from_contract_error(error) from error


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


def _preflight(plan: FederatedPlan) -> tuple[str, dict[str, Any]]:
    """Validate versions, fragment binds, and rendered merge SQL before I/O."""
    if plan.version != FEDERATED_PLAN_VERSION:
        raise ExecutionContractError(
            "Federated plan uses an unsupported version.",
            reason="artifact_version_unsupported",
            artifact="federated_plan",
            operation="execute",
            stage="preflight",
        )
    for index, fragment in enumerate(plan.fragments):
        artifact_name = f"fragment_{index}"
        if fragment.version != COMPILED_QUERY_VERSION:
            raise ExecutionContractError(
                "Compiled fragment uses an unsupported version.",
                reason="artifact_version_unsupported",
                artifact=artifact_name,
                operation="execute",
                stage="preflight",
            )
        requirements = fragment.binding_requirements
        if requirements is None:
            raise ExecutionContractError(
                "Compiled fragment is missing its required binding manifest.",
                reason="binding_manifest_missing",
                artifact=artifact_name,
                operation="execute",
                stage="preflight",
            )
        _execution_validate_bindings(
            fragment.sql,
            fragment.dialect,
            fragment.params,
            requirements,
            artifact=artifact_name,
        )
    for requirement in plan.merge_spec.merge_key_requirements:
        index = requirement.fragment_index
        names = tuple(requirement.columns)
        if (
            index < 0
            or index >= len(plan.fragments)
            or not names
            or len(set(names)) != len(names)
            or not set(names).issubset(plan.fragments[index].columns)
        ):
            raise ExecutionContractError(
                "Merge-key requirement does not match a fragment projection.",
                reason="merge_key_requirement_invalid",
                artifact="merge",
                names=names,
                references=(f"fragment:{index}",),
                operation="merge",
                stage="preflight",
            )
    merge_sql, merge_params = render_merge_sql(plan.merge_spec)
    merge_requirements = _execution_final_binding_requirements(
        merge_sql, Dialect.DUCKDB, merge_params
    )
    _execution_validate_bindings(
        merge_sql,
        Dialect.DUCKDB,
        merge_params,
        merge_requirements,
        artifact="merge",
    )
    _assert_merge_read_only(merge_sql)
    return merge_sql, merge_params


def _plan_analysis(plan: FederatedPlan) -> SemanticAnalysis:
    return plan.analysis


def _materialize_result(result: AdapterResult) -> _MaterializedAdapterResult:
    """Consume adapter rows once so evidence checks and merge share rows."""
    return _MaterializedAdapterResult(
        columns=list(result.columns),
        rows=[tuple(row) for row in result.rows],
    )


def _duckdb_key_value(value: object, duckdb_type: str, null_sentinel: object) -> object:
    """Normalize supported keys as DuckDB stores them; reject uncertain casts."""
    if value is None:
        return null_sentinel
    import datetime as dt

    if duckdb_type == "BOOLEAN":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.casefold() in {"true", "false"}:
            return value.casefold() == "true"
    if duckdb_type == "BIGINT":
        if isinstance(value, (int, bool)):
            return int(value)
        if isinstance(value, str):
            return int(value)
    if duckdb_type == "DOUBLE" and isinstance(value, (int, float, str)):
        converted = float(value)
        return ("DOUBLE", "NaN") if converted != converted else converted
    if duckdb_type == "VARCHAR" and isinstance(value, str):
        return value
    if duckdb_type == "DATE" and isinstance(value, (dt.date, dt.datetime)):
        return value.date() if isinstance(value, dt.datetime) else value
    if duckdb_type == "TIMESTAMP" and isinstance(value, dt.datetime):
        return value
    if duckdb_type == "TIME" and isinstance(value, dt.time):
        return value
    if duckdb_type == "BLOB" and isinstance(value, bytes):
        return value
    raise TypeError("value cannot be normalized without relying on an implicit cast")


def _validate_merge_keys(
    spec: MergeSpec,
    fragment_results: Sequence[AdapterResult | _MaterializedAdapterResult],
) -> tuple[MergeKeyValidationEvidence, ...]:
    evidence: list[MergeKeyValidationEvidence] = []
    for requirement in spec.merge_key_requirements:
        index = requirement.fragment_index
        if index < 0 or index >= len(fragment_results):
            raise ExecutionContractError(
                "Merge-key requirement refers to an absent materialized fragment.",
                reason="merge_key_requirement_invalid",
                artifact="merge",
                references=(f"fragment:{index}",),
                operation="merge",
                stage="preflight",
            )
        result = fragment_results[index]
        columns = tuple(requirement.columns)
        if not set(columns).issubset(result.columns):
            raise ExecutionContractError(
                "Merge-key columns are absent from the materialized fragment.",
                reason="merge_key_columns_mismatch",
                artifact=f"fragment_{index}",
                names=columns,
                references=(f"fragment:{index}",),
                operation="merge",
                stage="preflight",
            )
        positions = tuple(result.columns.index(name) for name in columns)
        types = _infer_column_types(list(result.columns), [tuple(row) for row in result.rows])
        null_sentinel = object()
        seen: set[tuple[object, ...]] = set()
        for row in result.rows:
            key = tuple(row[position] for position in positions)
            if not requirement.nulls_equal and any(value is None for value in key):
                continue
            try:
                normalized = tuple(
                    _duckdb_key_value(value, types[position], null_sentinel)
                    for position, value in zip(positions, key, strict=True)
                )
                duplicate = normalized in seen
                if not duplicate:
                    seen.add(normalized)
            except (TypeError, ValueError, OverflowError):
                raise ExecutionContractError(
                    "Materialized merge-key values cannot be validated.",
                    reason="merge_key_validation_uncheckable",
                    artifact=f"fragment_{index}",
                    names=columns,
                    references=(f"fragment:{index}",),
                    operation="merge",
                    stage="materialized_validation",
                ) from None
            if duplicate:
                raise ExecutionContractError(
                    "Materialized merge-key uniqueness declaration was violated.",
                    reason="merge_key_uniqueness_violation",
                    artifact=f"fragment_{index}",
                    names=columns,
                    references=(f"fragment:{index}",),
                    operation="merge",
                    stage="materialized_validation",
                )
        evidence.append(MergeKeyValidationEvidence(index, columns, "validated"))
    return tuple(evidence)


@dataclass
class _MaterializedAdapterResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


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


@dataclass(frozen=True)
class MergeKeyValidationEvidence:
    fragment_index: int
    columns: tuple[str, ...]
    status: Literal["validated"]


@dataclass
class ExecutionResult:
    """Final result of running a :class:`FederatedPlan`."""

    columns: list[str]
    column_meta: list[ColumnMeta]
    rows: list[tuple[Any, ...]]
    analysis: SemanticAnalysis = SemanticAnalysis.unavailable()
    validation_evidence: tuple[MergeKeyValidationEvidence, ...] = ()


class ExecutionRowIterator(Iterator[dict[str, Any]]):
    """Sync row iterator with immutable per-stream analysis and evidence."""

    def __init__(
        self,
        rows: Iterator[dict[str, Any]],
        analysis: SemanticAnalysis,
        validation_evidence: tuple[MergeKeyValidationEvidence, ...] = (),
    ) -> None:
        self.analysis = analysis
        self.validation_evidence = validation_evidence
        self._rows = rows

    def __iter__(self) -> ExecutionRowIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        return next(self._rows)

    def close(self) -> None:
        """Close the wrapped row iterator when it exposes a close method."""
        close = getattr(self._rows, "close", None)
        if close is not None:
            close()


class AsyncExecutionIterator(AsyncIterator[list[tuple[Any, ...]]]):
    """Async chunk iterator with per-stream analysis and validation evidence."""

    def __init__(
        self,
        chunks: AsyncIterator[list[tuple[Any, ...]]],
        analysis: SemanticAnalysis,
        validation_evidence: tuple[MergeKeyValidationEvidence, ...] = (),
    ) -> None:
        self.analysis = analysis
        self.validation_evidence = validation_evidence
        self._chunks = chunks

    def __aiter__(self) -> AsyncExecutionIterator:
        return self

    async def __anext__(self) -> list[tuple[Any, ...]]:
        return await self._chunks.__anext__()

    async def aclose(self) -> None:
        """Close the wrapped async generator when iteration ends early."""
        close = getattr(self._chunks, "aclose", None)
        if close is not None:
            await close()


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
        analysis=result.analysis,
        validation_evidence=result.validation_evidence,
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
        repr(getattr(spec, "merge_key_requirements", ())),
        repr(getattr(spec, "observed_fact_sources", ())),
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
        requirements = _execution_final_binding_requirements(sql, Dialect.DUCKDB, params)
        _execution_validate_bindings(sql, Dialect.DUCKDB, params, requirements, artifact="merge")
        materialized = [_materialize_result(result) for result in fragment_results]
        _validate_merge_keys(spec, materialized)
        con = duckdb.connect(":memory:")
        try:
            for i, result in enumerate(materialized):
                _load_fragment_into(con, i, result.columns, list(result.rows))
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
        merge_sql, merge_params = _preflight(plan)
        cache_enabled = self._cache_size > 0
        cache_key: tuple[Any, ...] | None = (
            self._cache_key(plan, cache_namespace) if cache_enabled else None
        )
        if cache_enabled and cache_key is not None:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if entry.expires_at is not None and self._clock() >= entry.expires_at:
                    del self._cache[cache_key]
                else:
                    self._cache.move_to_end(cache_key)
                    self._cache_hits += 1
                    self._fire_hook(plan, 0.0, cache_hit=True)
                    result = _isolate(entry.result)
                    result.columns = list(plan.columns)
                    result.column_meta = [replace(m) for m in plan.column_meta]
                    result.analysis = _plan_analysis(plan)
                    return result

        start = time.perf_counter()
        result = self._execute_uncached(plan, merge_sql, merge_params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if cache_enabled and cache_key is not None:
            expires_at = self._clock() + self._cache_ttl if self._cache_ttl is not None else None
            physical_result = ExecutionResult(
                columns=list(result.columns),
                column_meta=[replace(m) for m in result.column_meta],
                rows=list(result.rows),
                validation_evidence=result.validation_evidence,
            )
            self._cache[cache_key] = _CacheEntry(physical_result, expires_at)
            self._cache.move_to_end(cache_key)
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

    def _execute_uncached(
        self, plan: FederatedPlan, merge_sql: str, merge_params: dict[str, Any]
    ) -> ExecutionResult:
        _assert_fragments_read_only(plan)
        fragment_results: list[_MaterializedAdapterResult] = []
        for i, fragment in enumerate(plan.fragments):
            adapter = self._adapters.get(fragment.dialect)
            if adapter is None:
                raise EngineError(
                    f"No adapter registered for backend "
                    f"{fragment.dialect.value!r}. Call Engine.register("
                    f"Dialect.{fragment.dialect.name}, your_adapter) "
                    f"before running this plan."
                )
            result = _materialize_result(adapter.execute(fragment.sql, fragment.params))
            if list(result.columns) != list(fragment.columns):
                raise ExecutionContractError(
                    "Adapter columns do not match the compiled fragment projection.",
                    reason="output_columns_mismatch",
                    artifact=f"fragment_{i}",
                    names=tuple(fragment.columns),
                    references=(f"fragment:{i}",),
                    operation="execute",
                    stage="output_validation",
                )
            fragment_results.append(result)
        evidence = _validate_merge_keys(plan.merge_spec, fragment_results)

        if self._merge_engine is not None:
            merged = _materialize_result(
                self._merge_engine.merge(
                    cast("list[AdapterResult]", fragment_results), plan.merge_spec
                )
            )
            if list(merged.columns) != list(plan.columns):
                raise ExecutionContractError(
                    "Merge engine columns do not match the declared output projection.",
                    reason="output_columns_mismatch",
                    artifact="merge",
                    names=tuple(plan.columns),
                    operation="merge",
                    stage="output_validation",
                )
            return ExecutionResult(
                columns=list(plan.columns),
                column_meta=[replace(m) for m in plan.column_meta],
                rows=list(merged.rows),
                analysis=_plan_analysis(plan),
                validation_evidence=evidence,
            )

        self._reset_frag_tables(len(plan.fragments))
        for i, result in enumerate(fragment_results):
            self._load_fragment(i, result.columns, list(result.rows))

        self._warn_inline_once()
        merge_cursor = self._con.execute(merge_sql, dict(merge_params))
        columns = [item[0] for item in merge_cursor.description]
        if columns != list(plan.columns):
            raise ExecutionContractError(
                "Rendered merge columns do not match the declared output projection.",
                reason="output_columns_mismatch",
                artifact="merge",
                names=tuple(plan.columns),
                operation="merge",
                stage="output_validation",
            )
        rows = merge_cursor.fetchall()
        return ExecutionResult(
            columns=list(plan.columns),
            column_meta=[replace(m) for m in plan.column_meta],
            rows=rows,
            analysis=_plan_analysis(plan),
            validation_evidence=evidence,
        )

    def iter_rows(self, plan: FederatedPlan) -> ExecutionRowIterator:
        """Yield row dictionaries with per-iterator semantic metadata."""
        stream: ExecutionRowIterator

        def rows() -> Iterator[dict[str, Any]]:
            result = self.run(plan)
            stream.validation_evidence = result.validation_evidence
            for row in result.rows:
                yield dict(zip(result.columns, row, strict=True))

        stream = ExecutionRowIterator(rows(), _plan_analysis(plan))
        return stream

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
        merge_sql, merge_params = _preflight(plan)
        self._adapters_present(plan)
        _assert_fragments_read_only(plan)
        results = [
            _materialize_result(result)
            for result in await asyncio.gather(
                *(
                    self._adapters[frag.dialect].execute(frag.sql, frag.params)
                    for frag in plan.fragments
                )
            )
        ]
        for i, (fragment, result) in enumerate(zip(plan.fragments, results, strict=True)):
            self._validate_result(i, fragment, result)
        evidence = _validate_merge_keys(plan.merge_spec, results)

        if self._merge_engine is not None:
            merged = _materialize_result(
                await self._merge_engine.merge(
                    cast("list[AdapterResult]", results), plan.merge_spec
                )
            )
            if list(merged.columns) != list(plan.columns):
                raise ExecutionContractError(
                    "Merge engine columns do not match the declared output projection.",
                    reason="output_columns_mismatch",
                    artifact="merge",
                    names=tuple(plan.columns),
                    operation="merge",
                    stage="output_validation",
                )
            return ExecutionResult(
                columns=list(plan.columns),
                column_meta=[replace(m) for m in plan.column_meta],
                rows=list(merged.rows),
                analysis=_plan_analysis(plan),
                validation_evidence=evidence,
            )

        with self._merge_con(len(plan.fragments)) as con:
            for i, result in enumerate(results):
                _load_fragment_into(con, i, result.columns, list(result.rows))
            self._warn_inline_once()
            cursor = con.execute(merge_sql, dict(merge_params))
            columns = [item[0] for item in cursor.description]
            if columns != list(plan.columns):
                raise ExecutionContractError(
                    "Rendered merge columns do not match the declared output projection.",
                    reason="output_columns_mismatch",
                    artifact="merge",
                    names=tuple(plan.columns),
                    operation="merge",
                    stage="output_validation",
                )
            rows = cursor.fetchall()
        return ExecutionResult(
            columns=list(plan.columns),
            column_meta=[replace(m) for m in plan.column_meta],
            rows=rows,
            analysis=_plan_analysis(plan),
            validation_evidence=evidence,
        )

    def iter_run(
        self,
        plan: FederatedPlan,
        *,
        chunk_rows: int = 10_000,
    ) -> AsyncExecutionIterator:
        """Return streamed merge chunks with metadata scoped to this iterator."""
        stream: AsyncExecutionIterator

        async def chunks() -> AsyncIterator[list[tuple[Any, ...]]]:
            if chunk_rows <= 0:
                raise EngineError(f"iter_run: chunk_rows must be positive, got {chunk_rows!r}.")
            merge_sql, merge_params = _preflight(plan)
            self._adapters_present(plan)
            _assert_fragments_read_only(plan)
            self.last_iter_run_used_fast_path = False

            if _can_stream_single_fragment(plan):
                fragment = plan.fragments[0]
                self.last_iter_run_used_fast_path = True
                adapter_result = await self._adapters[fragment.dialect].execute(
                    fragment.sql, fragment.params
                )
                self._validate_result(0, fragment, adapter_result)
                if plan.merge_spec.merge_key_requirements:
                    materialized_result = _materialize_result(adapter_result)
                    stream.validation_evidence = _validate_merge_keys(
                        plan.merge_spec, [materialized_result]
                    )
                    row_iter: Iterator[Sequence[Any]] = iter(materialized_result.rows)
                else:
                    row_iter = iter(adapter_result.rows)
                while True:
                    chunk: list[tuple[Any, ...]] = []
                    for _ in range(chunk_rows):
                        try:
                            chunk.append(tuple(next(row_iter)))
                        except StopIteration:
                            break
                    if not chunk:
                        return
                    yield chunk

            adapter_results: list[AdapterResult] = await asyncio.gather(
                *(
                    self._adapters[frag.dialect].execute(frag.sql, frag.params)
                    for frag in plan.fragments
                )
            )
            results = [_materialize_result(result) for result in adapter_results]
            for i, (fragment, result) in enumerate(zip(plan.fragments, results, strict=True)):
                self._validate_result(i, fragment, result)
            stream.validation_evidence = _validate_merge_keys(plan.merge_spec, results)

            with self._merge_con(len(plan.fragments)) as con:
                for i, result in enumerate(results):
                    _load_fragment_into(con, i, result.columns, result.rows)
                self._warn_inline_once()
                cursor = con.execute(merge_sql, dict(merge_params))
                columns = [item[0] for item in cursor.description]
                if columns != list(plan.columns):
                    raise ExecutionContractError(
                        "Rendered merge columns do not match the declared output projection.",
                        reason="output_columns_mismatch",
                        artifact="merge",
                        names=tuple(plan.columns),
                        operation="merge",
                        stage="output_validation",
                    )
                while True:
                    chunk = await asyncio.to_thread(cursor.fetchmany, chunk_rows)
                    if not chunk:
                        return
                    yield [tuple(row) for row in chunk]

        stream = AsyncExecutionIterator(chunks(), _plan_analysis(plan))
        return stream

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

    def _validate_result(
        self,
        index: int,
        fragment: CompiledQuery,
        result: AdapterResult | _MaterializedAdapterResult,
    ) -> None:
        if list(result.columns) != list(fragment.columns):
            raise ExecutionContractError(
                "Adapter columns do not match the compiled fragment projection.",
                reason="output_columns_mismatch",
                artifact=f"fragment_{index}",
                names=tuple(fragment.columns),
                references=(f"fragment:{index}",),
                operation="execute",
                stage="output_validation",
            )


__all__ = [
    "AsyncExecutionIterator",
    "AsyncEngine",
    "AsyncMergeEngine",
    "DuckDBMergeEngine",
    "Engine",
    "ExecutionContractError",
    "ExecutionResult",
    "ExecutionRowIterator",
    "MergeEngine",
    "to_async_merge_engine",
]
