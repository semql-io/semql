# Reliable integration contracts

These changes add descriptive import/provenance information, opt-in execution controls, and shared adapter conformance. Programmer-authored catalog declarations remain trusted. No proof service, certification gate, mandatory digest, or host waiver is required to compile or execute a supported query.

## Loss-aware introspection

Prefer `semql_introspect.introspect_to_result(...)` when integrating programmatically. Its result contains:

- `cubes`: executable catalog definitions supported by the current model;
- `annotations`: existing heuristic classification explanations;
- `table_reports`: ordered `TableMappingReport` records describing source primary keys, mapped scalar keys, unsupported key components/foreign keys, and `MappingFact` records;
- `source_provenance`: optional source identity supplied by the probe.

`MappingStatus` distinguishes `mapped`, `inferred`, and `report_only`. A composite primary key stays an ordered tuple in the report; every component remains an identifier dimension, and `Cube.primary_key` is unset rather than falsely assigned to its first component. This does not restrict hand-authored cube definitions.

A composite foreign key becomes one conjunction join, not independent component joins. Scalar `Dimension.foreign_key` shorthand is emitted only when it faithfully references a scalar target primary key; compound joins retain their explicit predicate. Referenced-key ordinals are paired in constraint order. Aliases are allocated before constructing SQL references so collisions do not invalidate fields or joins.

`introspect_catalog(...)` still returns cubes, but emits `UserWarning` if doing so would conceal mapping losses. Use the full result when you need to handle those losses without warnings. Generated Python contains executable `CUBES` plus `SOURCE_PROVENANCE` and `TABLE_MAPPING_REPORTS`; these reports are descriptive and do not gate catalog construction.

### Custom probe migration

`SchemaProbe.list_primary_keys()` now returns `dict[str, tuple[str, ...]]`, preserving every ordered component. For example, `{"orders": ("tenant_id", "order_id")}` replaces the former scalar representation.

`ForeignKeyInfo` now represents a whole constraint with `from_table`, `from_columns`, `to_table`, `to_columns`, and `constraint_name`. Keep source and referenced column tuples paired in the constraint's declared order. Update custom probe implementations; no scalar-key compatibility adapter is provided. `IntrospectionResult` constructors also supply `table_reports`.

The built-in probe uses ANSI `information_schema` metadata. Native DuckDB behavior is exercised; source metadata support on other engines still depends on their dictionaries. No vendor-specific metadata importer or automatic metric-certification feature is added.

## Provenance and content integrity

```python
from semql import Catalog, CatalogContext, SourceProvenance

catalog = Catalog(
    cubes,
    source_provenance=SourceProvenance(
        source_format="information_schema",
        source_format_version="deployment-dictionary-v1",
        source_identity="analytics",
        source_snapshot="snapshot-42",
    ),
)
digest = catalog.content_digest
context = CatalogContext(namespace="analytics", semantic_revision="host-revision-7")
compiled = catalog.compile(query, catalog_context=context)
```

The coordinates have different meanings:

| Coordinate | Meaning |
|---|---|
| `SourceProvenance.source_format` / `source_format_version` | How the source metadata is represented |
| `source_identity` / `source_snapshot` | Which source and caller-labelled snapshot supplied it |
| `CatalogContext.semantic_revision` | Existing host-owned semantic revision |
| `CatalogSpec.schema_version` | SemQL catalog wire shape |
| `CatalogContentDigest.format_version` | Version of digest canonicalization |
| `CatalogContentDigest.value` | SHA-256 of the described serialized catalog content |

`CatalogSpec.source_provenance` survives catalog reconstruction and JSON round-trips. The digest is derived on demand, sorts mapping keys, preserves sequence order, and changes when serialized content changes even if the host revision does not. Opaque user metadata participates as content but is not interpreted. Construction-error presentation and lookup runtime loaders/enrichers are excluded. Non-finite or unsupported serialized values cannot be given a misleading JSON-content digest.

A digest does **not** identify runtime policy, scope functions, hooks, lookup code, or their external data. It is not an authorization proof, an execution cache key, or a replacement for semantic-analysis comparison and private bound context. Local analysis/output IDs remain artifact-local. Digest calculation is optional and never occurs as a new compile gate.

## Optional execution controls

```python
from semql_engine import AsyncEngine, Engine, ExecutionLimits

limits = ExecutionLimits(
    max_in_flight=4,
    admission_timeout=0.25,
    execution_timeout=10.0,
    max_fragment_rows=50_000,
    max_fragment_value_bytes=8_000_000,
    max_rows=10_000,
    max_value_bytes=2_000_000,
)
engine = Engine(adapters=adapters, limits=limits)
async_engine = AsyncEngine(adapters=async_adapters, limits=limits)
```

Every field defaults to `None`: existing unrestricted behavior remains available. Reusing an engine shares its admission capacity across its calls. Separate engine instances do not implicitly share a process-wide quota. Admission waiting and execution elapsed time are separate; the execution deadline starts after admission and uses a monotonic clock.

- Fragment row/value-byte bounds are checked incrementally while the engine consumes each adapter result, before appending the next materialized row.
- Final row/value-byte bounds apply to returned outputs, including synchronous cache hits and configured merge engines.
- `Engine.iter_rows` and `AsyncEngine.iter_run` honor configured limits. Async streams may have delivered earlier chunks before overflow or deadline failure; a raised error means the stream is incomplete, not a successful truncated result.
- Exact-cap complete results succeed. The next row causes `ExecutionContractError`; no complete result is silently truncated.
- Cohort and semi-join execution share one admission lease and deadline across their stages. Cohort assembly separately checks its composed output budget.
- `execute_entity(compiled, adapter, limits=limits)` applies fragment/output accounting and cooperative deadlines on SQL and custom row adapters. To share direct entity-call admission, pass the same `semql_engine.execution.Admission(limits)` object via `admission=` on each call; a settings object alone is not a shared runtime semaphore.

### What these controls do not promise

A synchronous call cannot be forcibly interrupted here. Its deadline is checked at cooperative boundaries. Async deadlines stop waiting and cancel request-owned coroutine work, but built-in sync-backed async adapters retain ownership of already-started worker operations. Their admission lease is held until all such operations actually finish, including after timeout; late results are cleaned up and late exceptions consumed. Custom adapters must implement an honest cancellation/lifetime contract. Configure driver query deadlines when server-side work must be stopped.

Engine limits are **not** database scan, driver prefetch, Python heap, RSS, encoded transport, HTTP request-body, or decompression limits. A driver may buffer rows before returning `AdapterResult`. Configure those other boundaries in the driver or host service rather than assuming an engine row cap covers them.

Value bytes are deterministic payload accounting: UTF-8 for text, raw buffer bytes, one byte for bool, sign-inclusive bit-length accounting for integers, eight bytes for float, textual Decimal and temporal representations, sixteen bytes for UUID, and recursive sums for supported list/tuple/dict values. NULL contributes zero. Object/container overhead is excluded. Cyclic or unsupported opaque values refuse only when byte accounting is requested; uncapped custom outputs are not newly restricted by byte traversal.

Acquired closeable row iterators are closed on completion, overflow, failure, or explicit stream closure. Caller-owned database connections are not closed.

## Preflight and typed failures

Missing source adapters, unsupported executable versions, final binding mismatches, and non-read-only source SQL are checked before executing any fragment. Static cohort stage plans and semi-join inner/outer plans are preflighted before selection I/O. Dynamically bound plans are checked again at their execution boundary.

Semi-join plans now use version **2**, with required keyword-only `outer_plan` retaining a compiler-derived structural preflight shape. Value-free binding predicates include cubes and backends reached only through the eventual semi-join filters; that template is inspected, not executed. The residual output is compiled separately because introducing a filter-only source can change federation strategy and output metadata. Recompile old plans rather than relabeling a version-1 artifact or fabricating a template. Custom constructors must supply the compiled preflight plan; other positional constructor parameters retain their order.

Resource failures have stable reasons including `execution_admission_timeout`, `execution_deadline_exceeded`, `execution_row_limit_exceeded`, `execution_value_byte_limit_exceeded`, and `execution_value_unaccountable`. Missing adapters use `adapter_missing`; entity adapter mismatches use `adapter_capability_missing` rather than an incidental `TypeError`. These remain catchable as `EngineError` and core `ContractError`. Use `error.to_public_payload()` outside trusted diagnostics; do not expose raw driver errors or binding values.

## Conformance

The shared adapter pack compares labels, width/schema where declared, duplicate-preserving rows, promised order, bindings, Decimal/NULL/empty outputs, aggregate values, and deterministic repeated calls. Its fault cases test the assertions themselves; those are not live-service evidence.

See the [adapter conformance matrix](../specs/adapter-conformance-matrix.md) for concrete adapter/shape cells, native versus stand-in boundaries, commands, and residual gaps. `SEMQL_CONFORMANCE_REQUIRED=1` turns absent required external fixtures into failures instead of optional skips. Actual-backend tests do not establish universal support for unexercised combinations.
