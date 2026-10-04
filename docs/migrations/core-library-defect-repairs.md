# Core-library defect repair migration

These changes repair the findings in [the defect handoff](../specs/core-library-code-design-defect-handoff-2026-10-02.md). They are unreleased and intentionally change several pre-v1 contracts.

## Frozen catalog values

Catalog value collections and identity/security collections are read-only `Sequence`/`Mapping` values, not mutable `list`/`dict` attributes. Normal list/dict constructor inputs and JSON arrays/objects remain supported. Construction detaches nested containers; mutate the caller's original data freely, but build a new model to change a catalog value:

```python
updated = dimension.model_copy(update={"required_roles": ["admin", "reader"]})
```

`model_copy(update=...)` validates and freezes the replacement state. Do not call `.append()`, `.update()`, or indexed assignment on model collections. Materialize `list(value)`/`dict(value)` only when a downstream API requires its own mutable container. Metadata remains opaque: SemQL does not interpret its keys or values.

Hashable value models can be dictionary/set keys only when their entire equality-relevant value graph is supported and immutable. Opaque mutable models/custom objects make the containing instance intentionally unhashable; they are not mutated or coerced to manufacture a stable key. `Cube` remains unhashable.

## Catalog spec schema 2

New catalog specs preserve `allow_mutations`, `max_list_limit`, and `max_mutation_rows`, including across JSON round-trips. Entity wire declarations carry explicit `entity_type` tags (`entity` or `mutable_entity`), preserving write declarations rather than serializing a mutable entity through the read-only base schema.

Do not upgrade ambiguous old mutation-enabled entity specs by merely changing their version number or inferring write authority from an incomplete payload. Rebuild them from the original trusted entity declarations. Untagged ordinary entity data is read-only; untagged write declarations and ambiguous schema-1 mutation-enabled entity specs are refused.

`Catalog.from_spec(runtime=...)` retains the supplied runtime's policy, scope functions, registry, error transform, and hooks. A runtime field of `None` falls back to its keyword argument; explicitly empty scope registries and hook lists override keyword values.

## Prompt caches

Dynamic policy-dependent cubes do not belong in cross-viewer static/invariant segments. Cache per-viewer authorized projections separately. Public field descriptions form the invariant baseline; authorized overlays win when combining descriptions. The no-viewer tooling mode remains unfiltered and must not be mistaken for an authorization decision for a later request.

The low-level `catalog_prompt_hash` accepts `policy`; catalog convenience functions thread `Catalog.policy` through it. Protected cube fields remain intact when prompt budgets cannot fit; a cube-local `Relations` marker is not a structural domain-section boundary.

## Backend strategies

Custom `DialectStrategy` implementations must implement `emit_null_safe_eq(left, right, value_type)` for comparison grouping keys. The predicate must be executable in the backend's full-join shape. PostgreSQL needs typed hash/merge-joinable equality with separate NULL identity; bare `IS NOT DISTINCT FROM` is not supported by its full join.

Dense fill retains every bucket intersecting the half-open interval and preserves the requested projection, including aliases and inline-derived columns. Missing ratios remain NULL. Comparison result filters apply after the completed output projection and before final ordering/limits.

## Adapter schema and ownership

`AdapterResult.column_types` is optional positional physical schema metadata, aligned with `columns`. Built-in adapters retain available driver metadata; custom adapters should declare types for empty/all-NULL columns. Decimal precision/scale must be retained, not guessed from a missing sample.

Legacy undeclared results are inferable only when observed supported values establish a coherent type. Ambiguous absent data raises `physical_schema_missing`; unsupported materialization types raise `physical_schema_unsupported`, rather than falling back to text.

Unbounded DB-API `NUMERIC`/`DECIMAL` declarations without precision/scale do not
inherit BigQuery defaults. Available decimal precision/scale is retained;
otherwise the same conservative observed-value inference applies. Python
`float` type codes materialize as `DOUBLE`, not 32-bit SQL `FLOAT`.

DuckDB's lossless decimal materialization limit is 38 digits of precision.
Higher declared precision, including BigQuery's default `BIGNUMERIC(76,38)`,
and over-precision legacy `Decimal` values fail with
`physical_schema_unsupported`; they are never mapped to integer `BIGNUM`
storage or rounded to fit.

The engine owns acquired row iterators, not the caller's connections. Close async streams explicitly on early exit. Fragment failures and caller cancellation drain request-owned async tasks before returning. Cancellation does not forcibly stop an already-running sync driver thread; provide driver-level timeouts/cancellation where needed.

Cached mutable cells are isolated at insertion and retrieval; unsupported opaque cell values are not cached. Non-cache execution avoids these cache-only copies.

## API audit

Griffe comparison of all eight package public modules against discovery commit `fc57a9b3714b3fad34fa90201ad5737ddff4f0a5` reported 20 changes in `semql` and none in the other seven root public modules. The reported core changes are immutable collection defaults, the `Metadata` mapping type, and corrected spec construction. That static audit does not detect every semantic or submodule protocol change; the migration obligations above also cover spec schema, copying/hashing behavior, backend strategies, adapter metadata, and cache/resource ownership.
