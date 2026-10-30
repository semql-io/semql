# Lessons for reliable semantic data integration

This note records integration design decisions for SemQL. Its recommendations preserve programmer control and distinguish existing contracts from optional additions; they are not universal correctness or production-performance claims. The implementation is described in the [migration guide](../migrations/reliable-integration.md).

## Executive synthesis

The most durable lesson is to keep semantic promises, evidence, and execution outcomes distinct. A catalog declaration can authorize a compiler decision without proving the underlying data conforms. An adapter can report metadata without making it executable semantics. A successful compile proves neither that a backend executes the SQL nor that it returns the intended rows. A green test harness proves only the behaviors it actually exercises.

For SemQL, this reinforces rather than replaces its existing boundaries: the pure compiler emits parameterized SQL; authorization is compiler-owned; executors and importers live in sibling packages; federation is caller-declared. The opportunities are to make evidence and unsupported/lost metadata more legible at integration boundaries, and to make verification claims track actual behavioral coverage.

## Vocabulary

- **Declaration**: a catalog or operator assertion about intended meaning or capability.
- **Evidence**: an observation from a source schema, runtime, fixture, or test that bears on a declaration.
- **Certification**: a deliberate decision that an assertion meets a defined evidentiary contract; not a synonym for “metadata was present.”
- **Reported metadata**: source information preserved for inspection but not translated into executable semantics.
- **Refusal**: an explicit, attributable statement that a request or source object cannot be represented or safely handled.
- **Conformance**: observed behavior of an implementation against a shared, bounded set of assertions.

## Lessons

### 1. Trust programmer-authored declarations

Catalog keys, relationship cardinalities, grains, and authored SQL are trusted
programmer inputs. Supported work must not be blocked by demands for independent
proof, a certification service, or host waivers. Existing authorization and
known-invalid execution checks remain intact.

Descriptive analysis can explain which declarations a plan used without adding
new obligations. Source-format validation and import-loss reports prevent
accidental mistranslation; they are not a requirement to prove programmer intent.

### 2. Translate metadata conservatively; distinguish executable from report-only

Physical schemas, vendor catalogs, and semantic-model APIs expose different kinds of information. Names, descriptions, types, keys, and metric-like objects do not all carry equivalent meaning. A free-text expression or vendor-owned measure is not automatically a SemQL `Measure`; a structural foreign key is not a cardinality proof.

At each importer boundary, enumerate what is carried, what is report-only, and what is unsupported. Preserve unsupported objects in a named refusal/report where practical rather than silently omitting them. Keep inference visibly heuristic and reviewable; do not convert a naming guess into a certified metric without a deliberate contract.

This applies directly to `semql-introspect`: heuristic guesses have annotations, while composite source keys now retain every component in typed reports instead of assigning the first component to the scalar catalog key field. Supported composite foreign-key joins remain executable conjunctions.

**Acceptance scenarios:**
- Introspecting a composite primary key records that the key is composite/unrepresented, rather than silently claiming the first column is the complete key.
- A source metric whose binding or filter is opaque/free text is reported as non-executable metadata, not converted into guessed SQL.
- Unknown or malformed closed-format fields fail with an entity/path-specific reason; forward-compatible non-semantic display fields may use a different policy.
- A mixed import reports supported and refused objects deterministically, then passes produced definitions through the same semantic consistency checks as hand-authored catalogs.

### 3. Version, identity, and content integrity are separate coordinates

A human-supplied version label is useful provenance, but it is not a content-integrity check. Source-format version controls which fields and semantics are valid; a snapshot/version label identifies the caller's chosen revision; a digest identifies content. Conflating them makes it difficult to tell whether two assembled catalogs are equivalent or merely share a label.

SemQL's serialized, versioned catalogs and compiler analysis already distinguish catalog revision from artifact-local identities. Preserve that separation. For importers and future cache or cross-process comparison contracts, state which identity is source identity, which is semantic revision, and which is a content digest. Do not treat a caller-supplied label as proof that content is unchanged.

**Acceptance scenarios:**
- Changing imported definitions while retaining an operator-supplied revision label changes any content digest.
- Changing the source format version changes validation/feature interpretation without silently redefining semantic catalog identity.
- Repeated imports of equivalent input produce stable normalized content independent of filesystem enumeration order.

### 4. Make bounds and deterministic ingestion part of the interface

Directory and remote-source ingestion need explicit limits on entry count, bytes, request size, and traversal. Deterministic ordering makes output reviewable and repeatable; bounded reads keep an accidental or hostile input from becoming unbounded work. These are not merely performance optimizations: they determine whether the same declared input produces a reproducible artifact.

Apply bounds at the earliest layer that has the needed information (for example, byte caps before decoding). State residuals separately: a cap on encoded frames does not automatically cap decoded memory; a request timeout does not necessarily stop work already running in a blocking thread.

SemQL's live introspection surface accepts a DB-API connection and reads schema rows. Any future batch/file/service importer should make resource bounds and deterministic ordering explicit, while preserving the caller-owned connection and the package's driver-agnostic contract.

### 5. Separate caller identity, operation permission, and source execution identity

Authenticating an API caller does not make a database connection act as that caller. Tool visibility, endpoint permission, row-level authorization, and database credential identity answer different questions. A missing user-scoped credential must not silently downgrade to a shared service credential if the contract promised user-scoped execution.

SemQL already separates request-scoped `AuthContext`, cube/field authorization, and row-scope predicates injected by the compiler. Preserve that distinction across MCP and engine integrations: a trusted viewer provider establishes identity; the executor's database identity is a separate deployment fact. Avoid describing operation-level access as proof of row-level data authorization.

**Acceptance scenarios:**
- A viewer denied a cube or row scope is refused before execution, including when the outer query has `OR` predicates.
- An absent per-viewer source credential never becomes success using a broader shared identity unless the source is explicitly configured and documented for shared-identity access.
- Logs and errors do not expose tokens, raw credential material, or unmasked principals; test rendered outputs rather than only inspecting exception types.

### 6. Timeouts, admission, cancellation, and size limits are different controls

A timeout response bounds how long the requester waits; it may not stop a synchronous database call in a worker thread. Admission limits work that is queued or running. Cancellation is only real when the underlying driver/work can be stopped. Input-size and output-size limits constrain different resources, and a truncated successful result can look like a valid but incorrect answer.

For SemQL executors, preserve explicit lifecycle ownership: cancel and await sibling tasks where possible, close iterators and connections on early exit, and do not claim driver cancellation unless the driver confirms it. Return a typed over-limit outcome instead of silently treating partial rows as complete.

**Acceptance scenarios:**
- With all worker slots occupied by work that outlives request timeout, new work is shed or bounded; slots remain held until underlying work completes.
- Early stream termination closes acquired iterators/adapters on success and failure.
- An output at the exact row cap succeeds only when completeness is known; one additional row produces a typed refusal rather than silent truncation.
- Request bytes, encoded payload, decoded-memory policy, and result rows/bytes have separately stated ceilings.

### 7. Treat refusals and unsupported shapes as first-class behavior

A semantic system should distinguish invalid input, unsupported-but-valid shape, authorization refusal, source capability refusal, internal contract failure, and incomplete evidence. A generic error or empty answer obscures whether the system rejected the request correctly or lost meaning.

SemQL already has typed compile errors, collected validation diagnostics, engine contract errors, capability checks, and safe public diagnostic rendering. Continue to keep caller-facing refusal separate from internal failure, and ensure unsupported combinations do not degrade into partial SQL or plausible results.

**Acceptance scenarios:**
- Unsupported multi-backend or lowering combinations fail before adapters are invoked.
- Error details identify a stable reason and relevant query/artifact location without leaking bind values or raw internal exception text.
- Tests assert both that a refusal occurs and that the relevant adapter was not called.

### 8. One conformance pack is valuable; a green harness is not a universal proof

Shared, port-generic test bodies reduce assertion drift across adapters. Useful independent properties include projected labels/schema, row content as a multiset, ordering only when promised, repeat determinism, and agreement between preflight and execution. Test the harness itself with fakes that deliberately distort outputs, but keep fake-harness proof distinct from actual-adapter proof.

Coverage inventories and architecture scans are fitness functions, not runtime evidence. A binding count does not prove assertions are meaningful; a named refusal does not prove a test triggers it; a source scan does not prove backend behavior. State what is not exercised beside the claim. Use actual backend execution for any advertised backend/shape cell, and an independent oracle for semantic correctness—two implementations agreeing can share the same defect.

SemQL already has actual-backend conformance tests and meaningful deterministic test suites. Expand them around published behavior, not package names: backend × query shape × federation/merge mode × combinations. Keep test coverage tied to emitted tests, and retain explicit exceptions for shapes not exercised.

**Acceptance scenarios:**
- Every supported adapter binds a shared suite of observable behaviors; removing a binding makes a structural gate fail.
- Harness meta-tests prove selected faults are detected; adapter-specific tests separately run against the actual implementation/backend.
- Each advertised capability cell has a real execution fixture and an independent expected result where correctness is at issue.
- Every test report distinguishes pass, explicit decline/not-run, and failure; required provisioned environments fail closed when absent.

## What not to copy

- Do not infer that a catalog primary key or cardinality declaration is runtime-validated merely because it is structurally checked.
- Do not treat a successful SQL parse, a golden SQL snapshot, or two agreeing strategies as proof of result correctness.
- Do not turn metadata import into speculative SQL generation. More inferred features are not automatically more useful than explicit report-only metadata.
- Do not claim a timeout cancels synchronous work, a frame cap bounds decoded memory, or a result cap is safe if it silently truncates.
- Do not equate test registration, static source scans, or declared capabilities with executed behavioral evidence.
- Do not transplant a foreign language, execution engine, service topology, or abstraction wholesale. Transfer the contracts and counterexamples; retain SemQL's current package boundaries and compiler philosophy.

## Adoption record

The accepted changes add loss-aware key/report handling, separate provenance
coordinates and optional content digests, opt-in executor controls, typed
preflight refusals, and shared observable adapter conformance. The
[migration guide](../migrations/reliable-integration.md) documents programmer-facing
APIs and intentional probe/semi-join plan changes. The
[conformance matrix](adapter-conformance-matrix.md) records actual execution
boundaries and remaining native-service gaps.

## Evidence boundary

The original introspection probe demonstrated a composite primary key being reduced to its first column; the implementation now preserves and reports that limitation. Verification covers bounded fixtures and public execution scenarios, not a broad security audit, production load test, or universal backend guarantee.
