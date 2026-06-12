# Test plan — SemQL workspace

Status: proposed · June 2026
Companion docs: `architecture-review-2026-06.md` (defects this plan must
catch), `entities.md` (M1–M5 each ship red/green against this plan),
`property-testing.md` (full Hypothesis strategy redesign + property
catalogue P1–P22/S1–S4 — expands §6 of this doc).

## 0. The one-sentence strategy

SemQL is a compiler whose only unacceptable outcome is a wrong result,
so the centre of gravity moves from *asserting on SQL text* (where the
suite is today) to *asserting on executed results against oracles*
(differential + metamorphic), with text snapshots kept as the
cheap-and-fast outer ring.

## 1. Current state (inventory, June 2026)

- 1,598 test functions / ~28.6k LoC across 6 packages; 1,429 of them in
  `semql` (pure compile, no I/O — correct per PHILOSOPHY).
- Styles in use: syrupy AMBR snapshots (SQL + plan repr), Hypothesis
  with custom `strategies.py` (600-identifier pool, catalog+query
  composite), 22 parametrize matrices, manual-`asyncio.run` async tests.
- Engine tests run against in-process DuckDB, with a dialect-translating
  adapter posing as PG and BQ. No testcontainers, no real backends.
- Tooling already present but underused: **sqlglot** (never used to
  *parse-validate* output), **mutmut** (configured, no baseline run),
  **pytest-cov** (`fail_under = 0`), CI workflow exists but is
  `workflow_dispatch`-only.
- Known holes the plan must close: every assertion is on SQL strings or
  substrings — nothing executes compiled SQL and checks rows; the
  query-path vs plan-path equality matrix covers one trivial shape
  (which is exactly how defect A1 survived); merge engines have no
  differential test (defect A3); zero performance or security tests.

## 2. Principles (how compilers are tested)

1. **The oracle problem is the design problem.** A compiler test is
   only as good as its source of truth. We use four oracles, cheapest
   first:
   - *Syntactic*: output parses under the target dialect (sqlglot).
   - *Snapshot*: output is byte-identical to a reviewed golden file.
   - *Differential*: two independent paths that must agree (query-path
     vs plan-path; dialect A vs dialect B executed on identical data;
     federated vs single-source; DuckDB merge vs Polars merge).
   - *Metamorphic*: one path, two related inputs whose outputs must
     satisfy a relation (add a filter → rows ⊆ original). This is how
     SQLancer finds bugs in production databases (TLP/NoREC) and it
     adapts directly to a semantic layer.
2. **Generate inputs, don't enumerate them.** Hand-written cases pin
   known behaviour; Hypothesis-generated catalogs+queries find the
   unknown interactions. Every invariant below gets both forms.
3. **Test the pipeline at its seams**: SemanticQuery → LogicalPlan →
   SQL → rows. Each seam gets an equality/derivation contract, because
   half-migrated seams are where A1-class bugs live.
4. **Mutation score, not line coverage, is the quality gate** for the
   compiler core. Line coverage says code ran; mutation score says a
   test would notice if it were wrong.
5. **Tiered cost**: PR tier must stay under ~5 min; expensive oracles
   (containers, long fuzzing, full mutation) run nightly.

## 3. Layer 1 — Unit tests

Scope: one module, no I/O, no cross-package imports. Mostly *keep and
extend* what exists. Additions:

| Area | New tests | Success criterion |
|---|---|---|
| Ref parsing | Centralise `"cube.field"` split (review B3), table-test every shape incl. typos, unicode, keywords | every malformed ref yields a typed error, never silent degradation |
| CNF | Hypothesis: `cnf(e)` ≡ `e` via truth-table evaluation over random assignments (≤6 vars, exhaustive) | round-trip equivalence holds for 10k generated trees |
| Temporal | granularity truncation per dialect; range endpoint parsing; the inclusive/half-open contract (defect A2) **decided then pinned** | one canonical convention, property-tested at boundaries (`==` start, `==` end, mixed date/timestamp precision) |
| Auth units | JWKS (alg=none, expiry, audience, kid rotation), X509Mapper domain collision (review B10), mask_roles ⊆ required_roles | each refusal has a test asserting the typed error code |
| Cursor codec (entities D9) | round-trip, tamper → typed refusal, version skew | property: decode(encode(x)) == x; flipped byte never decodes |
| Cost / prompt-budget | boundary cases at exactly-budget | off-by-one pinned |
| Error taxonomy | every public raise site returns a `code` (review B8) | parametrized walk over registered codes; no bare f-string errors in compile paths |

Conventions: pytest, strict mypy/pyright already in place. Add
`pytest-asyncio` and delete the manual `_run` helpers (divergence
between sync/async suites is review B7's symptom).

## 4. Layer 2 — Integration tests

Scope: whole-pipeline within one process; in-memory DuckDB allowed; no
network.

### 4.1 Compile pipeline (semql)

- **Equality matrix (widen — this is the A1 fix's regression net).**
  Parametrize `compile_query(q) == compile_plan(to_logical_plan(q))`
  over the full feature lattice: filters × having × segments ×
  ungrouped × left_joins × derived_measures × aliases × compare_split ×
  rollup × partition, each alone and in sampled pairs. Plus a
  Hypothesis version: for *any* generated query, the two paths emit
  identical SQL. ~1 day; would have caught A1 mechanically.
- **Dialect validity property.** For every generated (catalog, query,
  dialect): `sqlglot.parse(out.sql, dialect=...)` succeeds. Cheap, runs
  in the PR tier, catches emission bugs no snapshot covers.
- **Bind-params invariant (also a security test, §8).** Property: no
  user-supplied filter *value* appears as a literal substring in
  emitted SQL; all values arrive via `out.params`. Generate adversarial
  values (quotes, `--`, `;`, unicode).
- **Snapshot suite (keep, grow deliberately).** One snapshot per
  *feature*, per dialect — not per test. Review discipline: snapshot
  diffs are reviewed like code; `--snapshot-update` never runs in CI.
- **Serialization round-trips.** Catalog/SavedQuery/RowPlan:
  `from_dict(model_dump(x)) == x`, plus version-skew refusal tests
  (consumes review B5 when CatalogSpec lands).

### 4.2 Engine + federation (semql-engine)

- **Differential merge-engine test (defect A3's net).** ~~One harness:
  generate a FederatedPlan + fragment result sets, execute the merge
  through DuckDB merge *and* PolarsMergeEngine, assert identical rows.~~
  **Moot as of 2026-06-12 (decisions.md D6):** `PolarsMergeEngine` was
  removed, leaving one merge implementation (DuckDB merge SQL), so
  there is no second engine to differ from. Retain this harness *shape*
  as the entry gate for any future `MergeEngine`: the day a second
  implementation lands, it must run every `FederatedPlan` through both
  and assert identical rows (sorted, type-normalised), Hypothesis-driven
  over filter ops incl. in/not_in/is_null/contains — so an op one engine
  supports and the other silently skips fails permanently.
- **Federated ≡ single-source.** Load identical data into one DuckDB
  and into two DuckDBs split by the partition scheme; same
  SemanticQuery must return identical rows. This is the federation
  correctness oracle and the regression net for defect A2 (boundary
  rows) — include boundary-exact time ranges as explicit cases.
- **Cache contract**: hit/miss/TTL, mutation-after-return isolation
  (defect A5: assert cached result is not aliased), unhashable-params
  behaviour pinned as typed error not TypeError.
- **Adapter contract suite (consumes entities M3).** One reusable
  pytest class parametrized over every adapter: placeholder style,
  param binding, error mapping, cancellation, `execute_rows(RowPlan)`
  for row-capable adapters. New adapters inherit the suite — conformance
  by construction.

### 4.3 MCP (semql-mcp)

- In-process FastMCP test client (no subprocess): tool listing matches
  catalog, per-cube tool schemas, structured error propagation
  (review B8: attrs must survive, not flatten to a string).
- Auth: per-request viewer propagation (defect A6's red test first),
  refusal of client-asserted roles on http/sse.

## 5. Layer 3 — End-to-end tests

Scope: real backends, network, containers. Nightly tier.

- **Backend matrix via testcontainers**: Postgres and ClickHouse
  containers (cheap, reliable); DuckDB in-process; BigQuery/Snowflake
  excluded from CI — covered by (a) the sqlglot dialect-validity
  property and (b) a manual pre-release smoke script against real
  accounts (document in RELEASING).
- **The differential execution oracle (the flagship harness).** Seed
  every backend with the *same* generated dataset (Hypothesis-generated
  schema + rows, written via each adapter). For each generated
  SemanticQuery: compile per dialect, execute, normalise (sort, decimal
  coercion, NULL ordering), assert all backends agree, with DuckDB as
  the reference verdict when they don't. This is the Csmith/SQLancer
  pattern applied to a semantic layer and is the single highest-value
  harness in this plan: it tests dialect emission, param binding, type
  mapping, and adapters at once, against the only oracle that matters —
  rows.
- **Full-stack MCP e2e**: MCP client → tool call → compile → execute
  against the Postgres container → rows; including a scoped-viewer call
  and (post-entities M4/M5) a mutation preflight→confirm round-trip
  inside a rolled-back transaction.
- **sqllogictest-style corpus**: a directory of `.yaml` cases —
  `catalog:, query:, expected_rows:` — executed against every backend
  in the matrix. Contributors add a case per bug fix (the regression
  corpus grows monotonically, the way SQLite's does).

## 6. Correctness programme (cross-cutting)

The compiler-specific techniques, in adoption order:

1. **Metamorphic relations** (Hypothesis + DuckDB execution; PR tier
   with low example counts, nightly with high):
   - *Filter monotonicity*: adding a conjunct → rows(Q′) ⊆ rows(Q).
   - *TLP, adapted*: rows(Q) = rows(Q ∧ p) ⊎ rows(Q ∧ ¬p) ⊎
     rows(Q ∧ p IS NULL) for a generated predicate p — SQLancer's
     highest-yield oracle, near-free to implement on the DuckDB path.
   - *Limit containment*: rows(Q limit n) ⊆ rows(Q), |·| ≤ n.
   - *Partition additivity*: Σ measure over disjoint segment values =
     unsegmented total (for additive aggs only — REAGG_OK metadata
     already encodes which).
   - *Group/ungroup consistency*: Σ per-group counts = ungrouped count.
   - *Order irrelevance*: order_by changes sequence, never membership.
   - *Compare-split disjointness*: current and prior windows never
     overlap; union covers exactly the requested range.
   - *Join fan-out guard* (review B4, once built): one-to-one join
     leaves measure values unchanged.
   - *Auth containment*: rows(Q, viewer) ⊆ rows(Q, no-viewer), and
     every returned row satisfies the scope predicate re-evaluated
     client-side.
2. **Mutation testing as the core-quality gate.** mutmut is already
   configured: run the baseline, record the score, then gate in CI on
   *changed-lines mutation score* (fast) with a weekly full run
   (slow). Target: ≥85% killed on `compile.py`, `logical.py`, `cnf.py`,
   `_resolve.py`; surviving mutants are triaged into "add test" or
   "dead code — delete".
3. **Coverage ratchet, not threshold.** Replace `fail_under = 0` with a
   ratchet (CI fails if branch coverage drops below the last recorded
   value). Add `--cov-branch`. Apply per-package, including
   semql-engine (currently `--cov=semql` only measures the core).
4. **Hypothesis profiles**: `dev` (50 examples), `ci` (200, fixed
   seed-per-PR + `derandomize`), `nightly` (5,000 + swarm of the
   execution oracle). Persist the failure database
   (`.hypothesis/examples`) as a CI cache so found bugs re-run first.
5. **Crash-corpus discipline**: every Hypothesis/differential failure
   gets its shrunk repro committed as a deterministic regression test
   before the fix lands (red/green; matches the house rule).

## 7. Performance programme

Compile is in the request path of text-to-SQL loops; the budget story
must be tested, not assumed.

- **Tooling**: `pytest-benchmark` locally + **CodSpeed** in CI
  (instrumentation-based, so PR runners' noise doesn't flake it).
  Benchmarks live in `benchmarks/`, excluded from the default pytest
  run.
- **Compile-time budgets** (assert p50 in benchmark, alert on
  regression >15%):
  - simple query, 10-cube catalog: < 5 ms
  - 5-join, 20-filter query, 100-cube catalog: < 50 ms
  - catalog construction, 1,000 cubes: < 2 s (review B5's ~300-line
    constructor is the suspect; measure before refactoring it)
- **Pathology guards** (plain tests, not benchmarks): CNF on a 30-leaf
  nested OR tree completes < 1 s and the clause count is bounded
  (exponential-blowup tripwire); prompt-budget on a 1k-cube catalog;
  deep filter nesting doesn't hit recursion limits.
- **Memory**: `tracemalloc` snapshot test — compiling 1k queries leaks
  < 1 MB (catches the identity-keyed side-table class of leak,
  review B6); cached ExecutionResult memory bounded by cache size.
- **Engine micro-benchmarks**: cache hit vs miss; merge-engine rows/sec
  on 100k-row fragments (DuckDB vs Polars — also documents when to
  choose which).
- **Trend, don't just gate**: CodSpeed keeps history; review the graph
  at release time.

## 8. Security programme

Threat model first, in `docs/SECURITY.md` (one page): catalog authors
are **trusted** (they already write raw SQL — document this loudly);
query *values*, viewer identities, and MCP clients are **untrusted**;
the LLM constructing queries/mutations is untrusted-but-sandboxed by
compile-time gates.

- **Injection (untrusted values)**: the §4.1 bind-params property is
  the primary defence test. Extend the adversarial value pool with
  classic payloads (stacked queries, comment escapes, `OR 1=1`,
  unicode homoglyphs, null bytes) and assert: never in SQL text, always
  in params, and — on the e2e tier — actually executed harmlessly
  against the Postgres container.
- **Auth bypass suite (the crown jewels)**: property-based — for every
  generated query shape against a scoped cube with a viewer set, the
  emitted SQL contains the scope subquery (structural check via sqlglot
  AST, not substring), *and* on the execution tier every returned row
  satisfies the scope. Enumerate bypass attempts as named cases:
  via alias, via derived measure, via ungrouped mode, via federation
  merge SQL (review B1's `_lit` path is a known offender), via
  entities row-mode and cursor replay, via mutation preflight.
- **Mutation gates (post-entities M4)**: each of the four gates
  (global flag, entity ops, role policy, predicate_targeting) gets an
  enabled/disabled pair; pinned-column spoofing by the LLM → loud
  compile failure (spec already requires this); preflight `affects`
  count matches actual rows touched, tested in a rolled-back txn.
- **MCP surface**: per-request auth tests (defect A6); schema fuzzing
  of tool inputs (malformed JSON, wrong types, oversized payloads) →
  typed errors, never tracebacks-with-internals; error messages never
  leak connection strings or table DDL beyond the catalog's own
  descriptions.
- **Static + supply chain in CI**: `ruff` S-rules (flake8-bandit) on
  src (covers Bandit's checks without a second tool), `pip-audit`
  (or `uv` audit) weekly + on lockfile change, `zizmor` on workflow
  files, SBOM (`uv export` + cyclonedx) attached to releases. Secrets:
  gitleaks in pre-commit.
- **Fuzz the deserializers**: Hypothesis `from_type` + garbage-bytes
  fuzzing of Catalog-from-dict, SavedQuery, RowPlan, cursor codec —
  malformed input must yield typed validation errors, never partial
  objects.

## 9. CI tiers & infrastructure

Re-enable the workflow (it's already written) and split into tiers:

| Tier | Trigger | Contents | Budget |
|---|---|---|---|
| **PR** | every push | lint, mypy+pyright, unit + integration (Hypothesis `ci` profile), snapshot checks, dialect-validity property, changed-lines mutation score, CodSpeed | ≤ 5 min |
| **Merge** | main | PR tier + full coverage ratchet + API-breakage (griffe, already configured) + build smoke | ≤ 15 min |
| **Nightly** | cron | testcontainers matrix (PG, CH), differential execution oracle, metamorphic suite at `nightly` profile, full mutmut, memory tests, pip-audit | ≤ 2 h |
| **Release** | tag | nightly tier + manual BQ/Snowflake smoke script + SBOM | manual gate |

Mechanics:
- `pytest -m` markers: `unit` (default), `integration`, `e2e`
  (requires docker), `bench`, `security`. Unmarked = unit.
- One `conftest.py` per package (engine/mcp currently have none);
  shared catalog factories move to a `semql-testing` internal package
  so engine/mcp tests stop redefining fixtures.
- Flake policy: zero tolerated; a flaky test is quarantined with an
  issue link within 24 h (marker `quarantine`, excluded from gates,
  report job lists them so they can't rot silently).
- The 5 failing federate tests are resolved **before** tiers gate
  anything: fix or quarantine-with-issue each, so the suite returns to
  all-green truth (a suite that's "known red" trains everyone to
  ignore red).

## 10. Sequencing & success criteria

Ordered by leverage; each item is independently shippable and red/green.

1. **Triage the 5 red federate tests** → suite all-green. *(hours)*
2. **Equality matrix + dialect-validity property** (§4.1) → A1-class
   bugs impossible to reintroduce. *(1 day)*
3. **Differential merge harness** (§4.2) → A3 net. *(1 day)*
4. **Bind-params + auth-containment properties** (§8) → the two
   invariants PHILOSOPHY stakes the project on are machine-checked.
   *(1–2 days)*
5. **CI re-enable, tiered, ratcheted coverage, markers** (§9). *(1 day)*
6. **DuckDB-backed metamorphic suite** (§6.1 — TLP, monotonicity,
   additivity). First execution-based oracle, no containers needed.
   *(2–3 days)*
7. **mutmut baseline + changed-lines gate** (§6.2). *(1 day setup)*
8. **Testcontainers matrix + differential execution oracle** (§5).
   *(3–5 days; the flagship)*
9. **CodSpeed + budgets + pathology guards** (§7). *(1–2 days)*
10. **Adapter conformance suite + MCP auth/fuzz tests** (§4.2/4.3/§8) —
    aligned with entities M3–M5 so the contracts land tested. *(with
    entities work)*

Definition of done for the plan as a whole:
- A wrong-results bug of class A1/A2/A3 cannot merge: some harness in
  §4–§6 fails mechanically.
- Every PHILOSOPHY invariant names the test that enforces it.
- PR feedback ≤ 5 min; nightly produces a one-page report (mutation
  score, coverage, bench trend, quarantine list, oracle disagreements).
- A new backend or adapter is tested by inheriting suites, not by
  writing bespoke tests.
