# Roadmap reconciliation — June 2026

Status: active · supersedes `docs/notes/sequenced-open-backlog.md`
(which predates everything below and described Phases 0–5 as the
frontier; those landed).

## 0. Purpose and inputs

Five planning artifacts now make claims about what to build next, and
they were written without knowledge of each other. This doc reconciles
them into one sequence, resolves the six places they conflict, and
records the disposition of every open item.

Inputs reconciled:

| Source | Nature | State |
|---|---|---|
| `TODOS.org` | live backlog | ~13 genuinely open items; 3 stale entries (patched 2026-06-12) |
| `docs/specs/graphql-borrowed.md` | decision record | **closed** — both "implement"s shipped (I14, I15); rest rejected/deferred |
| `docs/specs/ktx-borrowed.md` + `ktx-ports.md` | inspiration + 6-milestone plan (M1–M6, ~3–4 wk) | not started; written before the architecture review |
| `docs/specs/entities.md` | row-mode read/write spec, milestones M1–M5 | design pinned (D1–D10); supersedes TODOS S8 |
| `docs/specs/architecture-review-2026-06.md` | defects A1–A6, debts B1–B10 | A-list unfixed on main |
| `docs/specs/test-plan.md` + `property-testing.md` | test programme | adopted as plan; absorbs three TODOS items (§R4) |
| `docs/notes/gap-analysis-proposal.md` + org notes | historical source material | mined; remaining value = the open carry-overs in TODOS.org |

Prefix key used below: A/B = architecture review; C/M = ktx candidates
/ milestones; I/P/S/# = TODOS.org series; entities M1–M5 = entities
spec milestones.

## 1. Conflict resolutions (R1–R6)

### R1 — "No HAVING" (ktx locked decision 1) vs `SemanticQuery.having` on main

**Context.** ktx-ports locks "No HAVING. User filters reference
dimensions and segments; aggregates only via `Measure.filter`." But
`having` exists today: exported, federated (`MergeSpec.having`),
handled by the SQL parser (I13 maps `HAVING`), property-tested, and
named in A1's dropped-fields list. Meanwhile `Measure.filter`
(catalog-time conditional aggregation) does **not** exist yet (ktx R3).

**The two are not substitutes.** `having` is a *query-time* post-
aggregate predicate ("show groups where `sum(revenue) > 1000`" — the
LLM invents the threshold per question). `Measure.filter` is
*catalog-time* conditional aggregation (`SUM(CASE WHEN approved …)`).
An LLM cannot express an ad-hoc threshold through a catalog field.

| Option | Pros | Cons |
|---|---|---|
| (a) Remove `having`, route everything through `Measure.filter` | one mechanism; matches locked-decision text literally | breaking; loses ad-hoc post-agg filtering entirely; churns federate/parser/tests; wrong tool for query-time intent |
| (b) Keep `having`; read the locked decision as scoped to the *locality emission path* (no WHERE/HAVING auto-classification inside per-group CTEs) | non-breaking; both features keep their distinct jobs; M3's emission stays simple | two aggregate-filtering surfaces to document; the locked decision's text is broader than this reading |
| (c) Keep both but deprecate `having` at v1 | gradual | defers rather than decides; v1 freeze makes it permanent anyway |

**Resolution: (b) — confirmed by maintainer 2026-06-12.** `having`
(query-time, LLM-facing) and `Measure.filter` (catalog-time,
author-facing) are different features; M3's locality path applies
`Measure.filter` inside per-group CTEs and applies `having` at the
outer SELECT, with no classifier deciding between them. Recorded as
`docs/decisions.md` D5.

### R2 — ktx M3 as a third emission path vs one-meaning-one-implementation

**Context.** ktx-ports M3 adds `_emit_locality_query` beside
`_emit_simple_query` and `_emit_compare_query` — on a compiler whose
plan path already drops filters (A1) and whose alias logic exists
twice (B1). Philosophy addition #2: every semantic decision is made in
exactly one place.

| Option | Pros | Cons |
|---|---|---|
| (a) Build M3 as planned, parallel emitter | fastest path to chasm-trap safety; isolated blast radius | third copy of projection/predicate emission; A1-class divergence becomes 3-way; every later feature pays 3× |
| (b) Build M3 as a LogicalPlan→LogicalPlan transform (partition into MeasureGroups) + the single emitter, after A1/B1 | one emitter; the partition transform is reusable by federation (same safe-merge math); permanent | blocked ~2 weeks behind A1+B1; M3 design needs a light rework from ktx's shape |
| (c) Hybrid: ktx M2 (Dijkstra/JoinPath/MeasureGroup, lives in `logical.py`) now; M3 emission after B1 | M2 is IR-aligned investment with standalone value (ambiguity warnings, `has_one_to_many`); no wasted work; M3 lands on one emitter | M3 still waits on B1 |

**Resolution: (b) — maintainer chose 2026-06-12, overriding the (c)
recommendation.** *All* of ktx M2+M3 waits for B1: the graph upgrade,
the fan-out warning, and the chasm-trap emission land together on the
completed IR. No interim fan-out warning ships beforehand (the
quick-win #13 early-warning idea is withdrawn); B4 is closed in one
coherent pass rather than two.

### R3 — Raw-row federation carryovers vs B1 (delete the parallel compiler)

**Context.** The carryover TODO (distributive-mode where/segments
lift; segment-SQL→BoolExpr at construction) invests more capability
into federate.py's ~700-LoC parallel compiler — the code B1 wants
deleted. Its CNF prerequisite is already met.

| Option | Pros | Cons |
|---|---|---|
| (a) Do carryovers now in federate.py | user-visible federation gaps close sooner | deepens the debt B1 pays off; everything written gets rewritten |
| (b) Do carryovers after B1, as plan transforms on the IR split-point | written once; segment-SQL→BoolExpr parse benefits scope-portability (entities §4) too | federation keeps refusing where-trees/segments in distributive mode for ~2–4 more weeks |

**Resolution: (b).** The refusals are loud, not wrong — PHILOSOPHY
tolerates a refusal far better than a rewrite-twice.

**Superseded 2026-06-13 → effectively (a), by way of D10.** When W2 ran,
deleting the parallel compiler (B1's core) was deferred to its own
post-W2 workstream (decisions.md D10): it changes no behaviour and is
large, so it's poor value at W2's tail. With B1 deferred, holding the
carryovers hostage to it would have kept distributive where-trees /
segments refused indefinitely for no gain — so the carryovers landed
in federate.py now (stage 5: `_route_where_distributive` + segment
routing + the cross-partition residual the merge already emitted). The
"rewrite-twice" cost (b) avoided is now owned by D10's workstream, which
must keep these tests byte-stable as it moves the routing onto the
split-point. Net: users get correct distributive federation now; the
IR migration pays the rewrite once, later, against a green oracle.

### R4 — Three test mechanisms converge on one harness (#47, I4, I6)

**Context.** #47 (pointblank assertions, deferred "until the compiler
has a local-execution test fixture"), I4 (`Cube.asserts` smoke/range
checks via `semql test`), and I6 (plan replay + golden-file
regression) each independently want what test-plan §5 builds: a local
execution fixture and a growing golden corpus.

**Resolution: build the harness once** (test-plan §5: DuckDB
metamorphic harness + sqllogictest-style YAML corpus), then:
- I6 becomes a thin feature: the corpus *is* the replay fixture set;
  `semql test replay` runs it against the current catalog.
- I4 becomes `Cube.asserts` compiled and executed *by* the harness
  runner in CI.
- #47 is satisfied-or-retired: its deferral condition is met by the
  harness; pointblank adds value only if its assertion vocabulary
  beats `Cube.asserts` — evaluate then, don't pre-commit.
  *(Confirmed 2026-06-12: retire #47 when I4 ships.)*

Pro: one execution substrate, three features. Con: none of the three
ships before the harness (test-plan sequencing item 6, ~2–3 days).

### R5 — S2 (typed-ref codegen) vs B3 (QualifiedRef type)

**Context.** Both attack stringly-typed `"cube.field"` refs. S2
generates per-catalog ref classes; B3 introduces the parsing/typing
discipline inside the library.

**Resolution: B3 first, S2 second — confirmed 2026-06-12.** Codegen
emitted against today's ad-hoc string conventions would bake the
inconsistency (qualified vs local vs either) into generated
artifacts. After B3, S2's generator emits `QualifiedRef`s and stays
trivial. S2 remains demand-gated.

### R6 — I10 (silent cross-cube type coercion) is mis-tiered

**Context.** Filed as a P3-ish DX item, but the behaviour on main —
timestamp-vs-date and uuid-vs-string comparisons silently coerce — is
a refusal-over-omission violation (philosophy addition #1) and can
return wrong rows.

**Resolution: promote to the correctness tier — confirmed
2026-06-12** (Workstream 1, after the A-list). Shape per the TODO:
explicit `CompileError` on type mismatch, with `Dimension.coerce_to`
as the opt-in escape hatch, and the resolution surfaced in the
planner prompt.

**Done 2026-06-12 (decisions.md D8).** Implemented as
`FederationError(reason="cross_cube_type_coercion")` raised in
`federate._parse_bridge` — the single funnel both merge paths route
through — when a cross-backend bridge join's two keys have disjoint
acceptable-type sets (`{type} ∪ {coerce_to}`). `Dimension.coerce_to`
added (rejects `coerce_to == type`); the opt-in renders next to the
dimension `type` in the catalog prompt. Scoped to the federated bridge
path; same-backend raw-SQL `on` joins remain the B2 escape hatch.
Covered by `test_i10_cross_cube_coercion.py`. This closes W1.

## 2. Workstreams with user stories

### W1 — Correctness on main (A1, A3, A2, A4, A5, I10)

> As a **data engineer debugging a production incident**, when SemQL
> emits SQL I need it to contain every predicate the query specified —
> a missing WHERE clause is the incident.

> As a **platform owner federating Postgres and ClickHouse**, I need
> the same FederatedPlan to return the same rows regardless of which
> merge engine is configured.

Contents: A1 compile_plan fidelity; A3 merge-engine silent op drops
(+ differential merge harness); A2 partition boundary/lexical
comparison; A4 fix-or-quarantine the 5 red federate tests; A5 cache
aliasing/unhashable params; I10 typed refusal on cross-cube coercion.
Each starts with a failing test (test-plan §10 items 1–4).

Pros: directly serves "wrong results are the only unacceptable
outcome"; everything else in this doc gets built on top of these code
paths. Cons: zero new features for ~1 week. **Effort: ~1 week.
Dependencies: none. This workstream gates all others.**

### W2 — IR completion + federation rebuild (B1, B6, split-point, R3 carryovers)

> As a **maintainer adding a compiler feature**, I want exactly one
> place where projection, predicates, and aliases are decided, so a
> feature added to the query path cannot silently miss the plan,
> federation, or row-mode paths.

> As a **user querying across two warehouses**, I want filters and
> segments to work in a federated query the same way they do in a
> single-source query.

Contents: finish IR adoption (emitter trusts the plan; alias logic
once; `Join.kind` honoured); B6 structural predicate resolution;
federation split-point replaces federate.py's parallel compiler;
carryovers land as plan transforms; `FederatedPlan` frozen+versioned
(naming-convention §1.4).

Pros: deletes ~700 LoC of duplicate compiler; unblocks entities M2
(RowPlan derivation), ktx M3, and federation correctness in one move.
Cons: the longest single item (~1–2 wk); pure refactor pressure with
user value arriving at the end. **Effort: ~2–3 weeks. Depends: W1
(A1 specifically).**

**Status 2026-06-13 — functionally complete; the parallel-compiler
deletion split off (D10).** Landed and green: B6 keyed-by-dimension
predicate resolution (stage 1); one alias convention via `output_alias`
(stage 2); `CompareSplit` load-bearing (stage 3b); `compile_plan` trusts
a prebuilt plan — the A1 finish, a rewritten scan / pushed-down predicate
survives to emission (stage 4); distributive where-tree + segment lift
with cross-partition merge residual — the R3 carryovers, 5 parked A4
tests now green (stage 5); `FederatedPlan` frozen + version-stamped
(stage 7). `Join.kind` honouring is resolved (D9 — emitter honours
`plan_join.kind`, graph re-roots off left-joined cubes, default joins
INNER). Parked: deleting the
parallel compiler + `_lit` removal (D10 → its own post-W2 workstream —
`partition_scans` + a plan-trusting `compile_plan` are the primitives it
will build on). The "user value at the end" risk was retired by landing
the carryovers in the current compiler rather than gating them on the
rewrite (see R3 supersession).

### W3 — Join-graph integrity / chasm trap (ktx M2 → M3, B4)

> As a **catalog author** declaring `relationship="one_to_many"`, I
> expect the compiler to *use* that fact — a revenue sum that doubles
> because of a fanned join is the canonical semantic-layer lie, and I
> declared exactly the metadata needed to prevent it.

> As an **LLM agent**, when I combine measures from two fact cubes
> sharing a dimension, I want either correct per-source aggregation or
> a refusal naming the problem — not a quietly inflated number.

Contents: ktx M2 (Dijkstra with 10× `one_to_many` weighting,
`JoinPath{is_ambiguous, has_one_to_many}`, `MeasureGroup`) with the
fan-out warning; ktx M3 chasm-trap partition as an IR transform +
per-group CTE emission through the single emitter (per R2); `Measure.
filter` field lands at M3 start (ktx R3); `having` kept per R1;
anchor selection stays implicit, no `anchor_cube` query field (ktx R2,
confirmed 2026-06-12).

Pros: closes B4 — the largest *modelled-but-unenforced* correctness
gap — in one coherent pass on the finished IR. Cons: M3 is the single
biggest feature build (~1–2 wk); emission shape (COALESCE over shared
keys, FULL JOIN) needs careful snapshot review; B4 stays open until
W2 completes (accepted per R2). **Effort: M2 ~3–4 days + M3 ~1–2
weeks, both after W2 (per R2). ktx M1/M5/M6 cheap wins (~2 days
total) are independent — ship anytime.**

### W4 — Entities: row-mode reads + mutations (entities M1–M5)

> As a **support agent's LLM**, I want `get order 42` and `list user
> 7's orders` through the same catalog, auth scope, and MCP surface as
> aggregates — without anyone writing SQL.

> As a **platform owner**, I want writes to be quadruple-gated and
> preflight-previewed, so an LLM can *propose* a mutation but only an
> explicit confirmation executes it.

Contents: per `docs/specs/entities.md` (decisions D1–D10 pinned).
M1 model+catalog wiring; M2 read compile (RowPlan derived from
LogicalPlan); M3 engine execution + adapter capability sub-protocols
(B7 overlap); M4 mutations; M5 MCP tools (needs A6 per-request auth).

Pros: the headline feature; spec already red/green-milestoned.
Cons: M2 inherits whatever the plan path does — building it before W1/W2
means deriving RowPlan from a lowering that drops filters. **Effort:
~2–3 weeks. Depends: M2 on W2; M5 on A6 (~2–3 days, can run parallel
with W2).**

### W5 — Test infrastructure (test-plan §10; absorbs #47/I4/I6 per R4)

> As a **contributor**, I want a failing test to exist before any bug
> fix and a harness that makes A1/A2/A3-class bugs unmergeable, so the
> suite — not vigilance — carries correctness.

Contents and order: test-plan §10 (equality matrix, dialect-validity,
merge differential, bind-params/auth properties, CI tiers, DuckDB
metamorphic harness, mutation baseline, testcontainers oracle), with
property-testing.md's strategy redesign. I6/I4/#47 land on the
harness (R4).

Pros: every other workstream's definition-of-done lives here. Cons:
test-only investment competes for the same weeks. **Effort:
incremental; items 1–5 ≈ 1 week, interleaved with W1.**

### W6 — DX & tooling (ktx M1/M5/M6, I2-adjacent lint, S2 after B3, I11, I8)

> As a **catalog author**, I want `lint_catalog` to tell me every
> place raw SQL enters my catalog and every join the compiler can't
> route, before an LLM finds out at question time.

Contents: ktx M1 cheap wins (Provenance, ValidationReport warnings
channel, reserved-identifier quoting, parse lru_cache, DIALECT blocks,
measure-filter helper); ktx M5 boundaries lint (frozen-model
allowlist, raw-SQL f-string fence, AuthContext import fence); ktx M6
nested-WITH validation; raw-SQL lint rule (quick-win #10); later: S2
codegen (after B3, per R5), I11 docs site, I8 DDL import.

Pros: cheap, independent, mostly ≤1 day each; the lint items
*enforce* this month's conventions. Cons: none material — this is the
filler track for blocked days. **Effort: ~3–4 days for the now-tier.**

### W7 — Engine maturity (P7b async parity, B7, per-driver adapters)

> As a **FastAPI service owner**, I want the async engine to be the
> sync engine's equal — same cache, same hooks, same observability —
> because production traffic runs on the async path.

Contents: P7b (AsyncEngine gains the P7 cache + on_execute hook,
fixing A5 hazards in the same pass); shared-core refactor so
sync/async stop diverging (B7); per-driver async adapters
(postgres/clickhouse/bigquery/snowflake extras) stay demand-gated.

Pros: parity removes a whole class of "works in tests (sync), differs
in prod (async)". Cons: adapter family balloons the CI matrix —
that's why it stays gated. **Effort: parity ~2–3 days. Depends: A5
fix shape (W1).**

### W8 — Demand-gated bets (unchanged disposition)

P8 catalog migrations (re-open: saved-query churn in a real
deployment); S6 dbt ingest (re-open: first dbt-shaped adopter); S1
pydantic-ai recipes (re-open: a team on pydantic-ai asks); I5 sub-cube
`contains` (re-open: the wide-raw/narrow-published pattern appears);
multi-cube mutations (entities §8 — designed, not built). Each entry
keeps its re-entry criterion in TODOS.org; nothing here schedules
them.

## 3. The unified sequence

```
Week 1      W1 correctness (A1→A3→A2→A4→A5, I10) ⟷ W5 items 1–5
            ∥ W6 now-tier (ktx M1/M5/M6, lint) on blocked days
            ∥ A6 per-request MCP auth (feeds W4-M5)
Weeks 2–4   W2 IR completion + federation rebuild
            ∥ entities M1 (model wiring — no plan-path dependency)
Weeks 4–6   W3: ktx M2 (graph + fan-out warning) → M3 chasm trap,
            all on the finished IR (per R2)
            ∥ entities M2–M3 (RowPlan + engine, on the trustworthy path)
Weeks 6–8   entities M4–M5 (mutations, MCP) ∥ W7 async parity
            ∥ federation carryovers as plan transforms (R3)
Ongoing     W5 remaining tiers (harness → containers oracle → mutation
            score); W8 stays parked
```

Calendar honesty: serial-worst-case ≈ 8 weeks; with the parallel
tracks above and one contributor ≈ 6–7; the ktx-ports estimate
(~3–4 wk) and entities estimate (~2–3 wk) were each written as if
alone in the world.

## 4. Disposition of stale artifacts (actioned 2026-06-12)

| Artifact | Disposition |
|---|---|
| `docs/notes/sequenced-open-backlog.md` | superseded by this doc; delete or archive on next notes cleanup |
| TODOS.org [P7] | split: DONE (sync Engine, commit 1e3c37b) + new [P7b] async-parity TODO |
| TODOS.org [S8] | superseded by `docs/specs/entities.md`; entry now points there |
| TODOS.org "Aspirational notes → real docs" | gains pointer to this doc re: backlog supersession |
| `docs/specs/graphql-borrowed.md` | closed; no action |
| `docs/notes/*.org` (3 files) | historical; re-key per the notes TODO when touched, no deadline |
| TODOS.org [#47] | retire when I4 ships on the execution harness (R4, confirmed) |
| TODOS.org [I10] | promoted to correctness tier W1 (R6, confirmed) |
| TODOS.org [S2] | ordering note added: blocked behind B3 (R5, confirmed) |
| `docs/notes/` graduation question | decided 2026-06-12: stays gitignored; specs/ is the public design history |

## 5. Decision log (interview, 2026-06-12)

All conflicts resolved with the maintainer; nothing in this doc is
provisional.

| # | Question | Decision |
|---|---|---|
| R1 | fate of `having` vs "No HAVING" | keep both, scoped: `having` query-time, `Measure.filter` catalog-time; no WHERE/HAVING classifier in the locality path (→ decisions.md D5) |
| R2 | ktx M3 emission-path placement | **everything after B1** — M2 graph upgrade, fan-out warning, and M3 chasm trap all land on the finished IR; no interim warning (overrides the hybrid recommendation) |
| R3 | raw-row federation carryovers | after B1, as plan transforms on the IR split-point |
| ktx-R2 | anchor semantics | implicit only; no `anchor_cube` query field in M3 |
| R4 | #47 pointblank | retire when I4 `Cube.asserts` ships on the execution harness |
| R5 | S2 codegen vs B3 QualifiedRef | B3 first; S2 stays demand-gated and emits typed refs |
| R6 | I10 silent cross-cube coercion | promoted to correctness tier (W1) |
| — | docs/notes/ graduation | stays gitignored; `docs/specs/` is the public design history |
