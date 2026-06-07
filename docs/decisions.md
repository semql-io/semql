# Design decisions

Pinned answers to recurring "should we add X?" questions. Each entry
records the *reason* for the call so a future revisit can weigh
whether the constraints have changed.

The format is loosely an ADR — context, decision, consequences —
condensed to a paragraph each.

---

## D1. PyYAML — no, not in core

**Context.** Some BI tooling (Cube.js, dbt, LookML) loads cube
definitions from YAML files. The question recurs: should `semql`
ship a YAML loader, or expose one through an optional extra
(`pip install semql[yaml]`)?

**Decision.** Neither. Python is the native catalogue language. A
YAML loader sits outside core and outside extras — if and when
demand is real, it ships as a separate package (`semql-yaml`).

**Why.** PHILOSOPHY.md is explicit: "Python is the native language
for cube definitions. Type safety, refactoring, and testing come
free." A YAML loader inside core invites the long tail of dynamic
loading concerns (schema validation against a remote registry,
templating, hot reload) and dilutes the type-checker payoff. The
out-of-tree shape preserves "dependencies you don't need should
cost nothing to avoid."

**Revisit.** If three independent users ship YAML loaders to bridge
non-Python services, and the loaders converge on a roughly common
schema, accept it as `semql-yaml`.

---

## D2. mypy and pyright — keep both, for now

**Context.** Both run under `just typecheck`. They overlap heavily;
neither subsumes the other. The dev loop runs ~1s slower than it
would with just one.

**Decision.** Keep both pre-v1. mypy stays the structural backbone
(strict mode catches the classic generics / variance bugs); pyright
catches more in narrow inference corners (Pydantic field
resolution, `runtime_checkable` Protocol drift). Together they have
caught real bugs that one alone missed during this codebase's
build-out — the cost is worth the redundancy.

**Why.** Pre-v1 is when type discipline pays the most: the surface
is moving, and a single bad inference can ripple into the public
API. Once the surface freezes at v1, the marginal value of the
second checker drops and we pick one.

**Revisit.** At v1 cut. Drop pyright if mypy's gaps haven't bitten
in the prior six months.

---

## D3. Fluent Interface on top of Pydantic — no

**Context.** Could we layer a builder-style `Cube.named("orders")
.with_table("orders").with_measure(...)` API on top of the
Pydantic constructor? Some users prefer chained-call ergonomics.

**Decision.** No. Pydantic kwargs are the only catalogue-authoring
API. There is no fluent layer, no shorthand factory, no DSL.

**Why.** Two reasons. First, kwarg construction with type hints +
default values is already short — every "extra" character is a
catch by the type checker. Second, a fluent layer is a parallel
public surface that has to be kept in sync with the data model on
every change; doubling the maintenance cost of every new field is
not worth the cosmetic gain. PHILOSOPHY.md: "Composes with your
stack — it does not own it."

**Revisit.** If a downstream tool needs a callable / chainable
shape (e.g. a no-code UI builder), they layer it themselves on top
of `Cube(...)`. Don't pull it into core.

---

## D4. Catalogue value types are frozen Pydantic models

**Context.** Should `Measure`, `Dimension`, `Cube`, `SemanticQuery`,
etc. be frozen?

**Decision.** Frozen. `model_config = ConfigDict(frozen=True)`
everywhere except `Cube` itself (which has too many fields and
internal cross-validation to lock down right now).

**Why.** Catalogue and spec objects are value types in the
Evans/Fowler sense — equality is structural, identity is
irrelevant, and mutation invites consistency bugs the compiler
can't catch (rebinding `query.measures` after compile started would
produce a result that disagrees with the query the caller built).
Frozen is the default for value objects.

**Revisit.** Only if a real performance scenario emerges where
copy-on-modify dominates. The Pydantic v2 `model_copy(update=...)`
shape covers the common "tweak one field" need without re-opening
mutation.
