---
name: semql-requirement-discovery
description: >
  Discover what a SemQL catalog should contain — before any Python
  gets written. Use upstream of `semql-cube`: when the user says
  "we want to answer revenue questions" or hands you a PRD and asks
  what cubes to build, this skill captures intent at the domain
  level and emits a requirements doc that `semql-cube` consumes.
---

# Discovering catalog requirements

This skill bridges **intent** ("we want analytics for our orders
process") to a **structured plan** the `semql-cube` skill can act
on. It stays out of Python: no SQL fragments, no alias choices, no
column types. Those decisions belong in `semql-cube`, where the
technical interview happens.

**Use this skill when:**
- The user hasn't decided what to build yet.
- The user hands you a PRD / feature brief / strategy doc and asks
  "what cubes do we need?"
- The user can describe what their users want to know, but hasn't
  mapped that to tables.

**Use `semql-cube` instead when:**
- The user already knows the entities and now needs the Python.
- The user is extending an existing catalog with one more measure
  or dimension.
- The conversation is about SQL fragments, granularities, ScopeFn
  predicates, or other technical details.

## What the skill produces

A markdown requirements document at `docs/requirements/<name>.md`
(or a path the user specifies). The doc is the contract with
`semql-cube` — that skill reads it and writes the Python.

The doc lists **intent-level** facts only:
- The domain and audience.
- The questions users will ask.
- The entities those questions imply.
- Which entities relate to which.
- How answers should land (chart / format hints from a closed enum).
- Who's allowed to see what (in business terms).

It does **not** list:
- Backend / dialect choices.
- SQL fragments or column expressions.
- Specific aggregations or granularities.
- Exact join predicates.
- ScopeFn implementations.

Those are `semql-cube`'s job. If the user wants to talk about them,
hand off.

## Inputs

Either or both:

1. **PRD documents** — paths or URLs the user names. Read them with
   whatever file-reading affordance the runtime gives you (Claude
   Code: `Read` tool; Codex / Gemini CLI: file-read; Copilot: paste
   into context). Cite specific sections in your follow-up questions
   so the user sees you're grounded ("section 3.2 mentions
   transactions; should that be a separate cube from orders?").
2. **Interview answers** — when the PRD is silent on an intent
   question, ask. Prefer a structured-choice mechanism when the
   runtime supports it (Claude Code: `AskUserQuestion`; CLI tools:
   numbered options the user picks via input). Fall back to plain
   prose questions otherwise.

## The interview — six domain-level passes

### 1. Domain and audience

Who's asking these questions? Founder reviewing weekly numbers, an
operations lead investigating incidents, a sales rep checking their
own pipeline? Audience drives auth (everyone vs role-scoped) and
view design (curated facade vs full catalog).

Capture in the doc:
- Domain (one line).
- Audience (roles or personas).
- Primary question shapes (headline / breakdown / compare /
  context).

### 2. The questions

Get **5-10 example questions** users will ask. Real ones, in their
voice. Examples:
- "What was revenue last quarter?"
- "Which regions outperformed?"
- "How does that compare to the previous quarter?"
- "How many orders contributed?"

If the user can't come up with five, the analytics ask is
underspecified. Flag it as a blocker; come back when there are
real questions.

### 3. The entities

For each question, identify the **business entity** it's about — not
the database table. "Revenue" implies orders / transactions /
invoices (pick one based on the user's mental model). "Active
users" implies customers + activity. Each entity becomes a cube.

Capture in the doc:
- Cube name (use the business-domain noun, plural lowercase:
  `orders`, `customers`, `tickets`).
- One-sentence description ("one row per …").
- Which questions reference this entity.

Don't ask for column lists or DDL. Those land in `semql-cube`.

### 4. Relationships

For each pair of entities a question crosses, capture **direction**
in business terms:
- "An order belongs to one customer; a customer has many orders" →
  many-to-one from `orders` to `customers`.
- "Each ticket has one assignee; an assignee has many tickets" →
  many-to-one from `tickets` to `employees`.

The user describes the relationship; `semql-cube` translates it to
a `Join` with a predicate.

Capture in the doc:
- For each entity: who they relate to and how (1:1, 1:N, N:1).

### 5. Presentation hints

For each entity (and each major measure / dimension on it), ask how
the answer should *land in front of the user*. Stay at the
intent level — chart-type names from the supported enum, format
names from the supported enum. Do NOT propose anything outside
these; SemQL won't render it.

**Supported chart types** (use these names exactly):
- `line_chart` — time series with granularity.
- `bar_chart` — categorical comparison.
- `pie_chart` — share of whole (small N only — flag if the entity
  has many categories).
- `data_table` — fallback / row listings / multi-measure detail.

**Supported formats** (for measures and presentation-hinted dimensions):
- `currency` — money. Pair with a unit like "USD" if multi-currency.
- `percent` — already a rate (0.12 = 12%).
- `integer` — counts, IDs.
- `duration` — seconds / minutes / hours; let the renderer humanise.
- `raw` — fallback.

Capture in the doc:
- **Default chart per entity** — `line_chart` / `bar_chart` / `pie_chart` /
  `data_table`, or "no default" if the questions span shapes.
- **Per-measure format** — only when not obvious from the name
  (`revenue` is clearly currency; `count_distinct_users` doesn't
  need a hint). Skip when defaults are fine.
- **Per-dimension format hint** — only for non-obvious ones
  (`duration_seconds` benefits from `format=duration`; `region`
  doesn't need a hint).

If the user says "we want a forecast / pivot / sparkline / heatmap"
— flag it. SemQL doesn't render those today. Either pick a
supported substitute or note it as out-of-scope in the open
questions section. **Don't invent chart-type names** to placate the
user.

When in doubt: data_table renders everything. Charts are upgrades,
not requirements.

### 6. Authorisation — at the policy level

Capture the **rules** the catalog has to honour, not the
implementation:

- **Role gates.** Which entities are restricted to which roles? In
  business terms: "finance ledger is finance team only."
- **Row-level visibility.** Are there entities everyone can see, but
  scoped to their slice? "Sales reps see only their team's orders."
  "Engineers see only their incidents."
- **Tenant model.** Is this multi-tenant? Per-tenant schemas, or
  shared tables with a tenant column?

Capture in the doc:
- Expected viewer roles (the role vocabulary).
- For each row-level rule: the rule in business terms + which
  entities it applies to (without writing the SQL).
- Tenant model (one line).

`semql-cube` translates "sales reps see their team's orders" into a
`ScopeFn` returning a `ScopePredicate` with the right SQL. You don't
write that SQL here.

## Output document

Write to `docs/requirements/<catalog_name>.md`. Use this structure
verbatim so `semql-cube` parses it predictably:

```markdown
# Catalog Requirements: <name>

## Context

- **Domain**: <one line>
- **Audience**: <roles or personas>
- **Primary question shapes**: <headline / breakdown / compare / context>

## Example questions

1. <question>
2. <question>
...

## Entities

### <cube_name>
- **Description**: <one sentence — "one row per …">
- **Relates to**:
  - many_to_one → `<other_entity>` ("a <this> belongs to one <other>")
- **Question references**: questions 1, 3, 5
- **Default chart**: line_chart | bar_chart | pie_chart | data_table | none
- **Presentation hints**:
  - Measure `<name>` → format `currency` | `percent` | `integer` | `duration`
  - Dimension `<name>` → format `duration` (etc — skip when obvious)

### <next_entity>
...

## Views (optional)

When the catalog grows past ~10 entities or there are repeated
question shapes, suggest curated facades:

### <view_name>
- **Description**: <one line>
- **Surfaces**: <which fields from which entities, business names>
- **Used for**: <which questions>

## Authorisation

- **Viewer roles**: [<role1>, <role2>, …]
- **Role gates**:
  - `<entity>` — restricted to roles [<role>, …]
- **Row-level rules** (business prose, not SQL):
  - <entity>: <rule, e.g. "sales reps see their team's rows; admins see all">
- **Tenant model**: schema | discriminator | none — <one-line rationale>

## Open questions

- <Anything the PRD / interview didn't resolve. semql-cube surfaces
  these before writing any Cube that depends on them.>
```

## Output report

After writing the doc, send the user a short report:

1. Path to the requirements document.
2. Entity count + view count.
3. Open questions that need resolving before `semql-cube` can run.
4. Suggested next step: invoke `semql-cube` pointing at the doc.

End with: "Want me to refine any of these — narrow down an entity's
scope, propose more views, sketch additional questions?" That's the
follow-up invitation. The user opts into another pass or moves on
to `semql-cube`.

## Interview etiquette

- Cite the PRD where you have it. "Section 3.2 mentions transactions
  separately from orders — should we model them as two cubes?"
- Ask in business terms, not technical ones. "Who's allowed to see
  this data?" not "What role string should we use?"
- Don't propose SQL, joins-as-predicates, or ScopeFn names. Those
  belong in `semql-cube`.
- Don't write Python in this skill. Output is the markdown doc.
- One bundle of related questions at a time. Use the runtime's
  structured-choice mechanism for closed questions when available
  (Claude Code: `AskUserQuestion`; CLI tools: a numbered prompt the
  user picks via input). Cap at four options per question.

## Common pitfalls

- **Skipping straight to schema.** If the user starts listing
  columns, gently redirect: "Let's nail down the questions first;
  the columns fall out of those."
- **Inventing the audience.** If the user doesn't say who's asking,
  ask. Audience is load-bearing for auth.
- **Treating auth as an afterthought.** Section 5 is mandatory.
  Catalogs without a clear auth story tend to ship one and then
  have to refactor.
- **One giant entity.** A "facts" entity that fuses orders +
  customers + products is an entity smell — split by business noun.

## See also

- `skills/semql-cube.md` — the downstream skill that writes Python
  from this skill's requirements doc.
- `PHILOSOPHY.md` — design invariants. The catalog-is-data,
  identity-is-caller-side, auth-is-compiler-side line is what
  Section 5 honours.
