# Prompt generation improvements

## Goal and scope

Improve all four requested outcomes: reliable query generation, smaller and more relevant context, evidence-grounded answers and follow-ups, and integrator control. Retain the existing role builders, typed model contracts, retrieval implementation, and compiler boundaries. No LLM client, embedding model, new router outcome, or chart-selection redesign.

Worktree: `.worktrees/feat/prompt-improvements`, branch `feat/prompt-generation`, based on current main (`7b4856a`). Preserve the original checkout's changes. No commit or push is authorized.

## Baseline

The original checkout's prompt suite passed: `uv run --frozen pytest packages/semql-prompt/tests --no-cov -p no:cacheprovider` produced 271 passed and 1 skipped. In-memory rendering demonstrated:

- Router prompt requests `path`, while `RouterDecision` requires `route_to`.
- A viewer-aware generator omits a finance measure, but the drilldown builder has no viewer parameter and renders it.
- Presenter and drilldown fence runtime data without explaining the fence in their fragments.
- A 400-token budget dropped the only cube and still returned an estimated 854 tokens.
- An unstructured 100-character string under a one-token budget returned 25 estimated tokens with `was_truncated=False`.

These are renderer/runtime findings, not measured model-quality results.

## Execution order and ownership

### 1. Role contracts, context, and builder controls

Primary file: `packages/semql-prompt/src/semql_prompt/prompt.py`.

- Correct `_ROUTER_OUTPUT_SCHEMA` to `route_to`; retain `RouterDecision`'s binary routing contract.
- Update `_SPEC_CONTRACT` from current `SemanticQuery`, `TimeWindow`, comparison, derived-measure, and join-filter contracts; expose supported features and important constraints concisely.
- Replace obsolete blanket cross-backend fallback advice with capability-aware guidance; do not imply raw SQL bypasses authorization or resolves ambiguous business meaning.
- Reuse catalog rendering/retrieval for `build_query_generator_prompt_fragment`, including glossary, relations, saved-query retrieval metadata, and hooks. Treat explicit router scope as authoritative; use retrieval only when no explicit scope is supplied.
- Add optional trusted `instructions` to role/planner builders, additive to mandatory instructions, not an arbitrary replacement template.
- Strengthen presenter guidance about supplied evidence, units, null versus zero, incomplete data, comparisons, and prose-only output. Fence labels and result-derived context and include role-appropriate trust-boundary instructions.
- Extend drilldown with viewer/policy, optional catalog, and optional current query. Reuse authorized field metadata and query instructions. Preserve base filters/time context; do not invent neighbor fields or broaden the authorized population. Filter focused-row references to available fields and reject unavailable anchor cubes without disclosing their content.
- Keep authorization/cache boundaries intact when filtering views, joins, and drill paths surfaced by affected builders.

### 2. Budget contract and content preservation

Primary file: `packages/semql-prompt/src/semql_prompt/prompt_budget.py`.

- Add explicit budget-fit reporting, retaining the documented approximate chars/4 estimator.
- Protect caller-designated required cubes from pruning; never infer importance from an unavailable priority field.
- Restrict pruning to known catalog/domain sections rather than arbitrary markdown headers, and respect fenced runtime content and section boundaries.
- Preserve mandatory output/trust instructions and unrelated caller prose. If protected/mandatory material cannot fit, report that rather than presenting it as successful enforcement.
- Make description pruning match renderer output only where it can safely distinguish descriptions from field/type/constraint information.
- Retain existing zero-budget behavior as an explicitly documented empty-result special case.
- Keep existing tests intact; new regression coverage must demonstrate consumer-visible boundaries and preservation, not weaker assertions.

### 3. Catalog entrypoints and integration

Primary file: `packages/semql-prompt/src/semql_prompt/catalog_tools.py`.

- Thread trusted `instructions` through `planner_prompt`, `sql_planner_prompt`, and `planner_prompt_segments`.
- Preserve static/overlay/ephemeral composition; custom instructions participate in the rendered segment rather than changing catalog identity rules silently.
- Inspect exported-symbol references before each signature change; update affected consumers only where required by an intentional contract change.

### 4. Verification

Run commands from the worktree with its own uv environment:

- `uv run --frozen pytest packages/semql-prompt/tests --no-cov -p no:cacheprovider`
- `uv run --frozen ruff format --check packages/semql-prompt/`
- `uv run --frozen ruff check packages/semql-prompt/`
- `uv run --frozen mypy packages/semql-prompt/`
- `uv run --frozen pyright packages/semql-prompt/`
- Target affected MCP/demonstration/core integration cases after resolving references; run broader tests if public behavior affects them.

Also run throwaway in-memory scripts against actual builders. Validate the router example with `RouterDecision`; compile representative comparison/segment/alias/derived-query examples; render viewer-restricted drilldown and retrieval-scoped generator prompts; assert protected fields and unavailable catalog targets are absent; exercise budget overflow, pinned cubes, code/data fences, catalog-section boundaries, zero budget, and unrelated caller headings; render all instruction-customized entrypoints. Inspect output and token estimates. No live model/provider invocation or accuracy claim without an evaluation dataset and explicitly configured model.

## Risks and acceptance

- Prompt/schema drift: generated examples and documented query shapes must validate/compile with current models.
- Unauthorized context: low-role viewers must not receive restricted fields, views, joins, or drill paths through changed surfaces.
- Prompt injection: runtime values remain descriptive data with neutralized closing delimiters and role-appropriate fence guidance.
- Destructive truncation: required content survives; impossible budgets are explicit, not silently treated as met.
- Relevance loss: explicit scope is never widened and ranked retrieval order survives; protected cubes cannot be removed by budget trimming.
- API usability: customization is additive and trusted; no duplicated framework or provider dependency.

After runtime verification, update the existing package README and root changelog with the final behavior and limitations. Record actual verification results here. Existing tests and verification assets must not be weakened to make changes pass.

## Verification results

- `just check`: passed—formatting, Ruff, mypy, Pyright, boundaries, 3,067 tests,
  and 83 snapshots; 27 skipped and one expected failure.
- Prompt-package checks: 285 passed and one skipped; all 21 package files were
  formatted, Ruff-clean, and type-clean.
- Public API comparison against `main`: no breaking changes reported for
  `semql_prompt`.
- Final in-memory smoke: all 15 assertions passed across typed router output,
  retrieval and explicit scope, viewer filtering, fenced presenter/drilldown
  context, comparison and derived-query compilation, additive instructions,
  and explicit budget fit. The compacted prompt fit at 1,105 estimated tokens
  after dropping two descriptions and the unprotected `customers` cube while
  preserving the protected `orders` cube, trust guidance, and output schema.
- No live model/provider quality claim was made; that requires an evaluation
  dataset and configured model.
