# Separate model-facing query and prompt repair hatches

## Goal and decisions

Support both user-selected repair targets as separate operations. SemQL builds repair prompts and validates accepted augmentation proposals; the host supplies any model client and owns retries, validation, execution, and acceptance. No model dependency, persistence, provider call, automatic retry, or arbitrary replacement template.

## Contract

New module: `packages/semql-prompt/src/semql_prompt/repair.py`. Public exports in existing `__init__.py`. Existing adaptive/render/compiler signatures stay unchanged.

- `RepairGuidance`: string enum values `catalog_references`, `required_filters`, `filter_types`, `time_ranges`, `supported_shapes`. Each expands to fixed SemQL-owned additive guidance, never arbitrary model-authored system instructions.
- `PromptExample`: frozen, extra-forbid Pydantic model with nonblank `question: str` and `query: SemanticQuery`.
- `PromptAugmentation`: frozen, extra-forbid Pydantic model with `guidance: tuple[RepairGuidance, ...] = ()` and `examples: tuple[PromptExample, ...] = ()`. Represents additions only; cannot edit authorization, output schema, SQL policy, user question, or catalog definitions.
- `RepairPrompt`: extends `BudgetResult` with `target: Literal['query', 'prompt']`. `output_model` property returns `QueryPlan` for query repair and `PromptAugmentation` for prompt improvement, usable with host structured-output clients.
- `build_query_repair_prompt(catalog, request: AdaptivePromptRequest, *, original_prompt: str, viewer=None, ctx=None, policy=None, count_tokens=None, instructions=None) -> RepairPrompt`.
- `build_prompt_repair_prompt` has the same signature and return type but target/output model differ.
- Both require a nonblank original_prompt, nonblank request.previous_output, and at least one safe request.diagnostics entry. They preserve those original/failed-output evidence blocks even for impossible budgets and fence them using `CatalogPrompt.ephemeral`. The caller must authorize disclosure of those arbitrary historical strings; SemQL cannot redact prior user/model data reliably.
- Query repair reuses the authorized adaptive query-generation contract and asks for a corrected QueryPlan. Safe diagnostics are sanitized by the existing adaptive renderer. User constraints/mandatory contracts remain fixed; authorization failures never authorize a bypass.
- Prompt repair treats the original prompt and newly rendered authorized query-generator context as fenced evidence rather than competing system instructions. The operative instruction requests only PromptAugmentation and includes its complete JSON schema. The model never returns an edited arbitrary full system prompt.
- Full combined string (schemas, evidence, guidance, catalog, runtime input) is counted with the callback or heuristic. Protected evidence and mandatory content survive an impossible budget with fits=False; optional adaptation context may be trimmed by the existing adaptive API. Data is never fed directly to the Markdown catalog trimmer.
- `apply_prompt_augmentation(catalog, request, augmentation, *, viewer=None, ctx=None, policy=None, count_tokens=None, instructions=None) -> AdaptivePromptResult`: explicit host acceptance step. Rebuild authorized selection first; compile every example with the catalog's viewer/runtime/context; reject unknown/unauthorized/non-compiling or out-of-selected-scope examples generically before rendering any proposal. Never include compiled SQL/params in examples. Fence question + SemanticQuery JSON as optional reference examples, expand only bounded fixed guidance as additive instructions, and rebuild through `build_adaptive_prompt`. No side effect or source mutation.

## Execution order

1. Resolve current diagnostic, adaptive-budget, and compiler touched-cube contracts inline; preserve all pre-existing local changes. HEAD matches fetched main, no pull required.
2. Implement new repair module. Parent owns API/export integration.
3. Independent slices: behavioral tests in `packages/semql-prompt/tests/test_prompt_repair.py`; standalone `demos/prompt_repair_demo.py` exercises public builders, manually supplied typed model outputs, acceptance/rejection, and corrected query compilation. No mocked/live model quality claims.
4. Run the full new repair test module, then full prompt-package suite. Run demo/public probes covering both output contracts, authorization, complete-token budgeting, hostile fence delimiters, and proposal validation.
5. Run scoped Ruff formatting/check, prompt-package mypy/Pyright, and repository boundary guards. Do not weaken tests or verification assets to make checks pass.
6. After runtime proof, update package README/changelog with host usage and trust boundaries; record observed verification here.

## Acceptance and edge cases

- Both functions include original prompt, failed output, sanitized failures, and explicit distinct output contracts.
- Tight budgets report inability to fit without deleting repair evidence or mandatory instructions.
- Historical prompt/failed output are untrusted; embedded closing delimiters cannot escape their fences.
- Prompt proposals cannot introduce arbitrary system instructions, widen scope, select inaccessible fields, or inject unvalidated SQL/examples.
- Example queries are compiled with caller identity and scope functions; every touched cube must be in the selected prompt scope. Generic rejection must not disclose hidden identifiers/parameter values.
- Existing adaptive entrypoint behavior is unchanged. This is an additive host-owned seam, not a second model runtime.

No commits, pushes, releases, or publication.

## Implementation and observed verification

- Added `repair.py` and seven public exports without changing existing adaptive,
  renderer, diagnostic, or compiler signatures. Added the two independently
  authored regression/demo slices and integrated them.
- Full new repair module: `uv run --frozen --python 3.12 pytest
  packages/semql-prompt/tests/test_prompt_repair.py --no-cov -p no:cacheprovider`
  passed **20 tests**. The first collection exposed pytest's reserved `request`
  parameter name; it was renamed before the complete module ran successfully.
- Full prompt package: `uv run --frozen --python 3.12 pytest
  packages/semql-prompt/tests --no-cov -p no:cacheprovider` passed **324 tests,
  2 skipped**. No full-workspace suite result is claimed.
- Scoped Ruff check passed; all four touched Python files passed format check.
  Mypy passed all seven prompt source files after adding an explicit optional
  local type annotation; Pyright reported zero errors or warnings.
- Tach, core-boundary, and frozen-model guards passed.
- `uv run --frozen --python 3.12 python -B demos/prompt_repair_demo.py` passed.
  It built both targets, retained fenced historical evidence and safe failure
  feedback, parsed explicitly caller-provided sample structured outputs,
  accepted a bounded in-scope augmentation, rejected an unavailable example
  generically, and compiled the corrected QueryPlan through `Catalog.compile`.
  The corrected SQL was
  `SELECT o.region AS region, SUM(o.amount) AS revenue FROM orders AS o GROUP BY region`.
- The demo's 4,000-token limit fits query repair but not prompt repair because
  the latter includes the full augmentation schema and historical evidence.
  It reports `fits=False` rather than dropping required content; a real host
  must decline the call or supply a larger model-appropriate budget.
- README documents output-model parsing, explicit host acceptance, complete
  budgeting, and historical-data disclosure ownership; changelog records the
  additive API.

No provider call, live model-quality claim, SQL execution, commit, push, release,
or publication was performed.

## Authorized commit and push verification

The user subsequently authorized pushing this work. The adaptive prompts and
both repair hatches are isolated on `feat/adaptive-prompt-repair` in
`.worktrees/feat/adaptive-prompt-repair`, based on fetched `origin/main` and
fast-forwarded to the already-pushed defect-repair commit `b47c14c`. Only the
11 feature files are included in the new commit; the original dirty checkout,
unrelated local documents, and prior repair history are preserved.

The isolated environment was initialized with
`uv sync --python 3.12 --all-extras --frozen`. With external conformance endpoint
variables unset, `UV_PYTHON=3.12 SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check`
passed formatting, Ruff, mypy (271 files), Pyright, repository boundary guards,
and **3,296 tests, 4 skips, 1 expected failure, 81 snapshots**. Real external
conformance used disposable PostgreSQL and ClickHouse containers. A test-only
callback was changed from an expression using `list.append()`'s return value
to an explicitly typed function so full-workspace mypy can check it.

Both public demos ran successfully from the isolated worktree. A scoped scan
found no private-key, AWS-access-key, credential-URL, GitHub-token, or JWT-shaped
matches in the transferred feature files. No model call, tag, release, or
package publication is authorized as part of this push.
