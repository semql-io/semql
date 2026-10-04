# Adaptive prompt generation and augmentation

## Agreed scope

Deterministic request-scoped adaptation in semql-prompt for question/catalog relevance, conversation and external reference context, and generation-failure feedback. No historical learning, persistence, LLM client, model invocation, new dependency, or automatic retry execution. The host continues to generate, validate, compile, and execute queries.

Promptify's [prompt builder](https://github.com/promptslab/Promptify/blob/main/promptify/prompts/builder.py) separates task instructions, domain context, examples, and output schemas. Reuse that composition principle, not its provider/execution stack or arbitrary replacement templates.

## Baseline and preservation

Original checkout has existing local core-library repairs, including prompt security and budget changes. Preserve them. HEAD and fetched origin/main both resolve to fc57a9b; no pull is necessary. Existing runtime smoke established query-dependent retrieval, explicit scope precedence, fenced augmentation, and protected-budget overflow. One exploratory assertion initially used the wrong delimiter spelling; the actual untrusted-data delimiter was read and its neutralization exercised successfully. No model-quality result is established.

## Contract

New module: packages/semql-prompt/src/semql_prompt/adaptive.py.

- AdaptivePromptRequest: nonblank question; immutable tuples of conversation and retrieved_snippets; optional previous_output; immutable tuple of safe Diagnostic values; optional explicit scope_to tuple of cube names.
- AdaptivePromptPolicy: positive top_k (default 5), positive max_tokens (default 8000).
- AdaptivePromptResult: BudgetResult fields plus selected_cubes and selection_reason (explicit, question, conversation, fallback, empty).
- build_adaptive_prompt(catalog, request, *, viewer=None, ctx=None, policy=None, count_tokens=None, instructions=None).

Use existing authorization and field-visibility gates before ranking. SQLiteBM25Retriever is the deterministic lexical backend; enrich index-only descriptions with visible field names/display names/descriptions. Never index field SQL or opaque metadata. The current question determines ranking; only when it has no matches may conversation determine ranking. Otherwise fall back to the full authorized catalog rather than silently emitting an empty scope. No-viewer mode preserves existing unfiltered catalog-tooling semantics.

Explicit scope is authoritative, deduplicated, and validated against the authorized exposed surface. Unavailable/unknown requests reject with a generic error that does not disclose identifiers. In automatic selection, safe diagnostic references to available cubes/fields retain those cubes. Find shortest authorized undirected join paths connecting selected cubes and retain intermediary cubes; this may exceed top_k. Filter rendered join targets to included cubes. Keep complete selected-cube field definitions; do not attempt field-level semantic dependency pruning. Relevant glossary entries are selected lexically from question/conversation; catalog prose is trusted host content.

Compose using build_query_generator_prompt_fragment. Runtime question, conversation, references, and previous output are fenced using CatalogPrompt.ephemeral. Raw runtime strings are caller-authorized data; SemQL cannot infer their access controls. Prior output is evidence of a failed attempt, never an executable plan or authority to widen scope. Typed Diagnostic payloads use existing safe public categories/repair hints and retain only authorized references. Authorization failures never trigger scope expansion or bypass advice.

Budget the complete combined string, including runtime content and trusted instructions. Drop optional retrieved snippets from the end, oldest conversation turns, then previous output before trimming optional catalog prose through PromptBudget. Never submit runtime data to the catalog Markdown trimmer. Protect all selected cubes, question, and diagnostic repair guidance. Impossible budgets preserve required material and return fits=False. Report whole-unit runtime removals plus existing catalog removals in dropped. Count model tokens using the host callback when supplied; label heuristic counts otherwise. Never silently send an overflowing prompt.

## Execution and verification

1. Implement the new module and public exports; existing renderer and compiler APIs remain unchanged.
2. Add behavioral regression coverage in packages/semql-prompt/tests/test_adaptive_prompt.py. Keep existing tests and verification assets unchanged.
3. Add demos/adaptive_prompt_demo.py exercising question selection, a conversational follow-up, external reference augmentation, and compile-failure feedback through the public API; compile the corrected semantic query against a real catalog.
4. Run the demo with `uv run --frozen --python 3.12 python -B demos/adaptive_prompt_demo.py`; expect demonstrated scope changes, safe diagnostic augmentation, and successful corrected compilation.
5. Run `uv run --frozen --python 3.12 pytest packages/semql-prompt/tests --no-cov -p no:cacheprovider`; expect all affected regressions and existing prompt tests to pass, with optional-provider skips reported honestly.
6. Run package Ruff format/check, mypy, and Pyright using uv and Python 3.12; run existing cross-package/frozen-model guards. No broad formatter or autofix over unrelated local repairs.
7. After runtime proof, document the public API, trust boundaries, host-owned loop, and budget behavior in packages/semql-prompt/README.md and CHANGELOG.md. Record exercised results here.

## Risks and acceptance

- Authorization: denied cube/field metadata cannot influence ranking or appear in prompt, selected_cubes, joins, or diagnostic references. Verify two viewers and dynamic policies against the same Catalog.
- Prompt injection: embedded closing delimiters are neutralized; hostile headings never control catalog trimming. Verify complete-question and repair-guidance survival with tight budgets.
- Relevance: current question outranks stale conversation; no-match fallback is explicit; explicit scope never expands through retrieval or diagnostics.
- Join correctness: automatic multi-cube selection preserves authorized bridge cubes, without adding unavailable targets or assuming a nonexistent relationship.
- Failure handling: safe typed diagnostic categories add actionable bounded guidance; raw error messages/parameter values are never serialized. Host owns retries and must revalidate each generated query.
- Budget correctness: callback counts the final complete prompt; removing context never slices a fence or JSON payload. Impossible required content remains intact with fits=False.
- State isolation: no adaptive cache or learning state; request inputs are immutable tuples. Reusing a catalog across viewers must not leak request content.

## Verification results

- Public API regression coverage exercises current-question precedence over
  stale conversation, conversation fallback, explicit-scope rejection,
  authorized shortest join bridges, static field roles, dynamic catalog
  policies, authorization-diagnostic redaction, fenced prompt-injection
  containment, complete-prompt model-token counting, protected question/cubes,
  and authorized fallback selection.
- `uv run --frozen --python 3.12 pytest packages/semql-prompt/tests --no-cov -p no:cacheprovider`:
  304 passed, 2 skipped.
- Ruff check passed for the new module, public exports, regression tests, and
  demo. Mypy passed for all six prompt-package source files. Pyright passed
  with 0 errors, 0 warnings, 0 informations.
- `uv run --frozen --python 3.12 python -B demos/adaptive_prompt_demo.py`
  demonstrated conversation-driven cube selection, reference augmentation,
  safe unknown-field feedback after a real compile failure, and successful
  compilation of the corrected query.
- No live model call or prompt-quality claim was made.
- `scripts/check_api_break.py --base fc57a9b` reports one pre-existing core API
  change at `semql.model.Metadata` (`dict[str, str]` to `Mapping[str, str]`);
  this feature does not modify `semql/model.py`. The script compares only
  `semql`, `semql_mcp`, and `semql_erd`, so it does not inspect the new
  `semql_prompt` exports. Its `--base main` attempt could not load any baseline
  because a `griffe-main` branch already exists; despite skipping all modules,
  it printed “No breaking changes.” The comparison was retried with the
  verified `fc57a9b` commit hash to expose the real result.

No commit, push, publication, or release is requested.
