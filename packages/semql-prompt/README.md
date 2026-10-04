# semql-prompt

LLM-facing prompt rendering for a [semql](https://github.com/semql-io/semql)
`Catalog`.

`semql`'s compiler is pure — it turns a `SemanticQuery` into SQL and never
renders a prompt. This package is the rendering layer on top:

- **Four-role prompt fragments** — `build_planner_prompt_fragment`,
  `build_router_prompt_fragment`, `build_presenter_prompt_fragment`,
  `build_drilldown_prompt_fragment`, `build_query_generator_prompt_fragment`.
- **Cacheable segments** — `CatalogPrompt` (viewer-invariant `static` +
  per-viewer `overlay`) for prompt-cache breakpoints, plus `prompt_hash`.
- **Tool-description projection** — `to_openai_tools` / `to_langchain_tools`
  / `to_openai_function` for function-calling clients, and
  `to_bedrock_converse_tools` for the Bedrock Converse API (object-rooted
  `inputSchema`).
- **Prompt-token budgeting** — `PromptBudget`, `apply_budget`,
  `estimate_tokens`; every result states whether its estimate actually fits.

## Install

```sh
pip install semql-prompt
```

## Quick start

The catalog-level conveniences take the catalog as their first argument:

```python
from semql import Catalog
from semql_prompt import planner_prompt, planner_prompt_segments, prompt_hash, to_openai_tools

text = planner_prompt(
    catalog,
    viewer=viewer,
    instructions="Prefer concise labels.",  # trusted application policy
)
segs = planner_prompt_segments(catalog)
key = prompt_hash(catalog)
tools = to_openai_tools(catalog, viewer=viewer)
```

Cacheable `static`/`invariant` content contains only viewer-independent
catalog information and public fields. A dynamic `Catalog.policy` moves
cube content into the authorized per-viewer projection; empty cube roles
do not bypass that policy. Authorized field descriptions override their
public baseline when combining projections. The documented no-viewer
mode remains unfiltered tooling access, not an authorization decision for
a later viewer-scoped request.

Role builders keep mandatory query, authorization, and untrusted-data guidance
while accepting optional trusted `instructions`. With no explicit router
`scope_to`, the Query Generator can use the existing catalog retriever
(`user_query`, `retriever`, `top_k`) to select relevant cubes.

## Adaptive request prompts

`build_adaptive_prompt` composes question-aware catalog context with a
conversation, retrieved reference snippets, and safe generation diagnostics.
It uses a deterministic lexical retriever and does not call an LLM, execute a
query, persist history, or manage retry loops:

```python
from semql_prompt import (
    AdaptivePromptPolicy,
    AdaptivePromptRequest,
    build_adaptive_prompt,
)

result = build_adaptive_prompt(
    catalog,
    AdaptivePromptRequest(
        question="Show the trend.",
        conversation=("Show recognized revenue by month and region.",),
        retrieved_snippets=("Finance defines recognized revenue from paid orders.",),
    ),
    viewer=viewer,
    policy=AdaptivePromptPolicy(top_k=5, max_tokens=8_000),
    count_tokens=model_tokenizer.count,
)
if not result.fits:
    raise ValueError("required adaptive prompt content exceeds the token budget")
prompt = result.text
```

The current question drives selection; conversation is consulted only when the
question matches no catalog cubes. With no matches, the full authorized,
prompt-exposed catalog is the explicit fallback. A caller-provided `scope_to`
is authoritative; unavailable or unauthorized names fail closed. Automatic
selection can include authorized join-bridge cubes and validated diagnostic
references, so `top_k` is a relevance target rather than a hard final count.

Conversation, question, prior output, and retrieved snippets are fenced as
untrusted data. Only SemQL `Diagnostic` values contribute repair guidance;
authorization diagnostics never widen scope. The host owns source authorization
for external snippets, model invocation, output validation, compilation, retry
limits, and execution. Pass a model-specific token counter when available;
otherwise `token_count` uses the chars/4 heuristic. The result reports the
selection reason, selected cubes, dropped optional context, and `fits` status.

### Separate repair hatches

Use `build_query_repair_prompt` for a corrected **`QueryPlan`**, and
`build_prompt_repair_prompt` for a **`PromptAugmentation`** proposal. Both
require the original prompt plus the failed output and typed diagnostics:

```python
from semql_prompt import (
    AdaptivePromptRequest,
    apply_prompt_augmentation,
    build_prompt_repair_prompt,
    build_query_repair_prompt,
)

failure = AdaptivePromptRequest(
    question=question,
    scope_to=("orders",),
    previous_output=failed_output,
    diagnostics=(safe_diagnostic,),
)
query_repair = build_query_repair_prompt(
    catalog, failure, original_prompt=original.text, viewer=viewer,
)
prompt_repair = build_prompt_repair_prompt(
    catalog, failure, original_prompt=original.text, viewer=viewer,
)
```

Choose one operation and check its `fits` before sending `text` to your
model client. `output_model` supplies its Pydantic structured-output schema.
The host invokes the model and parses the returned JSON with
`repair.output_model.model_validate_json(response_json)`. For query repair,
compile every returned `QueryPlan` step with the same identity, context, and
host constraints before execution. SemQL does not invoke a provider or retry.

For prompt improvement, the host reviews the proposal before calling
`apply_prompt_augmentation(catalog, failure, proposal, viewer=viewer)`.
`PromptAugmentation` accepts only bounded `RepairGuidance` values and typed
`PromptExample(question, query)` examples, not arbitrary instructions or a
replacement prompt. The acceptance step compiles each example with the caller's
identity and context, rejects unavailable or out-of-selected-scope examples,
then rebuilds through the authorized adaptive renderer. Examples remain fenced
reference data and may be dropped under budget pressure. Compilation proves
query readiness, not relevance; the host still reviews their meaning.

Both hatches preserve repair evidence and mandatory content even when the
budget cannot fit (`fits=False`). The token counter covers the complete final
prompt, including the output schema and historical evidence. Pass the same
`policy`, `ctx`, trusted `instructions`, and `count_tokens` to all steps.
Original prompts and failed outputs are fenced as untrusted data, but cannot
be reliably redacted: the host must authorize their disclosure to the model.
Do not forward raw exception messages or internal error payloads as diagnostics.

Run `uv run --python 3.12 python demos/prompt_repair_demo.py` for both operations,
explicitly caller-provided sample structured outputs, proposal acceptance and
rejection, and real corrected-query compilation without a model client.

Budgeting uses an approximate chars/4 estimate. Always check
`BudgetResult.fits`; it is `False` when protected cubes or mandatory
instructions alone exceed the requested limit:

```python
from semql_prompt import PromptBudget

result = PromptBudget(max_tokens=4_000).apply(
    text,
    protected_cubes=frozenset({"orders"}),
)
if not result.fits:
    raise ValueError("required prompt content exceeds the configured budget")
```

For the lower-level per-role fragment builders, see
[API reference](../../docs/api/semql_prompt.md).

### Prompt-budget results

`PromptBudget.apply(text)` reports `token_count`, `count_source`
(`"heuristic"` by default), and explicit `fits` status alongside
`was_truncated` and `dropped`. The default count is only the package's
chars/4 heuristic. Supply a model-specific counter without adding a dependency:

```python
from semql_prompt import PromptBudget

result = PromptBudget(max_tokens=8_000).apply(
    prompt,
    count_tokens=model_tokenizer.count,
)
```

The same `count_tokens` callback is accepted by `apply_budget`. Trimming
removes known optional catalog sections; if required or otherwise untrimmable
text remains over budget, it is preserved and `fits` is `False`.

Trimming optional domain prose requires recognized section boundaries.
An ordinary cube's `**Relations:**` marker never permits removing its
fields or following protected cubes. Missing domain context is left intact
when no safe structural trim is available.

## License

BSD-3-Clause.
