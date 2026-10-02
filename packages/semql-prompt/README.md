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

Role builders keep mandatory query, authorization, and untrusted-data guidance
while accepting optional trusted `instructions`. With no explicit router
`scope_to`, the Query Generator can use the existing catalog retriever
(`user_query`, `retriever`, `top_k`) to select relevant cubes.

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

## License

BSD-3-Clause.
