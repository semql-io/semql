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
  `estimate_tokens`.

## Install

```sh
pip install semql-prompt
```

## Quick start

The catalog-level conveniences take the catalog as their first argument:

```python
from semql import Catalog
from semql_prompt import planner_prompt, planner_prompt_segments, prompt_hash, to_openai_tools

text = planner_prompt(catalog, viewer=viewer)
segs = planner_prompt_segments(catalog)
key  = prompt_hash(catalog)
tools = to_openai_tools(catalog, viewer=viewer)
```

For the lower-level per-role fragment builders, see
[API reference](../../docs/api/semql_prompt.md).

## License

BSD-3-Clause.
