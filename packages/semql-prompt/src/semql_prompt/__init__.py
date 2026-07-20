"""Public surface of the semql-prompt package.

The LLM-facing rendering layer for a semql ``Catalog``: the typed
four-role planner / router / presenter / drilldown prompt fragments, the
cacheable two-segment ``CatalogPrompt``, tool-description projection
(OpenAI / LangChain), and prompt-token budgeting.

The ``semql`` compiler emits SQL and never renders a prompt; that lives
here. Catalog-level conveniences (``planner_prompt`` /
``planner_prompt_segments`` / ``prompt_hash`` / ``to_openai_tools`` /
``to_langchain_tools``) are functions taking the catalog as their first
argument.
"""

from __future__ import annotations

from semql_prompt.bedrock import to_bedrock_converse_tools
from semql_prompt.catalog_tools import (
    planner_prompt,
    planner_prompt_segments,
    prompt_hash,
    sql_planner_prompt,
    to_langchain_tools,
    to_openai_tools,
)
from semql_prompt.prompt import (
    CatalogPrompt,
    ToolDescriptionProjection,
    build_drilldown_prompt_fragment,
    build_planner_prompt_fragment,
    build_planner_prompt_segments,
    build_presenter_prompt_fragment,
    build_query_generator_prompt_fragment,
    build_router_prompt_fragment,
    build_sql_planner_prompt_fragment,
    catalog_prompt_hash,
    filter_tool_descriptions,
    project_tool_descriptions,
    render_catalog_block,
    render_catalog_segments,
    render_saved_query_tool_description,
    render_tool_description,
    to_openai_function,
)
from semql_prompt.prompt_budget import (
    BudgetResult,
    PromptBudget,
    apply_budget,
    estimate_tokens,
)

__all__ = [
    # prompt fragments + rendering
    "CatalogPrompt",
    "ToolDescriptionProjection",
    "build_drilldown_prompt_fragment",
    "build_planner_prompt_fragment",
    "build_planner_prompt_segments",
    "build_presenter_prompt_fragment",
    "build_query_generator_prompt_fragment",
    "build_router_prompt_fragment",
    "build_sql_planner_prompt_fragment",
    "catalog_prompt_hash",
    "filter_tool_descriptions",
    "project_tool_descriptions",
    "render_catalog_block",
    "render_catalog_segments",
    "render_saved_query_tool_description",
    "render_tool_description",
    "to_openai_function",
    # token budgeting
    "BudgetResult",
    "PromptBudget",
    "apply_budget",
    "estimate_tokens",
    # catalog-level conveniences
    "planner_prompt",
    "planner_prompt_segments",
    "prompt_hash",
    "sql_planner_prompt",
    "to_langchain_tools",
    "to_openai_tools",
    # Bedrock Converse adaptation
    "to_bedrock_converse_tools",
]
