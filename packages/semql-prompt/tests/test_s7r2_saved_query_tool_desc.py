"""S7-R2 — render_saved_query_tool_description + ToolDescriptionProjection saved queries.

New function: render_saved_query_tool_description(sq: SavedQuery) -> str
Format:
- Lead with sq.description or "Run the {name} saved query."
- [BETA] prefix when sq.stability == "beta"
- If sq.purpose non-empty: "Purpose: {purpose}."
- If sq.questions non-empty: slash-joined example questions line.
- Close with "Zero arguments — the query is pre-baked."

ToolDescriptionProjection gains:
- saved_query_invariant: dict[str, str]  — public saved queries (empty required_roles)
- saved_query_viewer_gated: dict[str, str] — role-gated saved queries the viewer sees
"""

from __future__ import annotations

from typing import Literal

from semql import SemanticQuery
from semql.spec import SavedQuery
from semql_prompt import project_tool_descriptions


def _sq(
    *,
    description: str = "",
    purpose: str = "",
    questions: list[str] | None = None,
    required_roles: list[str] | None = None,
    stability: Literal["stable", "beta", "deprecated"] = "stable",
) -> SavedQuery:
    return SavedQuery(
        name="weekly_revenue",
        query=SemanticQuery(measures=["orders.revenue"]),
        description=description,
        purpose=purpose,
        questions=questions or [],
        required_roles=required_roles or [],
        stability=stability,
    )


# ---------------------------------------------------------------------------
# render_saved_query_tool_description importable
# ---------------------------------------------------------------------------


def test_render_saved_query_tool_description_importable() -> None:
    from semql_prompt import render_saved_query_tool_description

    assert render_saved_query_tool_description is not None


def test_render_saved_query_tool_description_exported_from_semql_prompt() -> None:
    import semql_prompt

    assert hasattr(semql_prompt, "render_saved_query_tool_description")


# ---------------------------------------------------------------------------
# Format: basic
# ---------------------------------------------------------------------------


def test_default_description_when_empty() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(description="")
    result = render_saved_query_tool_description(sq)
    assert "weekly_revenue" in result
    assert "Run the" in result


def test_uses_sq_description_when_set() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(description="Revenue breakdown by week.")
    result = render_saved_query_tool_description(sq)
    assert "Revenue breakdown by week." in result


def test_zero_arg_footer_always_present() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq()
    result = render_saved_query_tool_description(sq)
    assert "Zero arguments" in result and "pre-baked" in result


def test_purpose_included_when_set() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(purpose="weekly ops report")
    result = render_saved_query_tool_description(sq)
    assert "weekly ops report" in result


def test_purpose_omitted_when_empty() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(purpose="")
    result = render_saved_query_tool_description(sq)
    assert "Purpose:" not in result


def test_questions_slash_joined_when_set() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(questions=["How much revenue last week?", "Weekly revenue trend?"])
    result = render_saved_query_tool_description(sq)
    assert "/" in result
    assert "How much revenue last week?" in result
    assert "Weekly revenue trend?" in result


def test_questions_omitted_when_empty() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(questions=[])
    result = render_saved_query_tool_description(sq)
    assert "Example questions" not in result


def test_beta_prefix_when_stability_beta() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(stability="beta")
    result = render_saved_query_tool_description(sq)
    assert result.startswith("[BETA]")


def test_no_beta_prefix_when_stable() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(stability="stable")
    result = render_saved_query_tool_description(sq)
    assert not result.startswith("[BETA]")


# ---------------------------------------------------------------------------
# ToolDescriptionProjection: saved_query_invariant and saved_query_viewer_gated
# ---------------------------------------------------------------------------


def test_tool_description_projection_has_saved_query_invariant() -> None:
    proj = project_tool_descriptions({})
    assert hasattr(proj, "saved_query_invariant")
    assert isinstance(proj.saved_query_invariant, dict)


def test_tool_description_projection_has_saved_query_viewer_gated() -> None:
    proj = project_tool_descriptions({})
    assert hasattr(proj, "saved_query_viewer_gated")
    assert isinstance(proj.saved_query_viewer_gated, dict)


def test_project_tool_descriptions_accepts_saved_queries_kwarg() -> None:
    sq = _sq()
    proj = project_tool_descriptions({}, saved_queries=[sq])
    assert "weekly_revenue" in proj.saved_query_invariant


def test_public_saved_query_goes_in_invariant() -> None:
    sq = _sq(required_roles=[])
    proj = project_tool_descriptions({}, saved_queries=[sq])
    assert "weekly_revenue" in proj.saved_query_invariant
    assert "weekly_revenue" not in proj.saved_query_viewer_gated


def test_saved_query_description_uses_render_saved_query_tool_description() -> None:
    from semql_prompt import render_saved_query_tool_description

    sq = _sq(description="My custom desc.")
    proj = project_tool_descriptions({}, saved_queries=[sq])
    assert proj.saved_query_invariant["weekly_revenue"] == render_saved_query_tool_description(sq)


def test_saved_queries_empty_when_none_provided() -> None:
    proj = project_tool_descriptions({})
    assert proj.saved_query_invariant == {}
    assert proj.saved_query_viewer_gated == {}


def test_all_includes_saved_queries() -> None:
    sq = _sq()
    proj = project_tool_descriptions({}, saved_queries=[sq])
    result = proj.all()
    assert "weekly_revenue" in result
