"""S7 slice 1 — LLM-grounding metadata on Cube / SavedQuery / Catalog.

Covers the new ``questions`` / ``keywords`` / ``relations`` /
``stability`` / ``replacement`` fields, ``GlossaryEntry``, and the
Catalog kwargs ``glossary`` / ``relations``. Validation rules: empty-
string refusal, length caps, acronym-preserving keyword dedupe,
replacement-points-at-real-cube. The lifecycle enforcement (deprecated
→ CompileError) lives in slice 2 — not tested here.
"""

from __future__ import annotations

from typing import Any

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    GlossaryEntry,
    Measure,
    SavedQuery,
    SemanticQuery,
)
from semql_prompt import planner_prompt, prompt_hash


def _cube(name: str = "orders", **kwargs: Any) -> Cube:  # noqa: ANN401 — test factory
    defaults: dict[str, Any] = {
        "backend": Dialect.POSTGRES,
        "table": f"public.{name}",
        "alias": name[0],
        "measures": [Measure(name="cnt", sql="*", agg="count")],
        "dimensions": [Dimension(name="status", sql="{o}.status", type="string")],
    }
    defaults.update(kwargs)
    return Cube(name=name, **defaults)


# ---------------------------------------------------------------------------
# Cube.questions
# ---------------------------------------------------------------------------


def test_cube_questions_default_empty() -> None:
    assert _cube().questions == []


def test_cube_questions_stored_verbatim() -> None:
    qs = ["How many orders shipped last week?", "Top 10 customers by LTV"]
    c = _cube(questions=qs)
    assert c.questions == qs


def test_cube_questions_refuses_empty_string() -> None:
    with pytest.raises(ValueError, match="questions"):
        _cube(questions=["valid", ""])


def test_cube_questions_refuses_whitespace_only() -> None:
    with pytest.raises(ValueError, match="questions"):
        _cube(questions=["   "])


def test_cube_questions_caps_at_200_chars() -> None:
    long_q = "x" * 201
    with pytest.raises(ValueError, match="200 chars"):
        _cube(questions=[long_q])


def test_cube_questions_accepts_200_chars_exactly() -> None:
    q = "x" * 200
    assert _cube(questions=[q]).questions == [q]


# ---------------------------------------------------------------------------
# Cube.keywords — acronym-preserving normalisation
# ---------------------------------------------------------------------------


def test_cube_keywords_default_empty() -> None:
    assert _cube().keywords == []


def test_cube_keywords_preserves_all_caps_acronyms() -> None:
    c = _cube(keywords=["AOV", "orders", "Transactions"])
    assert c.keywords == ["AOV", "orders", "transactions"]


def test_cube_keywords_dedupes_case_insensitively_first_form_wins() -> None:
    c = _cube(keywords=["AOV", "aov", "Aov", "orders", "Orders"])
    assert c.keywords == ["AOV", "orders"]


def test_cube_keywords_lowercase_dedupes_after_normalize() -> None:
    c = _cube(keywords=["Orders", "orders"])
    assert c.keywords == ["orders"]


def test_cube_keywords_refuses_empty_string() -> None:
    with pytest.raises(ValueError, match="keywords"):
        _cube(keywords=["valid", ""])


def test_cube_keywords_caps_at_50_chars() -> None:
    with pytest.raises(ValueError, match="50 chars"):
        _cube(keywords=["x" * 51])


def test_cube_keywords_preserves_order() -> None:
    c = _cube(keywords=["zeta", "alpha", "beta"])
    assert c.keywords == ["zeta", "alpha", "beta"]


# ---------------------------------------------------------------------------
# Cube.relations
# ---------------------------------------------------------------------------


def test_cube_relations_default_empty() -> None:
    assert _cube().relations == ""


def test_cube_relations_caps_at_2000_chars() -> None:
    with pytest.raises(ValueError, match="2000 chars"):
        _cube(relations="x" * 2001)


def test_cube_relations_accepts_long_narrative() -> None:
    text = "Orders only count once payment_status='paid'. " * 20
    assert _cube(relations=text).relations == text


# ---------------------------------------------------------------------------
# Cube.stability / Cube.replacement
# ---------------------------------------------------------------------------


def test_cube_stability_defaults_stable() -> None:
    assert _cube().stability == "stable"


def test_cube_stability_accepts_beta() -> None:
    assert _cube(stability="beta").stability == "beta"


def test_cube_stability_accepts_deprecated() -> None:
    assert _cube(stability="deprecated").stability == "deprecated"


def test_cube_stability_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        _cube(stability="experimental")


def test_cube_replacement_default_none() -> None:
    assert _cube().replacement is None


def test_cube_replacement_set_only_when_deprecated() -> None:
    """Setting ``replacement`` on a stable cube is a configuration error."""
    with pytest.raises(ValueError, match="replacement"):
        _cube(replacement="orders_v2")
    with pytest.raises(ValueError, match="replacement"):
        _cube(stability="beta", replacement="orders_v2")


def test_cube_replacement_allowed_when_deprecated() -> None:
    c = _cube(stability="deprecated", replacement="orders_v2")
    assert c.replacement == "orders_v2"


def test_cube_deprecated_no_replacement_ok() -> None:
    """Replacement is *optional*; a cube can be going away entirely."""
    c = _cube(stability="deprecated")
    assert c.stability == "deprecated"
    assert c.replacement is None


# ---------------------------------------------------------------------------
# GlossaryEntry
# ---------------------------------------------------------------------------


def test_glossary_entry_minimal() -> None:
    g = GlossaryEntry(term="ARR", definition="Annual recurring revenue.")
    assert g.term == "ARR"
    assert g.aliases == []


def test_glossary_entry_with_aliases() -> None:
    g = GlossaryEntry(
        term="ARR",
        definition="Annual recurring revenue.",
        aliases=["annual recurring revenue", "yearly subscription revenue"],
    )
    assert g.aliases == ["annual recurring revenue", "yearly subscription revenue"]


def test_glossary_entry_refuses_empty_term() -> None:
    with pytest.raises(ValueError, match="term"):
        GlossaryEntry(term="", definition="x")


def test_glossary_entry_refuses_empty_definition() -> None:
    with pytest.raises(ValueError, match="definition"):
        GlossaryEntry(term="ARR", definition="")


def test_glossary_entry_refuses_empty_alias() -> None:
    with pytest.raises(ValueError, match="aliases"):
        GlossaryEntry(term="ARR", definition="x", aliases=["valid", ""])


# ---------------------------------------------------------------------------
# Catalog.glossary
# ---------------------------------------------------------------------------


def test_catalog_glossary_default_empty() -> None:
    cat = Catalog([_cube()])
    assert cat.glossary == []


def test_catalog_glossary_stored() -> None:
    g = GlossaryEntry(term="ARR", definition="x")
    cat = Catalog([_cube()], glossary=[g])
    assert cat.glossary == [g]


def test_catalog_glossary_refuses_duplicate_term() -> None:
    with pytest.raises(ValueError, match="collides"):
        Catalog(
            [_cube()],
            glossary=[
                GlossaryEntry(term="ARR", definition="one"),
                GlossaryEntry(term="arr", definition="two"),  # case-insensitive collision
            ],
        )


def test_catalog_glossary_refuses_alias_term_collision() -> None:
    with pytest.raises(ValueError, match="collides"):
        Catalog(
            [_cube()],
            glossary=[
                GlossaryEntry(term="ARR", definition="one", aliases=["MRR"]),
                GlossaryEntry(term="MRR", definition="two"),
            ],
        )


def test_catalog_glossary_refuses_alias_alias_collision() -> None:
    with pytest.raises(ValueError, match="collides"):
        Catalog(
            [_cube()],
            glossary=[
                GlossaryEntry(term="ARR", definition="one", aliases=["yearly"]),
                GlossaryEntry(term="LTV", definition="two", aliases=["yearly"]),
            ],
        )


# ---------------------------------------------------------------------------
# Catalog.relations
# ---------------------------------------------------------------------------


def test_catalog_relations_default_empty() -> None:
    cat = Catalog([_cube()])
    assert cat.relations == ""


def test_catalog_relations_stored() -> None:
    rel = "orders ↔ shipments via order_id; accounts ↔ subscriptions via account_id."
    cat = Catalog([_cube()], relations=rel)
    assert cat.relations == rel


def test_catalog_relations_caps_at_2000_chars() -> None:
    with pytest.raises(ValueError, match="2000 chars"):
        Catalog([_cube()], relations="x" * 2001)


# ---------------------------------------------------------------------------
# Catalog cross-cube validations: replacement must exist
# ---------------------------------------------------------------------------


def test_catalog_refuses_replacement_pointing_at_unknown_cube() -> None:
    c = _cube(stability="deprecated", replacement="does_not_exist")
    with pytest.raises(ValueError, match="not in the catalog"):
        Catalog([c])


def test_catalog_accepts_valid_replacement() -> None:
    old = _cube(name="orders_v1", stability="deprecated", replacement="orders_v2")
    new = _cube(name="orders_v2")
    cat = Catalog([old, new])
    assert "orders_v1" in cat
    assert "orders_v2" in cat


# ---------------------------------------------------------------------------
# SavedQuery — same grounding fields as Cube
# ---------------------------------------------------------------------------


def _saved(name: str = "weekly_report", **kwargs: Any) -> SavedQuery:  # noqa: ANN401
    defaults: dict[str, Any] = {
        "query": SemanticQuery(measures=["orders.cnt"]),
        "description": "Weekly KPI",
    }
    defaults.update(kwargs)
    return SavedQuery(name=name, **defaults)


def test_savedquery_questions_default_empty() -> None:
    assert _saved().questions == []


def test_savedquery_questions_stored() -> None:
    qs = ["What were our weekly orders?"]
    assert _saved(questions=qs).questions == qs


def test_savedquery_keywords_normalised() -> None:
    sq = _saved(keywords=["WEEKLY", "weekly", "Report"])
    assert sq.keywords == ["WEEKLY", "report"]


def test_savedquery_purpose_default_empty() -> None:
    assert _saved().purpose == ""


def test_savedquery_purpose_stored() -> None:
    assert _saved(purpose="operational ops dashboard").purpose == "operational ops dashboard"


def test_savedquery_stability_default_stable() -> None:
    assert _saved().stability == "stable"


def test_savedquery_replacement_only_when_deprecated() -> None:
    with pytest.raises(ValueError, match="replacement"):
        _saved(replacement="weekly_report_v2")


def test_savedquery_replacement_allowed_when_deprecated() -> None:
    sq = _saved(stability="deprecated", replacement="weekly_report_v2")
    assert sq.replacement == "weekly_report_v2"


def test_catalog_refuses_savedquery_replacement_unknown() -> None:
    sq = _saved(stability="deprecated", replacement="ghost_query")
    with pytest.raises(ValueError, match="not in the catalog"):
        Catalog([_cube()], saved_queries=[sq])


def test_catalog_accepts_savedquery_replacement_existing() -> None:
    sq_old = _saved(name="report_v1", stability="deprecated", replacement="report_v2")
    sq_new = _saved(name="report_v2")
    cat = Catalog([_cube()], saved_queries=[sq_old, sq_new])
    assert "report_v1" in cat.saved_queries
    assert "report_v2" in cat.saved_queries


# ---------------------------------------------------------------------------
# Lifecycle enforcement — deprecated cube → CompileError
# ---------------------------------------------------------------------------


def test_deprecated_cube_refused_at_compile() -> None:
    """A query that touches a deprecated cube must fail compile with a
    clear error pointing at the replacement."""
    from semql import CompileError

    old = _cube(name="orders_v1", stability="deprecated", replacement="orders_v2")
    new = _cube(name="orders_v2")
    cat = Catalog([old, new])
    q = SemanticQuery(measures=["orders_v1.cnt"])
    with pytest.raises(CompileError, match="deprecated"):
        cat.compile(q)


def test_deprecated_error_names_replacement() -> None:
    from semql import CompileError

    old = _cube(name="orders_v1", stability="deprecated", replacement="orders_v2")
    new = _cube(name="orders_v2")
    cat = Catalog([old, new])
    q = SemanticQuery(measures=["orders_v1.cnt"])
    with pytest.raises(CompileError, match="orders_v2"):
        cat.compile(q)


def test_deprecated_no_replacement_error_message() -> None:
    from semql import CompileError

    old = _cube(name="orders_v1", stability="deprecated")
    cat = Catalog([old])
    q = SemanticQuery(measures=["orders_v1.cnt"])
    with pytest.raises(CompileError, match="no replacement"):
        cat.compile(q)


def test_beta_cube_compiles_fine() -> None:
    """``beta`` is informational — the compiler doesn't refuse."""
    c = _cube(name="orders_beta", stability="beta")
    cat = Catalog([c])
    q = SemanticQuery(measures=["orders_beta.cnt"])
    cat.compile(q)  # no raise


def test_stable_cube_compiles_fine() -> None:
    cat = Catalog([_cube()])
    q = SemanticQuery(measures=["orders.cnt"])
    cat.compile(q)


def test_validate_emits_deprecated_error() -> None:
    """``validate`` mirrors ``compile`` — surfaces deprecated cubes
    as an error code so the planner-feedback loop sees the same
    refusal the compiler does."""
    from semql import validate

    old = _cube(name="orders_v1", stability="deprecated", replacement="orders_v2")
    new = _cube(name="orders_v2")
    cat = Catalog([old, new])
    q = SemanticQuery(measures=["orders_v1.cnt"])
    errs = validate(q, cat)
    codes = [e.code for e in errs]
    assert "cube_deprecated" in codes


def test_validate_emits_beta_warning() -> None:
    from semql import validate

    c = _cube(name="orders_beta", stability="beta")
    cat = Catalog([c])
    q = SemanticQuery(measures=["orders_beta.cnt"])
    errs = validate(q, cat)
    codes = [e.code for e in errs]
    assert "cube_beta" in codes


def test_validate_no_lifecycle_noise_on_stable() -> None:
    from semql import validate

    cat = Catalog([_cube()])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    codes = [e.code for e in errs]
    assert "cube_beta" not in codes
    assert "cube_deprecated" not in codes


# ---------------------------------------------------------------------------
# Prompt rendering — Domain Context, per-cube grounding, deprecated filter
# ---------------------------------------------------------------------------


def test_prompt_omits_deprecated_cubes() -> None:
    """Deprecated cubes are hidden from the planner prompt entirely."""
    old = _cube(name="orders_v1", stability="deprecated", replacement="orders_v2")
    new = _cube(name="orders_v2")
    cat = Catalog([old, new])
    rendered = planner_prompt(
        cat,
    )
    assert "orders_v1" not in rendered
    assert "orders_v2" in rendered


def test_prompt_annotates_beta_cubes() -> None:
    c = _cube(name="orders_b", stability="beta")
    cat = Catalog([c])
    assert "[beta]" in planner_prompt(
        cat,
    )


def test_prompt_renders_per_cube_questions_and_keywords() -> None:
    c = _cube(
        name="orders",
        questions=["How many orders this week?", "Top customers by LTV"],
        keywords=["orders", "AOV", "LTV"],
    )
    cat = Catalog([c])
    p = planner_prompt(
        cat,
    )
    assert "Questions this cube answers" in p
    assert "How many orders this week?" in p
    assert "Top customers by LTV" in p
    assert "Keywords:" in p
    assert "AOV" in p and "LTV" in p


def test_prompt_renders_cube_relations() -> None:
    c = _cube(
        name="orders",
        relations="Orders only count once payment_status='paid'.",
    )
    cat = Catalog([c])
    p = planner_prompt(
        cat,
    )
    assert "Relations" in p
    assert "payment_status='paid'" in p


def test_prompt_renders_domain_context_glossary() -> None:
    g = GlossaryEntry(
        term="ARR",
        definition="Annual recurring revenue.",
        aliases=["annual recurring revenue"],
    )
    cat = Catalog([_cube()], glossary=[g])
    p = planner_prompt(
        cat,
    )
    assert "DOMAIN CONTEXT" in p
    assert "Glossary" in p
    assert "ARR" in p
    assert "Annual recurring revenue." in p
    assert "annual recurring revenue" in p  # alias surfaced


def test_prompt_renders_domain_context_relations() -> None:
    cat = Catalog(
        [_cube()],
        relations="Orders ↔ shipments via order_id.",
    )
    p = planner_prompt(
        cat,
    )
    assert "DOMAIN CONTEXT" in p
    assert "Orders ↔ shipments" in p


def test_prompt_no_domain_context_when_empty() -> None:
    cat = Catalog([_cube()])
    p = planner_prompt(
        cat,
    )
    assert "DOMAIN CONTEXT" not in p


def test_prompt_hash_includes_glossary() -> None:
    """Editing the glossary busts the prompt-cache hash."""
    base = Catalog([_cube()])
    with_glossary = Catalog(
        [_cube()],
        glossary=[GlossaryEntry(term="ARR", definition="x")],
    )
    assert prompt_hash(
        base,
    ) != prompt_hash(
        with_glossary,
    )


def test_prompt_hash_includes_catalog_relations() -> None:
    base = Catalog([_cube()])
    with_rel = Catalog([_cube()], relations="orders ↔ shipments")
    assert prompt_hash(
        base,
    ) != prompt_hash(
        with_rel,
    )


def test_tool_description_surfaces_questions() -> None:
    from semql_prompt import project_tool_descriptions

    c = _cube(name="orders", questions=["How many orders this week?"])
    cat = Catalog([c])
    proj = project_tool_descriptions(cat.as_dict())
    text = proj.invariant["orders"]
    assert "Example questions" in text
    assert "How many orders this week?" in text


def test_tool_description_truncates_relations() -> None:
    from semql_prompt import project_tool_descriptions

    long_rel = "Orders count as paid. " * 50
    c = _cube(name="orders", relations=long_rel)
    cat = Catalog([c])
    text = project_tool_descriptions(cat.as_dict()).invariant["orders"]
    assert "Notes:" in text
    notes_line = next(line for line in text.split("\n") if line.startswith("Notes:"))
    # 120-char cap + "Notes: " prefix + ellipsis
    assert len(notes_line) <= len("Notes: ") + 121


def test_tool_description_omits_deprecated_cubes() -> None:
    from semql_prompt import project_tool_descriptions

    cat = Catalog(
        [
            _cube(name="orders_v1", stability="deprecated", replacement="orders_v2"),
            _cube(name="orders_v2"),
        ]
    )
    proj = project_tool_descriptions(cat.as_dict())
    assert "orders_v1" not in proj.invariant
    assert "orders_v2" in proj.invariant


def test_tool_description_annotates_beta_cubes() -> None:
    from semql_prompt import project_tool_descriptions

    cat = Catalog([_cube(name="orders_b", stability="beta")])
    text = project_tool_descriptions(cat.as_dict()).invariant["orders_b"]
    assert "[BETA]" in text


def test_deprecated_error_lists_multiple_offenders() -> None:
    """When a query touches two deprecated cubes (via a join), the
    error message names both — saves a round-trip for the planner."""
    from semql import CompileError, Join

    # Two cubes deprecated. The second one is reachable from the first
    # via a join. The query joins them by referring to both.
    a = _cube(
        name="aaa",
        stability="deprecated",
        replacement="aaa_v2",
        joins=[Join(to="bbb", relationship="many_to_one", on="{a}.bid = {b}.id")],
    )
    a_repl = _cube(name="aaa_v2")
    b = _cube(name="bbb", alias="b", stability="deprecated")
    cat = Catalog([a, a_repl, b])
    q = SemanticQuery(measures=["aaa.cnt"], dimensions=["bbb.status"])
    with pytest.raises(CompileError) as exc:
        cat.compile(q)
    msg = str(exc.value)
    assert "aaa" in msg
    assert "bbb" in msg
