"""Tests for prompt fit status, counting, and safe catalog trimming."""

from __future__ import annotations

import time

import pytest
from semql import (
    AuthContext,
    Cube,
    Dialect,
    Dimension,
    Measure,
)
from semql_prompt import (
    CatalogPrompt,
    PromptBudget,
    apply_budget,
    estimate_tokens,
    render_catalog_block,
    render_catalog_segments,
)


def _orders(description: str = "Orders table — the main fact table") -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        description=description,
        measures=[
            Measure(
                name="revenue",
                sql="{o}.amount",
                agg="sum",
                unit="currency",
                description="Sum of all order amounts in the period",
            ),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="status",
                sql="{o}.status",
                type="string",
                description="The current status of the order",
            ),
        ],
    )


def _customers(description: str = "Customer dimension") -> Cube:
    return Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        description=description,
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


def _catalog(*cubes: Cube) -> dict[str, Cube]:
    return {c.name: c for c in cubes}


def test_render_catalog_block_under_budget_passes_through() -> None:
    """A small catalog under the budget renders unchanged."""
    prompt = render_catalog_block(_catalog(_orders(), _customers()))
    budget = PromptBudget(max_tokens=10_000)  # huge budget
    result = budget.apply(prompt)
    assert result.text == prompt
    assert not result.was_truncated
    assert not result.dropped


def test_render_catalog_block_over_budget_truncates() -> None:
    """A small budget truncates and reports what was dropped."""
    prompt = render_catalog_block(_catalog(_orders(), _customers()))
    budget = PromptBudget(max_tokens=10)  # absurdly small
    result = budget.apply(prompt)
    assert len(result.text) < len(prompt)
    assert result.was_truncated
    assert len(result.dropped) > 0


def test_prompt_budget_tracks_drops_structurally() -> None:
    """The truncation report names what was dropped, in order."""
    prompt = render_catalog_block(
        _catalog(_orders(), _customers()),
        relations="See also: cross-cube joins.",
    )
    result = PromptBudget(max_tokens=5).apply(prompt)
    assert result.was_truncated
    # At least one thing was dropped.
    assert len(result.dropped) >= 1
    # Drop names are stable strings we can assert against.
    for name in result.dropped:
        assert isinstance(name, str)


def test_prompt_budget_labels_count_and_explicit_fit() -> None:
    result = PromptBudget(max_tokens=10_000).apply("Hello world")
    assert result.token_count == estimate_tokens("Hello world")
    assert result.count_source == "heuristic"
    assert result.fits
    assert isinstance(result.token_count, int)


def test_prompt_budget_preserves_untrimmable_required_text_when_it_cannot_fit() -> None:
    text = "x" * 100
    result = PromptBudget(max_tokens=1).apply(text)
    assert result.text == text
    assert result.token_count == 25
    assert not result.fits
    assert not result.was_truncated


def test_prompt_budget_custom_counter_controls_fit_and_is_labeled() -> None:
    def count(text: str) -> int:
        return 1 if text else 0

    result = PromptBudget(max_tokens=1).apply("required instructions", count_tokens=count)
    assert result.text == "required instructions"
    assert result.token_count == 1
    assert result.count_source == "callback"
    assert result.fits


def test_prompt_budget_custom_counter_used_after_trimming() -> None:
    prompt = "required instructions\n\n## DOMAIN CONTEXT\n\n**Glossary:**\noptional vocabulary"

    def count(text: str) -> int:
        return 10 if "Glossary" in text else 2

    result = PromptBudget(max_tokens=2).apply(prompt, count_tokens=count)
    assert "required instructions" in result.text
    assert "**Glossary:**" not in result.text
    assert result.token_count == 2
    assert result.count_source == "callback"
    assert result.fits
    assert result.was_truncated


def test_prompt_budget_exact_boundary_fits() -> None:
    prompt = "four"
    result = PromptBudget(max_tokens=1).apply(prompt)
    assert result.token_count == 1
    assert result.fits


def test_prompt_budget_zero_budget_does_not_clear_required_text() -> None:
    required = "Required instructions"
    result = PromptBudget(max_tokens=0).apply(required)
    assert result.text == required
    assert not result.fits


def test_prompt_budget_negative_max_tokens_rejected() -> None:
    """A negative budget is a config error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PromptBudget(max_tokens=-1)


def test_render_catalog_prompt_applies_budget_to_static_and_overlay() -> None:
    """The two-segment ``render_catalog_segments`` is also subject to
    the budget when applied."""
    cubes = _catalog(_orders(), _customers())
    prompt: CatalogPrompt = render_catalog_segments(cubes)
    result = PromptBudget(max_tokens=5).apply(prompt.full())
    assert result.was_truncated


def test_prompt_budget_does_not_damage_instructions_to_force_fit() -> None:
    prompt = (
        "Catalog:\n\n"
        "### orders (postgres)\n  description: Optional cube description\n\n"
        "### Instructions\nRequired: never reveal private values.\n\n"
        "### sales\nA required view description."
    )
    result = PromptBudget(max_tokens=1).apply(prompt)
    assert "### Instructions\nRequired: never reveal private values." in result.text
    assert "### sales\nA required view description." in result.text
    assert not result.fits


def test_prompt_budget_preserves_required_prompt_contract_when_catalog_is_pruned() -> None:
    from semql_prompt import build_query_generator_prompt_fragment

    prompt = build_query_generator_prompt_fragment(_catalog(_orders(), _customers()))
    result = PromptBudget(max_tokens=10).apply(prompt)
    assert result.was_truncated
    assert "## Semantic path" in result.text
    assert "## Trust boundary" in result.text
    assert not result.fits


def test_prompt_budget_preserves_protected_cube_and_reports_unfit() -> None:
    prompt = render_catalog_block(_catalog(_orders(), _customers()))
    result = PromptBudget(max_tokens=5).apply(prompt, protected_cubes=frozenset({"orders"}))
    assert "### orders" in result.text
    assert not result.fits


def test_apply_budget_forwards_protected_cubes() -> None:
    prompt = render_catalog_block(_catalog(_orders(), _customers()))
    result = apply_budget(prompt, 5, protected_cubes=frozenset({"orders"}))
    assert "### orders" in result.text


def test_prompt_budget_prunes_rendered_domain_context_before_cubes() -> None:
    from semql import GlossaryEntry

    prompt = render_catalog_block(
        _catalog(_orders(), _customers()),
        glossary=[GlossaryEntry(term="GMV", definition="Gross merchandise value")],
        relations="Orders belong to customers.",
    )
    result = PromptBudget(max_tokens=5).apply(prompt)
    assert result.dropped[:2] == ("relations", "glossary")


def test_prompt_budget_does_not_prune_arbitrary_markdown_headings() -> None:
    prompt = "## Required instructions\n\n### orders\nNever remove this application policy."
    result = PromptBudget(max_tokens=1).apply(prompt)
    assert result.text == prompt
    assert result.dropped == ()
    assert not result.fits


def test_policy_and_viewer_compose_with_budget() -> None:
    """The budget operates on the rendered text, so it composes with
    the viewer/policy filters that ran first. Sanity check."""
    viewer = AuthContext(viewer_id="u1", roles=["public"])
    catalog = _catalog(_orders(), _customers())
    full = render_catalog_block(catalog, viewer=viewer)
    gated = render_catalog_block(
        catalog,
        viewer=viewer,
        policy=lambda cube, _v: cube.name in {"orders"},
    )
    # The gated prompt is shorter (one cube dropped) — verify the
    # budget trims it independently.
    result = PromptBudget(max_tokens=5).apply(gated)
    assert result.was_truncated
    # Sanity: the gated full prompt is no longer than the
    # unfiltered one.
    assert len(gated) <= len(full)


# ---------------------------------------------------------------------------
# Scale tripwire (W5/§7): a 1000-cube catalog must render and trim to fit a
# tight budget in bounded time. Structural guard — the loose wall-clock
# bound only trips on a quadratic/exponential regression, not on ordinary
# machine variance.
# ---------------------------------------------------------------------------


def test_prompt_budget_trims_thousand_cube_catalog_to_fit() -> None:
    catalog = {
        f"cube{i}": Cube(
            name=f"cube{i}",
            dialect=Dialect.POSTGRES,
            table=f"t{i}",
            alias=f"a{i}",
            primary_key="id",
            description=f"Cube number {i} — a synthetic fact table for scale testing",
            measures=[Measure(name="count", sql="*", agg="count", description="row count")],
            dimensions=[
                Dimension(name="id", sql=f"{{a{i}}}.id", type="number"),
                Dimension(name="key", sql=f"{{a{i}}}.key", type="string", description="a key"),
            ],
        )
        for i in range(1000)
    }

    start = time.perf_counter()
    prompt = render_catalog_block(catalog)
    result = PromptBudget(max_tokens=2_000).apply(prompt)
    elapsed = time.perf_counter() - start

    # The un-budgeted prompt genuinely overflows (proves the trim did work).
    assert estimate_tokens(prompt) > 2_000
    # ...and the trimmed prompt fits, having dropped whole cubes to get there.
    assert result.token_count <= 2_000
    assert result.fits
    assert result.was_truncated
    assert any(d.startswith("cube:") for d in result.dropped)
    assert elapsed < 5.0
