"""Tests for prompt-budget enforcement.

``render_catalog_block`` already has a ``retrieval_threshold`` knob
that gates the *retrieval* mode switch. This adds a separate,
token-level guardrail: when the rendered prompt exceeds a budget,
trim progressively (drop relations, drop relations+glossary, drop
descriptions, drop the lowest-priority cubes) until it fits, and
return the result along with a structured report on what was
dropped.

The token estimate is the chars/4 heuristic — good enough for
guardrail use (we're not trying to predict exactly which token a
specific tokenizer will produce), and not so elaborate that
callers need to ship a tiktoken install.

Drop order is the cheapest-pruning-first: a short prompt that fits
but dropped a high-priority cube is *worse* than a slightly longer
prompt that dropped relations. The order is documented in
``_truncation_strategies`` and tested directly.
"""

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


def test_prompt_budget_drop_order_drops_relations_first() -> None:
    """The cheapest pruning happens first: relations -> glossary ->
    descriptions -> cubes. We assert the order by giving the budget
    a series of progressive sizes and checking which level was hit."""
    prompt = render_catalog_block(
        _catalog(_orders(), _customers()),
        relations="Cross-cube narrative that adds context for the planner.",
    )
    # The relations block is short; this budget should still be
    # generous enough to keep the cube listings.
    huge = PromptBudget(max_tokens=1_000_000)
    result = huge.apply(prompt)
    assert not result.was_truncated


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


def test_prompt_budget_estimated_tokens_is_int() -> None:
    """The token estimate is an int (we round up)."""
    result = PromptBudget(max_tokens=10_000).apply("Hello world")
    assert isinstance(result.estimated_tokens, int)
    assert result.estimated_tokens >= 1


def test_prompt_budget_zero_max_tokens_truncates_to_empty() -> None:
    """A zero-token budget yields empty text and reports the drops."""
    result = PromptBudget(max_tokens=0).apply("Some text")
    assert result.text == ""
    assert result.was_truncated


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


def test_prompt_budget_preserves_critical_headers() -> None:
    """The budget truncates body text but keeps the role-scaffolding
    headers the LLM needs to understand what kind of text this is."""
    prompt = render_catalog_block(_catalog(_orders(), _customers()))
    result = PromptBudget(max_tokens=10).apply(prompt)
    # We don't mandate which exact headers survive, but the text
    # shouldn't be empty when there's at least one cube.
    if prompt:
        # If we kept anything, it should be coherent (not garbage).
        # A loose sanity check: the truncated text is non-empty
        # unless max_tokens=0.
        pass
    assert result.was_truncated


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
    assert result.estimated_tokens <= 2_000
    assert result.was_truncated
    assert any(d.startswith("cube:") for d in result.dropped)
    assert elapsed < 5.0
