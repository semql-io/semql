"""H6 — Ephemeral prompt segment on CatalogPrompt.

CatalogPrompt gains two new methods:
- ephemeral(current_date, retrieved_snippets, extra) -> str
  Per-request content; never cached. Returns "" when all args are None.
- full(**kwargs) -> str
  Convenience: static + overlay + ephemeral(...)

planner_prompt(Catalog, ) also grows the same three kwargs, delegating to
CatalogPrompt.full() internally.
"""

from __future__ import annotations

from semql_prompt import CatalogPrompt, planner_prompt


def _prompt(static: str = "STATIC", overlay: str = "") -> CatalogPrompt:
    return CatalogPrompt(static=static, overlay=overlay)


# ---------------------------------------------------------------------------
# CatalogPrompt.ephemeral()
# ---------------------------------------------------------------------------


def test_ephemeral_no_args_returns_empty_string() -> None:
    assert _prompt().ephemeral() == ""


def test_ephemeral_current_date_renders_context_block() -> None:
    out = _prompt().ephemeral(current_date="2026-06-08")
    assert "2026-06-08" in out
    assert "Current context" in out


def test_ephemeral_retrieved_snippets_renders_retrieved_block() -> None:
    out = _prompt().ephemeral(retrieved_snippets=["note 1", "note 2"])
    assert "Retrieved context" in out
    assert "note 1" in out
    assert "note 2" in out


def test_ephemeral_extra_appended_verbatim() -> None:
    out = _prompt().ephemeral(extra="My custom block\nSecond line")
    assert "My custom block" in out
    assert "Second line" in out


def test_ephemeral_only_non_empty_sections_appear() -> None:
    out = _prompt().ephemeral(current_date="2026-06-08")
    assert "Retrieved context" not in out


def test_ephemeral_all_three_args() -> None:
    out = _prompt().ephemeral(
        current_date="2026-06-08",
        retrieved_snippets=["note A"],
        extra="Extra stuff",
    )
    assert "2026-06-08" in out
    assert "note A" in out
    assert "Extra stuff" in out


# ---------------------------------------------------------------------------
# CatalogPrompt.full()
# ---------------------------------------------------------------------------


def test_full_no_args_equals_static_plus_overlay() -> None:
    p = _prompt(static="S", overlay="O")
    assert p.full() == p.joined()


def test_full_with_current_date_appends_ephemeral() -> None:
    p = _prompt(static="S", overlay="")
    out = p.full(current_date="2026-06-08")
    assert "S" in out
    assert "2026-06-08" in out


def test_full_equals_joined_plus_ephemeral() -> None:
    p = _prompt(static="S", overlay="O")
    expected = p.joined() + p.ephemeral(current_date="2026-06-08")
    assert p.full(current_date="2026-06-08") == expected


# ---------------------------------------------------------------------------
# planner_prompt(Catalog, ) delegation
# ---------------------------------------------------------------------------


def test_catalog_prompt_accepts_current_date_kwarg() -> None:
    from semql import Catalog, Cube, Dialect, Measure

    cat = Catalog(
        [
            Cube(
                name="orders",
                backend=Dialect.POSTGRES,
                table="public.orders",
                alias="o",
                measures=[Measure(name="cnt", sql="*", agg="count")],
            )
        ]
    )
    out = planner_prompt(cat, current_date="2026-06-08")
    assert "2026-06-08" in out


def test_catalog_prompt_no_date_unchanged() -> None:
    from semql import Catalog, Cube, Dialect, Measure

    cat = Catalog(
        [
            Cube(
                name="orders",
                backend=Dialect.POSTGRES,
                table="public.orders",
                alias="o",
                measures=[Measure(name="cnt", sql="*", agg="count")],
            )
        ]
    )
    without = planner_prompt(
        cat,
    )
    with_date = planner_prompt(cat, current_date="2026-06-08")
    # Base content is preserved; extra ephemeral block is appended.
    assert without in with_date or "orders" in with_date
    assert "2026-06-08" not in without
