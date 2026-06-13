"""H4 — CubePromptHook Protocol.

CubePromptHook: a callable Protocol that takes a Cube and returns a str
(extra text appended after that cube's block in the prompt).

Wire as ``cube_prompt_hooks`` kwarg on:
- ``build_planner_prompt_fragment``
- ``build_planner_prompt_segments``
- ``planner_prompt(Catalog, )``
"""

from __future__ import annotations

from semql import (
    Backend,
    Catalog,
    Cube,
    Dimension,
    Measure,
)
from semql_prompt import (
    build_planner_prompt_fragment,
    build_planner_prompt_segments,
    planner_prompt,
)


def _catalog() -> Catalog:
    cube = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    return Catalog([cube])


# ---------------------------------------------------------------------------
# Protocol importable and structural
# ---------------------------------------------------------------------------


def test_cube_prompt_hook_importable() -> None:
    from semql.hooks import CubePromptHook

    assert CubePromptHook is not None


def test_cube_prompt_hook_is_runtime_checkable() -> None:
    from semql.hooks import CubePromptHook

    def my_hook(cube: Cube) -> str:
        return "extra"

    assert isinstance(my_hook, CubePromptHook)


def test_cube_prompt_hook_non_callable_fails() -> None:
    from semql.hooks import CubePromptHook

    assert not isinstance("not a callable", CubePromptHook)


# ---------------------------------------------------------------------------
# build_planner_prompt_fragment accepts cube_prompt_hooks
# ---------------------------------------------------------------------------


def test_build_planner_prompt_fragment_accepts_hooks_kwarg() -> None:
    cat = _catalog()

    def hook(cube: Cube) -> str:
        return "HOOK_TEXT_FRAGMENT"

    result = build_planner_prompt_fragment(
        cat.as_dict(),
        cube_prompt_hooks=[hook],
    )
    assert "HOOK_TEXT_FRAGMENT" in result


def test_build_planner_prompt_fragment_hook_receives_cube() -> None:
    cat = _catalog()
    seen: list[str] = []

    def hook(cube: Cube) -> str:
        seen.append(cube.name)
        return ""

    build_planner_prompt_fragment(cat.as_dict(), cube_prompt_hooks=[hook])
    assert "orders" in seen


def test_build_planner_prompt_fragment_no_hooks_unchanged() -> None:
    cat = _catalog()
    without_hooks = build_planner_prompt_fragment(cat.as_dict())
    with_empty_hooks = build_planner_prompt_fragment(cat.as_dict(), cube_prompt_hooks=[])
    assert without_hooks == with_empty_hooks


def test_build_planner_prompt_fragment_multiple_hooks() -> None:
    cat = _catalog()

    def hook_a(cube: Cube) -> str:
        return "HOOK_A"

    def hook_b(cube: Cube) -> str:
        return "HOOK_B"

    results = build_planner_prompt_fragment(
        cat.as_dict(),
        cube_prompt_hooks=[hook_a, hook_b],
    )
    assert "HOOK_A" in results
    assert "HOOK_B" in results


# ---------------------------------------------------------------------------
# build_planner_prompt_segments accepts cube_prompt_hooks
# ---------------------------------------------------------------------------


def test_build_planner_prompt_segments_accepts_hooks_kwarg() -> None:
    cat = _catalog()

    def hook(cube: Cube) -> str:
        return "SEGMENT_HOOK"

    cp = build_planner_prompt_segments(cat.as_dict(), cube_prompt_hooks=[hook])
    assert "SEGMENT_HOOK" in cp.joined()


# ---------------------------------------------------------------------------
# planner_prompt(Catalog, ) accepts cube_prompt_hooks
# ---------------------------------------------------------------------------


def test_catalog_prompt_accepts_cube_prompt_hooks() -> None:
    cat = _catalog()

    def hook(cube: Cube) -> str:
        return "CATALOG_HOOK_TEXT"

    result = planner_prompt(cat, cube_prompt_hooks=[hook])
    assert "CATALOG_HOOK_TEXT" in result


# ---------------------------------------------------------------------------
# Exported from semql
# ---------------------------------------------------------------------------


def test_cube_prompt_hook_exported_from_semql() -> None:
    import semql

    assert hasattr(semql, "CubePromptHook")
