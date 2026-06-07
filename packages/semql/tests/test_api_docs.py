"""Smoke test for the griffe-driven API doc generator.

The generator at ``scripts/gen_api_docs.py`` walks each package's
public surface via griffe and emits one markdown file per package
under ``docs/api/``. This test only verifies that:

1. The script's ``render_package`` function returns a non-empty
   markdown string for each documented package.
2. The output contains the headings the planner / consumer will look
   for (one ``### `name``` line per public export).

The test deliberately doesn't pin exact body text — that would brick
the test every time anyone edits a docstring. The goal is to catch
*structural* regressions (an export silently disappearing, the
docstring renderer raising) without locking content.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_api_docs.py"


def _load_script_module() -> ModuleType:
    """Import the generator script as a module (not on sys.path)."""
    spec = importlib.util.spec_from_file_location("gen_api_docs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_api_docs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_api_docs() -> ModuleType:
    return _load_script_module()


def test_render_package_emits_markdown_for_semql(gen_api_docs: ModuleType) -> None:
    out = gen_api_docs.render_package("semql")
    assert out.startswith("# `semql` — API reference")
    # Headings for a sampling of core exports.
    for name in ("Catalog", "Cube", "SemanticQuery", "BoolExpr", "Measure"):
        assert f"### `{name}`" in out


def test_render_package_emits_markdown_for_semql_mcp(gen_api_docs: ModuleType) -> None:
    out = gen_api_docs.render_package("semql_mcp")
    assert out.startswith("# `semql_mcp` — API reference")
    assert "MCPServer" in out


def test_render_package_emits_markdown_for_semql_erd(gen_api_docs: ModuleType) -> None:
    out = gen_api_docs.render_package("semql_erd")
    assert out.startswith("# `semql_erd` — API reference")
    assert "render_dot" in out


def test_write_docs_creates_one_file_per_package(
    tmp_path: Path,
    gen_api_docs: ModuleType,
) -> None:
    written = gen_api_docs.write_docs(tmp_path)
    assert len(written) == 3
    names = {p.name for p in written}
    assert names == {"semql.md", "semql_mcp.md", "semql_erd.md"}
    for p in written:
        assert p.read_text().strip()  # non-empty
