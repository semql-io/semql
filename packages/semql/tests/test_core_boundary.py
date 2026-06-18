"""The core package must not import sibling packages (enforced by
scripts/check_core_boundary.py). These tests pin both directions: the guard
stays green on the real tree, and it actually catches a core -> sibling
import."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "check_core_boundary.py"

_spec = importlib.util.spec_from_file_location("check_core_boundary", _SCRIPT)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def test_core_imports_no_siblings() -> None:
    packages = guard.discover_packages(_REPO_ROOT / "packages")
    core = packages[guard.CORE_PACKAGE]
    siblings = {name for name in packages if name != guard.CORE_PACKAGE}
    assert siblings, "expected to discover sibling packages"
    assert guard.find_violations(core, siblings) == []


def test_main_passes_on_real_tree() -> None:
    assert guard.main() == 0


def test_detects_a_sibling_import(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("from semql.model import Cube\nimport os\n")
    (tmp_path / "bad.py").write_text("from semql_engine.adapter import Adapter\n")
    violations = guard.find_violations(tmp_path, {"semql_engine", "semql_mcp"})
    assert [(v[1], v[2]) for v in violations] == [(1, "semql_engine")]


def test_ignores_relative_and_aliased_safe_imports(tmp_path: Path) -> None:
    # Relative imports stay in-package; aliased core imports are fine.
    (tmp_path / "m.py").write_text(
        "from . import sibling\nimport semql.refs as r\nfrom semql import errors\n"
    )
    assert guard.find_violations(tmp_path, {"semql_engine"}) == []


@pytest.mark.parametrize(
    "src",
    [
        "import semql_mcp",
        "import semql_mcp.server as s",
        "from semql_mcp import server",
        "from semql_mcp.server import MCPServer",
    ],
)
def test_catches_every_import_form(tmp_path: Path, src: str) -> None:
    (tmp_path / "x.py").write_text(src + "\n")
    violations = guard.find_violations(tmp_path, {"semql_mcp"})
    assert len(violations) == 1 and violations[0][2] == "semql_mcp"
