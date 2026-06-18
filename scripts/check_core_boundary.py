#!/usr/bin/env python3
"""Fail if the core ``semql`` package imports any sibling package.

tach enforces boundaries *within* core, but the sibling packages
(``semql-engine``, ``semql-mcp``, ...) are separate distributions that tach
treats as external, so it cannot see a core -> sibling edge. This guard
covers that edge: core stays standalone, and the siblings (executors, MCP,
prompt rendering, ...) import core, never the reverse.

Non-brittle by construction:

- Imports are read with :mod:`ast`, not regex, so aliased, multi-line, and
  conditionally-imported forms are all handled and string/comment mentions
  are ignored.
- The sibling package names are discovered by scanning ``packages/*/src/``,
  not hardcoded — add or rename a package and the guard tracks it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CORE_PACKAGE = "semql"


def discover_packages(packages_dir: Path) -> dict[str, Path]:
    """Map import-name -> package source dir for every ``packages/*/src/<pkg>``."""
    found: dict[str, Path] = {}
    for src in sorted(packages_dir.glob("*/src")):
        for child in sorted(src.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                found[child.name] = child
    return found


def imported_top_levels(tree: ast.Module) -> set[tuple[str, int]]:
    """``(top-level imported name, line number)`` for every absolute import.

    Relative imports (``from . import x``) stay inside the package and are
    skipped.
    """
    out: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.name.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add((node.module.split(".", 1)[0], node.lineno))
    return out


def find_violations(core_dir: Path, forbidden: set[str]) -> list[tuple[Path, int, str]]:
    """Every (file, line, imported-name) where core imports a forbidden package."""
    violations: list[tuple[Path, int, str]] = []
    for py in sorted(core_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue  # not our error to raise; the type checker / tests own it
        for name, lineno in sorted(imported_top_levels(tree)):
            if name in forbidden:
                violations.append((py, lineno, name))
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    packages_dir = repo_root / "packages"
    packages = discover_packages(packages_dir)

    core_dir = packages.get(CORE_PACKAGE)
    if core_dir is None:
        print(
            f"check_core_boundary: core package {CORE_PACKAGE!r} not found under {packages_dir}",
            file=sys.stderr,
        )
        return 2

    siblings = {name for name in packages if name != CORE_PACKAGE}
    violations = find_violations(core_dir, siblings)
    if violations:
        print(
            f"Core package {CORE_PACKAGE!r} must not import sibling packages, but found:",
            file=sys.stderr,
        )
        for py, lineno, name in violations:
            print(f"  {py.relative_to(repo_root)}:{lineno}: imports {name!r}", file=sys.stderr)
        print(
            "Invert the dependency so the sibling imports core, or move the shared code into core.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
