#!/usr/bin/env -S uv run
"""Report breaking API changes between two git refs.

Loads each public module at ``--base`` (default: ``HEAD~5``) and at
``--head`` (default: working tree), runs ``griffe.find_breaking_changes``
on each pair, and prints the explanations.

Run from the repo root:

    uv run scripts/check_api_break.py --base 3aeb436 --head HEAD

Exits non-zero if any breakage is reported, so a CI wrapper can
gate merges on a clean diff. We don't wire that gate up yet — running
manually before publishing a release is enough for now.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import griffe

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES: tuple[tuple[str, str], ...] = (
    ("semql", "semql"),
    ("semql_mcp", "semql-mcp"),
    ("semql_erd", "semql-erd"),
)


def _search_paths_for_ref(ref: str | None) -> list[str]:
    """Per-package ``src/`` dirs for the given ref. ``None`` = working tree.

    For a non-None ref, ``griffe.load_git`` mounts the worktree itself,
    so we hand it relative repo-root-relative paths (it cd's into the
    worktree before resolving)."""
    if ref is None:
        return [str(REPO_ROOT / "packages" / pkg_dir / "src") for _, pkg_dir in PACKAGES]
    return [f"packages/{pkg_dir}/src" for _, pkg_dir in PACKAGES]


def _load(module: str, ref: str | None) -> griffe.Object | griffe.Alias:
    """Load ``module`` at ``ref`` (or working tree if ``ref`` is None)."""
    paths = _search_paths_for_ref(ref)
    if ref is None:
        return griffe.load(module, search_paths=paths)
    return griffe.load_git(module, ref=ref, search_paths=paths)


def check(base: str, head: str | None) -> int:
    """Return the count of breaking changes across all packages."""
    total = 0
    for module_name, _ in PACKAGES:
        try:
            old = _load(module_name, base)
        except Exception as exc:  # noqa: BLE001 — package may not exist at the base ref.
            print(f"# {module_name}: skipped — could not load at {base!r}: {exc}", file=sys.stderr)
            continue
        new = _load(module_name, head)
        breakages = list(griffe.find_breaking_changes(old, new))
        if not breakages:
            print(f"# {module_name}: no breaking changes vs {base}")
            continue
        total += len(breakages)
        print(f"# {module_name}: {len(breakages)} breaking change(s) vs {base}")
        for b in breakages:
            print(b.explain())
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="HEAD~5",
        help="The base git ref to compare against (default: HEAD~5).",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="The newer git ref. Default: working tree.",
    )
    args = parser.parse_args(argv)
    n = check(args.base, args.head)
    if n == 0:
        print("\nNo breaking changes.")
        return 0
    print(f"\n{n} breaking change(s) found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
