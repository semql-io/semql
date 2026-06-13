"""``python -m semql`` CLI.

Compiles a ``SemanticQuery`` JSON spec against a Catalog declared
in a Python module and prints the SQL + params to stdout. Useful
for ad-hoc cube authoring (no need to write a runner script) and
as a smoke target in CI.

    python -m semql --catalog mypkg.catalogs:default \\
        '{"measures": ["orders.revenue"], "dimensions": ["orders.region"]}'

The ``--catalog`` arg is ``module.path:attr`` — the module is
imported, the named attribute must be a ``Catalog`` instance.
Context substitutions for ``{schema}`` / ``{tenant}`` placeholders
go through repeated ``--context key=value`` flags.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from semql import Catalog, SemanticQuery


def _load_catalog(spec: str) -> Catalog:
    """Resolve ``module.path:attr`` into the named ``Catalog``."""
    if ":" not in spec:
        raise SystemExit(
            f"--catalog must be 'module.path:attr', got {spec!r}. "
            "Example: --catalog mypkg.catalogs:default"
        )
    module_path, attr = spec.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Could not import {module_path!r}: {exc}") from exc
    try:
        catalog = getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(f"Module {module_path!r} has no attribute {attr!r}.") from exc
    if not isinstance(catalog, Catalog):
        raise SystemExit(f"{spec!r} resolved to {type(catalog).__name__}, not semql.Catalog.")
    return catalog


def _parse_context(pairs: list[str]) -> dict[str, str]:
    """Parse repeated ``--context key=value`` flags into a dict."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--context expects key=value, got {pair!r}.")
        k, v = pair.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m semql",
        description="Compile a SemanticQuery JSON spec to SQL.",
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Catalog locator: module.path:attr (e.g. mypkg.catalogs:default).",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Compile-time context pair (key=value). May be repeated.",
    )
    parser.add_argument(
        "--params-format",
        choices=("comment", "json"),
        default="comment",
        help="How to print the params: as a SQL comment (default) or a JSON line.",
    )
    parser.add_argument(
        "spec",
        help="SemanticQuery as a JSON string. Use '-' to read from stdin.",
    )
    args = parser.parse_args(argv)

    raw_spec = sys.stdin.read() if args.spec == "-" else args.spec
    try:
        spec_dict: dict[str, Any] = json.loads(raw_spec)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--spec must be JSON: {exc}") from exc
    query = SemanticQuery.model_validate(spec_dict)

    catalog = _load_catalog(args.catalog)
    ctx = _parse_context(args.context)

    compiled = catalog.compile(query, context=ctx)

    print(compiled.sql)
    if args.params_format == "json":
        print(json.dumps(compiled.params, default=str))
    else:
        print(f"-- params: {json.dumps(compiled.params, default=str)}")
        print(f"-- columns: {compiled.columns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
