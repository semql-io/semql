#!/usr/bin/env python3
"""Install release wheels together and exercise them outside the workspace.

The helper validates the lockstep distribution contract from wheel metadata,
installs all eight wheels into a temporary virtual environment, and runs a
compile/serialization/execution smoke from a temporary working directory.
"""

from __future__ import annotations

import argparse
import email.parser
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path

_DISTRIBUTIONS = frozenset(
    {
        "semql",
        "semql-auth",
        "semql-engine",
        "semql-erd",
        "semql-introspect",
        "semql-mcp",
        "semql-prompt",
        "semql-validate-db",
    }
)
_EXPECTED_SIBLING_REQUIREMENTS: dict[str, frozenset[str]] = {
    "semql": frozenset(),
    "semql-auth": frozenset({"semql"}),
    "semql-engine": frozenset({"semql"}),
    "semql-erd": frozenset({"semql"}),
    "semql-introspect": frozenset({"semql"}),
    "semql-mcp": frozenset({"semql", "semql-prompt"}),
    "semql-prompt": frozenset({"semql"}),
    "semql-validate-db": frozenset({"semql"}),
}
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_RELEASE_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?$")


@dataclass(frozen=True)
class _Wheel:
    path: Path
    name: str
    version: str
    requirements: tuple[str, ...]


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_metadata(path: Path) -> _Wheel:
    with zipfile.ZipFile(path) as archive:
        metadata_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_members) != 1:
            raise ValueError(
                f"{path.name}: expected one .dist-info/METADATA member, "
                f"found {len(metadata_members)}"
            )
        payload = archive.read(metadata_members[0]).decode("utf-8")
    metadata = email.parser.Parser().parsestr(payload)
    name = _canonical_name(metadata["Name"] or "")
    version = metadata["Version"] or ""
    if not name or not version:
        raise ValueError(f"{path.name}: wheel metadata is missing Name or Version")
    return _Wheel(
        path=path.resolve(),
        name=name,
        version=version,
        requirements=tuple(metadata.get_all("Requires-Dist", [])),
    )


def _discover_wheels(dist_dir: Path) -> tuple[list[_Wheel], str]:
    wheels = [_wheel_metadata(path) for path in sorted(dist_dir.glob("*.whl"))]
    by_name: dict[str, list[_Wheel]] = {}
    for wheel in wheels:
        by_name.setdefault(wheel.name, []).append(wheel)

    missing = sorted(_DISTRIBUTIONS - by_name.keys())
    unexpected = sorted(by_name.keys() - _DISTRIBUTIONS)
    duplicates = sorted(name for name, items in by_name.items() if len(items) != 1)
    if missing or unexpected or duplicates:
        raise ValueError(
            "release wheel set mismatch: "
            f"missing={missing}, unexpected={unexpected}, duplicate={duplicates}"
        )

    versions = {wheel.version for wheel in wheels}
    if len(versions) != 1:
        raise ValueError(f"release wheels do not share one version: {sorted(versions)}")
    version = versions.pop()
    version_match = _RELEASE_VERSION.fullmatch(version)
    if version_match is None:
        raise ValueError(f"unsupported release version shape: {version!r}")
    major, minor, _ = (int(part) for part in version_match.groups())
    upper_bound = f"{major}.{minor + 1}"

    for wheel in wheels:
        sibling_requirements: dict[str, str] = {}
        for requirement in wheel.requirements:
            name_match = _REQUIREMENT_NAME.match(requirement)
            if name_match is None:
                raise ValueError(f"{wheel.name}: cannot parse requirement {requirement!r}")
            requirement_name = _canonical_name(name_match.group(1))
            if requirement_name in _DISTRIBUTIONS:
                sibling_requirements[requirement_name] = requirement

        expected = _EXPECTED_SIBLING_REQUIREMENTS[wheel.name]
        actual = frozenset(sibling_requirements)
        if actual != expected:
            raise ValueError(
                f"{wheel.name}: sibling requirements mismatch; expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )
        for sibling, requirement in sibling_requirements.items():
            name_match = _REQUIREMENT_NAME.match(requirement)
            assert name_match is not None
            requirement_body = requirement[len(name_match.group(1)) :].replace(" ", "")
            if not requirement_body.startswith(f">={version},<{upper_bound}"):
                raise ValueError(
                    f"{wheel.name}: {sibling} must use the lockstep range "
                    f">={version},<{upper_bound}; found {requirement!r}"
                )

    return wheels, version


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


_SMOKE_PROGRAM = r"""
import importlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

EXPECTED = {
    "semql": "semql",
    "semql-auth": "semql_auth",
    "semql-engine": "semql_engine",
    "semql-erd": "semql_erd",
    "semql-introspect": "semql_introspect",
    "semql-mcp": "semql_mcp",
    "semql-prompt": "semql_prompt",
    "semql-validate-db": "semql_validate_db",
}
expected_version = sys.argv[1]
prefix = Path(sys.prefix).resolve()
for distribution, module_name in EXPECTED.items():
    if version(distribution) != expected_version:
        raise AssertionError(
            f"{distribution}: installed {version(distribution)!r}, expected {expected_version!r}"
        )
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(prefix):
        raise AssertionError(f"{module_name} imported outside isolated environment: {origin}")

import duckdb
from semql import (
    Catalog, CatalogContext, Cube, Dialect, Dimension, Measure, SemanticQuery, compare_analysis,
)
from semql.compile import CompiledQuery
from semql_engine import DuckDBAdapter

connection = duckdb.connect(":memory:")
try:
    connection.execute("CREATE TABLE orders(region VARCHAR, amount INTEGER)")
    connection.execute("INSERT INTO orders VALUES ('EMEA', 100), ('EMEA', 50), ('US', 30)")
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    catalog = Catalog([cube])
    context = CatalogContext(namespace="release-smoke", semantic_revision=expected_version)
    plain = catalog.compile(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        catalog_context=context,
    )
    aliased = catalog.compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            aliases={"area": "orders.region", "net": "orders.revenue"},
        ),
        catalog_context=context,
    )
    if compare_analysis(plain.analysis, aliased.analysis, scope="result").outcome != "equivalent":
        raise AssertionError("alias change altered result semantics")

    def round_trip(compiled):
        wire = json.loads(json.dumps(compiled.model_dump()))
        return CompiledQuery.model_validate(wire)

    adapter = DuckDBAdapter(connection)
    plain_result = adapter.execute(round_trip(plain).sql, round_trip(plain).params)
    alias_artifact = round_trip(aliased)
    alias_result = adapter.execute(alias_artifact.sql, alias_artifact.params)
    plain_rows = sorted((str(region), int(revenue)) for region, revenue in plain_result.rows)
    alias_rows = sorted((str(area), int(net)) for area, net in alias_result.rows)
    expected_rows = [("EMEA", 150), ("US", 30)]
    if plain_result.columns != ["region", "revenue"]:
        raise AssertionError(f"unexpected plain columns: {plain_result.columns}")
    if alias_result.columns != ["area", "net"]:
        raise AssertionError(f"unexpected alias columns: {alias_result.columns}")
    if plain_rows != expected_rows or alias_rows != expected_rows or plain_rows != alias_rows:
        raise AssertionError(
            f"alias-equivalence mismatch: plain={plain_rows}, alias={alias_rows}, "
            f"expected={expected_rows}"
        )
finally:
    connection.close()

print(f"installed-wheel smoke passed for all eight distributions at {expected_version}")
"""


def smoke(dist_dir: Path) -> None:
    wheels, version = _discover_wheels(dist_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install and validate release wheels")

    with tempfile.TemporaryDirectory(prefix="semql-release-smoke-") as temp_name:
        temp = Path(temp_name)
        environment = _clean_environment()
        # Standalone macOS interpreters resolve libpython relative to their executable.
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(temp / "venv")
        python = _venv_python(temp / "venv")
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), *(str(w.path) for w in wheels)],
            cwd=temp,
            env=environment,
            check=True,
        )
        subprocess.run(
            [uv, "pip", "check", "--python", str(python)],
            cwd=temp,
            env=environment,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", _SMOKE_PROGRAM, version],
            cwd=temp,
            env=environment,
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing the eight built wheels (default: dist)",
    )
    args = parser.parse_args(argv)
    if not args.dist_dir.is_dir():
        parser.error(f"distribution directory does not exist: {args.dist_dir}")
    try:
        smoke(args.dist_dir)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"release artifact smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
