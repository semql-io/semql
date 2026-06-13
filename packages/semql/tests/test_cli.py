"""Tests for ``python -m semql`` — the compile-to-stdout CLI.

Drives ``semql.__main__.main`` directly with argv lists; captures
stdout via pytest's ``capsys`` fixture. No subprocess overhead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from semql.__main__ import main


@pytest.fixture(autouse=True)
def _make_test_catalog_importable(  # pyright: ignore[reportUnusedFunction] -- autouse fixture; pytest reads it via the decorator, pyright can't see that
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Drop a temp catalog module on sys.path so --catalog can find it."""
    catalog_module = tmp_path / "_semql_cli_test_catalog.py"
    catalog_module.write_text(
        """
from semql import Dialect, Catalog, Cube, Dimension, Measure

default = Catalog([
    Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
])
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType] -- pytest's MonkeyPatch.syspath_prepend lacks complete type info upstream
    # Ensure a re-import each test, so the catalog hangs off this tmp dir.
    sys.modules.pop("_semql_cli_test_catalog", None)
    return tmp_path


def test_cli_prints_sql_and_params(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "--catalog",
            "_semql_cli_test_catalog:default",
            '{"measures": ["orders.revenue"], "dimensions": ["orders.region"]}',
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "SELECT" in out
    assert "SUM(o.amount)" in out
    assert "-- params:" in out
    assert "-- columns:" in out


def test_cli_params_format_json_prints_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    spec = json.dumps(
        {
            "measures": ["orders.revenue"],
            "filters": [{"dimension": "orders.region", "op": "eq", "values": ["us"]}],
        }
    )
    rc = main(
        [
            "--catalog",
            "_semql_cli_test_catalog:default",
            "--params-format",
            "json",
            spec,
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.strip().split("\n")
    # Last line is the JSON params dict.
    params = json.loads(lines[-1])
    assert params == {"p0": "us"}


def test_cli_context_passes_through(capsys: pytest.CaptureFixture[str]) -> None:
    """``--context schema=prod`` should substitute ``{schema}`` in the
    catalog. We don't have a schema-templated cube in the fixture,
    so just verify the flag parses without error."""
    rc = main(
        [
            "--catalog",
            "_semql_cli_test_catalog:default",
            "--context",
            "schema=prod",
            "--context",
            "tenant=acme",
            '{"measures": ["orders.revenue"]}',
        ]
    )
    assert rc == 0


def test_cli_rejects_bad_catalog_locator() -> None:
    with pytest.raises(SystemExit, match=r"module.path:attr"):
        main(["--catalog", "no_colon_here", '{"measures": ["orders.revenue"]}'])


def test_cli_rejects_missing_module() -> None:
    with pytest.raises(SystemExit, match=r"Could not import"):
        main(
            [
                "--catalog",
                "non_existent_module:default",
                '{"measures": ["orders.revenue"]}',
            ]
        )


def test_cli_rejects_missing_attribute() -> None:
    with pytest.raises(SystemExit, match=r"has no attribute"):
        main(
            [
                "--catalog",
                "_semql_cli_test_catalog:not_there",
                '{"measures": ["orders.revenue"]}',
            ]
        )


def test_cli_rejects_non_catalog_attribute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the resolved attribute isn't a Catalog, fail clearly."""
    (tmp_path / "_semql_cli_test_catalog2.py").write_text("default = 'not a catalog'")
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType] -- pytest's MonkeyPatch.syspath_prepend lacks complete type info upstream
    sys.modules.pop("_semql_cli_test_catalog2", None)

    with pytest.raises(SystemExit, match=r"not semql.Catalog"):
        main(
            [
                "--catalog",
                "_semql_cli_test_catalog2:default",
                '{"measures": ["orders.revenue"]}',
            ]
        )


def test_cli_rejects_bad_json() -> None:
    with pytest.raises(SystemExit, match=r"--spec must be JSON"):
        main(
            [
                "--catalog",
                "_semql_cli_test_catalog:default",
                "not json",
            ]
        )


def test_cli_reads_stdin_when_spec_is_dash(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--spec -`` reads the JSON spec from stdin."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO('{"measures": ["orders.revenue"]}'))
    rc = main(["--catalog", "_semql_cli_test_catalog:default", "-"])
    assert rc == 0
    assert "SUM(o.amount)" in capsys.readouterr().out
