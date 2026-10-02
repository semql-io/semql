"""Execution-boundary tests for final SQL placeholder manifests."""

from __future__ import annotations

import pytest
from semql import Dialect
from semql.bindings import BindingRequirement, final_binding_requirements, validate_bindings
from semql.errors import ContractError


@pytest.mark.parametrize(
    ("dialect", "sql", "name", "driver_type"),
    [
        (Dialect.POSTGRES, "SELECT %(p0)s", "p0", None),
        (Dialect.BIGQUERY, "SELECT @p0", "p0", None),
        (Dialect.DUCKDB, "SELECT $p0", "p0", None),
        (Dialect.CLICKHOUSE, "SELECT {p0:String}", "p0", "String"),
    ],
)
def test_final_bindings_cover_named_dialect_syntax(
    dialect: Dialect, sql: str, name: str, driver_type: str | None
) -> None:
    assert final_binding_requirements(sql, dialect, {name: "private"}) == (
        BindingRequirement(name=name, driver_type=driver_type),
    )


def test_logical_and_driver_types_remain_distinct() -> None:
    requirement = BindingRequirement(name="p0", logical_type="time", driver_type="String")
    validate_bindings(
        "SELECT {p0:String}",
        Dialect.CLICKHOUSE,
        {"p0": "private"},
        (requirement,),
        artifact="fragment_0",
    )

    assert requirement.model_dump() == {
        "name": "p0",
        "logical_type": "time",
        "driver_type": "String",
    }

    with pytest.raises(ContractError) as raised:
        validate_bindings(
            "SELECT {p0:String}",
            Dialect.CLICKHOUSE,
            {"p0": "private"},
            (BindingRequirement("p0", logical_type="time", driver_type="DateTime64"),),
            artifact="fragment_0",
        )

    assert raised.value.reason == "binding_type_mismatch"


def test_literals_comments_and_quoted_identifiers_are_not_bindings() -> None:
    sql = "SELECT %(used)s, '%(literal)s', \"%(identifier)s\" -- %(comment)s\n/* %(block)s */"
    assert final_binding_requirements(sql, Dialect.POSTGRES, {"used": 7}) == (
        BindingRequirement("used"),
    )


def test_missing_or_renamed_binding_fails_without_disclosing_values() -> None:
    with pytest.raises(ContractError) as raised:
        validate_bindings(
            "SELECT @expected",
            Dialect.BIGQUERY,
            {"renamed": "secret-value"},
            (BindingRequirement("expected"),),
            artifact="fragment_0",
        )

    error = raised.value
    assert error.reason == "binding_values_mismatch"
    assert set(error.names) == {"expected", "renamed"}
    assert "secret-value" not in str(error)


def test_manifest_must_match_the_final_executable_projection() -> None:
    with pytest.raises(ContractError) as raised:
        validate_bindings(
            "SELECT $actual",
            Dialect.DUCKDB,
            {"actual": 1},
            (BindingRequirement("stale"),),
            artifact="fragment_0",
        )
    assert raised.value.reason == "binding_manifest_mismatch"
