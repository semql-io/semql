# pyright: reportPrivateImportUsage=false
"""Structural executable-bind manifests and final SQL binding validation.

The compiler owns allocation. This module checks the *rendered SQL* at the
executor boundary, so omitted or renamed placeholders fail before I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from semql.dialect import dialect_for
from semql.errors import ContractError
from semql.model import Dialect


@dataclass(frozen=True)
class BindingRequirement:
    """One executable named parameter required by final SQL."""

    name: str
    logical_type: str | None = None
    driver_type: str | None = None

    def model_dump(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "logical_type": self.logical_type,
            "driver_type": self.driver_type,
        }

    @classmethod
    def model_validate(cls, data: Mapping[str, object]) -> BindingRequirement:
        name = data.get("name")
        logical_type = data.get("logical_type")
        driver_type = data.get("driver_type")
        if not isinstance(name, str):
            raise ValueError("BindingRequirement.name must be a string.")
        if logical_type is not None and not isinstance(logical_type, str):
            raise ValueError("BindingRequirement.logical_type must be a string or None.")
        if driver_type is not None and not isinstance(driver_type, str):
            raise ValueError("BindingRequirement.driver_type must be a string or None.")
        return cls(name=name, logical_type=logical_type, driver_type=driver_type)


def _placeholder_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Placeholder):
        name = node.name
    else:
        value = node.args.get("this")
        name = value.name if isinstance(value, exp.Expression) else ""
    if not name:
        raise ValueError("positional-placeholder")
    return name


def final_binding_requirements(
    sql: str,
    dialect: Dialect,
    params: Mapping[str, Any],
) -> tuple[BindingRequirement, ...]:
    """Derive unique named placeholders in final SQL AST occurrence order.

    SQL literals, comments, and quoted identifiers are not traversed as
    placeholders. Parameter values are never inspected or copied into errors.
    """
    _ = params  # Bind values are deliberately never inspected.
    try:
        statements = sqlglot.parse(sql, dialect=dialect_for(dialect))
    except (ParseError, ValueError) as exc:
        raise ContractError(
            "Executable SQL could not be parsed to verify its binding manifest.",
            reason="binding_sql_unparseable",
            artifact="sql",
            operation="bind",
            stage="preflight",
        ) from exc
    requirements: dict[str, BindingRequirement] = {}
    for statement in statements:
        if statement is None:
            continue
        for node in statement.find_all(exp.Placeholder, exp.Parameter):
            try:
                name = _placeholder_name(node)
            except ValueError as exc:
                raise ContractError(
                    "Executable SQL contains a positional placeholder without a manifest name.",
                    reason="binding_positional_unsupported",
                    artifact="sql",
                    operation="bind",
                    stage="preflight",
                ) from exc
            kind = node.args.get("kind")
            driver_type = (
                kind.sql(dialect=dialect_for(dialect))
                if isinstance(kind, exp.DataType)
                else kind
                if isinstance(kind, str)
                else None
            )
            prior = requirements.get(name)
            if (
                prior is not None
                and prior.driver_type not in (None, driver_type)
                and driver_type is not None
            ):
                raise ContractError(
                    "Executable SQL declares conflicting driver types for a binding.",
                    reason="binding_type_conflict",
                    artifact="sql",
                    names=(name,),
                    operation="bind",
                    stage="preflight",
                )
            requirements[name] = BindingRequirement(
                name,
                driver_type=(
                    driver_type
                    if driver_type is not None
                    else (prior.driver_type if prior else None)
                ),
            )
    return tuple(requirements.values())


def validate_bindings(
    sql: str,
    dialect: Dialect,
    params: Mapping[str, Any],
    requirements: Sequence[BindingRequirement],
    *,
    artifact: str,
) -> None:
    """Fail closed if final SQL, its manifest, and bound names disagree."""
    actual = final_binding_requirements(sql, dialect, params)
    actual_by_name = {item.name: item for item in actual}
    declared_by_name: dict[str, BindingRequirement] = {}
    for item in requirements:
        if item.name in declared_by_name:
            raise ContractError(
                "Executable binding manifest contains a duplicate name.",
                reason="binding_manifest_duplicate",
                artifact=artifact,
                names=(item.name,),
                operation="bind",
                stage="preflight",
            )
        declared_by_name[item.name] = item
    if set(actual_by_name) != set(declared_by_name):
        raise ContractError(
            "Executable SQL binding manifest does not match its final placeholders.",
            reason="binding_manifest_mismatch",
            artifact=artifact,
            names=tuple(sorted(set(actual_by_name) ^ set(declared_by_name))),
            operation="bind",
            stage="preflight",
        )
    actual_names = set(actual_by_name)
    param_names = set(params)
    if actual_names != param_names:
        raise ContractError(
            "Executable SQL placeholders do not match supplied bindings.",
            reason="binding_values_mismatch",
            artifact=artifact,
            names=tuple(sorted(actual_names ^ param_names)),
            operation="bind",
            stage="preflight",
        )
    for name, required in declared_by_name.items():
        rendered_type = actual_by_name[name].driver_type
        if required.driver_type is not None and rendered_type != required.driver_type:
            raise ContractError(
                "Executable SQL binding driver type does not match its manifest.",
                reason="binding_type_mismatch",
                artifact=artifact,
                names=(name,),
                operation="bind",
                stage="preflight",
            )


__all__ = ["BindingRequirement", "final_binding_requirements", "validate_bindings"]
