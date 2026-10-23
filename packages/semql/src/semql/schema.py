"""Explicit JSON-schema projection profiles for model-facing contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel

from semql._schema import profile_schema


class SchemaProfile(StrEnum):
    """Decoder representation profile; not authorization or compiler support."""

    TOOL = "tool"
    STRICT = "strict"


class SemanticFeature(StrEnum):
    """SemanticQuery input features that hosts may disable explicitly."""

    WHERE = "where"
    SEMI_JOINS = "semi_joins"
    RANK = "rank"


@dataclass(frozen=True)
class SchemaFeatureRemoval:
    """One intentionally omitted semantic input feature and its schema path."""

    feature: str
    path: str


@dataclass(frozen=True)
class SchemaConstraintWeakening:
    """A schema constraint weakened for decoder compatibility."""

    path: str
    constraint: str
    detail: str


@dataclass(frozen=True)
class SchemaProjection:
    """Projected schema plus deterministic loss report and canonical validator."""

    _schema: dict[str, Any] = field(repr=False)
    profile: SchemaProfile
    output_model: type[BaseModel]
    removed_features: tuple[SchemaFeatureRemoval, ...] = ()
    weakened_constraints: tuple[SchemaConstraintWeakening, ...] = ()
    disabled_features: tuple[str, ...] = ()

    @property
    def schema(self) -> dict[str, Any]:
        """Return an isolated schema copy so callers cannot mutate the contract."""
        return deepcopy(self._schema)

    def validate_payload(self, payload: object) -> BaseModel:
        """Validate canonically, then reject omitted features on SemanticQuery nodes."""
        parsed = self.output_model.model_validate(payload)
        forbidden = set(self.disabled_features)
        if self.profile == SchemaProfile.STRICT:
            forbidden.update((SemanticFeature.WHERE.value, SemanticFeature.SEMI_JOINS.value))
        if forbidden:
            from semql.spec import SemanticQuery

            self._reject_semantic_features(parsed, forbidden, SemanticQuery)
        return parsed

    @classmethod
    def _reject_semantic_features(
        cls,
        value: object,
        forbidden: set[str],
        semantic_query_type: type[BaseModel],
        path: str = "$",
        seen: set[int] | None = None,
    ) -> None:
        if seen is None:
            seen = set()
        if isinstance(value, BaseModel):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, semantic_query_type):
                unavailable = forbidden.intersection(value.model_fields_set)
                if unavailable:
                    feature = sorted(unavailable)[0]
                    raise ValueError(f"feature {feature!r} is not available at {path}.{feature}")
            for name in type(value).model_fields:
                cls._reject_semantic_features(
                    getattr(value, name), forbidden, semantic_query_type, f"{path}.{name}", seen
                )
        elif isinstance(value, Mapping):
            mapping = cast("Mapping[str, object]", value)
            for name, child in mapping.items():
                cls._reject_semantic_features(
                    child, forbidden, semantic_query_type, f"{path}.{name}", seen
                )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sequence = cast("Sequence[object]", value)
            for index, child in enumerate(sequence):
                cls._reject_semantic_features(
                    child, forbidden, semantic_query_type, f"{path}[{index}]", seen
                )


def project_schema(
    output_model: type[BaseModel],
    profile: SchemaProfile = SchemaProfile.TOOL,
    *,
    disabled_features: Sequence[SemanticFeature | str] = (),
) -> SchemaProjection:
    """Project a canonical output model without weakening required payload shape.

    ``STRICT`` omits the recursive ``SemanticQuery.where`` and ``semi_joins``
    features and turns fixed tuples into arrays. Its report explicitly records
    those omissions and positional tuple-constraint weakening. Host-disabled
    feature names are a separate representation restriction, not authorization.
    """
    active_profile = SchemaProfile(profile)
    disabled = tuple(sorted({SemanticFeature(item).value for item in disabled_features}))
    unsupported = set(disabled).difference(feature.value for feature in SemanticFeature)
    if unsupported:
        raise ValueError(f"unsupported disabled feature(s): {', '.join(sorted(unsupported))}")
    removed_names = set(disabled)
    if active_profile == SchemaProfile.STRICT:
        removed_names.update((SemanticFeature.WHERE.value, SemanticFeature.SEMI_JOINS.value))
    canonical = output_model.model_json_schema()
    schema = profile_schema(
        canonical,
        removed_properties=frozenset(removed_names),
        strict=active_profile == SchemaProfile.STRICT,
    )
    removals = tuple(
        SchemaFeatureRemoval(feature=name, path=f"SemanticQuery.{name}")
        for name in sorted(removed_names)
        if _schema_has_property(canonical, name)
    )
    weakened: tuple[SchemaConstraintWeakening, ...] = ()
    if active_profile == SchemaProfile.STRICT:
        found: list[SchemaConstraintWeakening] = []
        _collect_tuple_weakenings(canonical, "$", found)
        weakened = tuple(found)
    return SchemaProjection(
        _schema=schema,
        profile=active_profile,
        output_model=output_model,
        removed_features=removals,
        weakened_constraints=weakened,
        disabled_features=disabled,
    )


def _schema_has_property(schema: dict[str, Any], name: str) -> bool:
    definitions = schema.get("$defs", {})
    nodes: list[object] = [schema]
    if isinstance(definitions, dict):
        nodes.extend(cast("dict[str, Any]", definitions).values())
    for node in nodes:
        if not isinstance(node, dict):
            continue
        mapping = cast("Mapping[str, object]", node)
        properties = mapping.get("properties")
        if (
            mapping.get("title") == "SemanticQuery"
            and isinstance(properties, dict)
            and name in properties
        ):
            return True
    return False


def _collect_tuple_weakenings(
    node: object, path: str, output: list[SchemaConstraintWeakening]
) -> None:
    if isinstance(node, dict):
        mapping = cast("Mapping[str, object]", node)
        if "prefixItems" in mapping:
            output.append(
                SchemaConstraintWeakening(
                    path=path,
                    constraint="tuple positional validation",
                    detail="fixed tuple positions are represented as a homogeneous or union array",
                )
            )
        for key in sorted(mapping):
            _collect_tuple_weakenings(mapping[key], f"{path}.{key}", output)
    elif isinstance(node, list):
        sequence = cast("Sequence[object]", node)
        for index, child in enumerate(sequence):
            _collect_tuple_weakenings(child, f"{path}[{index}]", output)


__all__ = [
    "SemanticFeature",
    "SchemaConstraintWeakening",
    "SchemaFeatureRemoval",
    "SchemaProfile",
    "SchemaProjection",
    "project_schema",
]
