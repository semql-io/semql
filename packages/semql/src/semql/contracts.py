"""Opt-in, versioned declarations for metric semantics.

Contracts describe trusted catalog intent; they do not verify physical data.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semql.errors import Diagnostic
from semql.model import GranularityLiteral
from semql.refs import is_qualified


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProofStatus(StrEnum):
    PROVED = "proved"
    DISPROVED = "disproved"
    NOT_ESTABLISHED = "not_established"


def conjunction(statuses: tuple[ProofStatus, ...]) -> ProofStatus:
    if ProofStatus.DISPROVED in statuses:
        return ProofStatus.DISPROVED
    if not statuses or ProofStatus.NOT_ESTABLISHED in statuses:
        return ProofStatus.NOT_ESTABLISHED
    return ProofStatus.PROVED


def _refs(values: tuple[str, ...] | None, label: str, *, nonempty: bool = False) -> None:
    if values is None:
        return
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must contain distinct references")
    if any(not is_qualified(value) for value in values):
        raise ValueError(f"{label} references must be qualified cube.field values")


class Obligation(_Frozen):
    code: str
    status: ProofStatus
    reference: str
    explanation: str
    assumptions: tuple[str, ...] = ()
    mandatory: bool = True

    def diagnostic(self) -> Diagnostic | None:
        if self.status == ProofStatus.PROVED:
            return None
        return Diagnostic(
            code=self.code,
            severity="error" if self.mandatory else "advisory",
            reason="missing_proof" if self.status == ProofStatus.NOT_ESTABLISHED else self.code,
            references=(self.reference,),
            operation="check_semantics",
            stage="semantic_check",
        )


class Advice(_Frozen):
    code: str
    reference: str
    recommendation: str
    preconditions: tuple[str, ...] = ()


class FunctionalDependency(_Frozen):
    determinant: tuple[str, ...] = Field(min_length=1)
    dependent: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_refs(self) -> FunctionalDependency:
        _refs(self.determinant, "functional dependency determinant", nonempty=True)
        _refs(self.dependent, "functional dependency dependent", nonempty=True)
        return self


class SourceGrain(_Frozen):
    keys: tuple[str, ...] = Field(min_length=1)
    dependencies: tuple[FunctionalDependency, ...] = ()

    @model_validator(mode="after")
    def valid_refs(self) -> SourceGrain:
        _refs(self.keys, "source grain keys", nonempty=True)
        return self

    def determines(self, target: tuple[str, ...]) -> ProofStatus:
        """Return conditional FD entailment; absence of proof is not data failure."""
        if not target:
            return ProofStatus.PROVED
        _refs(target, "functional dependency target", nonempty=True)
        if not set(target) <= set(self.keys) | {
            ref for fd in self.dependencies for ref in (*fd.determinant, *fd.dependent)
        }:
            return ProofStatus.NOT_ESTABLISHED
        closure = set(self.keys)
        changed = True
        while changed:
            changed = False
            for dependency in self.dependencies:
                if (
                    set(dependency.determinant) <= closure
                    and not set(dependency.dependent) <= closure
                ):
                    closure.update(dependency.dependent)
                    changed = True
        return ProofStatus.PROVED if set(target) <= closure else ProofStatus.NOT_ESTABLISHED


class MetricContract(_Frozen):
    reference: str
    revision: str = Field(min_length=1)
    schema_version: Literal[1] = 1
    source_grain: SourceGrain | None = None
    permitted_time_axes: tuple[str, ...] | None = None
    permitted_granularities: tuple[GranularityLiteral | None, ...] | None = None
    required_time_axis: bool = False
    calendar_ref: str | None = None
    required_segments: tuple[str, ...] = ()
    forbidden_segments: tuple[str, ...] = ()
    required_dimensions: tuple[str, ...] = ()
    non_additive_dimensions: tuple[str, ...] = ()
    temporal_behavior: Literal["flow", "snapshot", "unspecified"] = "unspecified"
    snapshot_axis: str | None = None
    preferred_time_axis: str | None = None
    preferred_granularity: GranularityLiteral | None = None

    @model_validator(mode="after")
    def coherent(self) -> MetricContract:
        if not is_qualified(self.reference):
            raise ValueError("metric reference must be qualified cube.field")
        for name in (
            "permitted_time_axes",
            "required_segments",
            "forbidden_segments",
            "required_dimensions",
            "non_additive_dimensions",
        ):
            _refs(getattr(self, name), name)
        if set(self.required_segments) & set(self.forbidden_segments):
            raise ValueError("a segment cannot be both required and forbidden")
        if self.temporal_behavior == "snapshot" and self.snapshot_axis is None:
            raise ValueError("snapshot contracts require a snapshot axis")
        for name in ("snapshot_axis", "preferred_time_axis"):
            value = getattr(self, name)
            if value is not None and not is_qualified(value):
                raise ValueError(f"{name} must be a qualified cube.field reference")
        if (
            self.preferred_time_axis
            and self.permitted_time_axes is not None
            and self.preferred_time_axis not in self.permitted_time_axes
        ):
            raise ValueError("preferred time axis must be permitted")
        if (
            self.snapshot_axis
            and self.permitted_time_axes is not None
            and self.snapshot_axis not in self.permitted_time_axes
        ):
            raise ValueError("snapshot axis must be permitted")
        if (
            self.preferred_granularity
            and self.permitted_granularities is not None
            and self.preferred_granularity not in self.permitted_granularities
        ):
            raise ValueError("preferred granularity must be permitted")
        if self.required_time_axis and self.permitted_time_axes == ():
            raise ValueError("a required time axis cannot have an empty permitted-axis set")
        if self.snapshot_axis is not None and self.temporal_behavior != "snapshot":
            raise ValueError("snapshot_axis is only valid for snapshot contracts")
        if self.permitted_granularities is not None and len(
            set(self.permitted_granularities)
        ) != len(self.permitted_granularities):
            raise ValueError("permitted granularities must be distinct")
        return self


class SemanticExpectations(_Frozen):
    metrics: tuple[str, ...] | None = None
    grain: tuple[str, ...] | None = None
    population_inclusion: Literal["observed_facts", "query_rows"] | None = None
    required_segments: tuple[str, ...] | None = None
    forbidden_segments: tuple[str, ...] | None = None
    comparison_basis: Literal["fixed_cohort", "independent_periods"] | None = None
    time_axis: str | None = None
    granularity: GranularityLiteral | None = None
    calendar_ref: str | None = None
    origin: Literal["independently_confirmed", "model_inferred"] = "model_inferred"
    evidence_ref: str | None = None

    @model_validator(mode="after")
    def valid_expectations(self) -> SemanticExpectations:
        _refs(self.metrics, "expected metrics")
        _refs(self.grain, "expected grain")
        _refs(self.required_segments, "expected required segments")
        _refs(self.forbidden_segments, "expected forbidden segments")
        if (
            self.required_segments is not None
            and self.forbidden_segments is not None
            and set(self.required_segments) & set(self.forbidden_segments)
        ):
            raise ValueError("an expected segment cannot be both required and forbidden")
        for name in ("time_axis",):
            value = getattr(self, name)
            if value is not None and not is_qualified(value):
                raise ValueError(f"{name} must be a qualified cube.field reference")
        if self.origin == "independently_confirmed" and not self.evidence_ref:
            raise ValueError("confirmed intent requires an independent evidence reference")
        return self


class SemanticCheck(_Frozen):
    structural_validity: ProofStatus
    obligations: tuple[Obligation, ...]
    advice: tuple[Advice, ...] = ()
    alignment: tuple[Obligation, ...] = ()
    intent_origin: str | None = None
    intent_evidence_ref: str | None = None
    confirmed_intent_alignment: ProofStatus = ProofStatus.NOT_ESTABLISHED
    execution_support: ProofStatus = ProofStatus.NOT_ESTABLISHED
    execution_explanation: str = "No lowering evidence supplied."
    applied_contracts: tuple[str, ...] = ()
    compiler_assumptions: tuple[str, ...] = ()
    require_confirmed_intent: bool = False

    @property
    def semantic_validity(self) -> ProofStatus:
        return conjunction(tuple(x.status for x in self.obligations if x.mandatory))

    @property
    def accepted(self) -> bool:
        return (
            self.structural_validity == ProofStatus.PROVED
            and self.semantic_validity == ProofStatus.PROVED
            and (
                not self.require_confirmed_intent
                or self.confirmed_intent_alignment == ProofStatus.PROVED
            )
        )


def validate_registry(
    contracts: Mapping[str, MetricContract],
) -> tuple[tuple[str, MetricContract], ...]:
    """Snapshot a caller mapping and reject keys that disagree with contracts."""
    items = tuple(contracts.items())
    for key, contract in items:
        if key != contract.reference:
            raise ValueError("contract registry key must equal its canonical reference")
    return items


__all__ = [
    "Advice",
    "FunctionalDependency",
    "MetricContract",
    "Obligation",
    "ProofStatus",
    "SemanticCheck",
    "SemanticExpectations",
    "SourceGrain",
]
