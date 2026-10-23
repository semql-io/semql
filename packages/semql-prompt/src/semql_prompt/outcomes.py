"""Strict untrusted planner outcome protocol and typed context requests."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, TypeAdapter, model_validator
from semql.errors import Diagnostic
from semql.model import FrozenModel
from semql.plan import QueryPlan

_BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
_Token = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")]


class OutcomeModel(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlanOutcome(OutcomeModel):
    kind: Literal["plan"] = "plan"
    plan: QueryPlan

    @model_validator(mode="after")
    def _require_steps(self) -> PlanOutcome:
        if not self.plan.steps:
            raise ValueError("plan outcome requires at least one query step")
        return self


class ContextRequest(OutcomeModel):
    request_id: _Token
    decision_id: _Token
    kind: Literal["definition", "lookup"]
    target: Annotated[
        str,
        StringConstraints(
            min_length=3,
            max_length=160,
            pattern=r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$",
        ),
    ]
    query: _BoundedText | None = None


class ContextCandidate(OutcomeModel):
    id: _Token
    label: _BoundedText
    value: _BoundedText
    match_kind: Literal["exact", "substring", "fuzzy", "definition"]
    authoritative: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _fuzzy_is_suggestion(self) -> ContextCandidate:
        if self.match_kind == "fuzzy" and self.authoritative:
            raise ValueError("fuzzy candidates are never authoritative")
        return self


class ContextResult(OutcomeModel):
    request_id: _Token
    decision_id: _Token
    status: Literal["resolved", "ambiguous", "not_found", "unavailable"]
    candidates: tuple[ContextCandidate, ...] = Field(default_factory=tuple, max_length=50)
    complete: bool = Field(strict=True)
    truncated: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _check_certainty(self) -> ContextResult:
        if self.truncated and self.complete:
            raise ValueError("truncated context candidates cannot be complete")
        if self.status == "resolved" and (
            len(self.candidates) != 1 or not self.complete or not self.candidates[0].authoritative
        ):
            raise ValueError("resolved context requires one complete authoritative candidate")
        if self.status == "ambiguous" and (
            not self.candidates
            or (len(self.candidates) == 1 and self.complete and self.candidates[0].authoritative)
        ):
            raise ValueError("ambiguous context must remain non-authoritative")
        if self.status == "not_found" and (self.candidates or not self.complete):
            raise ValueError("not_found requires a complete empty candidate set")
        if self.status == "unavailable" and (self.candidates or self.complete):
            raise ValueError("unavailable context cannot claim complete candidate data")
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("context candidate IDs must be unique")
        return self


class NeedContext(OutcomeModel):
    kind: Literal["need_context"] = "need_context"
    requests: tuple[ContextRequest, ...] = Field(min_length=1, max_length=8)
    results: tuple[ContextResult, ...] = Field(default_factory=tuple, max_length=8)

    @model_validator(mode="after")
    def _correlate_results(self) -> NeedContext:
        correlations = {request.request_id: request.decision_id for request in self.requests}
        decisions = [request.decision_id for request in self.requests]
        if len(correlations) != len(self.requests) or len(decisions) != len(set(decisions)):
            raise ValueError("context request and decision identifiers must be unique")
        result_ids = [result.request_id for result in self.results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("context results must be unique per request")
        if any(
            correlations.get(result.request_id) != result.decision_id for result in self.results
        ):
            raise ValueError("context result must correlate to its exact request")
        return self


class ClarificationChoice(OutcomeModel):
    id: _Token
    label: _BoundedText
    value: _BoundedText


class NeedUserClarification(OutcomeModel):
    kind: Literal["need_user_clarification"] = "need_user_clarification"
    decision_id: _Token
    prompt: _BoundedText
    choices: tuple[ClarificationChoice, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _unique_choice_ids(self) -> NeedUserClarification:
        ids = [choice.id for choice in self.choices]
        if len(set(ids)) != len(ids):
            raise ValueError("clarification choice ids must be unique")
        return self


class CannotPlan(OutcomeModel):
    kind: Literal["cannot_plan"] = "cannot_plan"
    diagnostics: tuple[Diagnostic, ...] = Field(min_length=1, max_length=16)


type PlannerOutcome = Annotated[
    PlanOutcome | NeedContext | NeedUserClarification | CannotPlan,
    Field(discriminator="kind"),
]


class PlannerResponse(OutcomeModel):
    """Object-root response; correlation is required when continuing host state."""

    outcome: PlannerOutcome
    request_id: _Token | None = None
    revision: Annotated[int, Field(ge=0, strict=True)] | None = None


PLANNER_RESPONSE_ADAPTER: TypeAdapter[PlannerResponse] = TypeAdapter(PlannerResponse)

__all__ = [
    "CannotPlan",
    "ClarificationChoice",
    "ContextCandidate",
    "ContextRequest",
    "ContextResult",
    "NeedContext",
    "NeedUserClarification",
    "PlanOutcome",
    "PlannerOutcome",
    "PlannerResponse",
    "PLANNER_RESPONSE_ADAPTER",
]
