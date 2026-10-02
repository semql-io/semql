"""Structured exception hierarchy for the semantic layer.

The compiler raises specific leaf classes — ``UnknownIdentifierError``,
``JoinPathError``, ``FilterTypeError``, ``PlaceholderError``,
``CrossDialectError``, ``PhaseDeferredError`` — so callers (MCP, API
layers, the planner retry loop) can branch on failure mode
programmatically. ``str(err)`` carries the human message; the leaf's
attributes carry the machine-readable structure.

Backwards compatibility:
- ``ResolveError`` and ``CompileError`` keep their existing identities;
  every new leaf subclasses ``CompileError``, so callers that
  ``except CompileError:`` still catch them.
- ``CompileError`` still subclasses ``ResolveError``, preserving the
  visualisation layer's ``except ResolveError:`` pattern.

Uniform error contract:
- ``to_payload`` / ``from_payload`` preserve lossless trusted internal error
  data, including values needed by authorized repair and round-trip callers.
- ``to_public_payload`` is a separate allowlisted rendering for external
  adapters; it never copies arbitrary exception text, SQL, bindings, or
  repair arguments.
- ``Diagnostic`` carries stable codes, severity, bounded reasons, semantic
  references, and typed operation/stage details.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator

_ErrorPayload = dict[str, Any]


class Diagnostic(BaseModel):
    """Safe, structured diagnostic suitable for external consumers."""

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["error", "warning", "advisory"] = "error"
    reason: str
    references: tuple[str, ...] = ()
    operation: str | None = None
    stage: str | None = None

    @field_validator("code", "operation", "stage")
    @classmethod
    def _bounded_token(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
            return "contract_violation"
        return value

    @field_validator("references")
    @classmethod
    def _safe_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            value
            for value in values
            if len(value) <= 128 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", value)
        )

    @field_validator("reason")
    @classmethod
    def _bounded_reason(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9_:-]{1,80}", value):
            return "contract_violation"
        return value

    def render_message(self) -> str:
        """Render a message from bounded, non-value-bearing diagnostic fields."""
        refs = f" ({', '.join(self.references)})" if self.references else ""
        return f"{self.reason.replace('_', ' ')}{refs}."

    def to_public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "reason": self.reason,
            "message": self.render_message(),
            "references": list(self.references),
        }
        if self.operation is not None:
            payload["operation"] = self.operation
        if self.stage is not None:
            payload["stage"] = self.stage
        return payload


class SemQLError(Exception):
    """Top-level base for every error raised by the semantic layer."""

    _payload_code: ClassVar[str] = ""

    def to_payload(self) -> _ErrorPayload:
        """Lossless trusted serialization; may contain sensitive exception text."""
        return {"code": self.code, "message": str(self)}

    def to_public_payload(self) -> dict[str, Any]:
        """Safe allowlisted rendering; never serializes exception text."""
        reason = "invalid_query" if isinstance(self, CompileError) else "semantic_error"
        if isinstance(self, AuthError):
            reason = "authentication_failed"
        elif isinstance(self, UnknownIdentifierError) or (
            isinstance(self, ResolveError) and not isinstance(self, CompileError)
        ):
            reason = "invalid_reference"
        return Diagnostic(code=self.code, reason=reason).to_public_payload()

    @property
    def code(self) -> str:
        return self._payload_code or type(self).__name__

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SemQLError:
        code = payload.get("code")
        if code is None:
            raise ValueError("Error payload is missing 'code'.")
        leaf = _PAYLOAD_DISPATCH.get(code)
        if leaf is None:
            raise ValueError(
                f"Unknown error code {code!r}. Known codes: "
                f"{sorted(_PAYLOAD_DISPATCH)}. Add the leaf class to "
                "semql.errors._PAYLOAD_DISPATCH if this is a new error."
            )
        return cast("SemQLError", leaf._from_payload(payload))  # type: ignore[attr-defined]


class ResolveError(SemQLError):
    """Identifier resolution failed (malformed reference, unknown cube,
    unknown field). Visualisation callers catch this directly."""


class CompileError(ResolveError):
    """Compilation failed. Subclasses ResolveError so visualisation
    callers keep working; specific leaves below carry structured attrs."""


class ContractError(CompileError):
    """Typed contract failure with sensitive internal and safe public forms."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        artifact: str | None = None,
        names: tuple[str, ...] = (),
        references: tuple[str, ...] = (),
        operation: str | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.artifact = artifact
        self.names = tuple(names)
        self.references = tuple(references)
        self.operation = operation
        self.stage = stage

    def to_payload(self) -> _ErrorPayload:
        payload: _ErrorPayload = {
            "code": self.code,
            "message": str(self),
            "reason": self.reason,
            "names": list(self.names),
            "references": list(self.references),
        }
        for key in ("artifact", "operation", "stage"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    def to_diagnostic(
        self, *, severity: Literal["error", "warning", "advisory"] = "error"
    ) -> Diagnostic:
        return Diagnostic(
            code=self.code,
            severity=severity,
            reason=self.reason,
            references=self.references,
            operation=self.operation,
            stage=self.stage,
        )

    def to_public_payload(self) -> dict[str, Any]:
        return self.to_diagnostic().to_public_payload()

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> ContractError:
        return cls(
            str(payload.get("message", "")),
            reason=str(payload.get("reason", "contract_violation")),
            artifact=payload.get("artifact"),
            names=tuple(str(value) for value in payload.get("names", ())),
            references=tuple(str(value) for value in payload.get("references", ())),
            operation=payload.get("operation"),
            stage=payload.get("stage"),
        )


class UnknownIdentifierError(CompileError):
    """Raised when a cube or field reference cannot be resolved.

    ``kind`` is ``"cube"`` or ``"field"``. ``name`` is the unknown
    identifier as it appeared in the query. ``cube`` is the parent
    cube name for field misses (``None`` for cube misses). ``hint``
    is the nearest catalog identifier if one was found, else None.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        name: str,
        cube: str | None = None,
        hint: str | None = None,
        valid_alternatives: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.name = name
        self.cube = cube
        self.hint = hint
        # Pre-computed alternative list (the LLM-friendly spelling of
        # ``hint``). When ``None`` we look up alternatives at
        # ``to_payload`` time from the message-context; the
        # constructor accepts a precomputed list so the compiler can
        # feed in the full set of cube/field names without re-running
        # ``closest_match`` (we already did so in the walker).
        self.valid_alternatives: list[str] = list(valid_alternatives) if valid_alternatives else []

    def to_payload(self) -> _ErrorPayload:
        payload: _ErrorPayload = {
            "code": self.code,
            "message": str(self),
            "kind": self.kind,
            "name": self.name,
        }
        if self.cube is not None:
            payload["cube"] = self.cube
        if self.hint is not None:
            payload["hint"] = self.hint
        if self.valid_alternatives:
            payload["valid_alternatives"] = list(self.valid_alternatives)
        return payload

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> UnknownIdentifierError:
        alts: Any = payload.get("valid_alternatives") or []
        return cls(
            str(payload.get("message", "")),
            kind=str(payload.get("kind", "field")),
            name=str(payload.get("name", "")),
            cube=payload.get("cube"),
            hint=payload.get("hint"),
            valid_alternatives=(
                [str(a) for a in cast("list[object]", alts)] if isinstance(alts, list) else None
            ),
        )


class JoinPathError(CompileError):
    """Raised when the catalog has no join path between two touched cubes."""

    def __init__(self, message: str, *, root_cube: str, target_cube: str) -> None:
        super().__init__(message)
        self.root_cube = root_cube
        self.target_cube = target_cube

    def to_payload(self) -> _ErrorPayload:
        return {
            "code": self.code,
            "message": str(self),
            "root_cube": self.root_cube,
            "target_cube": self.target_cube,
        }

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> JoinPathError:
        return cls(
            str(payload.get("message", "")),
            root_cube=str(payload.get("root_cube", "")),
            target_cube=str(payload.get("target_cube", "")),
        )


class FilterTypeError(CompileError):
    """Raised when a Filter's value doesn't match its dimension's type.

    Carries an LLM-repair affordance (``next_tool``) when the failing
    dimension has a registered ``Lookup`` — the caller can run the
    tool to resolve a free-text filter value to a canonical one. The
    construction site (in :mod:`semql.compile`) injects the lookup
    metadata at raise time; the envelope shape is fixed here.
    """

    def __init__(
        self,
        message: str,
        *,
        dimension: str,
        op: str,
        value: Any = None,  # noqa: ANN401 — Filter values are user-supplied literals
        next_tool: str | None = None,
        next_tool_args: dict[str, Any] | None = None,
        did_you_mean: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.op = op
        # User-supplied filter literal; ``Any`` by contract (see the
        # ``value`` param). Annotated so readers / pyright don't see an
        # implicit-Unknown attribute on every ``exc.value`` access.
        self.value: Any = value
        self.next_tool = next_tool
        self.next_tool_args = dict(next_tool_args) if next_tool_args else None
        self.did_you_mean: list[str] = list(did_you_mean) if did_you_mean else []

    def to_payload(self) -> _ErrorPayload:
        payload: _ErrorPayload = {
            "code": self.code,
            "message": str(self),
            "dimension": self.dimension,
            "op": self.op,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.next_tool is not None:
            payload["next_tool"] = self.next_tool
            if self.next_tool_args is not None:
                payload["next_tool_args"] = dict(self.next_tool_args)
        if self.did_you_mean:
            payload["did_you_mean"] = list(self.did_you_mean)
        return payload

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> FilterTypeError:
        return cls(
            str(payload.get("message", "")),
            dimension=str(payload.get("dimension", "")),
            op=str(payload.get("op", "")),
            value=payload.get("value"),
            next_tool=payload.get("next_tool"),
            next_tool_args=payload.get("next_tool_args"),
            did_you_mean=payload.get("did_you_mean"),
        )


class PlaceholderError(CompileError):
    """Raised when a ``{key}`` placeholder in catalog SQL is unknown."""

    def __init__(
        self,
        message: str,
        *,
        placeholder: str,
        known: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.placeholder = placeholder
        self.known = list(known) if known else []

    def to_payload(self) -> _ErrorPayload:
        return {
            "code": self.code,
            "message": str(self),
            "placeholder": self.placeholder,
            "known": list(self.known),
        }

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> PlaceholderError:
        return cls(
            str(payload.get("message", "")),
            placeholder=str(payload.get("placeholder", "")),
            known=payload.get("known"),
        )


class CrossDialectError(CompileError):
    """Raised when a single query touches multiple backends. The merge
    path is deferred (Phase 2)."""

    def __init__(self, message: str, *, backends: list[str]) -> None:
        super().__init__(message)
        self.backends = list(backends)

    def to_payload(self) -> _ErrorPayload:
        return {
            "code": self.code,
            "message": str(self),
            "backends": list(self.backends),
        }

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> CrossDialectError:
        b: Any = payload.get("backends") or []
        return cls(
            str(payload.get("message", "")),
            backends=[str(x) for x in cast("list[object]", b)] if isinstance(b, list) else [],
        )


class PhaseDeferredError(CompileError):
    """Raised when the query asks for a feature whose compiler support is
    deferred (e.g. ``compare`` windows)."""

    def __init__(self, message: str, *, feature: str) -> None:
        super().__init__(message)
        self.feature = feature

    def to_payload(self) -> _ErrorPayload:
        return {
            "code": self.code,
            "message": str(self),
            "feature": self.feature,
        }

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> PhaseDeferredError:
        return cls(str(payload.get("message", "")), feature=str(payload.get("feature", "")))


class FederationError(CompileError):
    """Raised when a cross-source query asks for something the v1
    federated compiler can't honour: a compound or expression join key,
    a Filter referencing multiple backends, a non-distributive
    aggregation (``count_distinct`` / ``min`` / ``max`` / ``ratio``),
    ``compare`` mode, or a boolean ``where`` tree. The in-process
    executor (``semql_engine.Engine``) can stream raw rows and handle
    most of these — sans-io callers can't, so we refuse early."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason

    def to_payload(self) -> _ErrorPayload:
        return {
            "code": self.code,
            "message": str(self),
            "reason": self.reason,
        }

    def to_public_payload(self) -> dict[str, Any]:
        return Diagnostic(
            code=self.code,
            reason=self.reason,
            operation="federation",
            stage="lowering",
        ).to_public_payload()

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> FederationError:
        return cls(str(payload.get("message", "")), reason=str(payload.get("reason", "")))


class AuthError(SemQLError):
    """Raised by ``TokenVerifier`` implementations on invalid, expired,
    or otherwise unverifiable bearer tokens.

    Deliberately a ``SemQLError`` (not a ``CompileError``) — token
    verification is a *transport-layer* concern that runs before the
    query is even constructed, so it sits outside the resolve/compile
    subtree. Callers that need a broad catch should use ``SemQLError``;
    the more specific ``except AuthError:`` is for handlers that want
    to surface a 401 / re-prompt-for-token UX.

    Carries an optional ``reason`` attribute (e.g. ``"expired"``,
    ``"bad_signature"``, ``"malformed"``) so callers can branch
    programmatically without parsing ``str(err)``.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason

    def to_payload(self) -> _ErrorPayload:
        payload: _ErrorPayload = {"code": self.code, "message": str(self)}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> AuthError:
        return cls(str(payload.get("message", "")), reason=payload.get("reason"))


# Dispatch table: code -> leaf class with a ``_from_payload`` constructor.
# Exhaustive over the public leaves; adding a new leaf without
# registering it here makes :meth:`SemQLError.from_payload` raise
# ``ValueError`` — the failure is loud, not silent. Keyed on the
# class name (the default ``code``); leaves that override ``code``
# must add a second entry here.
_PAYLOAD_DISPATCH: dict[str, type[SemQLError]] = {
    cls.__name__: cls
    for cls in (
        ContractError,
        UnknownIdentifierError,
        JoinPathError,
        FilterTypeError,
        PlaceholderError,
        CrossDialectError,
        PhaseDeferredError,
        FederationError,
        AuthError,
    )
}


def closest_match(
    name: str,
    candidates: Iterable[str],
    *,
    cutoff: float = 0.6,
) -> str | None:
    """Return the candidate closest to ``name`` by difflib ratio, or None.

    Used to enrich ``UnknownIdentifierError`` with a ``Did you mean ...``
    hint. ``cutoff`` is tuned to suppress wild guesses on short names.

    >>> closest_match("regin", ["region", "status"])
    'region'
    >>> closest_match("zzznotathing", ["region", "status"])
    """
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=cutoff)
    return matches[0] if matches else None


__all__ = [
    "CompileError",
    "AuthError",
    "ContractError",
    "CrossDialectError",
    "Diagnostic",
    "FederationError",
    "FilterTypeError",
    "JoinPathError",
    "PhaseDeferredError",
    "PlaceholderError",
    "ResolveError",
    "SemQLError",
    "UnknownIdentifierError",
    "closest_match",
]
