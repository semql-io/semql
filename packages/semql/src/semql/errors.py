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
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Any


class SemQLError(Exception):
    """Top-level base for every error raised by the semantic layer."""


class ResolveError(SemQLError):
    """Identifier resolution failed (malformed reference, unknown cube,
    unknown field). Visualisation callers catch this directly."""


class CompileError(ResolveError):
    """Compilation failed. Subclasses ResolveError so visualisation
    callers keep working; specific leaves below carry structured attrs."""


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
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.name = name
        self.cube = cube
        self.hint = hint


class JoinPathError(CompileError):
    """Raised when the catalog has no join path between two touched cubes."""

    def __init__(self, message: str, *, root_cube: str, target_cube: str) -> None:
        super().__init__(message)
        self.root_cube = root_cube
        self.target_cube = target_cube


class FilterTypeError(CompileError):
    """Raised when a Filter's value doesn't match its dimension's type."""

    def __init__(
        self,
        message: str,
        *,
        dimension: str,
        op: str,
        value: Any = None,  # noqa: ANN401 — Filter values are user-supplied literals
    ) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.op = op
        self.value = value


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


class CrossDialectError(CompileError):
    """Raised when a single query touches multiple backends. The merge
    path is deferred (Phase 2)."""

    def __init__(self, message: str, *, backends: list[str]) -> None:
        super().__init__(message)
        self.backends = list(backends)


class PhaseDeferredError(CompileError):
    """Raised when the query asks for a feature whose compiler support is
    deferred (e.g. ``compare`` windows)."""

    def __init__(self, message: str, *, feature: str) -> None:
        super().__init__(message)
        self.feature = feature


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


def closest_match(
    name: str,
    candidates: Iterable[str],
    *,
    cutoff: float = 0.6,
) -> str | None:
    """Return the candidate closest to ``name`` by difflib ratio, or None.

    Used to enrich ``UnknownIdentifierError`` with a ``Did you mean ...``
    hint. ``cutoff`` is tuned to suppress wild guesses on short names.
    """
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=cutoff)
    return matches[0] if matches else None


__all__ = [
    "CompileError",
    "AuthError",
    "CrossDialectError",
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
