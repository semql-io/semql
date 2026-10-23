"""Qualified reference type and the single parser for ``"cube.field"`` refs.

A ``SemanticQuery`` names catalog fields as qualified strings — ``cube.field``.
This module is the one place that knows that convention: the shape check and
the split live here instead of being repeated at every use site.

A qualified ref is exactly ``<cube>.<field>`` — two SQL identifiers joined by
a single dot. Bare names and multi-dot output columns (e.g. compare facets)
are *not* qualified refs; :func:`local_name` is the tolerant read for the
paths that legitimately accept an already-unqualified field name.
"""

from __future__ import annotations

import re

from semql.errors import ResolveError

__all__ = [
    "QualifiedRef",
    "parse_qualified_ref",
    "is_qualified",
    "cube_of",
    "field_of",
    "local_name",
]

# One dot, two SQL identifiers. Case-insensitive: catalog identifiers may be
# authored in any case and resolve case-sensitively downstream.
_QUALIFIED_RE = re.compile(r"^([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)$", re.IGNORECASE)


class QualifiedRef(str):
    """A ``cube.field`` reference.

    A ``str`` subclass: it equals its source text and drops into any place a
    ref string is expected (``SemanticQuery.measures`` is a sequence of strings),
    while exposing the validated ``cube`` / ``field`` halves. Construction
    raises :class:`~semql.errors.ResolveError` on a malformed ref, so a bad
    ref fails loudly at the boundary instead of silently degrading at a
    downstream ``.split(".")``.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> QualifiedRef:
        if _QUALIFIED_RE.match(value) is None:
            raise ResolveError(f"Field reference must be 'cube.field', got: {value!r}")
        return super().__new__(cls, value)

    @property
    def cube(self) -> str:
        return self.split(".", 1)[0]

    @property
    def field(self) -> str:
        return self.split(".", 1)[1]

    @classmethod
    def of(cls, cube: str, field: str) -> QualifiedRef:
        """Build a ref from its two halves (validates the result)."""
        return cls(f"{cube}.{field}")


def parse_qualified_ref(ref: str) -> QualifiedRef:
    """Parse ``ref`` into a :class:`QualifiedRef`, raising
    :class:`~semql.errors.ResolveError` if it is not exactly ``cube.field``."""
    return QualifiedRef(ref)


def is_qualified(ref: str) -> bool:
    """True iff ``ref`` is exactly ``cube.field`` (no raise)."""
    return _QUALIFIED_RE.match(ref) is not None


def cube_of(ref: str) -> str:
    """The cube half of a qualified ref (strict — raises on a bare name)."""
    return parse_qualified_ref(ref).cube


def field_of(ref: str) -> str:
    """The field half of a qualified ref (strict — raises on a bare name)."""
    return parse_qualified_ref(ref).field


def local_name(ref: str) -> str:
    """The field name, tolerant of an already-unqualified ref.

    Returns the part after the last dot, or the whole string when there is no
    dot — for the paths that legitimately receive either a qualified ref or a
    bare field name (e.g. names already stripped of their cube)."""
    return ref.rsplit(".", 1)[-1]
