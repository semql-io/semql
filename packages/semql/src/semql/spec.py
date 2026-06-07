"""The query spec the planner emits and the compiler consumes.

All identifiers are *qualified* — `cube.field`. The compiler resolves
them against the catalogue; unknown identifiers raise `CompileError`
naming the offending field.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FilterOp = Literal[
    "eq",
    "neq",
    "in",
    "not_in",
    "gt",
    "lt",
    "gte",
    "lte",
    "contains",
    "is_null",
    "not_null",
]


class TimeWindow(BaseModel):
    model_config = ConfigDict(frozen=True)
    dimension: str
    granularity: Literal["hour", "day", "week", "month"] | None = None
    range: tuple[str, str]
    # When set, every measure in the query gets ``COALESCE(measure,
    # fill_nulls_with)`` and the result has one row per truncated time
    # bucket in ``range`` even where the underlying data has gaps.
    # Requires ``granularity``; rejected when the query has any
    # non-time dimensions (cartesian fill = Phase B).
    fill_nulls_with: int | None = None


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True)
    dimension: str
    op: FilterOp
    values: list[str | int | float | bool] = []

    def validate_for_type(self, dim_type: str) -> None:
        """Compile-time type check. Raises ValueError on mismatch; the
        compiler re-raises as CompileError."""
        if self.op in ("is_null", "not_null"):
            return
        if not self.values:
            raise ValueError(
                f"Filter on {self.dimension!r} with op={self.op!r} requires at least one value."
            )
        for v in self.values:
            if dim_type == "number":
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Filter on numeric dimension {self.dimension!r} got "
                        f"non-numeric value {v!r}."
                    )
            elif dim_type == "bool":
                if not isinstance(v, bool):
                    raise ValueError(
                        f"Filter on bool dimension {self.dimension!r} got non-bool value {v!r}."
                    )
            elif dim_type == "time":
                if not isinstance(v, str):
                    raise ValueError(
                        f"Filter on time dimension {self.dimension!r} got non-string value {v!r}."
                    )
                try:
                    datetime.fromisoformat(v)
                except ValueError:
                    raise ValueError(
                        f"Filter on time dimension {self.dimension!r} got non-ISO-8601 value {v!r}."
                    ) from None
            elif dim_type == "uuid":
                if not isinstance(v, str):
                    raise ValueError(
                        f"Filter on uuid dimension {self.dimension!r} got non-string value {v!r}."
                    )
                try:
                    _uuid.UUID(v)
                except ValueError:
                    raise ValueError(
                        f"Filter on uuid dimension {self.dimension!r} got non-UUID value {v!r}."
                    ) from None
            elif dim_type == "string" and not isinstance(v, str):
                raise ValueError(
                    f"Filter on string dimension {self.dimension!r} got non-string value {v!r}."
                )


class BoolExpr(BaseModel):
    """Recursive boolean predicate tree over ``Filter`` leaves.

    ``filters`` (the flat list on ``SemanticQuery``) is implicit-AND;
    use ``BoolExpr`` when you need OR or NOT. Children are either
    nested ``BoolExpr`` nodes or ``Filter`` leaves. ``not`` takes
    exactly one child; ``and`` / ``or`` take two or more.
    """

    model_config = ConfigDict(frozen=True)
    op: Literal["and", "or", "not"]
    children: list[BoolExpr | Filter]

    @model_validator(mode="after")
    def _check_arity(self) -> BoolExpr:
        n = len(self.children)
        if self.op == "not" and n != 1:
            raise ValueError(f"BoolExpr(op='not') takes exactly one child; got {n}.")
        if self.op in ("and", "or") and n < 2:
            raise ValueError(f"BoolExpr(op={self.op!r}) requires at least two children; got {n}.")
        return self


class CompareWindow(BaseModel):
    """`previous_period` derives the prior window from the TimeWindow's
    range (same duration, immediately prior). `explicit` requires `range`."""

    model_config = ConfigDict(frozen=True)
    mode: Literal["previous_period", "explicit"] = "previous_period"
    range: tuple[str, str] | None = None


InlineDerivedOp = Literal["ratio", "sum", "diff"]


class InlineDerived(BaseModel):
    """Ad-hoc derived measure composed from existing catalog measures.

    Covers the exploratory shape an LLM or human reaches for in chat:
    "show me ``productive_time + active_time`` per region", or
    "``revenue / count``", or "``logged_minutes - active_minutes``"
    — without forcing a catalog change for a one-off composition.
    Stable named metrics still belong in the catalog as
    ``Measure(agg="ratio")`` (with their own ``numerator`` /
    ``denominator``) so every caller picks them up by name.

    ``operands`` are qualified ``"cube.measure"`` refs that the
    compiler resolves to their existing aggregation (``SUM`` /
    ``COUNT`` / ``AVG`` / ``COUNT(DISTINCT)`` / etc.). The outer
    SELECT then composes the named arithmetic operator over those
    aggregates.

    Op semantics:
    - ``"ratio"``: exactly two operands; emits
      ``<num_agg>(num_expr) / NULLIF(<den_agg>(den_expr), 0) AS name``
      — identical SQL shape to a pre-declared ratio measure.
    - ``"sum"``: two or more operands; emits
      ``<a_agg>(a) + <b_agg>(b) + ... AS name``.
    - ``"diff"``: exactly two operands; emits
      ``<a_agg>(a) - <b_agg>(b) AS name``.

    Phase A restriction: every operand must resolve to a measure on
    the same cube. Cross-cube refs raise at compile time with a hint
    to pre-declare the derivation in the catalog.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    op: InlineDerivedOp
    operands: list[str]

    @model_validator(mode="after")
    def _check_arity(self) -> InlineDerived:
        n = len(self.operands)
        if self.op == "ratio" and n != 2:
            raise ValueError(
                f"InlineDerived({self.name!r}, op='ratio'): requires "
                f"exactly two operands (numerator, denominator); got {n}."
            )
        if self.op == "diff" and n != 2:
            raise ValueError(
                f"InlineDerived({self.name!r}, op='diff'): requires "
                f"exactly two operands (minuend, subtrahend); got {n}."
            )
        if self.op == "sum" and n < 2:
            raise ValueError(
                f"InlineDerived({self.name!r}, op='sum'): requires at least two operands; got {n}."
            )
        return self


class SemanticQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    measures: list[str] = []
    dimensions: list[str] = []
    time_dimension: TimeWindow | None = None
    # Named, pre-defined predicates the planner references by name
    # (qualified as ``cube.segment``). Compose with ``filters`` via AND.
    segments: list[str] = []
    filters: list[Filter] = []
    # Boolean predicate tree — use when ``filters`` (implicit AND) isn't
    # expressive enough (OR / NOT). Composes with ``filters`` via AND.
    where: BoolExpr | None = None
    having: list[Filter] = []
    compare: CompareWindow | None = None
    order: list[tuple[str, Literal["asc", "desc"]]] = []
    limit: int | None = None
    offset: Annotated[int, Field(ge=0)] | None = None
    # Row-listing mode (no GROUP BY). Incompatible with `measures`.
    ungrouped: bool = False
    # Ad-hoc derived measures composed from existing catalog measures
    # — the exploratory case for ratios / sums / differences that
    # don't (yet) deserve a catalog entry. See :class:`InlineDerived`.
    derived_measures: list[InlineDerived] = []
    # Cube names that should be LEFT joined (rather than INNER joined)
    # along their declared edge. Pair with ``Filter(dimension=...,
    # op="is_null")`` to express anti-join / absent-row queries: rows
    # present in the FROM root but missing in the LEFT-joined cube.
    # Phase A: the left-joined cube can't be in ``dimensions`` (NULL
    # group keys are surprising); the compiler raises if it is.
    left_joins: list[str] = []

    @model_validator(mode="after")
    def _check_ungrouped_no_measures(self) -> SemanticQuery:
        if self.ungrouped and self.measures:
            raise ValueError(
                "ungrouped=True is incompatible with measures — measures "
                "imply aggregation, which ungrouped skips. Either drop "
                "the measures (row-listing mode) or set ungrouped=False "
                "(aggregated mode)."
            )
        return self

    def rewrite(self, op: object) -> SemanticQuery:
        """Apply a :class:`semql.RewriteOp` and return a new query.

        Convenience wrapper over :func:`semql.rewrite.rewrite`. ``op``
        is typed as ``object`` here because importing the RewriteOp
        union at module-load time would create a cycle
        (rewrite.py → spec.py); the implementation in
        :mod:`semql.rewrite` performs the dispatch and the static
        type-checkers see the precise op type through the function
        form ``rewrite(q, op)``."""
        from semql.rewrite import rewrite as _rewrite

        # Cast for mypy: the rewrite function is typed against the
        # closed-enum RewriteOp union. Runtime dispatches via isinstance.
        return _rewrite(self, op)  # type: ignore[arg-type]


class SavedQuery(BaseModel):
    """A pre-baked :class:`SemanticQuery` registered on a Catalog.

    Saved queries cover the "give me the standard quarterly revenue
    report" shape — the planner doesn't need to author one for a
    recurring ask. An MCP server auto-exposes each visible saved
    query as a zero-arg tool, so an LLM client just calls
    ``saved_<name>()``.

    Visibility follows ``required_roles`` the same ANY-match way
    ``Cube.required_roles`` does — a viewer with at least one matching
    role sees the query; an empty list means publicly accessible.

    ``description`` is surfaced to MCP clients as the tool docstring;
    a one-sentence "what this answers" is the right shape.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    query: SemanticQuery
    description: str = ""
    owner: str | None = None
    required_roles: list[str] = []


__all__ = [
    "BoolExpr",
    "CompareWindow",
    "Filter",
    "FilterOp",
    "InlineDerived",
    "InlineDerivedOp",
    "SavedQuery",
    "SemanticQuery",
    "TimeWindow",
]
