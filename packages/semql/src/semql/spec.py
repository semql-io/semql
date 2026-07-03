"""The query spec the planner emits and the compiler consumes.

All identifiers are *qualified* — `cube.field`. The compiler resolves
them against the catalog; unknown identifiers raise `CompileError`
naming the offending field.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ``parse_instant`` is a pure leaf utility — it lives in
# :mod:`semql.instant` so the catalog model (which needs it for
# time-partitioned source range checks) doesn't have to import
# the whole spec tree. Re-exported here for callers that imported
# it from ``semql.spec`` historically.
from semql._grounding import validate_keywords, validate_questions
from semql.instant import parse_instant
from semql.model import GranularityLiteral

if TYPE_CHECKING:
    # The cycle here is real: ``rewrite.py`` needs ``Filter`` /
    # ``TimeWindow`` as field types of the rewrite-op classes, and
    # the type-checker wants ``RewriteOp`` available where the
    # ``SemanticQuery.rewrite()`` method is declared. Both
    # directions cross, so neither module can top-level-import
    # the other without a lazy bridge. The runtime body of
    # ``rewrite()`` is a one-liner that delegates to the free
    # function imported lazily in the method body — see below.
    from semql.rewrite import RewriteOp


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
    dimension: str = Field(
        description=(
            "Qualified time-dimension name (e.g. 'orders.created_at') the window restricts."
        ),
    )
    granularity: GranularityLiteral | None = Field(
        default=None,
        description="Bucket size for time GROUP BY; required when fill_nulls_with is set.",
    )
    range: tuple[str, str] = Field(
        description=(
            "Half-open [start, end) ISO-8601 pair; emits `dim >= start AND dim < end`, "
            "compared by instant not text."
        ),
    )
    # When set, every measure in the query gets ``COALESCE(measure,
    # fill_nulls_with)`` and the result has one row per truncated time
    # bucket in ``range`` even where the underlying data has gaps.
    # Requires ``granularity``; rejected when the query has any
    # non-time dimensions (cartesian fill = Phase B).
    fill_nulls_with: int | None = Field(
        default=None,
        description=(
            "Constant returned for buckets with no rows; requires granularity, no non-time dims."
        ),
    )

    @model_validator(mode="after")
    def _check_range_order(self) -> TimeWindow:
        """Refuse a reversed range when both endpoints are real instants.

        Half-open ``[start, end)`` with ``start == end`` is an empty but
        legal window; ``start > end`` is almost always an authoring
        slip, so it is refused — but only when both endpoints parse as
        ISO-8601. Non-instant endpoints are left untouched: SemQL binds
        range values as parameters (the injection-safety model), so a
        malformed instant is the database's concern, not a construction
        error.
        """
        try:
            start, end = parse_instant(self.range[0]), parse_instant(self.range[1])
        except ValueError:
            return self
        if start > end:
            raise ValueError(
                f"TimeWindow.range is reversed: start {self.range[0]!r} is after "
                f"end {self.range[1]!r}. The range is half-open [start, end); use "
                "start <= end (equal endpoints denote an empty window)."
            )
        return self


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True)
    dimension: str = Field(
        description="Qualified dimension name (e.g. 'orders.region') the predicate applies to.",
    )
    op: FilterOp = Field(
        description=(
            "Operator: eq, neq, in, not_in, gt, lt, gte, lte, contains, is_null, not_null."
        ),
    )
    values: list[str | int | float | bool] = Field(
        default_factory=lambda: list[str | int | float | bool](),
        description=(
            "Predicate arguments — one for scalar ops, many for in/not_in, empty for is_null."
        ),
    )

    def validate_for_type(self, dim_type: str) -> None:
        """Compile-time type check. Raises ValueError on mismatch; the
        compiler re-raises as CompileError.

        >>> Filter(dimension="x", op="eq", values=["yes"]).validate_for_type("string")
        >>> Filter(dimension="x", op="eq", values=["yes"]).validate_for_type("bool")
        Traceback (most recent call last):
            ...
        ValueError: Filter on bool dimension 'x' got non-bool value 'yes'.
        """
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
            elif dim_type in ("time", "date"):
                if not isinstance(v, str):
                    raise ValueError(
                        f"Filter on {dim_type} dimension {self.dimension!r} got "
                        f"non-string value {v!r}."
                    )
                try:
                    datetime.fromisoformat(v)
                except ValueError:
                    raise ValueError(
                        f"Filter on {dim_type} dimension {self.dimension!r} got "
                        f"non-ISO-8601 value {v!r}."
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
    op: Literal["and", "or", "not"] = Field(
        description="Combinator: 'and' / 'or' (>=2 children) or 'not' (exactly one child).",
    )
    children: list[BoolExpr | Filter] = Field(
        description="Sub-expressions: nested BoolExpr nodes or Filter leaves.",
    )

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
    mode: Literal["previous_period", "explicit"] = Field(
        default="previous_period",
        description=(
            "'previous_period' derives prior from time_dimension.range; 'explicit' requires range."
        ),
    )
    range: tuple[str, str] | None = Field(
        default=None,
        description="Required for mode='explicit'; ignored for 'previous_period'.",
    )


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

    Operands may span cubes when the join graph connects them (C17):
    the compiler pulls each operand's cube into the FROM clause and
    validates the route is unambiguous and fan-safe. It refuses a
    derivation whose operand cube is unreachable, reachable two
    equal-cost ways (ambiguous — the value would depend on the route),
    or joined so that an additive operand fans out. Stable cross-cube
    metrics still belong in the catalog; this is the ad-hoc escape
    hatch.
    """

    model_config = ConfigDict(frozen=True)
    name: str = Field(
        description="Output column name for the derived measure.",
    )
    op: InlineDerivedOp = Field(
        description="Composition: 'ratio' (num/den), 'sum' (a+b+...), or 'diff' (a-b).",
    )
    operands: list[str] = Field(
        description=(
            "Qualified measure refs ('cube.measure'); may span cubes the "
            "join graph connects unambiguously and fan-safely."
        ),
    )
    dependencies: tuple[str, ...] = Field(
        default=(),
        description=(
            "Extra cube names to pull into the join graph for this "
            "derivation — e.g. a bridge cube linking the operand cubes "
            "that is not itself an operand. Operand cubes are joined "
            "automatically; this covers intermediaries they need."
        ),
    )

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


class SemiJoin(BaseModel):
    """Cross-backend semi-join: restrict an outer dimension to the value
    set produced by an inner query, shipped as a literal value list — never
    a cross-backend join.

    The inner ``source`` runs first on its own backend(s); its ``select``
    column is projected to a de-duplicated list and spliced into the outer
    query as an ``in`` / ``not_in`` ``Filter`` on ``dimension``. This is the
    answer to "metric X (backend A) restricted to the entities matching
    condition Y (backend B)" without forcing a federated join.

    AND-composed with the outer query's ``filters`` / ``where``. v1 is
    AND-only: a semi-join cannot be OR'd or NOT'd against other predicates
    (use ``op='not_in'`` for negation). The ``source`` may itself span
    backends, but must not contain its own ``semi_joins`` (no nesting in v1).

    The outer ``dimension`` and the inner ``select`` dimension must have
    compatible declared types — same coercion rule as a cross-backend bridge
    join (opt in via ``Dimension.coerce_to``). Otherwise the compiler refuses
    rather than silently coercing the value list.
    """

    model_config = ConfigDict(frozen=True)
    dimension: str = Field(
        description="Qualified outer dimension to constrain, e.g. 'activity.employee_id'.",
    )
    op: Literal["in", "not_in"] = Field(
        default="in",
        description="Membership test of the outer dimension against the inner value list.",
    )
    select: str = Field(
        description="Qualified inner dimension whose values form the list, e.g. 'employees.id'.",
    )
    source: SemanticQuery = Field(
        description="Inner query producing the value list; must project ``select`` as a dimension.",
    )

    @model_validator(mode="after")
    def _check(self) -> SemiJoin:
        if "." not in self.dimension:
            raise ValueError(
                f"SemiJoin.dimension must be a qualified 'cube.field' ref; got {self.dimension!r}."
            )
        if "." not in self.select:
            raise ValueError(
                f"SemiJoin.select must be a qualified 'cube.field' ref; got {self.select!r}."
            )
        if self.select not in self.source.dimensions:
            raise ValueError(
                f"SemiJoin.select {self.select!r} must be one of the inner query's "
                f"dimensions; got {self.source.dimensions!r}."
            )
        if self.source.semi_joins:
            raise ValueError(
                "SemiJoin.source must not itself contain semi_joins (v1 supports no nesting)."
            )
        return self


class SemanticQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    measures: list[str] = Field(
        default_factory=list,
        description=(
            "Qualified measure refs to aggregate (e.g. 'orders.revenue'); each grouped per its agg."
        ),
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Qualified dimension refs to group by (e.g. 'orders.region').",
    )
    time_dimension: TimeWindow | None = Field(
        default=None,
        description=(
            "Time window restricting rows; supplies the time bucket when granularity is set."
        ),
    )
    segments: list[str] = Field(
        default_factory=list,
        description=("Named pre-defined predicates as 'cube.segment'; AND-composed with filters."),
    )
    filters: list[Filter] = Field(
        default_factory=lambda: list[Filter](),
        description="Flat list of predicates; combined with implicit AND.",
    )
    where: BoolExpr | None = Field(
        default=None,
        description=(
            "Boolean predicate tree (use for OR / NOT); AND-composed with the filters list."
        ),
    )
    having: list[Filter] = Field(
        default_factory=lambda: list[Filter](),
        description=(
            "Post-aggregation filters on measure values; references a measure also in 'measures'."
        ),
    )
    compare: CompareWindow | None = Field(
        default=None,
        description=(
            "Adds <measure>_current / <measure>_prior / _delta / _pct_change columns per measure."
        ),
    )
    order: list[tuple[str, Literal["asc", "desc"]]] = Field(
        default_factory=lambda: list[tuple[str, Literal["asc", "desc"]]](),
        description="(field, 'asc'|'desc') pairs; field may be a measure or dimension name.",
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Max rows returned; required when ungrouped=True, applied after aggregation otherwise."
        ),
    )
    offset: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="Skip this many rows after ordering; pairs with limit for offset pagination.",
    )
    ungrouped: bool = Field(
        default=False,
        description=(
            "Row-listing mode (no GROUP BY); incompatible with measures; needs explicit limit."
        ),
    )
    derived_measures: list[InlineDerived] = Field(
        default_factory=lambda: list[InlineDerived](),
        description=(
            "Ad-hoc derived measures (ratio/sum/diff) composed inline from catalog measures."
        ),
    )
    semi_joins: list[SemiJoin] = Field(
        default_factory=lambda: list[SemiJoin](),
        description=(
            "Cross-backend semi-joins: restrict a dimension to the value set of an inner "
            "query, shipped as a value list (no join). AND-composed with filters/where."
        ),
    )
    left_joins: list[str] = Field(
        default_factory=list,
        description=(
            "Cubes to LEFT JOIN instead of INNER; pair with op='is_null' for absent-row queries."
        ),
    )
    # Output column aliases. Maps the output column name
    # (alias key) to a qualified field ref (alias value). Useful
    # when a dashboard needs the same field under two names in one
    # result. The alias key is the output column name; the alias
    # value is what the resolver would have returned. Alias keys
    # must be unique (Pydantic dict semantics), must resolve to a
    # declared field, and must not collide with existing output
    # column names (compile error).
    aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Output column aliases; ``{output_name: qualified_ref}``. "
            "The output column name is the alias key."
        ),
    )

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

    @classmethod
    def tool_json_schema(cls) -> dict[str, Any]:
        """Object-rooted JSON schema for LLM tool-calling parameters.

        ``model_json_schema()`` emits a root ``$ref`` because this model is
        recursive (``semi_joins[].source`` is itself a ``SemanticQuery``), and
        OpenAI / Anthropic / Bedrock tool-calling all require an object root.
        This returns the flattened, object-rooted form — keeping ``$defs`` and
        the recursive internal refs intact. Prefer this over a raw
        ``model_json_schema()`` anywhere the schema is shipped as a tool spec.
        """
        from semql._schema import flatten_root_ref

        return flatten_root_ref(cls.model_json_schema())

    def rewrite(self, op: RewriteOp) -> SemanticQuery:
        """Apply a :class:`semql.rewrite.RewriteOp` and return a new query.

        Convenience wrapper over the free function :func:`semql.rewrite.rewrite`.
        The two forms are equivalent — kept as sugar for chat-style
        ``q = q.rewrite(Drilldown(...))`` chains. Implementation lives
        in :mod:`semql.rewrite` so the dispatch table is in one place.

        Note: this import is intentionally lazy. ``spec`` and
        ``rewrite`` form a true module cycle (rewrite's op classes
        carry ``Filter`` / ``TimeWindow`` fields; spec's
        ``SemanticQuery.rewrite`` calls back into ``rewrite``).
        Either direction can be top-level — not both. The method
        body is the one place we reach across, and lazy import
        inside a method body is the standard Python idiom.
        """
        from semql.rewrite import rewrite as _rewrite

        return _rewrite(self, op)


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

    ``questions`` / ``keywords`` / ``purpose`` are LLM-grounding
    metadata. The planner / router indexes saved queries by
    these surfaces independently from cubes.

    ``stability`` / ``replacement`` mirror :attr:`Cube.stability` —
    ``deprecated`` causes the compiler to refuse the saved query.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    query: SemanticQuery
    description: str = ""
    owner: str | None = None
    required_roles: list[str] = []
    # Fully-baked NL questions this saved query answers — same shape
    # rules as ``Cube.questions``. Surface in the MCP tool description
    # so external agents pick the right saved query by capability.
    questions: list[str] = []
    # Free-text search tokens — same acronym-preserving normalisation
    # as ``Cube.keywords``.
    keywords: list[str] = []
    # One-line "why this exists": "operational dashboard", "weekly
    # ops report", "on-call latency check". Free-form.
    purpose: str = ""
    # Lifecycle tier — same semantics as ``Cube.stability``. The
    # compiler refuses to materialise a saved query that resolves to
    # a ``deprecated`` cube (downstream of cube lifecycle).
    stability: Literal["stable", "beta", "deprecated"] = "stable"
    # Successor pointer surfaced in the deprecation error message.
    replacement: str | None = None

    @model_validator(mode="after")
    def _check_grounding(self) -> SavedQuery:
        validate_questions("SavedQuery", self.name, self.questions)
        normalised = validate_keywords("SavedQuery", self.name, self.keywords)
        # Frozen model — bypass via object.__setattr__ since
        # model_copy(update=...) inside an after-validator would
        # re-trigger validation and recurse.
        if normalised != self.keywords:
            object.__setattr__(self, "keywords", normalised)
        if self.stability != "deprecated" and self.replacement is not None:
            raise ValueError(
                f"SavedQuery {self.name!r}: ``replacement`` may only "
                f"be set when ``stability='deprecated'`` (got stability="
                f"{self.stability!r})."
            )
        return self


# ``SemiJoin.source`` and ``SemanticQuery.semi_joins`` are mutually
# recursive; rebuild the forward ref now that both classes exist.
SemiJoin.model_rebuild()


class SemanticQueryDefaults(BaseModel):
    """Declarative compile defaults applied before compile hooks.

    Merge priority (highest wins): query's own value > per-call
    ``query_defaults`` > catalog ``query_defaults`` > ``None`` (no fill).

    A ``None`` field means "don't fill" — it never overrides an explicit
    ``None`` on the query. Pass ``SemanticQueryDefaults()`` (all-None)
    to opt in without filling anything; omit to get current behaviour.
    """

    model_config = ConfigDict(frozen=True)
    limit: int | None = None
    time_window: TimeWindow | None = None
    granularity: GranularityLiteral | None = None


def _apply_query_defaults(
    query: SemanticQuery,
    defaults: SemanticQueryDefaults | None,
) -> SemanticQuery:
    """Return a new SemanticQuery with defaults filled in where the
    query itself has no value. ``query`` is never mutated."""
    if defaults is None:
        return query
    updates: dict[str, object] = {}

    if defaults.limit is not None and query.limit is None:
        updates["limit"] = defaults.limit

    if defaults.time_window is not None and query.time_dimension is None:
        updates["time_dimension"] = defaults.time_window

    if (
        defaults.granularity is not None
        and query.time_dimension is not None
        and query.time_dimension.granularity is None
    ):
        new_td = query.time_dimension.model_copy(update={"granularity": defaults.granularity})
        updates["time_dimension"] = new_td

    if not updates:
        return query
    return query.model_copy(update=updates)


__all__ = [
    "BoolExpr",
    "CompareWindow",
    "Filter",
    "FilterOp",
    "InlineDerived",
    "InlineDerivedOp",
    "SavedQuery",
    "SemanticQuery",
    "SemanticQueryDefaults",
    "SemiJoin",
    "TimeWindow",
    "_apply_query_defaults",
]
