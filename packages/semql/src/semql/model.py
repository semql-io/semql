"""Type definitions for the semantic catalog.

A `Cube` declares one logical table: where its rows live (`backend`,
`table`), the always-on predicate that defines membership
(`base_predicate`), the measures/dimensions/time-dimensions exposed,
and the join edges to other cubes.

`expose_in_prompt` controls whether `render_catalog_block` includes
the cube in the system-prompt fragment shown to the planner. The
catalog is intentionally wider than the prompt — every cube the
compiler accepts doesn't need to be in the planner's vocabulary. Cubes
flagged `False` are reachable via joins from exposed cubes and still
compile cleanly when the planner names them; they just don't appear in
the catalog rendering.

`metadata` is a user-owned escape hatch (k8s-annotation flavoured): an
opaque ``dict[str, str]`` SemQL never reads, validates, or surfaces.
Callers can stash ownership tags, lineage IDs, presentation hints —
anything the platform shouldn't know about. Round-trips through
``model_copy``, ``model_dump`` / ``model_validate``, and serialisation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semql._grounding import (
    validate_keywords as _grounding_validate_keywords,
)
from semql._grounding import (
    validate_questions as _grounding_validate_questions,
)
from semql._grounding import (
    validate_relations as _grounding_validate_relations,
)
from semql.spec import parse_instant as _parse_instant


class Backend(StrEnum):
    POSTGRES = "postgres"
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    META = "meta"  # reflection over the catalog itself; see introspect.py


class Provenance(StrEnum):
    """How much a projected output column can be trusted (C3).

    Lets downstream consumers (MCP tools, the presenter / drilldown prompt
    roles) distinguish a value that came from an approved measure
    definition from one the planner composed ad hoc."""

    VERIFIED = "verified"  # a catalog-defined measure
    COMPOSED = "composed"  # a derived / inline (ad-hoc) measure expression
    DIMENSION = "dimension"  # a raw dimension / time-bucket column


StabilityLiteral = Literal["stable", "beta", "deprecated"]
"""Lifecycle hint for a Cube / SavedQuery. ``deprecated`` is refused by
the compiler (see S7 PRD); ``beta`` flows through with an annotation."""

AggLiteral = Literal[
    "sum",
    "count",
    "count_distinct",
    "avg",
    "min",
    "max",
    "ratio",
    # Quantile aggregations — non-distributive. ``median`` is the q=0.5
    # case; ``p75`` / ``p90`` / ``p95`` cover the long-tail diagnostic
    # cases. Pair with ``non_additive=True`` so callers don't naively
    # sum a median across grains.
    "median",
    "p75",
    "p90",
    "p95",
]
DimTypeLiteral = Literal["string", "number", "time", "bool", "uuid"]
DimCategoryLiteral = Literal["identity", "status", "pii", "metadata"]
# Resolved storage type carried on ``ColumnMeta`` so callers (visualisers,
# tool-schema renderers, downstream tooling) can reason about output
# columns without re-resolving against the catalog. Distinct from
# ``DimTypeLiteral`` — adds ``integer`` / ``float`` for measures, where
# the aggregate determines a tighter type than the raw column.
StorageType = Literal[
    "string",
    "integer",
    "float",
    "number",
    "time",
    "uuid",
    "bool",
]
GranularityLiteral = Literal["hour", "day", "week", "month"]
FormatLiteral = Literal["raw", "integer", "percent", "currency", "duration"]
ChartTypeLiteral = Literal["pie_chart", "bar_chart", "line_chart", "data_table"]

# Per-cube tenant isolation strategy:
# - schema       : tenant lives in its own DB schema; the cube's ``table``
#                  contains ``{tenant_schema}`` and the compile-time
#                  ``context`` substitutes it.
# - discriminator: a shared physical table; the compiler wraps the
#                  source in a subquery with ``WHERE <tenancy_column>
#                  = bind(tenant_value)`` inside the alias so outer
#                  ``OR`` predicates can't cross tenants.
# - none         : no tenant boundary (META reflection cubes, public
#                  lookups). The compiler emits no tenancy predicate.
TenancyMode = Literal["schema", "discriminator", "none"]

# k8s-annotation style: opaque string→string map. SemQL never touches
# the contents — the user owns the namespace and meaning.
Metadata = dict[str, str]


def _freeze(value: object) -> object:
    """Recursively turn a field value into a hashable, order-normalised key.

    Mirrors :class:`_HashableModel`'s contract: equal field values produce
    equal frozen keys, so equal models hash equal. ``dict`` items are
    sorted by key (insertion order is not part of value identity); ``set``
    becomes a ``frozenset``; nested models recurse through their fields."""
    if isinstance(value, BaseModel):
        return (
            type(value).__name__,
            tuple(_freeze(getattr(value, n)) for n in type(value).model_fields),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


class _HashableModel(BaseModel):
    """Base for frozen catalog models that restores the frozen→hashable
    contract.

    ``ConfigDict(frozen=True)`` is meant to make a model hashable, but
    Pydantic's generated ``__hash__`` hashes the raw field tuple and so
    raises ``unhashable type: 'list'`` / ``'dict'`` for any model carrying
    a collection field — which is most of them. We override ``__hash__``
    with a recursive value-based hash (:func:`_freeze`) so a ``Measure``
    can go in a ``set`` and a ``Join`` can be a dict key. Pydantic respects
    an inherited ``__hash__`` (it does not regenerate one when the subclass
    redeclares ``frozen=True``), and equality stays Pydantic's field-wise
    ``__eq__`` — so equal models still hash equal."""

    def __hash__(self) -> int:
        return hash(
            (type(self).__name__, tuple(_freeze(getattr(self, n)) for n in type(self).model_fields))
        )


class BaseField(_HashableModel):
    """Shared supertype for catalog fields.

    Every named, addressable artefact on a cube — measures, dimensions,
    time dimensions, segments — carries the same identity / presentation
    fields: a stable machine ``name``, the underlying ``sql`` fragment
    (``{alias}``-templated), a human-readable ``description``, an
    optional ``display_name`` for the prompt fragment, and an opaque
    ``metadata`` map for caller-owned tags.

    Subclasses add the type-specific surface (``agg`` on Measure,
    ``type`` on Dimension, ``granularities`` on TimeDimension).
    ``isinstance(fld, Measure)`` etc. still narrows correctly — the
    base type is a structural convenience, not a discriminator.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    sql: str
    description: str = ""
    display_name: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    # A1 — Field-level visibility. ANY-match semantics: a viewer with
    # at least one of the listed roles passes; an empty list means
    # the field is open to all viewers who can see the cube. Compiled
    # errors for missing roles are indistinguishable from "field
    # doesn't exist" so callers can't infer the field exists.
    required_roles: list[str] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        """Structural field-type tag, the SemQL equivalent of GraphQL ``__typename``.

        I15: ``"measure"`` / ``"dimension"`` / ``"time"`` / ``"segment"``.
        Lets consumers (MCP tool factory, planner, evaluator) branch on
        field type without importing the concrete ``Measure`` /
        ``Dimension`` / ``TimeDimension`` / ``Segment`` subclasses. Pairs
        with ``ColumnMeta.kind`` on ``CompiledQuery`` which uses the same
        vocabulary for output columns. Implemented as a property so it
        survives ``model_copy`` (the underlying class is unchanged).
        """
        from semql.model import Dimension, Measure, Segment, TimeDimension

        if isinstance(self, Measure):
            return "measure"
        if isinstance(self, TimeDimension):
            return "time"
        if isinstance(self, Segment):
            return "segment"
        if isinstance(self, Dimension):
            return "dimension"
        raise TypeError(
            f"Unknown BaseField subclass: {type(self).__name__}; "
            "extend BaseField.kind to add support."
        )


class Measure(BaseField):
    agg: AggLiteral
    # ``unit`` is the unit the column STORES (e.g. ``"seconds"``,
    # ``"bytes"``, ``"currency"``). ``display_unit`` is the unit a UI
    # should SHOW the value in (e.g. ``"hours"``). Splitting the two
    # is the unum-style separation between dimensional truth and
    # presentation: the compiler ignores both, downstream visualisers
    # call ``semql.units.convert`` to translate. Leave ``display_unit``
    # ``None`` to render in the storage unit unchanged.
    unit: str | None = None
    display_unit: str | None = None
    format: FormatLiteral | None = None
    # When True, summing this measure across a coarser time grain
    # yields a wrong answer (the canonical case is ``count_distinct``).
    # The flag is declarative for now — surfaced in the prompt fragment
    # so the planner LLM doesn't naively ask for rollups; compiler
    # refusal lands with the rollup work.
    non_additive: bool = False
    # Optional row-level predicate scoping this measure's aggregation:
    # ``COUNT(*) FILTER (WHERE <filter>)``. Same ``{alias}`` placeholder
    # convention as ``Segment.sql`` / ``base_predicate`` / ``Join.on``.
    # Lets one query ask "paid revenue vs pending revenue" without
    # three round-trips. sqlglot renders FILTER natively on PG / CH /
    # DuckDB / BigQuery and transpiles to ``COUNT(IFF(...))`` on
    # Snowflake.
    filter: str | None = None
    # For ``agg='ratio'`` only: the names of two other measures on the
    # *same cube* whose aggregates compose the ratio. The compiler emits
    # ``<num_agg> / NULLIF(<den_agg>, 0)``; the ``sql`` field is ignored
    # for ratio measures (set it to ``""`` by convention). Composes with
    # filtered measures — a filtered ratio is just a ratio of two
    # filtered sums.
    numerator: str | None = None
    denominator: str | None = None
    # A1 — Field-level masking. ``mask_roles`` names roles for which
    # the field compiles but the SQL is substituted by a constant
    # instead of the real expression. ``mask_value``: ``None`` → NULL,
    # a string → a SQL literal (``'REDACTED'``, ``'0'``, etc.). The
    # constructor enforces ``mask_roles ⊆ required_roles`` — you
    # can't mask a role that can't even see the field.
    mask_roles: list[str] = Field(default_factory=list)
    mask_value: str | None = None

    @model_validator(mode="after")
    def _check_ratio_consistency(self) -> Measure:
        if self.agg == "ratio":
            if not self.numerator or not self.denominator:
                raise ValueError(
                    f"Measure {self.name!r}: agg='ratio' requires both "
                    "numerator and denominator (names of other measures "
                    "on the same cube)."
                )
        else:
            if self.numerator is not None or self.denominator is not None:
                raise ValueError(
                    f"Measure {self.name!r}: numerator/denominator are "
                    f"only valid for agg='ratio'; got agg={self.agg!r}."
                )
        return self

    @model_validator(mode="after")
    def _check_mask_subset(self) -> Measure:
        if self.mask_roles and not set(self.mask_roles).issubset(set(self.required_roles)):
            raise ValueError(
                f"Measure {self.name!r}: mask_roles={self.mask_roles!r} "
                f"is not a subset of required_roles={self.required_roles!r}. "
                "A field that masks a role that can't even access the field "
                "is a configuration error."
            )
        return self


class Dimension(BaseField):
    type: DimTypeLiteral
    # Presentation hints — mirror ``Measure.unit`` / ``Measure.display_unit``
    # / ``Measure.format``. The compiler ignores all three; downstream
    # visualisers (charts, tables) use them to format the rendered value.
    # Example: a duration_seconds dimension might set ``unit="seconds"``,
    # ``display_unit="hours"``, ``format="duration"`` so a tabular cell
    # renders as "1.38 h" instead of "4980".
    unit: str | None = None
    display_unit: str | None = None
    format: FormatLiteral | None = None
    # Names another cube whose ``primary_key`` this dimension references.
    # The Catalog auto-derives a ``many_to_one`` Join from this cube's
    # FK to the named cube's PK — saving the catalog author the
    # repetition. An explicit Join with the same ``to`` wins.
    foreign_key: str | None = None
    # A1 — Field-level masking. Same shape as Measure.mask_*.
    mask_roles: list[str] = Field(default_factory=list)
    mask_value: str | None = None
    # I7 — Input aliases. An LLM might emit ``territory`` /
    # ``zone`` / ``area`` when the catalog names the dimension
    # ``region``. The resolver accepts any alias as a synonym for
    # the canonical name; the prompt renders the canonical name
    # only (no alias listing — the planner learns the canonical).
    # The field-hide / mask gates (A1) apply on the canonical
    # field; an alias is a synonym, not a separate auth surface.
    aliases: list[str] = Field(default_factory=list)
    # I10 — Cross-cube coercion opt-in. A federated bridge join whose two
    # keys have different ``type`` is refused (FederationError) rather
    # than silently coerced. ``coerce_to`` declares an *additional* type
    # this dimension is willing to be compared as, so a key of one type
    # can deliberately join a key of another. The comparison is allowed
    # when the two keys share at least one acceptable type
    # (own ``type`` ∪ ``coerce_to``). ``None`` = no coercion.
    coerce_to: DimTypeLiteral | None = None

    @model_validator(mode="after")
    def _check_coerce_to(self) -> Dimension:
        if self.coerce_to is not None and self.coerce_to == self.type:
            raise ValueError(
                f"Dimension {self.name!r}: coerce_to={self.coerce_to!r} equals the "
                f"dimension's own type, so it coerces nothing. Drop it, or set it to "
                f"the other type this key must compare against."
            )
        return self

    @model_validator(mode="after")
    def _check_mask_subset(self) -> Dimension:
        if self.mask_roles and not set(self.mask_roles).issubset(set(self.required_roles)):
            raise ValueError(
                f"Dimension {self.name!r}: mask_roles={self.mask_roles!r} "
                f"is not a subset of required_roles={self.required_roles!r}. "
                "A field that masks a role that can't even access the field "
                "is a configuration error."
            )
        return self


class TimeDimension(BaseField):
    """Time dimensions are separated so the compiler can apply
    `granularity` truncation when a query asks for it (hourly/daily/etc.
    rollups). Otherwise they're just dimensions of type `time`."""

    type: Literal["time"] = "time"
    granularities: tuple[GranularityLiteral, ...] = ("hour", "day", "week", "month")


class Segment(BaseField):
    """A named, reusable predicate over a cube's rows.

    Segments centralise business definitions ("active customers",
    "paid orders", "trial-tier accounts") so the planner names the
    segment instead of re-deriving the predicate. The compiler
    AND-composes referenced segments into the WHERE clause alongside
    ``filters``.

    The ``sql`` fragment uses the same ``{alias}`` placeholder
    convention as dimension / measure SQL — ``{o}.status = 'paid'``
    resolves to ``o.status = 'paid'`` at compile time."""


class Rollup(_HashableModel):
    """Pre-aggregated rollup table for a Cube.

    Declares a materialised table holding rows pre-grouped by a subset
    of the cube's dimensions and (optionally) a time-bucket dimension.
    The compiler matches a query whose dimensions / measures / time
    granularity match the rollup's grain and routes the SQL against
    ``physical_table`` instead of the cube's base table — a substantial
    speed-up on large fact tables.

    Phase-1 matching is exact-grain: every requested dim must appear
    in ``dimensions``, every requested measure in ``measures``, the
    query's time granularity (if any) must equal ``granularity``, and
    every filter must touch only stored dims / the time column. The
    compiler picks the smallest matching rollup; if none match, the
    base table is used.

    Column-naming convention: every stored column is named after its
    source field's local name. ``revenue`` (a SUM measure) is stored
    as a column literally named ``revenue``; ``region`` as ``region``;
    a ``started_at`` time dim materialised at ``day`` granularity is
    ``started_at_day``. The compiler relies on this naming.

    Measure-agg constraint: only re-aggregatable distributive aggs are
    allowed at registration (``sum``, ``count``, ``min``, ``max``).
    ``avg`` / ``ratio`` / ``count_distinct`` / percentile aggs are
    refused — re-aggregating them across a rollup grain is either
    wrong (avg-of-avgs) or impossible without storing additional
    state (sketches / decomposed components)."""

    model_config = ConfigDict(frozen=True)
    name: str
    physical_table: str
    alias: str = "r"
    dimensions: list[str] = []
    time_dimension: str | None = None
    granularity: GranularityLiteral | None = None
    measures: list[str] = []
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_time_pair(self) -> Rollup:
        if (self.time_dimension is None) != (self.granularity is None):
            raise ValueError(
                f"Rollup {self.name!r}: ``time_dimension`` and "
                "``granularity`` must both be set or both be unset — "
                "a time dim without a bucket grain is meaningless, "
                "and a grain without a time dim has no column to "
                "address."
            )
        return self


class GlossaryEntry(_HashableModel):
    """One catalog-wide vocabulary term + its definition + spelling aliases.

    Entries live on ``Catalog.glossary``. The retrieval index indexes
    each alias as its own document pointing at the same canonical
    definition — a misspelled term ("ARR" / "annual recurring revenue"
    / "yearly subscription revenue") still resolves to the same entry.
    Aliases get a separate vector / FTS5 row each; ``term`` and
    ``definition`` are combined into the entry's primary document.

    ``aliases`` may not contain empty strings; ``term`` may not be empty.
    """

    model_config = ConfigDict(frozen=True)
    term: str
    definition: str
    aliases: list[str] = []

    @model_validator(mode="after")
    def _check_shape(self) -> GlossaryEntry:
        if not self.term.strip():
            raise ValueError("GlossaryEntry.term cannot be empty.")
        if not self.definition.strip():
            raise ValueError(f"GlossaryEntry({self.term!r}): definition cannot be empty.")
        for a in self.aliases:
            if not a.strip():
                raise ValueError(
                    f"GlossaryEntry({self.term!r}): aliases cannot contain empty strings."
                )
        return self


class Join(_HashableModel):
    """A directed edge from one cube to another.

    `on` is a SQL fragment using `{alias}` placeholders that the
    compiler resolves to actual table aliases at compile time."""

    model_config = ConfigDict(frozen=True)
    to: str
    relationship: Literal["one_to_one", "one_to_many", "many_to_one"]
    on: str
    metadata: Metadata = Field(default_factory=dict)


class PhysicalTable(BaseModel):
    """A plain ``[schema.]table`` reference.

    Identical in meaning to the legacy ``Cube.table`` shorthand; goes
    through the same ``{alias}`` / ``{tenant_schema}`` / context
    substitution convention at compile time. Exists so ``Cube.source``
    can be a discriminated union with :class:`DerivedTable`."""

    model_config = ConfigDict(frozen=True)
    table: str


class NamedCTE(BaseModel):
    """A named CTE the compiler hoists into the outer ``WITH`` clause.

    A ``DerivedTable`` whose ``sql`` is a layered preamble can declare
    each CTE separately here, with the cube's main ``sql`` referencing
    them by name. The compiler collects every cube's CTEs across the
    query, deduplicates by ``name``, and emits a single ``WITH`` at the
    top of the final SELECT — so two cubes that share a CTE name MUST
    declare it identically (same SQL) or the catalog raises at
    construction time.

    ``sql`` uses the same placeholder convention as
    :attr:`DerivedTable.sql`: ``{tenant_schema}`` and ``{key}``
    substitution flow through ``resolve_sql`` at compile time."""

    model_config = ConfigDict(frozen=True)
    name: str
    sql: str


class DerivedTable(_HashableModel):
    """A cube whose physical source is a SQL expression, not a table.

    Emitted as ``(<sql>) AS <alias>`` inside the cube's isolation
    subquery — so ``tenancy`` / ``security_sql`` / ``scope`` wrappers
    apply over the derived rows the same way they apply over a real
    table.

    ``sql`` uses the same placeholder convention as ``Cube.table``:
    ``{tenant_schema}`` for schema-tenancy substitution and ``{key}``
    for compile-context substitution.

    ``with_ctes`` declares a preamble of named CTEs that get hoisted to
    a single outer ``WITH`` clause at compile time. The cube's main
    ``sql`` references them by bare name; the compiler resolves them.
    CTE names are global within a compiled query — the catalog enforces
    uniqueness across all cubes.

    DerivedTable is the second place raw SQL legitimately enters the
    catalog (the first is ``Measure.sql``). The compiler surfaces the
    resolved SQL of both the main ``sql`` and every CTE on
    :attr:`semql.compile.CompiledQuery.derived_sources` so static checks
    (``is_safe_select``, dialect snapshots) cover every raw fragment,
    not just the outer SELECT."""

    model_config = ConfigDict(frozen=True)
    sql: str
    with_ctes: list[NamedCTE] = []

    @model_validator(mode="after")
    def _check_unique_cte_names(self) -> DerivedTable:
        seen: set[str] = set()
        for cte in self.with_ctes:
            if cte.name in seen:
                raise ValueError(
                    f"DerivedTable: duplicate CTE name {cte.name!r} in "
                    "with_ctes. Each CTE within a derived source must "
                    "have a unique name."
                )
            seen.add(cte.name)
        return self


CubeSource = PhysicalTable | DerivedTable


class TimePartition(BaseModel):
    """Declares which ``TimeDimension`` on the cube drives source
    selection when the cube has multiple ``TimePartitionedSource``s.

    The same physical layout is reused by every source's half-open
    range; each source binds its own bounds. Routing is unambiguous
    — there is one routing dim per cube.

    Set this together with :attr:`Cube.physical_sources`; setting it
    alone is a configuration error."""

    model_config = ConfigDict(frozen=True)
    time_dimension: str


class TimePartitionedSource(_HashableModel):
    """One physical table in a cube's time-partitioned source set.

    A cube that historically has "old" and "new" tables (e.g. a
    monthly-rolled-up archive table and a per-row live table) declares
    one ``TimePartitionedSource`` per physical table. The compiler
    intersects the query's ``TimeWindow.range`` with each source's
    half-open range and ``UNION ALL``s the matches — the auth /
    tenancy / scope wrappers then wrap the union as a whole, so a
    single outer ``OR`` predicate can't bypass the per-source
    boundaries.

    ``range_start`` / ``range_end`` are ISO-8601 lower / upper
    bounds; ``None`` on a side means open-ended in that direction.
    Two sources MAY overlap (the compiler unions the overlap);
    ``range_start >= range_end`` is a construction error. Sources
    are matched by the cube's :class:`TimePartition` on the routing
    time dimension.

    ``column_renames`` maps a logical field name (as declared on
    the cube) to this source's physical column. The compiler emits
    the source as a ``SELECT <physical> AS <logical>, ... FROM
    <table>`` subquery, so the outer query and the planner see
    only logical names. The cube's field-level column map is the
    source of truth; missing renames fall back to the logical name
    verbatim.

    The single-source shorthand (``Cube.table="t"``) is unaffected
    — ``physical_sources`` is a strict opt-in."""

    model_config = ConfigDict(frozen=True)
    name: str
    table: str
    alias: str = "s"
    range_start: str | None = None
    range_end: str | None = None
    column_renames: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_range_ordering(self) -> TimePartitionedSource:
        if (
            self.range_start is not None
            and self.range_end is not None
            and not _parse_instant(self.range_start) < _parse_instant(self.range_end)
        ):
            raise ValueError(
                f"TimePartitionedSource {self.name!r}: range_start "
                f"{self.range_start!r} must be strictly less than "
                f"range_end {self.range_end!r} (compared by instant, not "
                "text). Half-open intervals with start == end are empty."
            )
        return self


class PartitionedScan(BaseModel):
    """The routing metadata for a time-partitioned cube after the
    plan→plan ``apply_partition_to_plan`` transform.

    A synthetic :class:`Cube` whose ``partitioned_scan`` is set
    represents "this cube is read through the matched physical
    sources, not through ``table`` / ``source``." The compiler's
    ``_from_clause_stage`` checks for a non-None ``partitioned_scan``
    and emits the unioned subquery in place of the cube's normal
    table reference.

    ``sources`` is the (sub)set of :class:`TimePartitionedSource` the
    query's ``TimeWindow.range`` intersected with. ``is_empty`` is a
    shortcut for the case where no source matched: the plan
    lowers to a zero-row subquery (``SELECT 1 WHERE FALSE``) and
    the outer query's predicates still apply, so the result is
    empty by construction.

    Lives on the cube value rather than as a free-floating IR node
    because the synthetic cube is what the existing emission path
    reads (``_from_clause_stage`` already dispatches on cube-level
    metadata for the ``physical_sources`` case). Putting
    ``PartitionedScan`` on the cube keeps the IR's ``scans`` list
    a flat list of :class:`Scan` nodes — a tagged-union surface
    would be a bigger contract change for marginal benefit.
    """

    model_config = ConfigDict(frozen=True)
    sources: tuple[TimePartitionedSource, ...] = ()
    is_empty: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> PartitionedScan:
        if self.is_empty and self.sources:
            raise ValueError(
                "PartitionedScan: is_empty=True and sources non-empty is "
                "self-contradictory. Set is_empty=False when the matched "
                "source list is non-empty."
            )
        return self


class Cube(BaseModel):
    name: str
    backend: Backend
    # Shorthand for a plain-table source: ``Cube(table="schema.t", ...)``
    # is equivalent to ``Cube(source=PhysicalTable(table="schema.t"), ...)``.
    # Exactly one of ``table`` / ``source`` must be specified; mixing a
    # non-empty ``table`` with a ``DerivedTable`` source raises.
    table: str = ""
    # Explicit source spec. Set this (instead of ``table``) when the cube's
    # physical rows come from a SQL expression rather than a real table —
    # e.g. a layered CTE preamble that surfaces derived columns the cube
    # then exposes as dimensions / measures. See :class:`DerivedTable`.
    source: CubeSource | None = None
    alias: str
    base_predicate: str | None = None
    measures: list[Measure] = []
    dimensions: list[Dimension] = []
    time_dimensions: list[TimeDimension] = []
    joins: list[Join] = []
    # Named, reusable predicates the planner can reference by name —
    # centralises business definitions instead of having the LLM
    # rederive a status / window / role filter every turn.
    segments: list[Segment] = []
    # Dimensions on this cube that MUST appear in a query's `filters`
    # (any operator, any value) before the compiler will accept the
    # query.
    required_filters: list[str] = []
    expose_in_prompt: bool = True
    description: str = ""
    display_name: str | None = None
    default_chart_type: ChartTypeLiteral | None = None
    metadata: Metadata = Field(default_factory=dict)
    # Tenant isolation strategy — see the ``TenancyMode`` docstring.
    # Defaults to ``"schema"`` so existing catalogs that rely on
    # ``{tenant_schema}`` substitution keep working.
    tenancy: TenancyMode = "schema"
    # The column that carries the tenant identifier in DISCRIMINATOR
    # mode. Required when ``tenancy == "discriminator"``; ignored
    # otherwise.
    tenancy_column: str | None = None
    # Caller-attached row-level security predicate. AND-composes with
    # the tenancy filter inside the isolation subquery, so an outer
    # predicate the planner emits cannot bypass it. May contain
    # ``{alias}`` placeholders (resolved to the cube's alias) and
    # ``{ctx.X}`` placeholders (bound from the compile-time
    # ``context`` dict, never inlined as a SQL literal).
    security_sql: str | None = None
    # Names the dimension on *this* cube that uniquely identifies a row.
    # Used by the Catalog to auto-derive ``many_to_one`` Joins from
    # other cubes' ``Dimension.foreign_key`` declarations.
    primary_key: str | None = None
    # Ordered dimension hierarchies a UI consumer can offer as drill
    # affordances. ``[["country", "state", "city"]]`` lets a frontend
    # rendering a result grouped by ``country`` show a "drill to state"
    # action. Multiple paths are allowed (alternate hierarchies). The
    # compiler ignores this field — it's pure metadata.
    drill_paths: list[list[str]] = []
    # Inherit measures / dimensions / time_dimensions / segments by
    # name from another cube in the same catalog. The child can
    # override a parent field by redeclaring with the same name; new
    # items append. Other settings (backend, table, alias,
    # base_predicate, tenancy, joins) are cube-specific and do not
    # inherit. Cycles raise at Catalog construction.
    extends: str | None = None
    # Roles required to see this cube. ANY-match: a viewer with at
    # least one of the listed roles can see it; empty list means
    # the cube is open. Filtering happens in ``iter_cubes(viewer=...)``
    # and the compiler refuses queries that touch a cube the viewer
    # cannot see. Static surface — for dynamic / programmable policy
    # use ``Catalog(policy=...)``.
    required_roles: list[str] = []
    # Names a ``ScopeFn`` registered on the Catalog. When set and the
    # caller passes a ``viewer``, the compiler calls the function and
    # injects the returned ``ScopePredicate`` inside this cube's
    # isolation subquery (alongside tenancy + security_sql) so it
    # can't be bypassed by an outer ``OR``. Decouples the row-level
    # rule from the cube definition — a single scope ("reportees of
    # viewer") can apply to N cubes that name the same scope.
    scope: str | None = None
    # Materialised rollup tables that pre-aggregate this cube at one
    # or more grains. The compiler matches a query against each
    # rollup; on a fit, the query is rewritten to read the rollup's
    # ``physical_table`` instead of ``table`` — typically orders of
    # magnitude faster on large fact tables. See :class:`Rollup` for
    # the matching rules and naming convention.
    rollups: list[Rollup] = []
    # LLM-grounding metadata (S7). Concrete NL questions a user might
    # literally ask of this cube — *not* templates, not noun fragments.
    # Spliced into the planner prompt (small catalog) or embedded for
    # top-k retrieval (large catalog). Surface in the MCP per-cube tool
    # description so external agents picking by capability see the
    # canonical phrasings.
    questions: list[str] = []
    # Free-text search tokens with acronym-preserving normalisation
    # applied at validation. ``"AOV"`` stays ``"AOV"`` (all-caps tokens
    # are acronyms); other tokens lowercase. Case-insensitive dedupe;
    # first form wins. Not validated against any controlled vocab.
    keywords: list[str] = []
    # Cube-*internal* narrative: cardinality / FK paths, business rules
    # / gotchas ("Orders only count once payment_status='paid'"),
    # anti-patterns, lineage / freshness. Cross-cube relationships go
    # on ``Catalog.relations`` so they live in one place. Capped at
    # 2000 chars; the first 120 chars appear in the MCP tool
    # description.
    relations: str = ""
    # Lifecycle tier. ``deprecated`` cubes trigger a ``CompileError``
    # at query compile time; ``beta`` flows through with a planner-
    # visible annotation. ``stable`` is the default.
    stability: StabilityLiteral = "stable"
    # Optional pointer at the successor cube when ``stability=
    # "deprecated"``. Surfaces in the CompileError message. Leave
    # ``None`` when the cube is going away entirely (no replacement).
    replacement: str | None = None
    # Optional declared row count. I1 cost estimation: when set,
    # ``estimate_cost`` uses this as the rows-scanned baseline for
    # queries that touch the cube. ``None`` means "unknown" — the
    # estimate is honest about the gap rather than lying with a
    # default. The value is a *declared* count, not a live
    # measurement; callers update it on a schedule.
    size_hint: int | None = None
    # Time-partitioned physical source set — see
    # :class:`TimePartition` / :class:`TimePartitionedSource`. The
    # compiler intersects the query's ``TimeWindow.range`` with each
    # source's half-open range and ``UNION ALL``s the matches.
    # Mutually exclusive with ``table`` / ``source``: a cube has
    # exactly one source declaration. The two single-source variants
    # (``table`` shorthand, ``source=PhysicalTable``) are unaffected.
    physical_sources: list[TimePartitionedSource] = []
    # Routing time dimension for the partition set. Required when
    # ``physical_sources`` is non-empty; ignored otherwise. Names a
    # ``TimeDimension`` on this cube.
    time_partition: TimePartition | None = None
    # Set by the plan→plan ``apply_partition_to_plan`` transform
    # when a time-partitioned cube is routed. The synthetic cube
    # the transform produces carries the matched source set here;
    # ``_from_clause_stage`` checks this and emits the unioned
    # subquery in place of ``table`` / ``source``. ``None`` for
    # the unpartitioned case (the common path).
    partitioned_scan: PartitionedScan | None = None

    @model_validator(mode="after")
    def _check_size_hint(self) -> Cube:
        if self.size_hint is not None and self.size_hint < 0:
            raise ValueError(
                f"Cube {self.name!r}: size_hint must be non-negative, got {self.size_hint}"
            )
        return self

    @model_validator(mode="after")
    def _check_physical_sources(self) -> Cube:
        """Validate the time-partitioned source set: uniqueness, the
        routing dim exists on the cube, every column rename maps to
        a real field, and ``time_partition`` is set when the source
        set is non-empty.

        Source / ``physical_sources`` mutual exclusivity is checked
        in :meth:`_check_source_consistency`; this validator only
        concerns the contents of the source set itself."""
        if not self.physical_sources:
            if self.time_partition is not None:
                raise ValueError(
                    f"Cube {self.name!r}: time_partition is set but "
                    "physical_sources is empty. Either drop time_partition "
                    "or populate physical_sources with at least one entry."
                )
            return self

        if self.time_partition is None:
            raise ValueError(
                f"Cube {self.name!r}: physical_sources is non-empty but "
                "time_partition is not set. The router needs to know which "
                "TimeDimension drives source selection."
            )

        td_names = {td.name for td in self.time_dimensions}
        if self.time_partition.time_dimension not in td_names:
            raise ValueError(
                f"Cube {self.name!r}: time_partition.time_dimension "
                f"{self.time_partition.time_dimension!r} is not a "
                "TimeDimension on this cube. Known: "
                f"{sorted(td_names)}."
            )

        field_names = self.field_names() | {td.name for td in self.time_dimensions}
        seen: set[str] = set()
        for src in self.physical_sources:
            if src.name in seen:
                raise ValueError(
                    f"Cube {self.name!r}: duplicate physical source name "
                    f"{src.name!r}. Source names must be unique within a cube."
                )
            seen.add(src.name)
            for logical_name in src.column_renames:
                if logical_name not in field_names:
                    raise ValueError(
                        f"Cube {self.name!r}, physical source {src.name!r}: "
                        f"column_renames key {logical_name!r} is not a field "
                        f"(measure / dimension / time dimension / segment) "
                        f"on this cube. Known: {sorted(field_names)}."
                    )
        return self
        return self

    @model_validator(mode="after")
    def _check_rollups(self) -> Cube:
        """Every rollup names existing measures/dimensions; aggs must be
        re-aggregatable; names unique; time_dim if set must be a real
        TimeDimension on this cube."""
        if not self.rollups:
            return self
        dim_by_name = {d.name: d for d in self.dimensions}
        measure_by_name = {m.name: m for m in self.measures}
        td_by_name = {td.name: td for td in self.time_dimensions}
        # Distributive aggs that re-aggregate trivially over a rollup
        # grain: SUM-of-SUMs, COUNT-of-COUNTs (as SUM), MIN, MAX.
        # Everything else (avg/ratio/count_distinct/percentiles) needs
        # extra state or a different aggregation entirely; refuse.
        REAGG_OK = {"sum", "count", "min", "max"}
        seen: set[str] = set()
        for r in self.rollups:
            if r.name in seen:
                raise ValueError(f"Cube {self.name!r}: duplicate rollup name {r.name!r}.")
            seen.add(r.name)
            for d in r.dimensions:
                if d not in dim_by_name:
                    raise ValueError(
                        f"Cube {self.name!r}, rollup {r.name!r}: "
                        f"dimension {d!r} is not declared on this cube. "
                        f"Known dimensions: {sorted(dim_by_name)}."
                    )
            for m in r.measures:
                if m not in measure_by_name:
                    raise ValueError(
                        f"Cube {self.name!r}, rollup {r.name!r}: "
                        f"measure {m!r} is not declared on this cube. "
                        f"Known measures: {sorted(measure_by_name)}."
                    )
                agg = measure_by_name[m].agg
                if agg not in REAGG_OK:
                    raise ValueError(
                        f"Cube {self.name!r}, rollup {r.name!r}: measure "
                        f"{m!r} has agg={agg!r}, which can't be re-aggregated "
                        f"over a rollup grain. Allowed aggs: "
                        f"{sorted(REAGG_OK)}. Drop the measure from the "
                        "rollup or store decomposed state in a separate rollup."
                    )
            if r.time_dimension is not None and r.time_dimension not in td_by_name:
                raise ValueError(
                    f"Cube {self.name!r}, rollup {r.name!r}: time_dimension "
                    f"{r.time_dimension!r} is not declared on this cube. "
                    f"Known time_dimensions: {sorted(td_by_name)}."
                )
            if (
                r.time_dimension is not None
                and r.granularity is not None
                and r.granularity not in td_by_name[r.time_dimension].granularities
            ):
                raise ValueError(
                    f"Cube {self.name!r}, rollup {r.name!r}: granularity "
                    f"{r.granularity!r} not permitted on time_dimension "
                    f"{r.time_dimension!r} (allowed: "
                    f"{td_by_name[r.time_dimension].granularities})."
                )
        return self

    @model_validator(mode="after")
    def _check_grounding(self) -> Cube:
        """Length caps + dedupe on the S7 grounding fields. Refuses
        ``deprecated`` without a self-consistent replacement field
        (validation that the replacement points at a *real* cube
        happens at Catalog construction — Cube doesn't know its
        siblings)."""
        _grounding_validate_questions("Cube", self.name, self.questions)
        # Normalised + deduped keywords replace the input list. Cube
        # isn't frozen, so direct assignment is fine.
        self.keywords = _grounding_validate_keywords("Cube", self.name, self.keywords)
        _grounding_validate_relations("Cube", self.name, self.relations)
        if self.stability != "deprecated" and self.replacement is not None:
            raise ValueError(
                f"Cube {self.name!r}: ``replacement`` may only be set "
                f"when ``stability='deprecated'`` (got stability="
                f"{self.stability!r})."
            )
        return self

    @model_validator(mode="after")
    def _check_drill_paths(self) -> Cube:
        dim_names = {d.name for d in self.dimensions}
        for i, path in enumerate(self.drill_paths):
            if not path:
                raise ValueError(
                    f"Cube {self.name!r}, drill_paths[{i}]: empty path. "
                    "Each drill path must list at least one dimension."
                )
            if len(set(path)) != len(path):
                raise ValueError(
                    f"Cube {self.name!r}, drill_paths[{i}]: duplicate "
                    f"dimensions in path {path!r}. Hierarchies must be strict."
                )
            for dim in path:
                if dim not in dim_names:
                    raise ValueError(
                        f"Cube {self.name!r}, drill_paths[{i}]: unknown "
                        f"dimension {dim!r}. Known dimensions on this cube: "
                        f"{sorted(dim_names)}."
                    )
        return self

    @model_validator(mode="after")
    def _check_source_consistency(self) -> Cube:
        """Exactly one source declaration: ``table`` (shorthand),
        ``source`` (explicit ``PhysicalTable`` / ``DerivedTable``), or
        ``physical_sources`` (time-partitioned set). Setting both
        ``table`` and ``source`` is OK only when ``source`` is a
        ``PhysicalTable`` whose ``table`` matches; mixing a ``table``
        value with a ``DerivedTable`` is rejected.

        ``physical_sources`` is mutually exclusive with both ``table``
        and ``source`` — a cube has one source declaration, full stop."""
        has_table = bool(self.table)
        has_source = self.source is not None
        has_physical = bool(self.physical_sources)
        if not has_table and not has_source and not has_physical:
            raise ValueError(
                f"Cube {self.name!r}: must declare one source: ``table=`` "
                "(plain table reference), ``source=DerivedTable(sql=...)`` "
                "(derived source), or ``physical_sources=[...]`` "
                "(time-partitioned set)."
            )
        if has_physical and (has_table or has_source):
            raise ValueError(
                f"Cube {self.name!r}: cannot set ``physical_sources`` "
                "together with ``table`` or ``source``. Pick one source "
                "declaration."
            )
        if has_table and has_source:
            if isinstance(self.source, PhysicalTable):
                if self.source.table != self.table:
                    raise ValueError(
                        f"Cube {self.name!r}: ``table={self.table!r}`` "
                        "disagrees with ``source=PhysicalTable(table="
                        f"{self.source.table!r})``. Specify only one."
                    )
            else:
                raise ValueError(
                    f"Cube {self.name!r}: cannot set both ``table`` and "
                    "``source=DerivedTable(...)``. Drop ``table`` when "
                    "using a derived source."
                )
        return self

    @model_validator(mode="after")
    def _check_tenancy_consistency(self) -> Cube:
        if self.tenancy == "discriminator":
            if not self.tenancy_column:
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' but "
                    "has no tenancy_column — the compiler can't emit a "
                    "WHERE predicate without a column to filter on."
                )
            # ``{tenant_schema}`` is the schema-tenancy substitution marker —
            # it has no meaning under discriminator tenancy. Check every
            # place a source SQL can live.
            offenders: list[str] = []
            if "{tenant_schema}" in self.table:
                offenders.append("table")
            if isinstance(self.source, PhysicalTable) and "{tenant_schema}" in self.source.table:
                offenders.append("source.table")
            if isinstance(self.source, DerivedTable) and "{tenant_schema}" in self.source.sql:
                offenders.append("source.sql")
            if isinstance(self.source, DerivedTable):
                for cte in self.source.with_ctes:
                    if "{tenant_schema}" in cte.sql:
                        offenders.append(f"source.with_ctes[{cte.name!r}]")
                        break
            if offenders:
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' "
                    f"but ``{offenders[0]}`` contains '{{tenant_schema}}'. "
                    "The two isolation strategies are mutually exclusive "
                    "— pick one."
                )
        return self

    @property
    def resolved_source(self) -> CubeSource:
        """Canonical single-source spec.

        Returns ``self.source`` when explicitly set, otherwise wraps
        ``self.table`` in a :class:`PhysicalTable`. Use this from the
        compiler / backend dialect so the ``table`` / ``source``
        shorthand distinction stays a model concern.

        For cubes with ``physical_sources`` this property is not
        meaningful — a partitioned cube has N physical tables, not
        one. It raises so callers that forgot to branch on
        ``physical_sources`` fail loudly rather than silently
        emitting a wrong FROM clause."""
        if self.physical_sources:
            raise ValueError(
                f"Cube {self.name!r}: resolved_source is not defined for "
                "cubes with physical_sources. Use the time-partitioned "
                "compile path (dialect.emit_physical_sources / "
                "compile._resolve_physical_sources) instead."
            )
        if self.source is not None:
            return self.source
        return PhysicalTable(table=self.table)

    def field_names(self) -> set[str]:
        names: set[str] = set()
        names.update(m.name for m in self.measures)
        names.update(d.name for d in self.dimensions)
        names.update(td.name for td in self.time_dimensions)
        return names


class ScopePredicate(_HashableModel):
    """The output of a ``ScopeFn`` — a SQL predicate the compiler injects
    inside a cube's tenancy/security wrapper so it can't be bypassed by
    outer ``OR`` clauses.

    ``sql`` uses the same ``{alias}`` / ``{ctx.X}`` placeholder convention
    as ``security_sql``. ``{ctx.X}`` keys MUST appear in ``ctx_keys`` so
    the compiler can validate the resolution context up front instead
    of surfacing a placeholder error mid-emission.

    Use ``ScopePredicate(sql="1=1", ctx_keys=[])`` (or just return None
    from the ScopeFn) when the viewer should see every row of the cube.
    """

    model_config = ConfigDict(frozen=True)
    sql: str
    ctx_keys: list[str] = Field(default_factory=list)


class AuthContext(_HashableModel):
    """Identity + roles a viewer carries into a request.

    Threads through ``Catalog.compile`` / ``semql_prompt.planner_prompt`` / ``iter_cubes``
    as the ``viewer=`` kwarg. Two effects:

    - **Cube visibility**: ``iter_cubes(viewer=...)`` skips cubes whose
      ``required_roles`` don't intersect the viewer's roles. Prompt
      fragments and MCP tool auto-registration shrink to what the
      viewer is allowed to see.
    - **Row-level scoping**: ``viewer_id`` auto-flattens to
      ``ctx.viewer_id`` inside ``security_sql`` substitution, so
      cubes that scope to "rows owned by the viewer" can declare
      ``security_sql="{t}.assignee_id = {ctx.viewer_id}"`` once and
      have it bound as a parameter (never as a SQL literal).

    ``metadata`` is the same caller-owned escape hatch the catalog
    types carry: opaque string→string the platform never reads.
    """

    model_config = ConfigDict(frozen=True)
    viewer_id: str
    roles: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)
    # A2 — typed bag for arbitrary JWT claims / auth attributes. Unlike
    # ``metadata`` (str→str), ``attrs`` preserves the original types
    # (list, bool, int) so ScopeFns can branch on structured claim values
    # without decoding them from strings first.
    attrs: dict[str, Any] = Field(default_factory=dict)


class ResolutionContext(_HashableModel):
    """The context handed to :class:`Lookup` loaders.

    Loaders are pure functions of this context — same input, same
    output — so callers can cache responses keyed by it. ``viewer``
    flows from ``semql_prompt.planner_prompt(viewer=...)``; ``context`` mirrors the
    compile-time substitution dict (typically ``{"tenant_schema":
    ..., "tenant": ...}``)."""

    model_config = ConfigDict(frozen=True)
    viewer: AuthContext | None = None
    context: dict[str, str] = Field(default_factory=dict)


LookupValues = Sequence[str] | Mapping[str, str]
"""Loader return type — either a flat list of canonical values, or a
``{value: human label}`` mapping for value→label rendering."""

LookupLoader = Callable[[ResolutionContext], LookupValues]
"""A function from :class:`ResolutionContext` to lookup values. Fires
when ``semql_prompt.planner_prompt(...)`` renders a dynamic ``Lookup`` — never from
the compiler."""


from typing import Protocol as _Protocol  # noqa: E402 — after ResolutionContext
from typing import runtime_checkable  # noqa: E402


@runtime_checkable
class LookupEnricher(_Protocol):
    """Optional extension for :data:`LookupLoader` callables that support
    batch ID→label resolution after a query executes.

    Implement this alongside a loader callable when the lookup source is
    too large to load in full but can efficiently resolve a specific batch
    of IDs (DB query by PK, cache lookup, REST API batch endpoint).

    ``enrich`` returns a ``{id: label}`` mapping. Missing IDs are filled
    by the caller with the raw ID value — never raises for unknown IDs.
    """

    def enrich(
        self,
        ids: list[str],
        ctx: ResolutionContext,
    ) -> dict[str, str]: ...


class Lookup(_HashableModel):
    """A finite set of valid values for a string dimension.

    Surfaces dimension values to the planner so "Show me sales in EMEA"
    binds to a concrete predicate without the LLM having to guess. Two
    flavours:

    - **Static**: ``values=("EMEA", "APAC", "NA")``. Values live in the
      catalog.
    - **Dynamic**: ``loader=lambda ctx: db.fetch_regions(...)``. The
      loader fires when ``semql_prompt.planner_prompt(...)`` renders the catalog
      block, so the rendered values can vary per viewer / tenant.

    ``max_inline`` caps how many values are inlined into the prompt.
    Beyond it the rendered catalog tells the planner to call a
    ``resolve_<dim>`` tool (or :func:`semql.lookups.resolve`) instead.

    Loaders are an I/O entry point — they live in
    ``semql_prompt.planner_prompt(...)``, never in ``Catalog.compile(...)``. The
    compiler stays sans-io."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    # Qualified ``cube.dim`` reference. The dimension must exist and
    # have type ``"string"``; validation runs at ``Catalog`` construction.
    dimension: str
    # Static value set. ``None`` means "use the loader". Tuples (not
    # lists) so the Lookup hashes cleanly and stays immutable.
    values: tuple[str, ...] | None = None
    # Optional human label per value. ``("EMEA",)`` + ``{"EMEA": "Europe,
    # Middle East & Africa"}`` renders both the canonical id and the label.
    # Loader-backed lookups can return a Mapping to populate this dynamically.
    labels: dict[str, str] | None = None
    loader: LookupLoader | None = None
    max_inline: int = 50
    description: str = ""

    @model_validator(mode="after")
    def _check_source(self) -> Lookup:
        has_values = self.values is not None
        has_loader = self.loader is not None
        if not has_values and not has_loader:
            raise ValueError(
                f"Lookup({self.dimension!r}): must declare either "
                "``values=`` (static) or ``loader=`` (dynamic)."
            )
        if has_values and has_loader:
            raise ValueError(
                f"Lookup({self.dimension!r}): ``values`` and ``loader`` are mutually exclusive."
            )
        if "." not in self.dimension or self.dimension.count(".") != 1:
            raise ValueError(
                f"Lookup.dimension {self.dimension!r} must be qualified as ``cube.dim``."
            )
        if self.max_inline < 0:
            raise ValueError(f"Lookup({self.dimension!r}): max_inline must be >= 0.")
        return self


class View(_HashableModel):
    """A curated catalog facade.

    A view exposes a renamed subset of measures / dimensions drawn
    from one or more underlying cubes. Two practical uses:

    - **Prompt trimming.** In a 30-cube catalog, expose a 5-field
      view for a question shape and the planner prompt shrinks
      proportionally.
    - **Disambiguation.** When the same dim name (``identity_id``)
      exists on multiple cubes, the view picks one explicitly.

    ``fields`` maps the view's local name to a qualified
    ``cube.field`` reference. Renaming is allowed — ``{"net_revenue":
    "orders.revenue"}`` exposes the underlying ``revenue`` measure
    as ``view.net_revenue``.

    Views live in the ``Catalog`` alongside cubes; their names share
    a namespace with cube names (no collisions allowed).
    """

    model_config = ConfigDict(frozen=True)
    name: str
    fields: dict[str, str]
    description: str = ""
    display_name: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_fields(self) -> View:
        if not self.fields:
            raise ValueError(
                f"View {self.name!r}: fields cannot be empty. "
                "A view must expose at least one cube.field reference."
            )
        for local, target in self.fields.items():
            if "." not in target or target.count(".") != 1:
                raise ValueError(
                    f"View {self.name!r}, field {local!r}: target "
                    f"{target!r} must be qualified as 'cube.field'."
                )
        return self


class Entity(_HashableModel):
    """A first-class business-object declaration — the prompt / product
    counterpart to a :class:`Cube`.

    An entity names the *thing* a user is asking about ("User",
    "Order", "LeaveInstance") and the physical cubes that materialise
    it. The split is intentional: a :class:`Cube` answers "what's the
    SQL surface?" (backend, table, joins, measures); an ``Entity``
    answers "what's the business object?" and is the right unit for
    prompt fragments, entity-resolution glue, and product-level
    vocabulary. ``cubes`` is a list because a single business object
    can be composite — ``User = UserInfo + Identity`` is two
    physical cubes under one entity, joined by a shared key.

    The compiler ignores ``Entity`` entirely. ``Cube`` is the unit
    of compilation; ``Entity`` is descriptive metadata that
    downstream tools (MCP, prompt renderers, ER diagrams) can
    iterate to attach the right business vocabulary. The Catalog
    validates the entity's references at construction time so
    callers can trust the surface.

    Fields
    ------
    ``cubes``
        One or more cube names the entity spans. Names are checked
        against the catalog at construction time; listing a
        non-existent cube is a configuration error, not a runtime
        surprise.
    ``key``
        Optional ``"cube.dim"`` reference naming the primary key
        for entity-typed fetches ("show me user 42"). The
        referenced dimension must exist on one of ``cubes``;
        otherwise construction refuses. ``None`` for entities that
        are exploratory ("Show me a sample of users") and don't
        yet have a stable key.
    ``fields``
        Optional local→qualified (``"cube.field"``) rename map.
        Mirrors :class:`View` semantics but on the entity scope —
        a caller can address ``checkout_user.email`` instead of
        ``users.email`` to disambiguate when the same dim name
        appears on multiple cubes the entity spans. Empty by
        default.
    ``questions``, ``keywords``, ``description``, ``display_name``,
    ``metadata``
        Prompt and product metadata. Same shape as the same-named
        fields on :class:`Cube`; ``Entity`` is the unit a planner
        prompt fragment will iterate, so it carries the same
        vocabulary.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    cubes: list[str]
    key: str | None = None
    description: str = ""
    display_name: str | None = None
    questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)
    fields: dict[str, str] = Field(default_factory=dict)

    @property
    def kind(self) -> str:
        """Structural type tag — ``"entity"``. Lets consumers branch on
        type without importing the class. Mirrors :attr:`BaseField.kind`."""
        return "entity"

    @model_validator(mode="after")
    def _check_name(self) -> Entity:
        if not self.name.strip():
            raise ValueError("Entity.name cannot be empty.")
        return self

    @model_validator(mode="after")
    def _check_cubes(self) -> Entity:
        if not self.cubes:
            raise ValueError(
                f"Entity {self.name!r}: ``cubes`` cannot be empty. "
                "An entity must name at least one cube it spans."
            )
        seen: set[str] = set()
        for c in self.cubes:
            if c in seen:
                raise ValueError(
                    f"Entity {self.name!r}: cube {c!r} is listed twice. "
                    "Each cube may appear at most once."
                )
            seen.add(c)
            if not c.strip():
                raise ValueError(f"Entity {self.name!r}: ``cubes`` cannot contain empty strings.")
        return self

    @model_validator(mode="after")
    def _check_key_qualified(self) -> Entity:
        if self.key is None:
            return self
        if "." not in self.key or self.key.count(".") != 1:
            raise ValueError(
                f"Entity {self.name!r}: key={self.key!r} must be qualified as 'cube.dim'."
            )
        return self

    @model_validator(mode="after")
    def _check_fields_qualified(self) -> Entity:
        for local, target in self.fields.items():
            if "." not in target or target.count(".") != 1:
                raise ValueError(
                    f"Entity {self.name!r}, field {local!r}: target "
                    f"{target!r} must be qualified as 'cube.field'."
                )
        return self

    @model_validator(mode="after")
    def _check_grounding(self) -> Entity:
        # Reuse the catalog-wide grounding validators so an Entity
        # carries the same vocabulary shape as a Cube — a downstream
        # prompt renderer can iterate entities with the same
        # normalisation it applies to cubes. The keyword dedupe
        # normally rewrites the input list (``Cube`` does that),
        # but ``Entity`` is frozen and Pydantic refuses assignment
        # to a frozen model. We call the validator for its
        # validation side-effects (empty-entry check, length cap)
        # and let the user pass a pre-deduped list. The Catalog
        # wraps the entity in a copy-and-replace path when dedupe
        # matters for prompt hashing.
        _grounding_validate_questions("Entity", self.name, self.questions)
        _grounding_validate_keywords("Entity", self.name, self.keywords)
        return self


__all__ = [
    "AggLiteral",
    "AuthContext",
    "Backend",
    "BaseField",
    "ChartTypeLiteral",
    "Cube",
    "CubeSource",
    "DerivedTable",
    "DimTypeLiteral",
    "Dimension",
    "Entity",
    "FormatLiteral",
    "GlossaryEntry",
    "GranularityLiteral",
    "Join",
    "Lookup",
    "LookupEnricher",
    "LookupLoader",
    "LookupValues",
    "Measure",
    "Metadata",
    "NamedCTE",
    "Provenance",
    "ResolutionContext",
    "ScopePredicate",
    "Segment",
    "StabilityLiteral",
    "StorageType",
    "PhysicalTable",
    "TenancyMode",
    "TimeDimension",
    "TimePartition",
    "TimePartitionedSource",
    "View",
]
