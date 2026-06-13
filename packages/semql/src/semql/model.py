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

import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from semql._grounding import (
    validate_keywords as _grounding_validate_keywords,
)
from semql._grounding import (
    validate_questions as _grounding_validate_questions,
)
from semql._grounding import (
    validate_relations as _grounding_validate_relations,
)

# ``parse_instant`` is a leaf utility. Importing it from ``semql.spec``
# would create a model ↔ spec cycle (the spec tree itself depends on
# model types). It lives at :mod:`semql.instant` precisely so the
# model layer can reach it without the round-trip.
from semql.instant import parse_instant as _parse_instant


class Dialect(StrEnum):
    """The SQL dialect a :class:`Cube` emits for.

    >>> from semql.model import Dialect
    >>> Dialect.POSTGRES.value
    'postgres'
    >>> Dialect.META.value
    'meta'
    """

    POSTGRES = "postgres"
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    # R1 analytics engines — first-class (registered by default).
    REDSHIFT = "redshift"
    TRINO = "trino"
    DATABRICKS = "databricks"
    # R1 OLTP engines — experimental / opt-in (see ``experimental_dialects``).
    # sqlglot transpiles their date_trunc / percentile to best-effort forms
    # we can't exercise in CI, so they're not registered by default.
    SQLSERVER = "sqlserver"
    MYSQL = "mysql"
    ORACLE = "oracle"
    META = "meta"  # reflection over the catalog itself; see introspect.py


class Provenance(StrEnum):
    """How much a projected output column can be trusted.

    Lets downstream consumers (MCP tools, the presenter / drilldown prompt
    roles) distinguish a value that came from an approved measure
    definition from one the planner composed ad hoc."""

    VERIFIED = "verified"  # a catalog-defined measure
    COMPOSED = "composed"  # a derived / inline (ad-hoc) measure expression
    DIMENSION = "dimension"  # a raw dimension / time-bucket column


class Op(StrEnum):
    """A mutation operation a :class:`MutableEntity` may permit.

    A StrEnum so it round-trips through JSON and lives comfortably in a
    ``frozenset`` (``MutableEntity.operations``) and in spec values."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


StabilityLiteral = Literal["stable", "beta", "deprecated"]
"""Lifecycle hint for a Cube / SavedQuery. ``deprecated`` is refused by
the compiler; ``beta`` flows through with an annotation."""

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
# ``time`` is a timestamp (instant, possibly zoned); ``date`` is a
# calendar date with no time-of-day or zone. The split lets the compiler
# refuse sub-day truncation on a date and skip the timezone shift a
# zoned timestamp would get.
DimTypeLiteral = Literal["string", "number", "time", "date", "bool", "uuid"]
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
    "date",
    "uuid",
    "bool",
]
GranularityLiteral = Literal["second", "minute", "hour", "day", "week", "month", "quarter", "year"]
# Which weekday a ``week`` bucket starts on. ``monday`` (ISO default) keeps
# every dialect consistent; ``sunday`` shifts the boundary. See
# ``Cube.week_start`` and ``DialectStrategy.trunc``.
WeekStartLiteral = Literal["monday", "sunday"]

# The type of an entity write field. Aliases ``DimTypeLiteral`` so the
# write surface and the read surface share one type vocabulary — a
# mutable field's declared type is the same notion as a dimension's.
FieldType = DimTypeLiteral
FormatLiteral = Literal["raw", "integer", "percent", "currency", "duration"]
ChartTypeLiteral = Literal[
    "pie_chart",
    "bar_chart",
    "line_chart",
    "data_table",
    # Added in the chart-vocabulary expansion. ``scatter_chart`` plots two
    # measures against each other; ``area_chart`` is a stacked time series
    # (multi-measure composition over time); ``stacked_bar_chart`` breaks a
    # primary dimension down by a second; ``histogram`` is a frequency
    # distribution over a numeric dimension.
    "scatter_chart",
    "area_chart",
    "stacked_bar_chart",
    "histogram",
]

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

# ``{ctx.X}`` placeholder keys inside ``security_sql`` — must mirror the
# ``_CTX_PLACEHOLDER_RE`` the compiler resolves with, so construction-time
# validation sees exactly the keys the compiler will look up.
_CTX_KEY_RE = re.compile(r"\{ctx\.([a-z_][a-z0-9_]*)\}")


class RawSQL(str):
    """An author-supplied raw-SQL fragment — the explicit escape hatch.

    SemQL's principle is "when raw SQL is used, SemQL says so." Every
    field that carries hand-written SQL — ``BaseField.sql``,
    ``Measure.filter``, ``Cube.base_predicate`` / ``security_sql``,
    ``Join.on``, ``DerivedTable`` / ``NamedCTE`` bodies,
    ``ScopePredicate.sql``, ``mask_value`` — is marked ``RawSQL`` so the
    trust boundary is visible at runtime (``isinstance(value, RawSQL)``)
    and in the model's JSON schema, rather than being implied by silence.

    It is a ``str`` subclass, so it parses, ``{alias}``-substitutes and
    serialises exactly like the string it wraps. Plain strings assigned
    to these fields are coerced on construction, so existing catalogs
    need no change.

    >>> isinstance(RawSQL("{o}.amount"), str)
    True
    >>> RawSQL("{o}.amount") == "{o}.amount"
    True
    """

    __slots__ = ()


def _as_raw_sql(value: str) -> RawSQL:
    return value if type(value) is RawSQL else RawSQL(value)


# Field annotation for raw-SQL entry points. To the type-checker it is a
# plain ``str`` (so ``sql="{o}.x"`` still type-checks at every call site);
# at runtime the validator coerces the value to :class:`RawSQL` so the
# marker is real and inspectable.
_Raw = Annotated[str, AfterValidator(_as_raw_sql)]


def _freeze(value: object) -> object:
    """Recursively turn a field value into a hashable, order-normalised key.

    Mirrors :class:`_HashableModel`'s contract: equal field values produce
    equal frozen keys, so equal models hash equal. ``dict`` items are
    sorted by key (insertion order is not part of value identity); ``set``
    becomes a ``frozenset``; nested models recurse through their fields.

    >>> from semql.model import _freeze
    >>> _freeze({"b": 1, "a": 2}) == _freeze({"a": 2, "b": 1})
    True
    >>> _freeze([1, 2, 3])
    (1, 2, 3)
    """
    if isinstance(value, BaseModel):
        return (
            type(value).__name__,
            tuple(_freeze(getattr(value, n)) for n in type(value).model_fields),
        )
    # ``value`` is ``object``; isinstance-narrowing to a bare container
    # leaves the element type unknown, but every element is itself an
    # ``object`` we recurse into — cast says so explicitly.
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in cast("Sequence[object]", value))
    if isinstance(value, dict):
        items = cast("dict[str, object]", value).items()
        return tuple(sorted((k, _freeze(v)) for k, v in items))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in cast("frozenset[object]", value))
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
    sql: _Raw
    description: str = ""
    display_name: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    # Field-level visibility. ANY-match semantics: a viewer with
    # at least one of the listed roles passes; an empty list means
    # the field is open to all viewers who can see the cube. Compiled
    # errors for missing roles are indistinguishable from "field
    # doesn't exist" so callers can't infer the field exists.
    required_roles: list[str] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        """Structural field-type tag, the SemQL equivalent of GraphQL ``__typename``.

        ``"measure"`` / ``"dimension"`` / ``"time"`` / ``"segment"``.
        Lets consumers (MCP tool factory, planner, evaluator) branch on
        field type without importing the concrete ``Measure`` /
        ``Dimension`` / ``TimeDimension`` / ``Segment`` subclasses. Pairs
        with ``ColumnMeta.kind`` on ``CompiledQuery`` which uses the same
        vocabulary for output columns. Implemented as a property so it
        survives ``model_copy`` (the underlying class is unchanged).

        >>> from semql.model import Measure, Dimension, Segment
        >>> Measure(name="r", sql="{a}.a", agg="sum").kind
        'measure'
        >>> Dimension(name="x", sql="{a}.x", type="string").kind
        'dimension'
        >>> Segment(name="s", sql="{a}.x = 1").kind
        'segment'
        """
        # Lazy self-import: ``Measure`` / ``Dimension`` / ``TimeDimension``
        # / ``Segment`` are subclasses declared further down this
        # module, so they don't exist when ``BaseField`` is defined.
        # ``isinstance`` is checked at call time, not class-body time,
        # so a deferred import works. The aliases are then in module
        # scope and the import resolves to the same object each call.
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
    filter: _Raw | None = None
    # For ``agg='ratio'`` only: the names of two other measures on the
    # *same cube* whose aggregates compose the ratio. The compiler emits
    # ``<num_agg> / NULLIF(<den_agg>, 0)``; the ``sql`` field is ignored
    # for ratio measures (set it to ``""`` by convention). Composes with
    # filtered measures — a filtered ratio is just a ratio of two
    # filtered sums.
    numerator: str | None = None
    denominator: str | None = None
    # Field-level masking. ``mask_roles`` names roles for which
    # the field compiles but the SQL is substituted by a constant
    # instead of the real expression. ``mask_value``: ``None`` → NULL,
    # a string → a SQL literal (``'REDACTED'``, ``'0'``, etc.). The
    # constructor enforces ``mask_roles ⊆ required_roles`` — you
    # can't mask a role that can't even see the field.
    mask_roles: list[str] = Field(default_factory=list)
    mask_value: _Raw | None = None

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
    # Field-level masking. Same shape as Measure.mask_*.
    mask_roles: list[str] = Field(default_factory=list)
    mask_value: _Raw | None = None
    # Input aliases. An LLM might emit ``territory`` /
    # ``zone`` / ``area`` when the catalog names the dimension
    # ``region``. The resolver accepts any alias as a synonym for
    # the canonical name; the prompt renders the canonical name
    # only (no alias listing — the planner learns the canonical).
    # The field-hide / mask gates apply on the canonical
    # field; an alias is a synonym, not a separate auth surface.
    aliases: list[str] = Field(default_factory=list)
    # Cross-cube coercion opt-in. A federated bridge join whose two
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

    # ``time`` = timestamp (sub-day grains allowed); ``date`` = calendar
    # date (no hour grain, no timezone shift — see DimTypeLiteral).
    type: Literal["time", "date"] = "time"
    granularities: tuple[GranularityLiteral, ...] = (
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
    )

    @model_validator(mode="before")
    @classmethod
    def _date_has_no_sub_day_grain(cls, data: Any) -> Any:  # noqa: ANN401
        """A ``date`` time-dimension cannot be truncated below a day.

        When the caller doesn't pin ``granularities``, default a date to
        the day-and-coarser set (the class default carries the sub-day
        grains ``second`` / ``minute`` / ``hour``, which are meaningless
        for a date). When they pin one explicitly, refuse any sub-day
        grain rather than silently dropping it.

        ``data`` is the raw pre-validation input (pydantic ``mode="before"``)
        — a ``dict`` for the usual kwargs path, but possibly an already-built
        model or other shape, hence ``Any``.
        """
        if not isinstance(data, dict):
            return data
        fields = cast("dict[str, Any]", data)
        if fields.get("type") != "date":
            return fields
        if "granularities" not in fields:
            return {**fields, "granularities": ("day", "week", "month", "quarter", "year")}
        sub_day = {"second", "minute", "hour"} & set(fields["granularities"])
        if sub_day:
            raise ValueError(
                f"a date TimeDimension cannot have sub-day granularity "
                f"{sorted(sub_day)}: a calendar date has no time-of-day to "
                "truncate."
            )
        return fields


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
    on: _Raw
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
    sql: _Raw


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
    (``is_read_only_statement``, dialect snapshots) cover every raw fragment,
    not just the outer SELECT."""

    model_config = ConfigDict(frozen=True)
    sql: _Raw
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
    # Frozen like every other catalog value type (AGENTS.md): a cube must
    # not be able to drift out of sync with the catalog that validated it.
    model_config = ConfigDict(frozen=True)

    name: str
    dialect: Dialect
    # IANA timezone (e.g. "America/New_York") the cube's timestamps are
    # truncated in. When set, time-dimension granularity buckets are
    # computed after converting the column to this zone — the compiler
    # emits the dialect's timezone-shift form (``AT TIME ZONE`` on
    # Postgres/DuckDB, ``TIMESTAMP(DATETIME(...))`` on BigQuery,
    # ``CONVERT_TIMEZONE`` on Snowflake, the native 2nd arg on
    # ClickHouse). ``None`` truncates in the column's own zone (UTC for a
    # timestamptz), the prior behaviour.
    timezone: str | None = None
    # Which weekday a ``week`` truncation bucket starts on. ``monday`` (the
    # ISO-8601 default) is consistent across every dialect; ``sunday`` shifts
    # the boundary — emitted as a ±1-day shift around the Monday-native
    # ``date_trunc('week', …)`` on SQL backends, or ``toStartOfWeek``'s mode
    # argument on ClickHouse. Only ``week`` grain is affected.
    week_start: WeekStartLiteral = "monday"
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
    base_predicate: _Raw | None = None
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
    # Defaults to ``"none"``: a cube that says nothing about tenancy gets
    # *no* isolation, and says so honestly (the old ``"schema"`` default
    # silently emitted nothing unless the table happened to contain
    # ``{tenant_schema}``). Opt in explicitly to ``"schema"`` /
    # ``"discriminator"``; ``Catalog(strict_tenancy=True)`` flags cubes
    # left at ``"none"`` with no other access control.
    tenancy: TenancyMode = "none"
    # Columns that carry the tenant identifier in DISCRIMINATOR mode. Each
    # is AND-composed into one bound predicate inside the isolation
    # subquery, so composite tenant keys (e.g. ``["org_id", "region"]``)
    # need no ``security_sql`` workaround. Required (non-empty) when
    # ``tenancy == "discriminator"``; ignored otherwise. At compile time
    # each column ``C`` binds to the value found under key ``C`` in the
    # resolution context / ``viewer.attrs`` (a single-column cube also
    # accepts the canonical ``tenant`` key, which ``viewer.tenant``
    # populates).
    tenancy_columns: list[str] = []
    # Caller-attached row-level security predicate. AND-composes with
    # the tenancy filter inside the isolation subquery, so an outer
    # predicate the planner emits cannot bypass it. May contain
    # ``{alias}`` placeholders (resolved to the cube's alias) and
    # ``{ctx.X}`` placeholders (bound from the compile-time
    # ``context`` dict, never inlined as a SQL literal).
    security_sql: _Raw | None = None
    # The ``{ctx.X}`` keys ``security_sql`` needs — the cube-level mirror
    # of :attr:`ScopePredicate.ctx_keys`. Every ``{ctx.X}`` used in
    # ``security_sql`` (other than the auto-flattened ``viewer_id``) MUST
    # be declared here, validated at construction so a typo is a build
    # error rather than a per-request PlaceholderError. The compiler also
    # checks the declared keys against the resolution context up front,
    # so a missing context value fails before emission with a clear
    # message instead of mid-query.
    security_ctx_keys: list[str] = []
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
    # LLM-grounding metadata. Concrete NL questions a user might
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
    # Optional declared row count. Cost estimation: when set,
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

    @model_validator(mode="before")
    @classmethod
    def _normalise_keywords(cls, data: Any) -> Any:  # noqa: ANN401 — pydantic before-validator raw input
        """Normalise + dedupe ``keywords`` before model construction.

        Done here (not in an after-validator) because Cube is frozen, so
        ``self.keywords`` can't be reassigned post-construction."""
        if not isinstance(data, dict):
            return data
        fields = cast("dict[str, Any]", data)
        if not fields.get("keywords"):
            return fields
        name = str(fields.get("name", ""))
        return {
            **fields,
            "keywords": _grounding_validate_keywords("Cube", name, fields["keywords"]),
        }

    @model_validator(mode="after")
    def _check_grounding(self) -> Cube:
        """Length caps + dedupe on the grounding fields. Refuses
        ``deprecated`` without a self-consistent replacement field
        (validation that the replacement points at a *real* cube
        happens at Catalog construction — Cube doesn't know its
        siblings)."""
        _grounding_validate_questions("Cube", self.name, self.questions)
        # Keyword normalisation/dedupe happens in a ``mode="before"``
        # validator (``_normalise_keywords``) because Cube is frozen and
        # an after-validator can't reassign ``self.keywords``.
        _grounding_validate_relations("Cube", self.name, self.relations)
        if self.stability != "deprecated" and self.replacement is not None:
            raise ValueError(
                f"Cube {self.name!r}: ``replacement`` may only be set "
                f"when ``stability='deprecated'`` (got stability="
                f"{self.stability!r})."
            )
        return self

    @model_validator(mode="after")
    def _check_required_filters_exist(self) -> Cube:
        """Each ``required_filters`` entry must name a real dimension or
        time-dimension on this cube. A typo (``["regn"]``) otherwise
        constructs cleanly and makes the cube permanently un-queryable
        with a misleading per-query 'requires a filter on regn' error."""
        field_names = {d.name for d in self.dimensions} | {td.name for td in self.time_dimensions}
        for req in self.required_filters:
            if req not in field_names:
                raise ValueError(
                    f"Cube {self.name!r}: required_filters entry {req!r} does not "
                    f"name a dimension or time_dimension on the cube "
                    f"(known: {sorted(field_names)})."
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
        # Schema-tenancy isolates by substituting ``{tenant_schema}`` into
        # the source. A schema-mode cube whose source never mentions the
        # placeholder would emit no isolation at all — the old silent-inert
        # default. Require it explicitly (the mirror of the discriminator
        # check below).
        if self.tenancy == "schema" and "{tenant_schema}" not in self._all_source_sql():
            raise ValueError(
                f"Cube {self.name!r} declares tenancy='schema' but no "
                "source SQL contains '{tenant_schema}' — it would emit "
                "zero tenant isolation. Add the placeholder, or set "
                "tenancy='none' if the cube is genuinely unscoped."
            )
        if self.tenancy == "discriminator":
            if not self.tenancy_columns:
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' but "
                    "has no tenancy_columns — the compiler can't emit a "
                    "WHERE predicate without a column to filter on."
                )
            # ``{tenant_schema}`` is the schema-tenancy substitution marker —
            # it has no meaning under discriminator tenancy. The two
            # isolation strategies are mutually exclusive.
            if "{tenant_schema}" in self._all_source_sql():
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' "
                    "but its source SQL contains '{tenant_schema}'. The two "
                    "isolation strategies are mutually exclusive — pick one."
                )
        return self

    @model_validator(mode="after")
    def _check_security_ctx_keys(self) -> Cube:
        """Every ``{ctx.X}`` in ``security_sql`` must be declared in
        ``security_ctx_keys`` (the cube mirror of ScopePredicate.ctx_keys)
        — except ``viewer_id``, which the compiler auto-flattens from the
        viewer. Catches a placeholder typo at construction instead of as a
        per-request PlaceholderError."""
        if not self.security_sql:
            return self
        used = set(_CTX_KEY_RE.findall(self.security_sql))
        # ``viewer_id`` is always available when a viewer is present —
        # the compiler auto-binds ``ctx.viewer_id``. Don't force authors
        # to redeclare it.
        undeclared = sorted(used - set(self.security_ctx_keys) - {"viewer_id"})
        if undeclared:
            raise ValueError(
                f"Cube {self.name!r}: security_sql uses {{ctx.X}} keys "
                f"{undeclared} that are not declared in security_ctx_keys="
                f"{self.security_ctx_keys!r}. Declare them (or fix a typo) so "
                "the context requirement is validated at construction."
            )
        return self

    def _all_source_sql(self) -> str:
        """Every place a source SQL string can live, concatenated.

        Used by tenancy validation to test for the ``{tenant_schema}``
        marker across the table shorthand, an explicit ``PhysicalTable``
        / ``DerivedTable`` source (plus its hoisted CTEs), and every
        time-partitioned physical source."""
        parts: list[str] = [self.table]
        if isinstance(self.source, PhysicalTable):
            parts.append(self.source.table)
        elif isinstance(self.source, DerivedTable):
            parts.append(self.source.sql)
            parts.extend(cte.sql for cte in self.source.with_ctes)
        parts.extend(src.table for src in self.physical_sources)
        return "\n".join(parts)

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
    sql: _Raw
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
    - **Tenancy**: ``tenant`` is the canonical tenant identifier
      carried *by the identity* rather than threaded through the loose
      ``context`` dict. When set, ``compile`` uses it as the default
      ``{tenant_schema}`` substitution (schema mode) and the default
      single-column discriminator value, and the audit hook records it.
      Making tenancy a property of *who is asking* — not of a context
      argument the application has to remember to pass — removes the
      class of bug where a plumbing slip silently crosses tenants. An
      explicit ``context`` value still overrides it.

    ``metadata`` is the same caller-owned escape hatch the catalog
    types carry: opaque string→string the platform never reads.
    """

    model_config = ConfigDict(frozen=True)
    viewer_id: str
    # Canonical tenant identifier for this identity (see class docstring).
    # ``None`` means the identity carries no tenant — a cube that requires
    # tenancy then refuses unless the value is supplied via ``context``.
    tenant: str | None = None
    roles: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)
    # Typed bag for arbitrary JWT claims / auth attributes. Unlike
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

    ``Entity`` carries two surfaces. As prompt / product metadata it
    names the business object for MCP, prompt renderers and ER
    diagrams. As a *read* surface (``key``, ``list_filters``,
    ``default_order``) it parameterises row-mode fetch/list compilation
    — "show me order 42", "list this user's open orders" — which lower
    to the compiler's ungrouped row-listing mode. Analytic aggregation
    still goes through :class:`SemanticQuery`; the entity surface is the
    point-lookup / short-list escape hatch.

    References are validated *by the Catalog* at construction time: the
    bare ``Entity(...)`` constructor only checks format (qualified
    ``cube.dim`` shape); whether those cubes and dimensions actually
    exist is checked when the entity is passed to ``Catalog(entities=[...])``
    (or ``CatalogSpec.from_iterables``). An ``Entity`` built in isolation
    is therefore format-valid but ungrounded until it joins a catalog.

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
    # Read surface (row-mode fetch/list). ``list_filters`` allowlists the
    # qualified ``cube.dim`` references an ``EntityList`` may filter on;
    # anything not listed routes to the analytic layer. ``default_order``
    # is the ``"cube.dim [asc|desc]"`` ordering a list falls back to (and
    # the keyset-cursor anchor). Both are format-checked here and
    # resolved against real dimensions by the Catalog.
    list_filters: list[str] = Field(default_factory=list)
    default_order: str | None = None
    # When True, this entity is served by a row-capable adapter (REST/KV/
    # custom store), not by raw-SQL execution. Two consequences (D1/§4):
    # it must be single-cube (the adapter contract is one table-shaped
    # source), and its cube scope must be *structured* — a raw ``security_sql``
    # / ``scope`` fragment can't be ported to a non-SQL backend, so the
    # row-mode compiler refuses it. False (default) = ordinary SQL backend.
    custom_backend: bool = False

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
    def _check_list_filters_qualified(self) -> Entity:
        for ref in self.list_filters:
            if "." not in ref or ref.count(".") != 1:
                raise ValueError(
                    f"Entity {self.name!r}: list_filter {ref!r} must be qualified as 'cube.dim'."
                )
        return self

    @model_validator(mode="after")
    def _check_default_order_shape(self) -> Entity:
        if self.default_order is None:
            return self
        parts = self.default_order.split()
        if len(parts) not in (1, 2):
            raise ValueError(
                f"Entity {self.name!r}: default_order={self.default_order!r} "
                "must be 'cube.dim' optionally followed by 'asc' or 'desc'."
            )
        ref = parts[0]
        if "." not in ref or ref.count(".") != 1:
            raise ValueError(
                f"Entity {self.name!r}: default_order dimension {ref!r} must "
                "be qualified as 'cube.dim'."
            )
        if len(parts) == 2 and parts[1].lower() not in ("asc", "desc"):
            raise ValueError(
                f"Entity {self.name!r}: default_order direction {parts[1]!r} "
                "must be 'asc' or 'desc'."
            )
        return self

    @model_validator(mode="after")
    def _check_custom_backend_single_cube(self) -> Entity:
        if self.custom_backend and len(self.cubes) != 1:
            raise ValueError(
                f"Entity {self.name!r}: custom_backend entities must span "
                f"exactly one cube (the adapter contract is one table-shaped "
                f"source); got {self.cubes!r}."
            )
        return self

    @model_validator(mode="after")
    def _check_grounding(self) -> Entity:
        # Reuse the catalog-wide grounding validators so an Entity carries
        # the same vocabulary shape as a Cube — a downstream prompt
        # renderer can iterate entities with the same normalisation it
        # applies to cubes. The keyword dedupe normally rewrites the input
        # list (``Cube`` does that), but ``Entity`` is frozen and Pydantic
        # refuses assignment to a frozen model, so we call the validators
        # only for their side-effects (empty-entry check, length cap) and
        # the caller passes a pre-deduped list.
        _grounding_validate_questions("Entity", self.name, self.questions)
        _grounding_validate_keywords("Entity", self.name, self.keywords)
        return self


class CtxRef(_HashableModel):
    """A reference to an :class:`AuthContext` attribute, used to pin a
    column to a ctx-derived value the LLM cannot supply.

    ``MutableEntity.pinned_values`` maps a target column to a ``CtxRef``;
    at mutation-compile the value is read from the viewer's ``AuthContext``
    (``getattr(viewer, attr)`` / ``viewer.attrs[attr]``) and bound as a
    parameter — never taken from LLM-supplied ``values``. Naming the attr
    explicitly keeps the col→attr direction unambiguous and leaves room to
    grow (transforms, defaults) without changing the field type."""

    model_config = ConfigDict(frozen=True)
    attr: str

    @model_validator(mode="after")
    def _check_attr(self) -> CtxRef:
        if not self.attr.strip():
            raise ValueError("CtxRef.attr cannot be empty.")
        return self


class MutableField(_HashableModel):
    """A single writable field on a :class:`MutableEntity`.

    Declares the column's write contract: its ``type`` (shared vocabulary
    with dimensions), whether it is ``required`` on insert, whether it may
    be ``nullable``, and whether it is ``immutable`` (settable on insert
    but refused on update)."""

    model_config = ConfigDict(frozen=True)
    type: FieldType
    required: bool = False
    nullable: bool = True
    immutable: bool = False


class MutableEntity(Entity):
    """An :class:`Entity` that additionally declares a *write* surface.

    Mutations are opt-in at every layer (see ``Catalog.allow_mutations``
    and the per-operation ``operations`` set); a ``MutableEntity`` is the
    model-layer declaration of *what* may be written and *how* it is
    targeted. v1 writes target exactly one cube (``target_cube``); PK
    targeting is the default and predicate targeting is opt-in
    (``predicate_targeting``). ``pinned_values`` force ctx-derived columns
    (tenant, owner) the LLM cannot set.

    A ``MutableEntity`` requires a ``key`` (PK targeting needs it). The
    ``target_cube ∈ cubes`` rule and pinned/mutable disjointness are
    checked here; field existence on ``target_cube`` is resolved against
    the real cube at Catalog construction."""

    model_config = ConfigDict(frozen=True)
    target_cube: str
    operations: frozenset[Op]
    mutable_fields: dict[str, MutableField] = Field(default_factory=dict)
    pinned_values: dict[str, CtxRef] = Field(default_factory=dict)
    predicate_targeting: bool = False

    @model_validator(mode="after")
    def _check_requires_key(self) -> MutableEntity:
        if self.key is None:
            raise ValueError(
                f"MutableEntity {self.name!r}: key is required — PK targeting "
                "needs a key. Declare key='cube.dim'."
            )
        return self

    @model_validator(mode="after")
    def _check_target_cube_listed(self) -> MutableEntity:
        if self.target_cube not in self.cubes:
            raise ValueError(
                f"MutableEntity {self.name!r}: target_cube={self.target_cube!r} "
                f"must be one of the entity's cubes {self.cubes!r}."
            )
        return self

    @model_validator(mode="after")
    def _check_operations_nonempty(self) -> MutableEntity:
        if not self.operations:
            raise ValueError(
                f"MutableEntity {self.name!r}: operations cannot be empty — "
                "declare at least one of insert/update/delete/upsert."
            )
        return self

    @model_validator(mode="after")
    def _check_pinned_not_mutable(self) -> MutableEntity:
        clash = set(self.pinned_values) & set(self.mutable_fields)
        if clash:
            raise ValueError(
                f"MutableEntity {self.name!r}: column(s) {sorted(clash)} are "
                "both pinned and mutable. A pinned (ctx-derived) column is "
                "not LLM-supplied — remove it from mutable_fields."
            )
        return self


__all__ = [
    "AggLiteral",
    "AuthContext",
    "Dialect",
    "BaseField",
    "ChartTypeLiteral",
    "Cube",
    "CtxRef",
    "CubeSource",
    "DerivedTable",
    "DimTypeLiteral",
    "Dimension",
    "Entity",
    "FieldType",
    "FormatLiteral",
    "GlossaryEntry",
    "GranularityLiteral",
    "WeekStartLiteral",
    "Join",
    "Lookup",
    "LookupEnricher",
    "LookupLoader",
    "LookupValues",
    "Measure",
    "Metadata",
    "MutableEntity",
    "MutableField",
    "NamedCTE",
    "Op",
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
