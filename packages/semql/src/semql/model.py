"""Type definitions for the semantic catalogue.

A `Cube` declares one logical table: where its rows live (`backend`,
`table`), the always-on predicate that defines membership
(`base_predicate`), the measures/dimensions/time-dimensions exposed,
and the join edges to other cubes.

`expose_in_prompt` controls whether `render_catalogue_block` includes
the cube in the system-prompt fragment shown to the planner. The
catalogue is intentionally wider than the prompt — every cube the
compiler accepts doesn't need to be in the planner's vocabulary. Cubes
flagged `False` are reachable via joins from exposed cubes and still
compile cleanly when the planner names them; they just don't appear in
the catalogue rendering.

`metadata` is a user-owned escape hatch (k8s-annotation flavoured): an
opaque ``dict[str, str]`` SemQL never reads, validates, or surfaces.
Callers can stash ownership tags, lineage IDs, presentation hints —
anything the platform shouldn't know about. Round-trips through
``model_copy``, ``model_dump`` / ``model_validate``, and serialisation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Backend(StrEnum):
    POSTGRES = "postgres"
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    META = "meta"  # reflection over the catalogue itself; see introspect.py


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


class BaseField(BaseModel):
    """Shared supertype for catalogue fields.

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
    # FK to the named cube's PK — saving the catalogue author the
    # repetition. An explicit Join with the same ``to`` wins.
    foreign_key: str | None = None


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


class Join(BaseModel):
    """A directed edge from one cube to another.

    `on` is a SQL fragment using `{alias}` placeholders that the
    compiler resolves to actual table aliases at compile time."""

    model_config = ConfigDict(frozen=True)
    to: str
    relationship: Literal["one_to_one", "one_to_many", "many_to_one"]
    on: str
    metadata: Metadata = Field(default_factory=dict)


class TableRef(BaseModel):
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


class DerivedTable(BaseModel):
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
    catalogue (the first is ``Measure.sql``). The compiler surfaces the
    resolved SQL of both the main ``sql`` and every CTE on
    :attr:`semql.compile.Compiled.derived_sources` so static checks
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


CubeSource = TableRef | DerivedTable


class Cube(BaseModel):
    name: str
    backend: Backend
    # Shorthand for a plain-table source: ``Cube(table="schema.t", ...)``
    # is equivalent to ``Cube(source=TableRef(table="schema.t"), ...)``.
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
    # Defaults to ``"schema"`` so existing catalogues that rely on
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
        """Exactly one source declaration: ``table`` (shorthand) or
        ``source`` (explicit). Setting both is OK only when ``source`` is
        a ``TableRef`` whose ``table`` matches; mixing a ``table`` value
        with a ``DerivedTable`` is rejected."""
        has_table = bool(self.table)
        has_source = self.source is not None
        if not has_table and not has_source:
            raise ValueError(
                f"Cube {self.name!r}: must declare either ``table=`` "
                "(plain table reference) or ``source=DerivedTable(sql=...)`` "
                "(derived source)."
            )
        if has_table and has_source:
            if isinstance(self.source, TableRef):
                if self.source.table != self.table:
                    raise ValueError(
                        f"Cube {self.name!r}: ``table={self.table!r}`` "
                        "disagrees with ``source=TableRef(table="
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
            if isinstance(self.source, TableRef) and "{tenant_schema}" in self.source.table:
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
        """Canonical source spec.

        Returns ``self.source`` when explicitly set, otherwise wraps
        ``self.table`` in a :class:`TableRef`. Use this from the compiler
        / backend strategy so the ``table`` / ``source`` shorthand
        distinction stays a model concern."""
        if self.source is not None:
            return self.source
        return TableRef(table=self.table)

    def field_names(self) -> set[str]:
        names: set[str] = set()
        names.update(m.name for m in self.measures)
        names.update(d.name for d in self.dimensions)
        names.update(td.name for td in self.time_dimensions)
        return names


class ScopePredicate(BaseModel):
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


class AuthContext(BaseModel):
    """Identity + roles a viewer carries into a request.

    Threads through ``Catalog.compile`` / ``Catalog.prompt`` / ``iter_cubes``
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

    ``metadata`` is the same caller-owned escape hatch the catalogue
    types carry: opaque string→string the platform never reads.
    """

    model_config = ConfigDict(frozen=True)
    viewer_id: str
    roles: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)


class ResolutionContext(BaseModel):
    """The context handed to :class:`Lookup` loaders.

    Loaders are pure functions of this context — same input, same
    output — so callers can cache responses keyed by it. ``viewer``
    flows from ``Catalog.prompt(viewer=...)``; ``context`` mirrors the
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
when ``Catalog.prompt(...)`` renders a dynamic ``Lookup`` — never from
the compiler."""


class Lookup(BaseModel):
    """A finite set of valid values for a string dimension.

    Surfaces dimension values to the planner so "Show me sales in EMEA"
    binds to a concrete predicate without the LLM having to guess. Two
    flavours:

    - **Static**: ``values=("EMEA", "APAC", "NA")``. Values live in the
      catalogue.
    - **Dynamic**: ``loader=lambda ctx: db.fetch_regions(...)``. The
      loader fires when ``Catalog.prompt(...)`` renders the catalogue
      block, so the rendered values can vary per viewer / tenant.

    ``max_inline`` caps how many values are inlined into the prompt.
    Beyond it the rendered catalogue tells the planner to call a
    ``resolve_<dim>`` tool (or :func:`semql.lookups.resolve`) instead.

    Loaders are an I/O entry point — they live in
    ``Catalog.prompt(...)``, never in ``Catalog.compile(...)``. The
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


class View(BaseModel):
    """A curated catalogue facade.

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
    "FormatLiteral",
    "GranularityLiteral",
    "Join",
    "Lookup",
    "LookupLoader",
    "LookupValues",
    "Measure",
    "Metadata",
    "NamedCTE",
    "ResolutionContext",
    "ScopePredicate",
    "Segment",
    "TableRef",
    "TenancyMode",
    "TimeDimension",
    "View",
]
