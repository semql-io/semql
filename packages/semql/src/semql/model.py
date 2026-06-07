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


AggLiteral = Literal["sum", "count", "count_distinct", "avg", "min", "max"]
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
    unit: str | None = None
    format: FormatLiteral | None = None
    # When True, summing this measure across a coarser time grain
    # yields a wrong answer (the canonical case is ``count_distinct``).
    # The flag is declarative for now — surfaced in the prompt fragment
    # so the planner LLM doesn't naively ask for rollups; compiler
    # refusal lands with the rollup work.
    non_additive: bool = False


class Dimension(BaseField):
    type: DimTypeLiteral


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


class Cube(BaseModel):
    name: str
    backend: Backend
    table: str
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

    @model_validator(mode="after")
    def _check_tenancy_consistency(self) -> Cube:
        if self.tenancy == "discriminator":
            if not self.tenancy_column:
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' but "
                    "has no tenancy_column — the compiler can't emit a "
                    "WHERE predicate without a column to filter on."
                )
            if "{tenant_schema}" in self.table:
                raise ValueError(
                    f"Cube {self.name!r} declares tenancy='discriminator' "
                    "but its table contains '{tenant_schema}'. The two "
                    "isolation strategies are mutually exclusive — pick "
                    "one."
                )
        return self

    def field_names(self) -> set[str]:
        names: set[str] = set()
        names.update(m.name for m in self.measures)
        names.update(d.name for d in self.dimensions)
        names.update(td.name for td in self.time_dimensions)
        return names


__all__ = [
    "AggLiteral",
    "Backend",
    "BaseField",
    "ChartTypeLiteral",
    "Cube",
    "DimTypeLiteral",
    "Dimension",
    "FormatLiteral",
    "GranularityLiteral",
    "Join",
    "Measure",
    "Metadata",
    "Segment",
    "TenancyMode",
    "TimeDimension",
]
