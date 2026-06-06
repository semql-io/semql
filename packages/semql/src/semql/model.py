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


class Measure(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    sql: str
    agg: AggLiteral
    unit: str | None = None
    description: str = ""
    display_name: str | None = None
    format: FormatLiteral | None = None
    metadata: Metadata = Field(default_factory=dict)


class Dimension(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    sql: str
    type: DimTypeLiteral
    description: str = ""
    display_name: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class TimeDimension(BaseModel):
    """Time dimensions are separated so the compiler can apply
    `granularity` truncation when a query asks for it (hourly/daily/etc.
    rollups). Otherwise they're just dimensions of type `time`."""

    model_config = ConfigDict(frozen=True)
    name: str
    sql: str
    type: Literal["time"] = "time"
    granularities: tuple[GranularityLiteral, ...] = ("hour", "day", "week", "month")
    description: str = ""
    display_name: str | None = None
    metadata: Metadata = Field(default_factory=dict)


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
    "ChartTypeLiteral",
    "Cube",
    "DimTypeLiteral",
    "Dimension",
    "FormatLiteral",
    "GranularityLiteral",
    "Join",
    "Measure",
    "Metadata",
    "TenancyMode",
    "TimeDimension",
]
