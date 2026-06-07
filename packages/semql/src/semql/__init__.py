"""Public surface of the semantic layer.

Most users only need ``Catalog``, ``Cube``, the field types
(``Measure`` / ``Dimension`` / ``TimeDimension``), and
``SemanticQuery``. The rest is exported for callers building their own
tooling on top of the compiler (custom validators, MCP servers,
prompt rendering, etc.).
"""

from __future__ import annotations

from semql.catalog import Catalog
from semql.compile import MAX_UNGROUPED_ROWS, Compiled, compile_query
from semql.errors import (
    CompileError,
    CrossBackendError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    SemQLError,
    UnknownIdentifierError,
)
from semql.introspect import (
    CATALOG_CUBES,
    CATALOG_DIMENSIONS,
    CATALOG_MEASURES,
    META_CUBES,
)
from semql.model import (
    AggLiteral,
    Backend,
    ChartTypeLiteral,
    Cube,
    Dimension,
    DimTypeLiteral,
    FormatLiteral,
    GranularityLiteral,
    Join,
    Measure,
    Segment,
    TimeDimension,
)
from semql.prompt import (
    build_planner_prompt_fragment,
    build_router_prompt_fragment,
    render_catalogue_block,
)
from semql.safe import is_safe_select
from semql.spec import (
    CompareWindow,
    Filter,
    FilterOp,
    SemanticQuery,
    TimeWindow,
)
from semql.validate import ValidationError, validate
from semql.visualize import VizColumn, VizDecision, decide_visualization

__all__ = [
    "AggLiteral",
    "Backend",
    "CATALOG_CUBES",
    "CATALOG_DIMENSIONS",
    "CATALOG_MEASURES",
    "Catalog",
    "ChartTypeLiteral",
    "CompareWindow",
    "CompileError",
    "Compiled",
    "CrossBackendError",
    "Cube",
    "DimTypeLiteral",
    "Dimension",
    "Filter",
    "FilterOp",
    "FilterTypeError",
    "FormatLiteral",
    "GranularityLiteral",
    "Join",
    "JoinPathError",
    "MAX_UNGROUPED_ROWS",
    "META_CUBES",
    "Measure",
    "PhaseDeferredError",
    "PlaceholderError",
    "ResolveError",
    "Segment",
    "SemQLError",
    "SemanticQuery",
    "TimeDimension",
    "TimeWindow",
    "UnknownIdentifierError",
    "ValidationError",
    "VizColumn",
    "VizDecision",
    "build_planner_prompt_fragment",
    "build_router_prompt_fragment",
    "compile_query",
    "decide_visualization",
    "is_safe_select",
    "render_catalogue_block",
    "validate",
]
