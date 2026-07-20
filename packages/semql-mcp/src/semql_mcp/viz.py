# pyright: reportUnusedFunction=false
# FastMCP's @mcp.resource / @mcp.tool decorators register the
# wrapped function with the server; pyright sees the local name as
# "unused" because it can't follow the decorator's side effect.
"""MCP Apps integration for ``semql.visualize.decide_visualization``.

Wires the visualiser into the MCP Apps spec (the host-iframe extension
to MCP) so an MCP client with a UI surface can render a chart for
the result of a query. The surface is marked **BETA** in the tool
description and on the resource annotation — the recommendation logic
is stable but the rendered shape and the per-host protocol details
are still settling, and we want consumers to be able to see at a
glance that the shape may change.

What this module exposes
------------------------

A resource (HTML, served at ``ui://semql/chart``) that an MCP Apps
host renders inside an iframe. The page reads the latest tool
result via the standard MCP Apps ``window.openai.toolOutput`` channel
and renders it as a small inline-SVG chart (bar, line, pie, or a
data table — driven by the ``chart_type`` field on the
``VizDecision`` payload). The HTML lives in
``semql_mcp/chart_template.html`` and is loaded via
:mod:`importlib.resources`; the Python module only contains the
registration glue.

A tool (``query_visualize``) that compiles a ``SemanticQuery`` and
returns the corresponding ``VizDecision`` (the resource is just
a renderer for that payload). The tool is registered with
``AppConfig(visibility=["app"])`` so the LLM-side of a chat client
doesn't see it; only an MCP Apps-aware host that can pair the tool
with the iframe resource sees the affordance. ``model``-visibility
clients (plain LLM chat) keep the chat-only path.

The visualizer is the **stable** layer; the iframe template and the
MCP Apps data-channel conventions are the part most likely to drift,
which is what the BETA label flags. A future "1.0" version of this
will lock the iframe shape and graduate the surface to ``stable``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from importlib.resources import files
from typing import Any

from fastmcp import FastMCP
from fastmcp.apps.config import AppConfig
from semql import (
    AuthContext,
    Catalog,
    SemanticQuery,
    ShapeStats,
    VizChartType,
    VizDecision,
    decide_visualization,
)

# Structural aliases — duplicated here so this module doesn't have
# to import from :mod:`semql_mcp.server` (which would create a
# circular import: the server imports from this module to register
# the tools). The canonical home of these types is ``server.py``;
# the ``MCPServer`` re-exports them on the public surface so callers
# still see one type per concept.
_Executor = Callable[[str, dict[str, Any]], list[dict[str, Any]]]
_ViewerProvider = Callable[[], AuthContext | None]

VIZ_BETA_NOTICE = (
    "[BETA] Visualisation is a beta surface. The recommendation logic "
    "(`decide_visualization`) is stable, but the rendered shape and "
    "the MCP Apps data-channel conventions may change before 1.0."
)

CHART_RESOURCE_URI = "ui://semql/chart"
"""URI the iframe resource is served at. The MCP Apps spec recommends
the ``ui://`` scheme for UI resources; the host pairs the tool with
the resource by matching this URI on the tool's ``app`` config."""


def _chart_html() -> str:
    """Load the chart iframe HTML from the package template asset.

    The HTML is intentionally kept in a ``.html`` file rather than
    embedded as a Python string so the JS / CSS / SVG can be edited
    in a single tool (a real HTML editor) without dealing with
    Python string-literal escaping, and so the linter can stay
    focused on the Python glue. Loaded once at module import so the
    per-call cost is zero."""
    return files("semql_mcp").joinpath("chart_template.html").read_text(encoding="utf-8")


def _viz_to_payload(
    decision: VizDecision, *, rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Serialise a :class:`VizDecision` (and optional rows) to the dict
    the iframe consumes.

    ``rows`` is the caller-executed result set (the visualiser is
    compile-only; the caller runs the SQL). The iframe is the one place
    where the chart is actually drawn, so it needs the row data to do
    that — the visualiser's sans-I/O invariant is preserved by keeping
    the row-fetching in the caller's hands.
    """
    payload: dict[str, Any] = {
        "chart_type": decision.chart_type,
        "title": decision.title,
        "x_axis": decision.x_axis,
        "y_axes": decision.y_axes,
        "series": decision.series,
        "size_axis": decision.size_axis,
        "columns": [asdict(c) for c in decision.columns],
        "reason": {
            "kind": decision.reason.kind,
            "note": decision.reason.note,
            "alternatives": decision.reason.alternatives,
        },
        "confidence": decision.confidence,
        "candidates": [
            {
                "chart_type": c.chart_type,
                "confidence": c.confidence,
                "reason": {
                    "kind": c.reason.kind,
                    "note": c.reason.note,
                    "alternatives": c.reason.alternatives,
                },
            }
            for c in decision.candidates
        ],
        "features": asdict(decision.features),
        "hints": asdict(decision.hints),
        "_stability": "beta",
    }
    if rows is not None:
        payload["rows"] = rows
    return payload


def _resolve_shape_stats(raw: dict[str, Any] | None) -> ShapeStats | None:
    if not raw:
        return None
    return ShapeStats(
        has_negatives=raw.get("has_negatives"),
        measure_min=raw.get("measure_min"),
        measure_max=raw.get("measure_max"),
        n_distinct_categories=raw.get("n_distinct_categories"),
        null_rate=raw.get("null_rate"),
        is_sparse=raw.get("is_sparse"),
    )


def _run_viz(
    catalog: Catalog,
    resolve_viewer: _ViewerProvider,
    spec: SemanticQuery,
    n_rows: int,
    shape_stats: dict[str, Any] | None,
    supported_charts: list[str] | None,
    context: dict[str, str] | None,
    executor: _Executor | None,
    debug: bool,
) -> dict[str, Any]:
    """Compile + decide + (optionally) execute; build the iframe payload.

    The shared body of :func:`register_visualization_tools`'s tool —
    extracted so the inline function closure doesn't get a thousand
    lines deep, and so the error handling is in one place."""
    from semql.errors import SemQLError

    try:
        viewer = resolve_viewer()
        compiled = catalog.compile(spec, context=context, viewer=viewer)
        stats = _resolve_shape_stats(shape_stats)
        # Cast at the boundary: the wire format is a list of strings
        # (the MCP JSON schema can't express ``VizChartType``), so we
        # accept whatever the client sent and trust the renderer to
        # not pass an unknown value.
        supported: frozenset[VizChartType] | None = (
            frozenset(supported_charts) if supported_charts else None  # type: ignore[arg-type]
        )
        decision = decide_visualization(
            spec,
            compiled,
            n_rows=n_rows,
            catalog=catalog.as_dict(),
            shape_stats=stats,
            supported_charts=supported,
        )
    except SemQLError as exc:
        return {"error": exc.to_payload(), "_stability": "beta"}
    except Exception as exc:
        if debug:
            return {
                "error": {"code": type(exc).__name__, "message": str(exc)},
                "_stability": "beta",
            }
        return {
            "error": {
                "code": "ExecutionError",
                "message": "Visualization failed. Enable server debug mode to see details.",
            },
            "_stability": "beta",
        }
    rows: list[dict[str, Any]] | None = None
    if executor is not None:
        try:
            from semql.safe import is_read_only_statement

            if is_read_only_statement(compiled.sql, dialect=compiled.dialect.value):
                rows = executor(compiled.sql, compiled.params)
            else:
                # Refuse silently; the payload still carries the
                # decision so a text-only consumer can show prose.
                rows = None
        except Exception:
            rows = None
    payload = _viz_to_payload(decision, rows=rows)
    # Always include the SQL envelope so the host (or a tool
    # client) can re-execute against its own backend when the
    # server was constructed without one.
    payload["sql"] = compiled.sql
    payload["params"] = compiled.params
    return payload


def register_visualization_tools(
    mcp: FastMCP,
    catalog: Catalog,
    resolve_viewer: _ViewerProvider,
    debug: bool = False,
    *,
    executor: _Executor | None = None,
) -> None:
    """Wire ``query_visualize`` + the ``ui://semql/chart`` resource.

    The tool is registered with ``visibility=["app"]`` so an
    MCP Apps-aware host surfaces the chart affordance and a plain
    chat client (model-only) doesn't see it. The resource is the
    sandboxed-iframe HTML the host loads to render the payload.

    ``executor`` is the same callable the server uses for
    ``query_execute``: ``(sql, params) -> rows``. When supplied,
    the visualisation tool runs the SQL after compiling and
    attaches the row data to the iframe payload, so the
    chart is actually drawable. Without an executor the
    tool returns the ``VizDecision`` only — the caller is
    expected to execute the SQL themselves and feed the rows
    to the iframe via a side channel.
    """

    # The chart resource: a single static HTML page the host loads.
    @mcp.resource(
        CHART_RESOURCE_URI,
        name="semql_chart",
        title="semql chart",
        description=(
            "Sandboxed-iframe HTML that renders a VizDecision payload. " + VIZ_BETA_NOTICE
        ),
        mime_type="text/html",
    )
    def chart_resource() -> str:  # pragma: no cover — host-rendered iframe
        return _chart_html()

    @mcp.tool(
        name="query_visualize",
        description=(
            f"{VIZ_BETA_NOTICE} Compile a SemanticQuery, run the SQL "
            "(if an executor is configured), and return a chart decision "
            "for the result. The decision pairs with the chart iframe at "
            f"{CHART_RESOURCE_URI}; an MCP Apps host renders the chart "
            "next to the model-side answer."
        ),
        app=AppConfig(
            resource_uri=CHART_RESOURCE_URI,
            visibility=["app"],
        ),
    )
    def query_visualize(
        spec: SemanticQuery,
        n_rows: int = 0,
        shape_stats: dict[str, Any] | None = None,
        supported_charts: list[str] | None = None,
        context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return _run_viz(
            catalog=catalog,
            resolve_viewer=resolve_viewer,
            spec=spec,
            n_rows=n_rows,
            shape_stats=shape_stats,
            supported_charts=supported_charts,
            context=context,
            executor=executor,
            debug=debug,
        )


__all__ = [
    "CHART_RESOURCE_URI",
    "VIZ_BETA_NOTICE",
    "register_visualization_tools",
]
