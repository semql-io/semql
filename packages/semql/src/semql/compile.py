# pyright: reportPrivateImportUsage=false
# sqlglot's AST types (Expression, Placeholder, ...) are imported via
# ``from sqlglot import exp``; they aren't in ``sqlglot.expressions.__all__``
# but are public by convention and by sqlglot's type stubs.
"""Pure compiler from `SemanticQuery` to backend SQL.

The compiler has no I/O. It reads the catalogue, resolves identifiers,
emits a parameterised SQL string + params dict + output column list,
and raises ``CompileError`` (or a more specific leaf) on unknown
references, unreachable joins, or unsupported shapes.

The body composes a sqlglot ``exp.Select`` AST and renders it via the
target backend's dialect. Per-cube fragments (dim SQL, measure SQL,
``base_predicate``, ``Join.on``) are parsed by sqlglot under the
catalogue cube's declared backend; dialect-specific shapes
(``placeholder``, ``trunc``, ``contains``, ``emit_source``) come from
the ``BackendStrategy`` and slot into the AST as nodes.

Scope (Phase 1):
- Single-backend queries (cross-backend rejected).
- ``base_predicate`` lifted to the outer WHERE.
- Parameterised ``Filter.values`` (positional names ``p0``, ``p1``, …).
- Time-window pre-resolved as ISO strings, exclusive end.
- Granularity truncation for time dimensions.
- ``ungrouped=True`` row listings with a hard 1000-row cap.
- ``having`` on measure aliases (bare or qualified).
- ``context`` dict substitutes ``{key}`` placeholders in table/SQL strings.

Out of scope (deferred):
- ``compare`` CTE shell (current/prior FULL OUTER JOIN).
- Cross-backend merge.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime
from typing import Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from semql._resolve import resolve_field as _resolve_field_raw
from semql.backend import BackendStrategy, strategy_for

# Importing from semql.dialect also registers the ClickHouse placeholder
# override against the ``clickhouse`` dialect name (side effect on import).
from semql.dialect import dialect_for
from semql.errors import (
    CompileError,
    CrossBackendError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    UnknownIdentifierError,
)
from semql.introspect import PolicyFn, ScopeFn, viewer_sees
from semql.model import (
    AuthContext,
    Backend,
    Cube,
    DerivedTable,
    Dimension,
    FormatLiteral,
    Join,
    Measure,
    ScopePredicate,
    Segment,
    TimeDimension,
    View,
)
from semql.spec import BoolExpr, Filter, InlineDerived, SemanticQuery

MAX_UNGROUPED_ROWS = 1000


ColumnKind = Literal["measure", "dimension", "time", "computed"]


@dataclass
class ColumnMeta:
    """Per-output-column type + presentation metadata.

    Sits on :class:`Compiled` in the same order as ``columns`` so a
    consumer (dashboard, MCP server, presenter LLM) can render a result
    row without re-resolving against the catalogue. ``kind`` tags the
    column's role; ``unit`` / ``display_unit`` / ``format`` mirror the
    fields on ``Measure`` / ``Dimension``; ``display_name`` carries the
    human-friendly label (catalog ``display_name`` or a humanised
    fallback) so visualisers don't need to call ``humanize`` themselves.

    ``computed`` covers compare-mode derivatives (``foo_delta``,
    ``foo_pct_change``) — values the catalogue doesn't directly name
    but the compiler emits. Their ``format`` is set inline (e.g.
    ``"percent"`` for pct_change) so renderers don't need to guess.
    """

    name: str
    kind: ColumnKind
    display_name: str = ""
    unit: str | None = None
    display_unit: str | None = None
    format: FormatLiteral | None = None


@dataclass
class Compiled:
    backend: Backend
    sql: str
    params: dict[str, Any]
    columns: list[str]
    column_meta: list[ColumnMeta] = dc_field(default_factory=lambda: [])
    # Names of every cube the query touched, in first-mention order.
    # Lets downstream tools (visualiser, MCP envelope) avoid re-running
    # the resolver against the catalogue for cube-level facts.
    touched_cube_names: list[str] = dc_field(default_factory=lambda: [])
    # Resolved SQL of every ``DerivedTable`` source the query touched,
    # in first-mention order matching ``touched_cube_names``. Surfaces
    # the *second* place raw SQL legitimately enters the catalogue
    # (the first is the outer ``sql``) so callers running
    # :func:`semql.safe.is_safe_select` or dialect snapshots see every
    # raw fragment, not just the compiler-generated SELECT. Plain-table
    # cubes contribute nothing here.
    derived_sources: list[str] = dc_field(default_factory=lambda: [])


def _collect_derived_sources(
    touched: list[Cube],
    resolve_sql: Callable[[str], str],
) -> list[str]:
    """Materialize the resolved SQL of every ``DerivedTable`` source the
    query touched. Plain-table cubes contribute nothing.

    Order matches ``touched`` (first-mention). For each derived cube,
    any ``with_ctes`` are emitted first (declaration order) followed by
    the main ``sql`` — so a static checker like ``is_safe_select``
    walking ``derived_sources`` sees the same fragments that actually
    enter the compiled query."""
    out: list[str] = []
    for cube in touched:
        src = cube.resolved_source
        if isinstance(src, DerivedTable):
            for cte in src.with_ctes:
                out.append(resolve_sql(cte.sql))
            out.append(resolve_sql(src.sql))
    return out


def _collect_hoisted_ctes(
    touched: list[Cube],
    resolve_sql: Callable[[str], str],
) -> list[tuple[str, str]]:
    """Collect every CTE every touched cube declares, deduped by name.

    Returns ``[(name, resolved_sql), ...]`` in first-mention order. The
    catalog already enforces cube-cross uniqueness at construction
    time, so the dedup here only matters when the same cube is touched
    twice (e.g. self-join) — its CTEs collapse to one ``WITH`` entry."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for cube in touched:
        src = cube.resolved_source
        if not isinstance(src, DerivedTable):
            continue
        for cte in src.with_ctes:
            if cte.name in seen:
                continue
            seen.add(cte.name)
            out.append((cte.name, resolve_sql(cte.sql)))
    return out


def _apply_with_clause(
    select_node: exp.Select,
    ctes: list[tuple[str, str]],
    backend: Backend,
) -> exp.Select:
    """Attach a ``WITH`` clause hoisting every ``ctes`` entry to the front
    of ``select_node``.

    Uses sqlglot's ``Select.with_`` helper which returns a new node
    carrying the CTE. ``as_`` strings get re-parsed in the cube's
    dialect so backend-specific shapes (ClickHouse ARRAY JOIN, BigQuery
    UNNEST) survive the round-trip. No-op when ``ctes`` is empty."""
    if not ctes:
        return select_node
    dialect = dialect_for(backend)
    out: exp.Select = select_node
    for name, body_sql in ctes:
        out = out.with_(name, as_=body_sql, dialect=dialect)
    return out


def _resolve_field(
    qualified: str,
    catalog: dict[str, Cube],
) -> tuple[Cube, Measure | Dimension | TimeDimension]:
    try:
        return _resolve_field_raw(qualified, catalog)
    except CompileError:
        raise
    except ResolveError as exc:
        raise CompileError(str(exc)) from exc


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


def _build_column_meta(
    columns: list[str],
    dim_fields: list[tuple[Cube, Dimension]],
    dim_col_names: list[str],
    measure_fields: list[tuple[Cube, Measure]],
    measure_col_names: list[str],
    time_dim: TimeDimension | None,
    time_col_name: str | None,
    is_compare: bool,
) -> list[ColumnMeta]:
    """Assemble :class:`ColumnMeta` for every entry in ``columns``.

    Lookup table keyed by column name (the alias the compiler emits) →
    ColumnMeta. Compare-mode columns (``foo_current``, ``foo_prior``,
    ``foo_delta``, ``foo_pct_change``) carry the underlying measure's
    unit/display_unit/format; ``pct_change`` overrides format to
    ``"percent"`` regardless of the measure's declared format. The
    computed derivatives' ``display_name`` is the parent measure's
    label suffixed with the derivative role (e.g. ``"Revenue (% change)"``).
    """
    by_name: dict[str, ColumnMeta] = {}

    for (_cube, dim), col_name in zip(dim_fields, dim_col_names, strict=True):
        by_name[col_name] = ColumnMeta(
            name=col_name,
            kind="dimension",
            display_name=dim.display_name or _humanize(col_name),
            unit=dim.unit,
            display_unit=dim.display_unit,
            format=dim.format,
        )

    if time_dim is not None and time_col_name is not None:
        # Drop the granularity suffix (``_day`` / ``_week`` / ...) from
        # the display label — readers don't need "Created At Day", they
        # need "Created At" plotted on a time axis.
        by_name[time_col_name] = ColumnMeta(
            name=time_col_name,
            kind="time",
            display_name=time_dim.display_name or _humanize(time_dim.name),
        )

    for (_cube, m), col_name in zip(measure_fields, measure_col_names, strict=True):
        m_label = m.display_name or _humanize(col_name)
        by_name[col_name] = ColumnMeta(
            name=col_name,
            kind="measure",
            display_name=m_label,
            unit=m.unit,
            display_unit=m.display_unit,
            format=m.format,
        )
        if is_compare:
            # Compare mode emits four derivatives per measure. The first
            # three keep the measure's units; pct_change is dimensionless
            # so its format is always "percent".
            suffix_label = {
                "_current": " (current)",
                "_prior": " (prior)",
                "_delta": " (delta)",
            }
            for suffix, label_tail in suffix_label.items():
                derived_name = col_name + suffix
                by_name[derived_name] = ColumnMeta(
                    name=derived_name,
                    kind="computed",
                    display_name=m_label + label_tail,
                    unit=m.unit,
                    display_unit=m.display_unit,
                    format=m.format,
                )
            pct_name = col_name + "_pct_change"
            by_name[pct_name] = ColumnMeta(
                name=pct_name,
                kind="computed",
                display_name=m_label + " (% change)",
                format="percent",
            )

    out: list[ColumnMeta] = []
    for c in columns:
        out.append(by_name.get(c) or ColumnMeta(name=c, kind="computed", display_name=_humanize(c)))
    return out


# ---------------------------------------------------------------------------
# Cube graph BFS — find a single join path from a root through `joins` edges.
# ---------------------------------------------------------------------------


def _find_join_path(
    root: str,
    target: str,
    catalog: dict[str, Cube],
    *,
    bidirectional: bool = False,
) -> list[tuple[str, Join]]:
    """BFS for a join path between two cubes.

    Returns a list of ``(next_cube_name, Join)`` pairs — each pair
    records the cube we land at and the ``Join`` whose ``on`` clause
    AND-composes into the LEFT JOIN predicate. ``next_cube_name`` is
    *not always* ``Join.to``: under bidirectional traversal the same
    edge can be walked in reverse (the spine→facts pattern), in which
    case ``next_cube_name`` is the source cube of the declared Join.
    The ``on`` clause is symmetric in alias placeholders, so the
    direction matters for FROM-clause emission but not for the SQL
    predicate itself.

    Default is forward-only: only edges declared on ``cube.joins`` get
    walked, which matches the auto-inferred FK→PK direction the
    catalog seeds. ``bidirectional=True`` also walks the reverse of
    every declared edge — needed for spine→facts patterns (anti-join /
    absent-row queries) where the FK lives on the fact cube but the
    spine is the FROM root."""
    if root == target:
        return []
    visited: set[str] = {root}
    queue: list[tuple[str, list[tuple[str, Join]]]] = [(root, [])]
    while queue:
        current, path = queue.pop(0)
        # Forward edges declared on ``current``.
        for j in catalog[current].joins:
            if j.to in visited:
                continue
            new_path = path + [(j.to, j)]
            if j.to == target:
                return new_path
            visited.add(j.to)
            queue.append((j.to, new_path))
        # Reverse edges: cubes that declare ``Join(to=current, ...)`` —
        # walked only when bidirectional traversal is explicitly enabled.
        if bidirectional:
            for other_name, other_cube in catalog.items():
                if other_name in visited:
                    continue
                for j in other_cube.joins:
                    if j.to != current:
                        continue
                    new_path = path + [(other_name, j)]
                    if other_name == target:
                        return new_path
                    visited.add(other_name)
                    queue.append((other_name, new_path))
                    break
    raise JoinPathError(
        f"No join path from cube {root!r} to {target!r}. "
        "Declare a Join in the catalogue or restructure the query.",
        root_cube=root,
        target_cube=target,
    )


# ---------------------------------------------------------------------------
# Placeholder substitution — `{key}` in SQL fragments resolved from
# `cube_aliases` (alias → alias, cube_name → alias) and `context`.
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
# ``{ctx.X}`` — caller-context placeholders inside ``security_sql``.
# Substituted with a bound parameter so the value never appears as a
# SQL literal.
_CTX_PLACEHOLDER_RE = re.compile(r"\{ctx\.([a-z_][a-z0-9_]*)\}")


def _resolve_sql(
    sql: str,
    cube_aliases: dict[str, str],
    context: dict[str, str],
) -> str:
    """Resolve ``{key}`` placeholders in a SQL fragment.

    Priority: cube aliases (both alias and cube name map to alias), then
    caller-supplied ``context``. ClickHouse typed param placeholders like
    ``{p0:String}`` are left untouched — the inner regex requires a plain
    identifier (no ``:Type`` suffix). Unknown placeholders raise
    ``PlaceholderError``."""
    lookup: dict[str, str] = dict(context)
    for cube_name, alias in cube_aliases.items():
        lookup[alias] = alias
        lookup[cube_name] = alias

    def _repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in lookup:
            raise PlaceholderError(
                f"Unknown placeholder {{{name}}} in catalogue SQL. Known: {sorted(lookup)}.",
                placeholder=name,
                known=sorted(lookup),
            )
        return lookup[name]

    return _PLACEHOLDER_RE.sub(_repl, sql)


# ---------------------------------------------------------------------------
# Aggregation — measure agg → sqlglot node
# ---------------------------------------------------------------------------


# Percentile aggregation literals → continuous quantile values for the
# strategy's ``emit_percentile`` hook. ``median`` is the q=0.5 case;
# the p75 / p90 / p95 covers the long-tail diagnostic shape.
_PERCENTILE_AGGS: dict[str, float] = {
    "median": 0.5,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
}


def _build_inline_derived_expr(
    ir: InlineDerived,
    env: _CompileEnv,
    already_declared: list[str],
) -> exp.Expression:
    """Compose the SELECT expression for one :class:`InlineDerived`.

    Resolves each operand to its catalog measure, runs the measure
    through ``env.build_measure_expr`` so the measure's own
    aggregation (``SUM`` / ``COUNT`` / ``AVG`` / ``COUNT(DISTINCT)`` /
    etc.) wraps it, then combines the aggregates by ``op``:

    - ``ratio``: ``num_agg / NULLIF(den_agg, 0)``.
    - ``sum``: ``a_agg + b_agg + ...``.
    - ``diff``: ``a_agg - b_agg``.

    Phase A restriction: every operand must resolve to a measure on
    the same cube. The first operand's cube is the anchor; subsequent
    operands referencing a different cube raise.
    """
    if ir.name in already_declared:
        raise CompileError(
            f"InlineDerived name {ir.name!r} collides with an output "
            f"column already in the query. Pick a unique name."
        )

    resolved_operands: list[tuple[Cube, Measure]] = []
    anchor_cube: Cube | None = None
    for operand_ref in ir.operands:
        cube, fld = _resolve_field(operand_ref, env.catalog)
        if not isinstance(fld, Measure):
            raise CompileError(
                f"InlineDerived({ir.name!r}): operand {operand_ref!r} is "
                f"not a measure on cube {cube.name!r}."
            )
        if fld.agg == "ratio":
            raise CompileError(
                f"InlineDerived({ir.name!r}): operand {operand_ref!r} is "
                "itself a ratio measure. Use leaf (sum / count / avg / "
                "min / max / ...) measures as operands; nested-ratio "
                "composition is not supported."
            )
        if anchor_cube is None:
            anchor_cube = cube
        elif cube is not anchor_cube:
            raise CompileError(
                f"InlineDerived({ir.name!r}): operand {operand_ref!r} is "
                f"on cube {cube.name!r}, but earlier operands are on "
                f"{anchor_cube.name!r}. Phase A requires every operand "
                "to live on the same cube — pre-declare cross-cube "
                "derivations in the catalog."
            )
        resolved_operands.append((cube, fld))

    nodes = [env.build_measure_expr(cube, m) for cube, m in resolved_operands]

    if ir.op == "ratio":
        num, den = nodes
        return exp.Div(
            this=num,
            expression=exp.Anonymous(
                this="NULLIF",
                expressions=[den, exp.Literal.number(0)],
            ),
        )
    if ir.op == "sum":
        acc = nodes[0]
        for n in nodes[1:]:
            acc = exp.Add(this=acc, expression=n)
        return acc
    if ir.op == "diff":
        left, right = nodes
        return exp.Sub(this=left, expression=right)
    raise CompileError(  # pragma: no cover — InlineDerivedOp is closed
        f"InlineDerived({ir.name!r}): unknown op {ir.op!r}."
    )


def _agg_node(m: Measure, expr_node: exp.Expression) -> exp.Expression:
    """Wrap an inner expression in the measure's aggregate function."""
    agg = m.agg
    if agg == "sum":
        return exp.Sum(this=expr_node)
    if agg == "count":
        return exp.Count(this=expr_node)
    if agg == "count_distinct":
        # sqlglot's exp.Count(..., distinct=True) is ignored on emit;
        # COUNT(DISTINCT x) requires wrapping the arg in exp.Distinct.
        return exp.Count(this=exp.Distinct(expressions=[expr_node]))
    if agg == "avg":
        return exp.Avg(this=expr_node)
    if agg == "min":
        return exp.Min(this=expr_node)
    if agg == "max":
        return exp.Max(this=expr_node)
    if agg == "ratio":
        # Ratio measures are composed in build_inner — _agg_node isn't
        # the right seam (it sees one expression, ratios need two).
        raise CompileError(  # pragma: no cover — caller routes around this.
            f"Measure {m.name!r}: ratio aggregation is computed in build_inner, not _agg_node."
        )
    raise CompileError(f"Unsupported aggregate: {agg!r}")  # pragma: no cover


_FILTER_BINOPS: dict[str, type[exp.Expression]] = {
    "eq": exp.EQ,
    "neq": exp.NEQ,
    "gt": exp.GT,
    "lt": exp.LT,
    "gte": exp.GTE,
    "lte": exp.LTE,
}


def _filter_node(
    f: Filter,
    field: exp.Expression,
    field_type: str,
    strategy: BackendStrategy,
    bind: Callable[[Any, str], exp.Placeholder],
) -> exp.Expression:
    """Build a predicate node for a Filter.

    Dialect-specific shapes (``contains``) go through the strategy. Type
    checks raise ``FilterTypeError`` with structured attrs."""
    op = f.op
    if op == "is_null":
        return exp.Is(this=field, expression=exp.Null())
    if op == "not_null":
        return exp.Not(this=exp.Is(this=field, expression=exp.Null()))

    try:
        f.validate_for_type(field_type)
    except ValueError as exc:
        raise FilterTypeError(
            str(exc),
            dimension=f.dimension,
            op=f.op,
            value=f.values[0] if f.values else None,
        ) from exc

    if op == "contains":
        return strategy.emit_contains(field, str(f.values[0]), bind)

    placeholders: list[exp.Placeholder] = [bind(v, field_type) for v in f.values]

    if op in _FILTER_BINOPS:
        return _FILTER_BINOPS[op](this=field, expression=placeholders[0])

    in_args: list[exp.Expression] = list(placeholders)
    if op == "in":
        return exp.In(this=field, expressions=in_args)
    if op == "not_in":
        return exp.Not(this=exp.In(this=field, expressions=in_args))

    raise CompileError(f"Unsupported filter op: {op!r}")  # pragma: no cover


def _walk_where_leaves(expr: BoolExpr | Filter) -> list[Filter]:
    """Return all ``Filter`` leaves from a where tree, depth-first."""
    if isinstance(expr, Filter):
        return [expr]
    leaves: list[Filter] = []
    for child in expr.children:
        leaves.extend(_walk_where_leaves(child))
    return leaves


def _compile_where_tree(
    expr: BoolExpr | Filter,
    leaf_to_node: Callable[[Filter], exp.Expression],
) -> exp.Expression:
    """Build a sqlglot predicate node from a where tree.

    ``leaf_to_node`` resolves each Filter leaf to its compiled AST
    fragment (the caller owns the bind closure + type resolution)."""
    if isinstance(expr, Filter):
        return leaf_to_node(expr)
    children_nodes = [_compile_where_tree(c, leaf_to_node) for c in expr.children]
    if expr.op == "not":
        return exp.Not(this=children_nodes[0])
    combiner: type[exp.Expression] = exp.And if expr.op == "and" else exp.Or
    result = children_nodes[0]
    for n in children_nodes[1:]:
        result = combiner(this=result, expression=n)
    return result


def _parse_fragment(sql: str, dialect: str) -> exp.Expression:
    """Parse a catalogue SQL fragment into a sqlglot AST node.

    Used for dim/measure/time-dimension expressions, ``Join.on``, and
    ``base_predicate``. Any parse failure surfaces as a ``CompileError``
    naming the offending fragment so the catalogue author can fix it."""
    try:
        return sqlglot.parse_one(sql, dialect=dialect)  # type: ignore[return-value]
    except ParseError as exc:
        raise CompileError(
            f"Could not parse catalogue SQL fragment {sql!r} under dialect {dialect!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Compile stages — extracted from ``compile_query`` for testability and to
# make the orchestration legible. Each stage is a pure function over its
# inputs; ``compile_query`` glues them together. POSA "Pipes and Filters".
# ---------------------------------------------------------------------------


@dataclass
class _ResolvedFields:
    """Field-level resolution output: every ``cube.field`` reference in
    the query mapped to a concrete ``(Cube, Field)`` pair, plus the
    ordered list of cubes the query touches."""

    measure_fields: list[tuple[Cube, Measure]]
    dim_fields: list[tuple[Cube, Dimension]]
    time_cube: Cube | None
    time_dim: TimeDimension | None
    filter_resolutions: list[tuple[Filter, Cube, Dimension | Measure | TimeDimension]]
    where_leaf_resolutions: dict[int, tuple[Cube, Dimension | Measure | TimeDimension]]
    segment_resolutions: list[tuple[Cube, Segment]]
    touched: list[Cube]


def _validate_query_invariants(
    q: SemanticQuery,
    *,
    allow_unbounded_ungrouped: bool,
) -> None:
    """Pre-flight checks that don't depend on the catalog.

    Raises ``CompileError`` for compare-mode shape mismatches, lonely
    ``offset`` without ``limit``, ungrouped queries above the row cap
    (unless explicitly allowed), and empty queries."""
    if q.compare is not None:
        if q.time_dimension is None:
            raise CompileError(
                "compare requires a time_dimension on the query — the "
                "current and prior windows are derived from it."
            )
        if not q.measures:
            raise CompileError(
                "compare requires at least one measure — without a "
                "measure there is nothing to delta between current and prior."
            )
        if q.ungrouped:
            raise CompileError(
                "compare is incompatible with ungrouped=True — compare aggregates by definition."
            )
        if q.compare.mode == "explicit" and q.compare.range is None:
            raise CompileError(
                "compare(mode='explicit') requires a range. Pass "
                "CompareWindow(mode='explicit', range=(start, end))."
            )

    if q.offset is not None and q.offset > 0 and q.limit is None:
        raise CompileError(
            "SemanticQuery has offset set without limit. "
            "OFFSET is only meaningful in combination with LIMIT."
        )

    if (
        q.ungrouped
        and not allow_unbounded_ungrouped
        and (q.limit is None or q.limit > MAX_UNGROUPED_ROWS)
    ):
        raise CompileError(
            f"Ungrouped query requires limit <= {MAX_UNGROUPED_ROWS}. Got limit={q.limit}."
        )

    if not q.measures and not q.dimensions and q.time_dimension is None:
        raise CompileError(
            "SemanticQuery is empty — at least one measure, dimension, "
            "or time_dimension is required."
        )


def _make_view_resolver(
    catalog: dict[str, Cube],
    views_map: dict[str, View],
) -> Callable[[str], tuple[Cube, Measure | Dimension | TimeDimension]]:
    """Build the per-call view-aware resolver.

    Rewrites ``view.local_name`` references to the underlying
    ``cube.field`` BUT re-aliases the returned field so the SELECT
    output column uses the view's local name (not the underlying
    field name)."""

    def _resolve(
        qualified: str,
    ) -> tuple[Cube, Measure | Dimension | TimeDimension]:
        if "." in qualified:
            prefix, local = qualified.split(".", 1)
            if prefix in views_map:
                view = views_map[prefix]
                if local not in view.fields:
                    raise CompileError(
                        f"View {prefix!r} has no field {local!r}. "
                        f"Known fields on this view: {sorted(view.fields)}."
                    )
                cube, fld = _resolve_field(view.fields[local], catalog)
                return cube, fld.model_copy(update={"name": local})
        return _resolve_field(qualified, catalog)

    return _resolve


def _resolve_query_fields(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    views_map: dict[str, View],
) -> _ResolvedFields:
    """Resolve every ``cube.field`` reference in ``q`` to its catalog
    entry and collect the ordered set of touched cubes.

    Also resolves segment references and where-tree leaves so all
    referenced cubes land in ``touched`` (the join-graph builder
    needs the full set up front)."""
    resolve_with_views = _make_view_resolver(catalog, views_map)

    measure_fields: list[tuple[Cube, Measure]] = []
    for ref in q.measures:
        cube, fld = resolve_with_views(ref)
        if not isinstance(fld, Measure):
            raise CompileError(f"{ref!r} is not a measure on cube {cube.name!r}.")
        measure_fields.append((cube, fld))

    dim_fields: list[tuple[Cube, Dimension]] = []
    for ref in q.dimensions:
        cube, fld = resolve_with_views(ref)
        if not isinstance(fld, Dimension):
            raise CompileError(f"{ref!r} is not a dimension on cube {cube.name!r}.")
        dim_fields.append((cube, fld))

    time_cube: Cube | None = None
    time_dim: TimeDimension | None = None
    if q.time_dimension is not None:
        tcube, tfld = resolve_with_views(q.time_dimension.dimension)
        if not isinstance(tfld, TimeDimension):
            raise CompileError(f"{q.time_dimension.dimension!r} is not a time dimension.")
        gran = q.time_dimension.granularity
        if gran is not None and gran not in tfld.granularities:
            raise CompileError(
                f"Granularity {gran!r} not supported on {q.time_dimension.dimension!r}. "
                f"Allowed: {tfld.granularities}."
            )
        time_cube, time_dim = tcube, tfld

    touched: list[Cube] = []
    for c, _ in [*measure_fields, *dim_fields]:
        if c not in touched:
            touched.append(c)
    if time_cube is not None and time_cube not in touched:
        touched.append(time_cube)

    filter_resolutions: list[tuple[Filter, Cube, Dimension | Measure | TimeDimension]] = []
    for f in q.filters:
        c, fld = resolve_with_views(f.dimension)
        filter_resolutions.append((f, c, fld))
        if c not in touched:
            touched.append(c)

    where_leaves: list[Filter] = _walk_where_leaves(q.where) if q.where is not None else []
    where_leaf_resolutions: dict[int, tuple[Cube, Dimension | Measure | TimeDimension]] = {}
    for leaf in where_leaves:
        c, fld = resolve_with_views(leaf.dimension)
        where_leaf_resolutions[id(leaf)] = (c, fld)
        if c not in touched:
            touched.append(c)

    segment_resolutions: list[tuple[Cube, Segment]] = []
    for seg_ref in q.segments:
        if "." not in seg_ref:
            raise CompileError(
                f"Segment reference {seg_ref!r} must be qualified as 'cube.segment'."
            )
        cube_name, seg_name = seg_ref.rsplit(".", 1)
        if cube_name not in catalog:
            raise CompileError(f"Segment reference {seg_ref!r}: unknown cube {cube_name!r}.")
        cube_obj = catalog[cube_name]
        match = next((s for s in cube_obj.segments if s.name == seg_name), None)
        if match is None:
            known = ", ".join(s.name for s in cube_obj.segments) or "(none)"
            raise CompileError(
                f"Segment reference {seg_ref!r}: cube {cube_name!r} has no segment "
                f"{seg_name!r}. Known segments: {known}."
            )
        segment_resolutions.append((cube_obj, match))
        if cube_obj not in touched:
            touched.append(cube_obj)

    if not touched:
        raise CompileError("Could not determine any cubes from the query.")

    return _ResolvedFields(
        measure_fields=measure_fields,
        dim_fields=dim_fields,
        time_cube=time_cube,
        time_dim=time_dim,
        filter_resolutions=filter_resolutions,
        where_leaf_resolutions=where_leaf_resolutions,
        segment_resolutions=segment_resolutions,
        touched=touched,
    )


def _check_viewer_authorization(
    touched: list[Cube],
    viewer: AuthContext | None,
    policy: PolicyFn | None,
) -> None:
    """Refuse queries that touch a cube the viewer can't see."""
    if viewer is None:
        return
    forbidden = [c.name for c in touched if not viewer_sees(c, viewer, policy)]
    if forbidden:
        raise CompileError(
            f"Query touches cubes the viewer is not authorised to see: "
            f"{sorted(forbidden)}. Check Cube.required_roles or the "
            "Catalog's policy override."
        )


def _check_required_filters(touched: list[Cube], q: SemanticQuery) -> None:
    """Every ``Cube.required_filters`` must be referenced by a Filter
    on the query — refuse early so the compiler doesn't have to emit
    a query the cube author marked as unsafe-without-scope."""
    filter_dims = {f.dimension for f in q.filters}
    for cube in touched:
        for req in cube.required_filters:
            if f"{cube.name}.{req}" not in filter_dims:
                raise CompileError(
                    f"Cube {cube.name!r} requires a filter on {req!r}. "
                    f"Add Filter(dimension='{cube.name}.{req}', op=..., values=[...])."
                )


def _pick_single_backend(touched: list[Cube]) -> Backend:
    """Single-backend gate. Cross-backend queries route to
    :func:`semql.compile_federated_query` — non-federated compile is
    one-backend-only."""
    backends = {c.backend for c in touched}
    if len(backends) > 1:
        backend_names = sorted(b.value for b in backends)
        raise CrossBackendError(
            "Cross-backend queries are not yet supported (Phase 2). "
            f"Touched backends: {backend_names}.",
            backends=backend_names,
        )
    return next(iter(backends))


class _CompileEnv:
    """All state ``compile_query`` carries across its stages.

    Built once after the prelude has resolved fields and the join
    graph. Owns the parameter binder + memo, the placeholder /
    resolver / parser methods (formerly closures), and the inner-SELECT
    builders (``build_measure_expr``, ``build_inner``). Emission
    helpers take a single env arg instead of fourteen positional
    parameters; the closures-as-methods migration replaces the
    shared-mutable-locals style of the old monolith.
    """

    def __init__(
        self,
        q: SemanticQuery,
        catalog: dict[str, Cube],
        *,
        context: dict[str, str] | None,
        group_by_alias: bool,
        having_alias: bool,
        strategies: dict[Backend, BackendStrategy] | None,
        views: dict[str, View] | None,
        viewer: AuthContext | None,
        policy: PolicyFn | None,
        scope_fns: dict[str, ScopeFn] | None,
        allow_unbounded_ungrouped: bool,
    ) -> None:
        _validate_query_invariants(q, allow_unbounded_ungrouped=allow_unbounded_ungrouped)

        self.q = q
        self.catalog = catalog
        self.group_by_alias = group_by_alias
        self.having_alias = having_alias
        self.strategies = strategies
        self.viewer = viewer
        self.policy = policy
        self.scope_fns = scope_fns

        # Compile context: caller-supplied substitutions + viewer
        # auto-flattening for ``{ctx.viewer_id}``.
        ctx = dict(context or {})
        if viewer is not None:
            ctx.setdefault("ctx.viewer_id", viewer.viewer_id)
        self.ctx = ctx
        self.views_map: dict[str, View] = views or {}

        resolved = _resolve_query_fields(q, catalog, self.views_map)
        self.measure_fields = resolved.measure_fields
        self.dim_fields = resolved.dim_fields
        self.time_cube = resolved.time_cube
        self.time_dim = resolved.time_dim
        self.filter_resolutions = resolved.filter_resolutions
        self.where_leaf_resolutions = resolved.where_leaf_resolutions
        self.segment_resolutions = resolved.segment_resolutions
        self.touched = resolved.touched

        _check_viewer_authorization(self.touched, viewer, policy)
        _check_required_filters(self.touched, q)

        self.backend = _pick_single_backend(self.touched)
        self.left_join_cubes: set[str] = set(q.left_joins)

        # Anti-join support: cubes named in ``q.left_joins`` get
        # bidirectional join-graph traversal so spine→facts queries
        # (FK on the fact side) can reach the fact cube. Validate
        # each name names a real cube up front; the join-graph BFS
        # raises later if a name was bogus or unreachable.
        for name in self.left_join_cubes:
            if name not in catalog:
                raise CompileError(
                    f"left_joins: cube {name!r} is not in the catalog. "
                    f"Known cubes: {sorted(catalog)}."
                )

        # Refuse a left-joined cube also appearing in dimensions —
        # GROUP BY NULL is a confusing silent gotcha for the
        # anti-join pattern.
        if self.left_join_cubes:
            dim_cubes = {c.name for c, _ in resolved.dim_fields}
            offending = self.left_join_cubes & dim_cubes
            if offending:
                raise CompileError(
                    f"left_joins: cube(s) {sorted(offending)} appear "
                    "in the query's ``dimensions`` — a LEFT-joined "
                    "cube's columns are NULL on absent rows, so GROUP "
                    "BY them gets a confusing NULL bucket. Drop those "
                    "dimensions and use ``Filter(op='is_null')`` on a "
                    "left-joined cube's column to express the anti-join."
                )

        self.cubes_in_from, self.join_edges = _build_join_graph(
            self.touched, catalog, left_join_cubes=self.left_join_cubes
        )
        self.root = self.cubes_in_from[0]

        self.cube_aliases: dict[str, str] = {c.name: c.alias for c in self.cubes_in_from}
        self.strategy = strategy_for(self.backend, strategies)
        self.dialect = dialect_for(self.backend)

        # Param binder state. Memoise ``(value, dim_type) → placeholder``
        # so a filter value referenced in both compare-mode CTEs binds
        # once — the database round-trip stays cheap and the planner's
        # intent (one filter, one bound value) is preserved.
        self.params: dict[str, Any] = {}
        self._binds: dict[tuple[Any, str], str] = {}

        # Output-column allocation. Collision-prefix any name that
        # appears more than once across dims + measures so each output
        # column is unique.
        proposed: list[str] = [d.name for _, d in self.dim_fields]
        proposed.extend(m.name for _, m in self.measure_fields)
        counts = Counter(proposed)
        self._collisions: set[str] = {n for n, c in counts.items() if c > 1}
        self.dim_col_names: list[str] = [self._col_name(c, d.name) for c, d in self.dim_fields]
        self.measure_col_names: list[str] = [
            self._col_name(c, m.name) for c, m in self.measure_fields
        ]

        self.has_time_breakdown: bool = (
            self.time_cube is not None
            and self.time_dim is not None
            and q.time_dimension is not None
            and q.time_dimension.granularity is not None
        )
        self.time_col_name: str | None = None
        if self.has_time_breakdown:
            assert self.time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            self.time_col_name = f"{self.time_dim.name}_{granularity}"

    # ------------------------------------------------------------------
    # Helpers — were inline closures inside ``compile_query`` before.
    # ------------------------------------------------------------------

    def _col_name(self, cube: Cube, field_name: str) -> str:
        return f"{cube.name}_{field_name}" if field_name in self._collisions else field_name

    def bind(self, value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        """Allocate (or reuse) a parameter placeholder for ``value``."""
        key = (value, dim_type)
        if key in self._binds:
            return self.strategy.placeholder(self._binds[key], dim_type)
        name = f"p{len(self.params)}"
        self.params[name] = value
        self._binds[key] = name
        return self.strategy.placeholder(name, dim_type)

    def resolve_in_ctx(self, sql: str) -> str:
        """Apply ``{alias}`` / ``{ctx.X}`` substitution to a raw SQL fragment."""
        return _resolve_sql(sql, self.cube_aliases, self.ctx)

    def parse(self, sql: str) -> exp.Expression:
        """Resolve + parse a SQL fragment in the env's dialect."""
        return _parse_fragment(self.resolve_in_ctx(sql), self.dialect)

    def _resolve_security_sql(self, cube: Cube, raw: str) -> str:
        """Substitute ``{alias}`` and ``{ctx.X}`` in a ``security_sql``
        fragment; ``{ctx.X}`` values bind as parameters so they never
        appear as SQL literals."""
        resolved = self.resolve_in_ctx(raw)

        def _ctx_repl(m: re.Match[str]) -> str:
            key = "ctx." + m.group(1)
            if key not in self.ctx:
                raise CompileError(
                    f"Cube {cube.name!r} security_sql references "
                    f"{{{key}}} but no {key!r} value was provided in "
                    "compile context."
                )
            return self.bind(self.ctx[key], "string").sql(
                dialect=self.dialect, normalize_functions=False
            )

        return _CTX_PLACEHOLDER_RE.sub(_ctx_repl, resolved)

    def wrap_for_tenancy(self, cube: Cube, source: exp.Expression) -> exp.Expression:
        """Wrap a cube's FROM source in an isolation subquery.

        Tenancy + ``security_sql`` + ScopeFn-injected predicates AND
        compose *inside* the alias the outer query sees so a malformed
        outer ``OR`` can't smuggle in rows the policy excludes."""
        predicates: list[exp.Expression] = []

        if cube.tenancy == "discriminator":
            if "tenant" not in self.ctx:
                raise CompileError(
                    f"Cube {cube.name!r} declares tenancy='discriminator' but "
                    "no 'tenant' value was provided in compile context. "
                    "Pass context={'tenant': <tenant_id>, ...}."
                )
            assert cube.tenancy_column is not None
            predicates.append(
                exp.EQ(
                    this=exp.column(cube.tenancy_column, table=cube.alias),
                    expression=self.bind(self.ctx["tenant"], "string"),
                )
            )

        if cube.security_sql:
            resolved_sql = self._resolve_security_sql(cube, cube.security_sql)
            predicates.append(_parse_fragment(resolved_sql, self.dialect))

        # ScopeFn-injected row-level predicate. Only fires when both
        # ``viewer`` and ``cube.scope`` are set and a function is
        # registered under that scope name.
        if self.viewer is not None and cube.scope is not None and self.scope_fns is not None:
            fn = self.scope_fns.get(cube.scope)
            if fn is not None:
                pred: ScopePredicate | None = fn(cube, self.viewer)
                if pred is not None:
                    missing = [k for k in pred.ctx_keys if k not in self.ctx]
                    if missing:
                        raise CompileError(
                            f"Cube {cube.name!r} scope {cube.scope!r} declares "
                            f"ctx_keys={pred.ctx_keys!r} but the following are not in "
                            f"the resolution context: {missing}."
                        )
                    predicates.append(
                        _parse_fragment(self._resolve_security_sql(cube, pred.sql), self.dialect)
                    )

        if not predicates:
            return source

        where_expr = predicates[0]
        for p in predicates[1:]:
            where_expr = exp.And(this=where_expr, expression=p)

        inner = exp.Select().select(exp.Star()).from_(source).where(where_expr)
        return exp.Subquery(
            this=inner,
            alias=exp.TableAlias(this=exp.to_identifier(cube.alias)),
        )

    def build_measure_expr(self, cube_owner: Cube, m: Measure) -> exp.Expression:
        """Compose the SELECT expression for one measure.

        Ratio measures recurse into a Div over their numerator /
        denominator (resolved by name on the same cube); everything
        else flows through ``_agg_node`` with an optional ``FILTER``
        wrapper for filtered measures."""
        if m.agg == "ratio":
            assert m.numerator is not None and m.denominator is not None
            num_m = next(
                (x for x in cube_owner.measures if x.name == m.numerator),
                None,
            )
            if num_m is None:
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio numerator "
                    f"{m.numerator!r} is not a measure on cube {cube_owner.name!r}."
                )
            den_m = next(
                (x for x in cube_owner.measures if x.name == m.denominator),
                None,
            )
            if den_m is None:
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio denominator "
                    f"{m.denominator!r} is not a measure on cube {cube_owner.name!r}."
                )
            if num_m.agg == "ratio" or den_m.agg == "ratio":
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio measures "
                    "cannot reference another ratio measure. Use leaf "
                    "(sum / count / avg / ...) measures as numerator "
                    "and denominator."
                )
            num_node = self.build_measure_expr(cube_owner, num_m)
            den_node = self.build_measure_expr(cube_owner, den_m)
            return exp.Div(
                this=num_node,
                expression=exp.Anonymous(
                    this="NULLIF",
                    expressions=[den_node, exp.Literal.number(0)],
                ),
            )

        if m.sql == "*":
            inner: exp.Expression = exp.Star()
        else:
            inner = self.parse(m.sql)

        # Percentile family routes through the strategy because the
        # SQL shape varies per dialect (PERCENTILE_CONT WITHIN GROUP
        # on PG/DuckDB/Snowflake; APPROX_QUANTILES[OFFSET] on BigQuery;
        # quantile(q)(expr) on ClickHouse). Plain aggs use the
        # dialect-agnostic exp.Sum / Count / etc. nodes via _agg_node.
        if m.agg in _PERCENTILE_AGGS:
            q_val = _PERCENTILE_AGGS[m.agg]
            agg: exp.Expression = self.strategy.emit_percentile(q_val, inner)
        else:
            agg = _agg_node(m, inner)
        if m.filter:
            agg = exp.Filter(
                this=agg,
                expression=exp.Where(this=self.parse(m.filter)),
            )
        return agg

    def build_inner(self, time_range: tuple[str, str]) -> exp.Select:
        """Build the inner aggregating Select.

        Filter values bind via ``self.bind`` so they dedupe across the
        current / prior compare-mode CTEs; only the per-CTE time-window
        binds differ.

        Orchestrates four named stages so each emission concern is a
        named method a reader can navigate to: FROM clause →
        projection → WHERE predicates → GROUP BY."""
        sel = exp.Select()
        sel = self._from_clause_stage(sel)
        sel = self._projection_stage(sel)
        sel = self._predicate_stage(sel, time_range)
        sel = self._group_by_stage(sel)
        return sel

    # ------------------------------------------------------------------
    # build_inner stages — each takes/returns the in-progress Select so
    # snapshot tests can pin per-stage IR if it ever earns its keep.
    # ------------------------------------------------------------------

    def _from_clause_stage(self, sel: exp.Select) -> exp.Select:
        """Attach the FROM root + LEFT JOINs across the catalog's join
        graph. Per-cube isolation subqueries (tenancy / security_sql /
        scope) wrap each source via ``wrap_for_tenancy``."""
        sel = sel.from_(
            self.wrap_for_tenancy(
                self.root,
                self.strategy.emit_source(self.root, self.catalog, self.resolve_in_ctx),
            )
        )
        for _, tgt, j in self.join_edges:
            tgt_strategy = strategy_for(tgt.backend, self.strategies)
            target_source = self.wrap_for_tenancy(
                tgt, tgt_strategy.emit_source(tgt, self.catalog, self.resolve_in_ctx)
            )
            sel = sel.join(target_source, on=self.parse(j.on), join_type="left")
        return sel

    def _projection_stage(self, sel: exp.Select) -> exp.Select:
        """Emit the SELECT list: dimensions, optional bucketed time
        dimension, measures. Adds DISTINCT for ungrouped row-listing
        queries with no measures."""
        q = self.q
        for (_cube, dim), col_name in zip(self.dim_fields, self.dim_col_names, strict=True):
            sel = sel.select(exp.alias_(self.parse(dim.sql), col_name))

        if self.has_time_breakdown:
            assert self.time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            assert self.time_col_name is not None
            trunc_node = self.strategy.trunc(granularity, self.parse(self.time_dim.sql))
            sel = sel.select(exp.alias_(trunc_node, self.time_col_name))

        for (cube_owner, m), col_name in zip(
            self.measure_fields, self.measure_col_names, strict=True
        ):
            sel = sel.select(exp.alias_(self.build_measure_expr(cube_owner, m), col_name))

        if not q.ungrouped and not self.measure_fields:
            sel.set("distinct", exp.Distinct())
        return sel

    def _predicate_stage(self, sel: exp.Select, time_range: tuple[str, str]) -> exp.Select:
        """AND-compose every WHERE-clause source: per-cube
        ``base_predicate`` (excluding META cubes), flat ``filters``,
        named segments, the boolean ``where`` tree, and the
        time-window range bounds."""
        q = self.q
        where_terms: list[exp.Expression] = []

        for cube_w in self.cubes_in_from:
            if cube_w.base_predicate and cube_w.backend is not Backend.META:
                where_terms.append(self.parse(cube_w.base_predicate))

        for f, _cube, fld in self.filter_resolutions:
            where_terms.append(self._filter_term(f, fld))

        for _seg_cube, segment in self.segment_resolutions:
            where_terms.append(self.parse(segment.sql))

        if q.where is not None:

            def _leaf_to_node(leaf: Filter) -> exp.Expression:
                _cube, fld = self.where_leaf_resolutions[id(leaf)]
                return self._filter_term(leaf, fld)

            where_terms.append(_compile_where_tree(q.where, _leaf_to_node))

        if self.time_dim is not None:
            where_terms.append(
                exp.GTE(
                    this=self.parse(self.time_dim.sql),
                    expression=self.bind(time_range[0], "time"),
                )
            )
            where_terms.append(
                exp.LT(
                    this=self.parse(self.time_dim.sql),
                    expression=self.bind(time_range[1], "time"),
                )
            )

        if where_terms:
            sel = sel.where(*where_terms)
        return sel

    def _filter_term(
        self,
        f: Filter,
        fld: Dimension | Measure | TimeDimension,
    ) -> exp.Expression:
        """Render one flat-filter / where-tree-leaf into a SQL predicate.

        Shared between ``filters`` and ``where`` so both paths use the
        same field-type lookup convention (Dimension.type wins; time
        dims bind as ``time``; measures fall back to ``string`` since
        the compiler doesn't model their value type)."""
        fld_node = self.parse(fld.sql)
        if isinstance(fld, Dimension):
            fld_type = fld.type
        elif isinstance(fld, TimeDimension):
            fld_type = "time"
        else:
            fld_type = "string"
        return _filter_node(f, fld_node, fld_type, self.strategy, self.bind)

    def _group_by_stage(self, sel: exp.Select) -> exp.Select:
        """Emit GROUP BY when the query has measures and isn't
        ``ungrouped``. Groups by every projected dimension; the bucket
        column when a time breakdown is in play. ``group_by_alias``
        controls whether GROUP BY references the SELECT alias or
        repeats the resolved expression."""
        q = self.q
        if q.ungrouped or not self.measure_fields:
            return sel

        for i, (_cube, dim) in enumerate(self.dim_fields):
            if self.group_by_alias:
                sel = sel.group_by(exp.column(self.dim_col_names[i]))
            else:
                sel = sel.group_by(self.parse(dim.sql))

        if self.has_time_breakdown:
            assert self.time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            assert self.time_col_name is not None
            if self.group_by_alias:
                sel = sel.group_by(exp.column(self.time_col_name))
            else:
                sel = sel.group_by(self.strategy.trunc(granularity, self.parse(self.time_dim.sql)))
        return sel

    def emit(self) -> Compiled:
        """Dispatch to the compare or non-compare emission helper."""
        if self.q.compare is not None:
            return _emit_compare_query(self)
        return _emit_simple_query(self)


# Synthetic compare-mode output refs: ``compare.<measure>.<facet>``
# where ``facet`` is one of ``current`` / ``prior`` / ``delta`` /
# ``pct_change``. Rewrites to the existing ``<col>_<facet>`` outer
# column. Lets ``order`` / ``having`` reference compare-mode
# derivatives without the caller having to know the column-aliasing
# convention. F3 in the gap analysis.
_COMPARE_FACETS: tuple[str, ...] = ("current", "prior", "delta", "pct_change")


def _resolve_compare_outer_ref(
    ref: str,
    outer_columns: list[str],
    measure_col_names: list[str],
    *,
    what: str,
) -> str:
    """Translate a compare-mode ``order`` / ``having`` reference into an
    actual outer-SELECT column name.

    Accepts two shapes:
    - The raw underscore alias (``revenue_delta``) — must be in
      ``outer_columns``.
    - The synthetic ``compare.<measure>.<facet>`` form — rewritten to
      ``<measure>_<facet>`` after validating ``<measure>`` is a measure
      in the query and ``<facet>`` is one of the four supported
      derivatives.

    Raises ``CompileError`` for any other shape so callers don't
    accidentally smuggle a raw inner-CTE column ref through to the
    outer SELECT."""
    if ref in outer_columns:
        return ref
    if ref.startswith("compare.") and ref.count(".") == 2:
        _, measure_name, facet = ref.split(".", 2)
        if facet not in _COMPARE_FACETS:
            raise CompileError(
                f"{what} {ref!r}: unknown compare facet {facet!r}. "
                f"Supported: {', '.join(_COMPARE_FACETS)}."
            )
        if measure_name not in measure_col_names:
            raise CompileError(
                f"{what} {ref!r}: measure {measure_name!r} is not in this "
                f"query's measures. Add it to ``measures`` or pick from "
                f"{measure_col_names}."
            )
        return f"{measure_name}_{facet}"
    raise CompileError(
        f"{what} {ref!r}: in compare mode, references must be an outer "
        f"output column (e.g. {measure_col_names[0]}_delta) or the "
        f"synthetic compare.<measure>.<facet> form. Got {ref!r}."
    )


def _emit_compare_query(env: _CompileEnv) -> Compiled:
    """Compose the current/prior FULL OUTER JOIN compare-mode output.

    Two CTEs (``current`` / ``prior``) wrap the same inner select with
    different time ranges; the outer SELECT COALESCEs the dim columns,
    projects per-measure ``<col>_current`` / ``<col>_prior`` /
    ``<col>_delta`` / ``<col>_pct_change``, and joins on every
    grouping column. ``pct_change`` guards divide-by-zero via
    ``CASE WHEN prior > 0 THEN ... END``.
    """
    q = env.q
    dim_col_names = env.dim_col_names
    measure_col_names = env.measure_col_names
    time_col_name = env.time_col_name

    assert q.compare is not None and q.time_dimension is not None
    current_range = q.time_dimension.range
    if q.compare.mode == "previous_period":
        cs = datetime.fromisoformat(current_range[0])
        ce = datetime.fromisoformat(current_range[1])
        duration = ce - cs
        prior_range: tuple[str, str] = (
            (cs - duration).isoformat(),
            cs.isoformat(),
        )
    else:
        assert q.compare.range is not None
        prior_range = q.compare.range

    current_inner = env.build_inner(current_range)
    prior_inner = env.build_inner(prior_range)

    outer = exp.Select()
    outer = outer.with_("current", current_inner)
    outer = outer.with_("prior", prior_inner)

    outer_columns: list[str] = []
    for col_name in dim_col_names:
        outer = outer.select(
            exp.alias_(
                exp.Anonymous(
                    this="COALESCE",
                    expressions=[
                        exp.column(col_name, table="current"),
                        exp.column(col_name, table="prior"),
                    ],
                ),
                col_name,
            )
        )
        outer_columns.append(col_name)
    if env.has_time_breakdown:
        assert time_col_name is not None
        outer = outer.select(
            exp.alias_(
                exp.Anonymous(
                    this="COALESCE",
                    expressions=[
                        exp.column(time_col_name, table="current"),
                        exp.column(time_col_name, table="prior"),
                    ],
                ),
                time_col_name,
            )
        )
        outer_columns.append(time_col_name)

    for col_name in measure_col_names:
        cur_ref = exp.column(col_name, table="current")
        pri_ref = exp.column(col_name, table="prior")
        cur_col = f"{col_name}_current"
        pri_col = f"{col_name}_prior"
        delta_col = f"{col_name}_delta"
        pct_col = f"{col_name}_pct_change"

        outer = outer.select(exp.alias_(cur_ref.copy(), cur_col))
        outer = outer.select(exp.alias_(pri_ref.copy(), pri_col))

        # delta: COALESCE both to 0 so the diff is meaningful when one
        # period is missing the entity.
        coalesced_cur = exp.Anonymous(
            this="COALESCE", expressions=[cur_ref.copy(), exp.Literal.number(0)]
        )
        coalesced_pri = exp.Anonymous(
            this="COALESCE", expressions=[pri_ref.copy(), exp.Literal.number(0)]
        )
        outer = outer.select(
            exp.alias_(
                exp.Sub(this=coalesced_cur, expression=coalesced_pri),
                delta_col,
            )
        )

        # pct_change: divide-by-zero guard. ``exp.Paren`` around the
        # Sub is load-bearing — sqlglot's renderer doesn't infer
        # precedence for binary ops, so ``Mul(Sub(a, b), 100)`` would
        # render as ``a - b * 100`` without the explicit paren.
        pct_expr = exp.Case(
            ifs=[
                exp.If(
                    this=exp.GT(this=pri_ref.copy(), expression=exp.Literal.number(0)),
                    true=exp.Div(
                        this=exp.Mul(
                            this=exp.Paren(
                                this=exp.Sub(
                                    this=cur_ref.copy(),
                                    expression=pri_ref.copy(),
                                )
                            ),
                            expression=exp.Literal.number(100.0),
                        ),
                        expression=pri_ref.copy(),
                    ),
                )
            ],
            default=exp.Null(),
        )
        outer = outer.select(exp.alias_(pct_expr, pct_col))

        outer_columns.extend([cur_col, pri_col, delta_col, pct_col])

    outer = outer.from_(exp.to_table("current"))
    join_dims = dim_col_names + ([time_col_name] if time_col_name else [])
    if join_dims:
        on_expr: exp.Expression | None = None
        for jd in join_dims:
            eq = exp.EQ(
                this=exp.column(jd, table="current"),
                expression=exp.column(jd, table="prior"),
            )
            on_expr = eq if on_expr is None else exp.And(this=on_expr, expression=eq)
        outer = outer.join(exp.to_table("prior"), on=on_expr, join_type="full outer")
    else:
        # No dims, no time bucket — cross product of single rows.
        outer = outer.join(
            exp.to_table("prior"),
            on=exp.Boolean(this=True),
            join_type="full outer",
        )

    for ref, direction in q.order:
        col = _resolve_compare_outer_ref(ref, outer_columns, measure_col_names, what="ORDER BY")
        outer = outer.order_by(exp.Ordered(this=exp.column(col), desc=(direction == "desc")))

    # HAVING — compare mode supports it against any outer column,
    # including the synthetic ``compare.<measure>.delta`` form. Most
    # dialects accept ``HAVING`` without ``GROUP BY`` as a top-level
    # row filter; the FULL OUTER JOIN's COALESCE'd-dim row identity
    # makes that exactly the right shape.
    for hf in q.having:
        col = _resolve_compare_outer_ref(
            hf.dimension, outer_columns, measure_col_names, what="HAVING"
        )
        outer = outer.having(_filter_node(hf, exp.column(col), "number", env.strategy, env.bind))

    if q.limit is not None:
        outer = outer.limit(int(q.limit))
    if q.offset is not None and q.offset > 0:
        outer = outer.offset(int(q.offset))

    outer = _apply_with_clause(
        outer, _collect_hoisted_ctes(env.touched, env.resolve_in_ctx), env.backend
    )
    sql = outer.sql(dialect=env.dialect, pretty=False, normalize_functions=False)
    cm = _build_column_meta(
        outer_columns,
        env.dim_fields,
        dim_col_names,
        env.measure_fields,
        measure_col_names,
        env.time_dim,
        time_col_name,
        is_compare=True,
    )
    return Compiled(
        backend=env.backend,
        sql=sql,
        params=env.params,
        columns=outer_columns,
        column_meta=cm,
        touched_cube_names=[c.name for c in env.touched],
        derived_sources=_collect_derived_sources(env.touched, env.resolve_in_ctx),
    )


def _emit_simple_query(env: _CompileEnv) -> Compiled:
    """Compose the single-SELECT (non-compare) output.

    Builds the inner aggregating SELECT via ``env.build_inner``,
    applies ``HAVING`` against a measure alias map, optionally wraps
    in a time-spine LEFT JOIN for ``fill_nulls_with``, then layers
    ``ORDER BY`` / ``LIMIT`` / ``OFFSET`` / hoisted CTEs.
    """
    q = env.q
    dim_col_names = env.dim_col_names
    measure_col_names = env.measure_col_names
    time_col_name = env.time_col_name

    assert q.time_dimension is None or env.time_dim is not None
    time_range_for_query: tuple[str, str] | None = (
        q.time_dimension.range if q.time_dimension is not None else None
    )
    select_node = env.build_inner(time_range_for_query or ("", ""))

    columns: list[str] = list(dim_col_names)
    if env.has_time_breakdown and time_col_name is not None:
        columns.append(time_col_name)
    columns.extend(measure_col_names)

    if not columns:
        raise CompileError("Compiled query has no SELECT projections.")

    # Map every measure ref (bare and qualified) to its agg expression so
    # HAVING / ORDER BY can address them without re-resolving.
    measure_alias_map: dict[str, exp.Expression] = {}
    for (cube_owner, m), col_name in zip(env.measure_fields, measure_col_names, strict=True):
        agg_node = env.build_measure_expr(cube_owner, m)
        measure_alias_map[m.name] = agg_node
        if col_name != m.name:
            measure_alias_map[col_name] = agg_node

    # Inline derived measures — ad-hoc ratios / sums / diffs composed
    # at query time over existing catalog measures. Each gets a fresh
    # SELECT projection + an alias-map entry so order / having can
    # address them by the derived measure's ``name``.
    for ir in q.derived_measures:
        node = _build_inline_derived_expr(ir, env, columns)
        select_node = select_node.select(exp.alias_(node, ir.name))
        columns.append(ir.name)
        measure_alias_map[ir.name] = node

    for hf in q.having:
        if hf.dimension.startswith("compare."):
            raise CompileError(
                f"HAVING {hf.dimension!r}: ``compare.<measure>.<facet>`` "
                "references are only valid when the query sets "
                "``compare=CompareWindow(...)``."
            )
        lookup_name = hf.dimension
        if lookup_name not in measure_alias_map and "." in lookup_name:
            lookup_name = lookup_name.rsplit(".", 1)[-1]
        if lookup_name not in measure_alias_map:
            raise CompileError(
                f"HAVING references {hf.dimension!r}, which is not a measure in this query."
            )
        target_node: exp.Expression
        if env.having_alias:
            target_node = exp.column(lookup_name)
        else:
            target_node = measure_alias_map[lookup_name].copy()
        select_node = select_node.having(
            _filter_node(hf, target_node, "number", env.strategy, env.bind)
        )

    # Time spine — wrap the aggregation in a CTE and LEFT JOIN a
    # spine CTE so every bucket in [start, end) gets a row.
    fill_value: int | None = (
        q.time_dimension.fill_nulls_with if q.time_dimension is not None else None
    )
    if fill_value is not None:
        if not env.has_time_breakdown:
            raise CompileError(
                "TimeWindow.fill_nulls_with requires a granularity on the "
                "time_dimension — the spine has nothing to step over otherwise."
            )
        if not env.measure_fields:
            raise CompileError(
                "TimeWindow.fill_nulls_with has no effect without measures — "
                "the COALESCE wraps a measure value, and a query with no "
                "measures has nothing to fill."
            )
        if env.dim_fields:
            raise CompileError(
                "TimeWindow.fill_nulls_with does not yet support non-time "
                "dimensions (Phase B: spine × dimension cartesian fill). "
                "Drop the non-time dimensions or run the query without fill."
            )
        if q.ungrouped:
            raise CompileError(
                "TimeWindow.fill_nulls_with is incompatible with ungrouped=True — "
                "spine fill aggregates by definition."
            )
        assert time_range_for_query is not None
        assert time_col_name is not None
        granularity_val = q.time_dimension.granularity if q.time_dimension is not None else None
        assert granularity_val is not None
        start_ph = env.bind(time_range_for_query[0], "time")
        end_ph = env.bind(time_range_for_query[1], "time")
        spine_inner = env.strategy.emit_time_spine(granularity_val, start_ph, end_ph, time_col_name)

        outer = exp.Select()
        outer = outer.with_("agg", select_node)
        outer = outer.with_("spine", spine_inner)
        outer = outer.select(exp.column(time_col_name, table="spine"))
        for col_name in measure_col_names:
            outer = outer.select(
                exp.alias_(
                    exp.Anonymous(
                        this="COALESCE",
                        expressions=[
                            exp.column(col_name, table="agg"),
                            exp.Literal.number(fill_value),
                        ],
                    ),
                    col_name,
                )
            )
        outer = outer.from_(exp.to_table("spine"))
        outer = outer.join(
            exp.to_table("agg"),
            on=exp.EQ(
                this=exp.column(time_col_name, table="spine"),
                expression=exp.column(time_col_name, table="agg"),
            ),
            join_type="left",
        )
        select_node = outer

    for ref, direction in q.order:
        if ref.startswith("compare."):
            raise CompileError(
                f"ORDER BY {ref!r}: ``compare.<measure>.<facet>`` "
                "references are only valid when the query sets "
                "``compare=CompareWindow(...)``."
            )
        if ref in measure_alias_map or ref in columns:
            order_target: exp.Expression = exp.column(ref)
        else:
            try:
                _, fld = _resolve_field(ref, env.catalog)
            except CompileError as exc:
                raise CompileError(
                    f"ORDER BY {ref!r}: must reference an output column or "
                    f"a known cube.field. ({exc})"
                ) from exc
            order_target = env.parse(fld.sql)
        select_node = select_node.order_by(
            exp.Ordered(this=order_target, desc=(direction == "desc"))
        )

    if q.limit is not None:
        select_node = select_node.limit(int(q.limit))
    if q.offset is not None and q.offset > 0:
        select_node = select_node.offset(int(q.offset))

    select_node = _apply_with_clause(
        select_node,
        _collect_hoisted_ctes(env.touched, env.resolve_in_ctx),
        env.backend,
    )
    sql = select_node.sql(dialect=env.dialect, pretty=False, normalize_functions=False)
    cm = _build_column_meta(
        columns,
        env.dim_fields,
        dim_col_names,
        env.measure_fields,
        measure_col_names,
        env.time_dim,
        time_col_name,
        is_compare=False,
    )
    return Compiled(
        backend=env.backend,
        sql=sql,
        params=env.params,
        columns=columns,
        column_meta=cm,
        touched_cube_names=[c.name for c in env.touched],
        derived_sources=_collect_derived_sources(env.touched, env.resolve_in_ctx),
    )


def _build_join_graph(
    touched: list[Cube],
    catalog: dict[str, Cube],
    *,
    left_join_cubes: set[str] | None = None,
) -> tuple[list[Cube], list[tuple[Cube, Cube, Join]]]:
    """BFS the catalog's Join edges from the first touched cube to
    every other touched cube. Returns ``(cubes_in_from, join_edges)``
    in the order the FROM clause + JOINs should be emitted.

    When a target cube is named in ``left_join_cubes``, the BFS walks
    edges bidirectionally — needed for spine→facts anti-join patterns
    where the FK lives on the fact cube but the FROM root is the
    spine. All other targets stay forward-only so normal queries can't
    accidentally find a surprising reverse path."""
    left_set: set[str] = left_join_cubes or set()
    root = touched[0]
    join_edges: list[tuple[Cube, Cube, Join]] = []
    cubes_in_from: list[Cube] = [root]
    for c in touched:
        if c is root:
            continue
        path = _find_join_path(
            root.name,
            c.name,
            catalog,
            bidirectional=c.name in left_set,
        )
        cursor = root
        for next_name, j in path:
            tgt = catalog[next_name]
            if tgt not in cubes_in_from:
                join_edges.append((cursor, tgt, j))
                cubes_in_from.append(tgt)
            cursor = tgt
    return cubes_in_from, join_edges


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compile_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    strategies: dict[Backend, BackendStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    _allow_unbounded_ungrouped: bool = False,
) -> Compiled:
    """Compile a SemanticQuery to a Compiled bundle.

    ``catalog`` — dict of cube name → Cube (build from ``Catalog.as_dict()``).
    ``context`` — optional string substitutions applied to ``{key}``
        placeholders in table names and SQL expressions
        (e.g. ``{"schema": "mydb"}``).
    ``group_by_alias`` — when True (default), GROUP BY references the
        SELECT output alias. Set False to repeat the resolved expression.
    ``having_alias`` — when False (default), HAVING repeats the aggregate
        expression. Set True only when you control the backend.
    ``strategies`` — optional per-backend strategy overrides. Pass a
        ``RecordingStrategy`` for tests; pass a custom Snowflake / BigQuery
        adapter for out-of-tree backends without touching the global registry.
    ``viewer`` — optional ``AuthContext``. When set, queries touching a
        cube the viewer can't see (``Cube.required_roles`` ANY-match +
        optional ``policy``) raise ``CompileError`` before SQL emission.
        ``viewer.viewer_id`` auto-binds to ``ctx.viewer_id`` so
        ``security_sql`` referencing it gets a parameter.
    ``policy`` — optional custom-visibility predicate (cube, viewer) → bool.
        AND-composes with ``Cube.required_roles``.
    ``scope_fns`` — registry of named ``ScopeFn`` callables. When a
        ``Cube.scope`` names a key in this dict, the compiler calls the
        function with ``(cube, viewer)`` and AND-injects the returned
        ``ScopePredicate`` inside the cube's isolation subquery.
    """
    env = _CompileEnv(
        q,
        catalog,
        context=context,
        group_by_alias=group_by_alias,
        having_alias=having_alias,
        strategies=strategies,
        views=views,
        viewer=viewer,
        policy=policy,
        scope_fns=scope_fns,
        allow_unbounded_ungrouped=_allow_unbounded_ungrouped,
    )
    return env.emit()


__all__ = [
    "Compiled",
    "CompileError",
    "CrossBackendError",
    "FilterTypeError",
    "JoinPathError",
    "MAX_UNGROUPED_ROWS",
    "PhaseDeferredError",
    "PlaceholderError",
    "UnknownIdentifierError",
    "compile_query",
]
