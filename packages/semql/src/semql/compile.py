# pyright: reportPrivateImportUsage=false
# sqlglot's AST types (Expression, Placeholder, ...) are imported via
# ``from sqlglot import exp``; they aren't in ``sqlglot.expressions.__all__``
# but are public by convention and by sqlglot's type stubs.
"""Pure compiler from `SemanticQuery` to backend SQL.

The compiler has no I/O. It reads the catalog, resolves identifiers,
emits a parameterised SQL string + params dict + output column list,
and raises ``CompileError`` (or a more specific leaf) on unknown
references, unreachable joins, or unsupported shapes.

The body composes a sqlglot ``exp.Select`` AST and renders it via the
target backend's dialect. Per-cube fragments (dim SQL, measure SQL,
``base_predicate``, ``Join.on``) are parsed by sqlglot under the
catalog cube's declared backend; dialect-specific shapes
(``placeholder``, ``trunc``, ``contains``, ``emit_source``) come from
the ``DialectStrategy`` and slot into the AST as nodes.

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

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from semql._resolve import (
    _ResolvedFields,
    walk_query_fields,
)
from semql._resolve import (
    resolve_field as _resolve_field_raw,
)
from semql.backend import DialectStrategy, dialect_for

# Importing from semql.dialect also registers the ClickHouse placeholder
# override against the ``clickhouse`` dialect name (side effect on import).
from semql.dialect import dialect_for as sqlglot_dialect_for
from semql.errors import (
    CompileError,
    CrossDialectError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    UnknownIdentifierError,
)
from semql.introspect import PolicyFn, ScopeFn, viewer_sees
from semql.logical import LogicalPlan, output_alias, output_column_collisions
from semql.model import (
    AggLiteral,
    AuthContext,
    Cube,
    DerivedTable,
    Dialect,
    Dimension,
    FormatLiteral,
    Measure,
    ScopePredicate,
    Segment,
    StorageType,
    TimeDimension,
    TimePartitionedSource,
    View,
)
from semql.rollup import apply_rollup, pick_rollup
from semql.spec import BoolExpr, Filter, InlineDerived, SemanticQuery

MAX_UNGROUPED_ROWS = 1000


ColumnKind = Literal["measure", "dimension", "time", "computed"]


def _infer_measure_storage_type(agg: AggLiteral) -> StorageType:
    """Pick the tightest storage type implied by a measure's aggregate.

    ``count`` / ``count_distinct`` always produce non-negative integers.
    ``avg`` / ``ratio`` / quantiles always produce floats. ``sum`` /
    ``min`` / ``max`` pass the source column's type through — we'd need
    to parse the underlying SQL to tell int vs float, so we fall back
    to the generic ``number`` literal (which still satisfies
    ``Filter.validate_for_type``)."""
    if agg in ("count", "count_distinct"):
        return "integer"
    if agg in ("avg", "ratio", "median", "p75", "p90", "p95"):
        return "float"
    return "number"


@dataclass
class ColumnMeta:
    """Per-output-column type + presentation metadata.

    Sits on :class:`CompiledQuery` in the same order as ``columns`` so a
    consumer (dashboard, MCP server, presenter LLM) can render a result
    row without re-resolving against the catalog. ``kind`` tags the
    column's role; ``storage_type`` carries the tightest type the
    compiler can infer (passed through from ``Dimension.type`` for
    dimensions, derived from ``agg`` for measures, ``"time"`` for time
    columns); ``unit`` / ``display_unit`` / ``format`` mirror the
    fields on ``Measure`` / ``Dimension``; ``display_name`` carries the
    human-friendly label (catalog ``display_name`` or a humanised
    fallback) so visualisers don't need to call ``humanize`` themselves.

    ``computed`` covers compare-mode derivatives (``foo_delta``,
    ``foo_pct_change``) — values the catalog doesn't directly name
    but the compiler emits. Their ``format`` is set inline (e.g.
    ``"percent"`` for pct_change) so renderers don't need to guess.
    """

    name: str
    kind: ColumnKind
    display_name: str = ""
    unit: str | None = None
    display_unit: str | None = None
    format: FormatLiteral | None = None
    storage_type: StorageType | None = None
    # A1 — True when the column's SQL was substituted by a mask
    # constant (NULL or ``mask_value``) rather than the real field
    # expression. Downstream renderers (chart, table) use this to
    # suppress cell rendering or show a lock icon.
    masked: bool = False

    def model_dump(self) -> dict[str, Any]:
        """JSON-safe dict view of this ColumnMeta (I9 round-trip helper)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "display_name": self.display_name,
            "unit": self.unit,
            "display_unit": self.display_unit,
            "format": self.format,
            "storage_type": self.storage_type,
            "masked": self.masked,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> ColumnMeta:
        """Reconstruct a ColumnMeta from a ``model_dump()`` payload."""
        return cls(
            name=data["name"],
            kind=data["kind"],
            display_name=data.get("display_name", ""),
            unit=data.get("unit"),
            display_unit=data.get("display_unit"),
            format=data.get("format"),
            storage_type=data.get("storage_type"),
            masked=data.get("masked", False),
        )


@dataclass
class CompiledQuery:
    backend: Dialect
    sql: str
    params: dict[str, Any]
    columns: list[str]
    column_meta: list[ColumnMeta] = dc_field(default_factory=lambda: [])
    # Names of every cube the query touched, in first-mention order.
    # Lets downstream tools (visualiser, MCP envelope) avoid re-running
    # the resolver against the catalog for cube-level facts.
    touched_cube_names: list[str] = dc_field(default_factory=lambda: [])
    # Resolved SQL of every ``DerivedTable`` source the query touched,
    # in first-mention order matching ``touched_cube_names``. Surfaces
    # the *second* place raw SQL legitimately enters the catalog
    # (the first is the outer ``sql``) so callers running
    # :func:`semql.safe.is_read_only_statement` or dialect snapshots see every
    # raw fragment, not just the compiler-generated SELECT. Plain-table
    # cubes contribute nothing here.
    derived_sources: list[str] = dc_field(default_factory=lambda: [])
    # Name of the rollup the compiler routed the query against, when
    # rollup routing fired. ``None`` means the query went to the base
    # tables. See :mod:`semql.rollup` for the matching rules.
    applied_rollup: str | None = None
    # Names of the physical sources a partitioned cube (#48) actually
    # scanned, in declaration order. ``()`` for cubes without
    # ``physical_sources`` or when the query's time range fell outside
    # every source (a zero-row subquery was emitted instead). See
    # :mod:`semql.partition` for the routing rules.
    physical_sources_hit: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        """JSON-safe dict view of this CompiledQuery (I9 round-trip helper).

        Stable across versions for the same query + catalog. The shape
        matches Pydantic's ``BaseModel.model_dump`` so callers writing
        tool-call payloads / eval-loop fixtures can swap to a plain dict
        without changing the call site.
        """
        return {
            "backend": self.backend.value,
            "sql": self.sql,
            "params": dict(self.params),
            "columns": list(self.columns),
            "column_meta": [m.model_dump() for m in self.column_meta],
            "touched_cube_names": list(self.touched_cube_names),
            "derived_sources": list(self.derived_sources),
            "applied_rollup": self.applied_rollup,
            "physical_sources_hit": list(self.physical_sources_hit),
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> CompiledQuery:
        """Reconstruct a CompiledQuery from a ``model_dump()`` payload.

        Pairs with :meth:`model_dump` for I9 — the round-trip is
        byte-stable: ``CompiledQuery.model_validate(cq.model_dump())``
        equals ``cq`` field-for-field.
        """
        backend = data["backend"]
        if isinstance(backend, str):
            backend = Dialect(backend)
        column_meta = [ColumnMeta.model_validate(m) for m in data.get("column_meta", [])]
        return cls(
            backend=backend,
            sql=data["sql"],
            params=dict(data.get("params", {})),
            columns=list(data.get("columns", [])),
            column_meta=column_meta,
            touched_cube_names=list(data.get("touched_cube_names", [])),
            derived_sources=list(data.get("derived_sources", [])),
            applied_rollup=data.get("applied_rollup"),
            physical_sources_hit=tuple(data.get("physical_sources_hit", ())),
        )


def _collect_derived_sources(
    touched: list[Cube],
    resolve_sql: Callable[[str], str],
) -> list[str]:
    """Materialize the resolved SQL of every ``DerivedTable`` source the
    query touched. Plain-table cubes contribute nothing.

    Order matches ``touched`` (first-mention). For each derived cube,
    any ``with_ctes`` are emitted first (declaration order) followed by
    the main ``sql`` — so a static checker like ``is_read_only_statement``
    walking ``derived_sources`` sees the same fragments that actually
    enter the compiled query."""
    out: list[str] = []
    for cube in touched:
        if cube.physical_sources:
            # Time-partitioned cubes (#48) have no single
            # ``DerivedTable`` to materialise — each physical
            # source is a plain table reference. The matched
            # sources' SQL is captured by the dialect's emit
            # path and is also surfaced via
            # ``physical_sources_hit``; nothing to add here.
            continue
        src = cube.resolved_source
        if isinstance(src, DerivedTable):
            for cte in src.with_ctes:
                out.append(resolve_sql(cte.sql))
            out.append(resolve_sql(src.sql))
    return out


def _empty_source_for(cube: Cube) -> exp.Subquery:
    """Build a zero-row subquery aliased to ``cube.alias``.

    Used as a fallback when a cube with ``physical_sources`` has
    no source that intersects the query's time range. The
    outer query's predicates still apply, so the result is
    empty by construction — no rows leak across the time
    boundary, and the cube is still queryable (the rest of the
    pipeline continues to run)."""
    inner = exp.Select().select(exp.Literal.number(1)).where(exp.Boolean(this=False))
    return exp.Subquery(
        this=inner,
        alias=exp.TableAlias(this=exp.to_identifier(cube.alias)),
    )


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
        if cube.physical_sources:
            # Time-partitioned cubes (#48) have no single
            # ``CubeSource`` to resolve — the source set is a
            # list of physical tables, each without CTEs.
            continue
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
    backend: Dialect,
) -> exp.Select:
    """Attach a ``WITH`` clause hoisting every ``ctes`` entry to the front
    of ``select_node``.

    Uses sqlglot's ``Select.with_`` helper which returns a new node
    carrying the CTE. ``as_`` strings get re-parsed in the cube's
    dialect so backend-specific shapes (ClickHouse ARRAY JOIN, BigQuery
    UNNEST) survive the round-trip. No-op when ``ctes`` is empty."""
    if not ctes:
        return select_node
    dialect = sqlglot_dialect_for(backend)
    out: exp.Select = select_node
    for name, body_sql in ctes:
        out = out.with_(name, as_=body_sql, dialect=dialect)
    return out


def _resolve_field(
    qualified: str,
    catalog: dict[str, Cube],
) -> tuple[Cube, Measure | Dimension | TimeDimension | Segment]:
    try:
        return _resolve_field_raw(qualified, catalog)
    except CompileError:
        raise
    except ResolveError as exc:
        raise CompileError(str(exc)) from exc


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


def _apply_mask_metadata(
    column_meta: list[ColumnMeta],
    env: _CompileEnv,
) -> list[ColumnMeta]:
    """A1 — flip ``ColumnMeta.masked`` for every field the viewer has
    a mask role on. Returns a new list; ``ColumnMeta`` is frozen, so
    we rebuild each masked entry.
    """
    if env.viewer is None:
        return column_meta
    masked_pairs: set[tuple[str, str]] = set()  # (col_name, kind)
    for (_cube, dim), col_name in zip(env.dim_fields, env.dim_col_names, strict=True):
        if env.field_is_masked(dim):
            masked_pairs.add((col_name, "dimension"))
    for (_cube, m), col_name in zip(env.measure_fields, env.measure_col_names, strict=True):
        if env.field_is_masked(m):
            masked_pairs.add((col_name, "measure"))
    if not masked_pairs:
        return column_meta
    out: list[ColumnMeta] = []
    for cm in column_meta:
        if (cm.name, cm.kind) in masked_pairs:
            out.append(
                ColumnMeta(
                    name=cm.name,
                    kind=cm.kind,
                    display_name=cm.display_name,
                    unit=cm.unit,
                    display_unit=cm.display_unit,
                    format=cm.format,
                    storage_type=cm.storage_type,
                    masked=True,
                )
            )
        else:
            out.append(cm)
    return out


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
            storage_type=dim.type,
        )

    if time_dim is not None and time_col_name is not None:
        # Drop the granularity suffix (``_day`` / ``_week`` / ...) from
        # the display label — readers don't need "Created At Day", they
        # need "Created At" plotted on a time axis.
        by_name[time_col_name] = ColumnMeta(
            name=time_col_name,
            kind="time",
            display_name=time_dim.display_name or _humanize(time_dim.name),
            storage_type="time",
        )

    for (_cube, m), col_name in zip(measure_fields, measure_col_names, strict=True):
        m_label = m.display_name or _humanize(col_name)
        m_storage = _infer_measure_storage_type(m.agg)
        by_name[col_name] = ColumnMeta(
            name=col_name,
            kind="measure",
            display_name=m_label,
            unit=m.unit,
            display_unit=m.display_unit,
            format=m.format,
            storage_type=m_storage,
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
                    storage_type=m_storage,
                )
            pct_name = col_name + "_pct_change"
            by_name[pct_name] = ColumnMeta(
                name=pct_name,
                kind="computed",
                display_name=m_label + " (% change)",
                format="percent",
                storage_type="float",
            )

    out: list[ColumnMeta] = []
    for c in columns:
        out.append(by_name.get(c) or ColumnMeta(name=c, kind="computed", display_name=_humanize(c)))
    return out


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
                f"Unknown placeholder {{{name}}} in catalog SQL. Known: {sorted(lookup)}.",
                placeholder=name,
                known=sorted(lookup),
            )
        return lookup[name]

    return _PLACEHOLDER_RE.sub(_repl, sql)


# ---------------------------------------------------------------------------
# Aggregation — measure agg → sqlglot node
# ---------------------------------------------------------------------------


# Percentile aggregation literals → continuous quantile values for the
# dialect's ``emit_percentile`` hook. ``median`` is the q=0.5 case;
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
    dialect: DialectStrategy,
    bind: Callable[[Any, str], exp.Placeholder],
) -> exp.Expression:
    """Build a predicate node for a Filter.

    Dialect-specific shapes (``contains``) go through the dialect. Type
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
        return dialect.emit_contains(field, str(f.values[0]), bind)

    placeholders: list[exp.Placeholder] = [bind(v, field_type) for v in f.values]

    if op in _FILTER_BINOPS:
        return _FILTER_BINOPS[op](this=field, expression=placeholders[0])

    in_args: list[exp.Expression] = list(placeholders)
    if op == "in":
        return exp.In(this=field, expressions=in_args)
    if op == "not_in":
        return exp.Not(this=exp.In(this=field, expressions=in_args))

    raise CompileError(f"Unsupported filter op: {op!r}")  # pragma: no cover


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


# SQL reserved words that must be quoted when used as bare identifiers.
# sqlglot's own per-dialect ``RESERVED_KEYWORDS`` are inconsistent — empty
# for Postgres/ClickHouse, only partial for DuckDB — so a fragment like
# ``{t}.order`` emits unquoted, invalid SQL under Postgres. We quote against
# this curated set ourselves (C7, ktx-ports M1). Source: ANSI SQL:2016
# reserved words plus the common dialect keywords that collide with real
# column names. Quoting a reserved word is safe in every supported dialect;
# ordinary identifiers (``region``, ``amount``) are left untouched.
_RESERVED_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "all",
        "and",
        "any",
        "are",
        "array",
        "as",
        "asc",
        "asensitive",
        "asymmetric",
        "at",
        "authorization",
        "begin",
        "between",
        "both",
        "by",
        "case",
        "cast",
        "check",
        "collate",
        "column",
        "commit",
        "condition",
        "connect",
        "constraint",
        "create",
        "cross",
        "cube",
        "current",
        "current_date",
        "current_time",
        "current_timestamp",
        "current_user",
        "cursor",
        "date",
        "day",
        "dec",
        "decimal",
        "declare",
        "default",
        "delete",
        "desc",
        "describe",
        "deterministic",
        "distinct",
        "do",
        "drop",
        "each",
        "element",
        "else",
        "elseif",
        "end",
        "escape",
        "except",
        "exec",
        "execute",
        "exists",
        "exit",
        "external",
        "false",
        "fetch",
        "filter",
        "for",
        "foreign",
        "from",
        "full",
        "function",
        "grant",
        "group",
        "grouping",
        "having",
        "hour",
        "if",
        "in",
        "index",
        "inner",
        "inout",
        "insensitive",
        "insert",
        "intersect",
        "interval",
        "into",
        "is",
        "join",
        "key",
        "language",
        "leading",
        "leave",
        "left",
        "level",
        "like",
        "limit",
        "loop",
        "match",
        "merge",
        "minute",
        "month",
        "natural",
        "not",
        "null",
        "of",
        "offset",
        "on",
        "only",
        "open",
        "or",
        "order",
        "outer",
        "over",
        "partition",
        "position",
        "primary",
        "procedure",
        "range",
        "rank",
        "references",
        "rename",
        "repeat",
        "replace",
        "reset",
        "return",
        "returns",
        "revoke",
        "right",
        "rollback",
        "rollup",
        "row",
        "rows",
        "schema",
        "scroll",
        "second",
        "select",
        "sensitive",
        "session_user",
        "set",
        "similar",
        "some",
        "specific",
        "sql",
        "start",
        "symmetric",
        "system",
        "system_user",
        "table",
        "then",
        "time",
        "timestamp",
        "to",
        "trailing",
        "trigger",
        "true",
        "union",
        "unique",
        "unnest",
        "until",
        "update",
        "user",
        "using",
        "value",
        "values",
        "when",
        "whenever",
        "where",
        "while",
        "window",
        "with",
        "within",
        "year",
    }
)


def _quote_reserved_identifiers(node: exp.Expression) -> exp.Expression:
    """Force-quote identifiers whose name is a SQL reserved word.

    Walks the parsed fragment and marks any unquoted ``exp.Identifier``
    matching :data:`_RESERVED_IDENTIFIERS` as quoted, so the generator
    emits it as a delimited identifier under any dialect. Identifiers the
    author already quoted, and string literals, are left as-is."""
    for ident in node.find_all(exp.Identifier):
        if not ident.args.get("quoted") and str(ident.this).lower() in _RESERVED_IDENTIFIERS:
            ident.set("quoted", True)
    return node


@functools.lru_cache(maxsize=256)
def _parse_fragment_cached(sql: str, dialect: str) -> exp.Expression:
    """Parse + reserved-word-quote a fragment once per ``(sql, dialect)`` (C9).

    The same ``expr`` strings recur many times across a compilation; this
    memoises the parse. ``lru_cache`` does not cache exceptions, so a bad
    fragment still raises ``CompileError`` on every call. Callers must not
    mutate the returned node — use :func:`_parse_fragment`, which copies."""
    try:
        # sqlglot's parse_one is stubbed to return the ``Expr`` TypeVar, not
        # bare ``Expression``; narrow it here (same quirk the prior code
        # suppressed with type: ignore[return-value]).
        parsed: exp.Expression = sqlglot.parse_one(sql, dialect=dialect)  # type: ignore[assignment]
    except ParseError as exc:
        raise CompileError(
            f"Could not parse catalog SQL fragment {sql!r} under dialect {dialect!r}: {exc}"
        ) from exc
    return _quote_reserved_identifiers(parsed)


def _parse_fragment(sql: str, dialect: str) -> exp.Expression:
    """Parse a catalog SQL fragment into a sqlglot AST node.

    Used for dim/measure/time-dimension expressions, ``Join.on``, and
    ``base_predicate``. Reserved-word identifiers are force-quoted (C7) so
    a column named after a keyword emits valid SQL.

    Returns an independent ``.copy()`` of the cached parse (C9): callers
    reparent the node into a larger expression and the reserved-word pass
    mutates it, so each caller needs its own tree."""
    return _parse_fragment_cached(sql, dialect).copy()


# ---------------------------------------------------------------------------
# Compile stages — extracted from ``compile_query`` for testability and to
# make the orchestration legible. Each stage is a pure function over its
# inputs; ``compile_query`` glues them together. POSA "Pipes and Filters".
# ---------------------------------------------------------------------------


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


def _resolve_query_fields(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    views_map: dict[str, View],
) -> _ResolvedFields:
    """Resolve every ``cube.field`` reference in ``q`` to its catalog
    entry and collect the ordered set of touched cubes.

    Thin wrapper over :func:`semql._resolve.walk_query_fields`. The
    shared walker accumulates per-reference diagnostics without
    raising; this wrapper translates them into the compile-time error
    contract: a single combined ``CompileError`` listing every
    problem, or — when exactly one diagnostic carries a typed source
    (``FilterTypeError`` / ``UnknownIdentifierError``) — that typed
    exception standalone so UIs branching on the leaf class still
    receive it."""
    resolved, diagnostics = walk_query_fields(q, catalog, views_map=views_map)
    if diagnostics:
        if len(diagnostics) == 1:
            src = diagnostics[0].source
            if isinstance(src, CompileError):
                raise src
            raise CompileError(diagnostics[0].message)
        lines = [f"  - {d.message}" for d in diagnostics]
        raise CompileError(
            f"SemanticQuery has {len(diagnostics)} resolution errors:\n" + "\n".join(lines)
        )

    if not resolved.touched:
        raise CompileError("Could not determine any cubes from the query.")

    return resolved


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


def _check_field_visibility(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    viewer: AuthContext | None,
) -> None:
    """A1 — Field hide gate. Refuse queries that touch a field the
    viewer's roles don't intersect with ``field.required_roles``.

    The error message is indistinguishable from "field doesn't exist"
    so callers can't infer the field exists from the error shape.
    Empty ``required_roles`` is open to all viewers who can see the
    cube; ``viewer=None`` (unauthed path) bypasses the gate so
    catalog-level tooling keeps working.
    """
    if viewer is None:
        return

    def _sees(field_required: list[str]) -> bool:
        if not field_required:
            return True
        return any(r in viewer.roles for r in field_required)

    # Walk every ref: measures, dimensions, time_dimension, segments,
    # filters (dimension), where-tree leaves, having (dimension), order.
    refs: list[str] = []
    refs.extend(q.measures)
    refs.extend(q.dimensions)
    if q.time_dimension is not None:
        refs.append(q.time_dimension.dimension)
    refs.extend(q.segments)
    for f in q.filters:
        refs.append(f.dimension)
    if q.where is not None:
        refs.extend(_walk_where_dims(q.where))
    for hf in q.having:
        refs.append(hf.dimension)
    for ref, _direction in q.order:
        refs.append(ref)
    for ir in q.derived_measures:
        # InlineDerived operands are qualified ``cube.measure`` refs
        # (or unqualified names that resolve to the touched cubes).
        # Push them through the same field-visibility gate so a
        # viewer without the operand's required role can't probe
        # through a derived measure.
        for op in ir.operands:
            if "." in op:
                refs.append(op)

    for ref in refs:
        cube_name, _, field_name = ref.partition(".")
        if not cube_name or not field_name:
            continue
        cube = catalog.get(cube_name)
        if cube is None:
            continue
        field = _find_field(cube, field_name)
        if field is None:
            continue
        if not _sees(field.required_roles):
            # Indistinguishable from "field doesn't exist".
            hint = _closest_match(
                field_name, [f.name for f in cube.measures + cube.dimensions + cube.time_dimensions]
            )
            known = sorted(f.name for f in cube.measures + cube.dimensions + cube.time_dimensions)
            suffix = f" Did you mean {hint!r}?" if hint else ""
            raise UnknownIdentifierError(
                f"Unknown field {field_name!r} on cube {cube_name!r}. "
                f"Known fields: {known}.{suffix}",
                kind="field",
                name=field_name,
                cube=cube_name,
                hint=hint,
            )


def _walk_where_dims(node: BoolExpr | Filter) -> list[str]:
    """Walk a BoolExpr / Filter tree and collect every ``Filter.dimension`` ref."""
    from semql.spec import BoolExpr, Filter

    out: list[str] = []
    if isinstance(node, Filter):
        out.append(node.dimension)
    elif isinstance(node, BoolExpr):  # pyright: ignore[reportUnnecessaryIsInstance] — defensive
        for c in node.children:
            out.extend(_walk_where_dims(c))
    return out


def _find_field(
    cube: Cube, field_name: str
) -> Measure | Dimension | TimeDimension | Segment | None:
    """Return the field on ``cube`` matching ``field_name``, or None."""
    for m in cube.measures:
        if m.name == field_name:
            return m
    for d in cube.dimensions:
        if d.name == field_name:
            return d
    for td in cube.time_dimensions:
        if td.name == field_name:
            return td
    for s in cube.segments:
        if s.name == field_name:
            return s
    return None


def _closest_match(name: str, candidates: list[str]) -> str | None:
    """Tiny typo-tolerance for the field-hide gate (no info leak — we
    only suggest names from the visible set)."""
    if not candidates:
        return None
    name_lower = name.lower()
    # Prefix match first (the most common case).
    for c in candidates:
        if c.lower().startswith(name_lower[:3]):
            return c
    return candidates[0]


def _check_lifecycle(touched: list[Cube]) -> None:
    """Refuse queries that touch a ``deprecated`` cube. ``beta`` flows
    through unchanged — the planner sees a "beta" annotation in the
    prompt fragment (slice 3) and chooses whether to use it.

    The error names every deprecated cube touched in one shot (instead
    of stopping at the first), so a query against two deprecated cubes
    gets one error listing both. The replacement pointer is surfaced
    per cube — clients can route users to the successor."""
    offenders = [c for c in touched if c.stability == "deprecated"]
    if not offenders:
        return
    parts: list[str] = []
    for c in offenders:
        if c.replacement is not None:
            parts.append(f"{c.name!r} (use {c.replacement!r} instead)")
        else:
            parts.append(f"{c.name!r} (no replacement; remove the reference)")
    raise CompileError("Query touches deprecated cube(s): " + ", ".join(parts) + ".")


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


def _check_alias_uniqueness(touched: list[Cube]) -> None:
    """Refuse a query whose participating cubes share a SQL alias.

    The compiler emits ``FROM a AS <alias> JOIN b AS <alias> ON ...``
    verbatim; if two distinct cubes joined in one query carry the same
    ``alias``, every ``{alias}.col`` reference is ambiguous — silently
    wrong SQL. This fires per query rather than catalog-wide, so variant
    cubes that share an alias but are never co-queried stay legal."""
    owner: dict[str, str] = {}
    for c in touched:
        prior = owner.get(c.alias)
        if prior is not None and prior != c.name:
            raise CompileError(
                f"Query joins cubes {prior!r} and {c.name!r}, which share "
                f"SQL alias {c.alias!r} — every {{{c.alias}}} reference would "
                f"be ambiguous. Give one cube a distinct alias."
            )
        owner[c.alias] = c.name


# Aggregates whose value is inflated when a join duplicates the rows of
# the cube the measure lives on. ``min`` / ``max`` / ``count_distinct``
# are invariant under row duplication and stay safe; ``avg`` / ``ratio`` /
# quantiles are distorted only by *uneven* duplication and are left to the
# join-graph rework (W3) rather than refused with this coarser check.
_FAN_OUT_SENSITIVE_AGGS: frozenset[str] = frozenset({"sum", "count"})


def _check_fan_out(
    measure_fields: list[tuple[Cube, Measure]],
    join_edges: list[tuple[Cube, Cube, Any]],
) -> None:
    """Refuse a query whose join graph fans out an additive measure.

    A ``one_to_many`` / ``many_to_one`` join duplicates the rows of its
    "one" side; ``SUM`` / ``COUNT`` over a measure on that cube then
    double-counts — the canonical semantic-layer wrong result. The
    cardinality is read from ``Join.relationship`` (this is its first
    reader) and from ``Join.to`` rather than the plan's left/right
    assignment, so spine-rooting that flips an edge can't fool it: the
    "one" side is intrinsic to the declared relationship.

    Scoped to a single query's join edges, so a cube whose measures only
    fan out under *some other* join still aggregates fine on its own."""
    duplicated: dict[str, tuple[str, str]] = {}  # cube -> (other cube, relationship)
    for left, right, join in join_edges:
        target = getattr(join, "to", None)
        rel = getattr(join, "relationship", None)
        names = {left.name, right.name}
        if target not in names or len(names) != 2:
            continue  # self-join or malformed edge — not a plain fan-out
        declarer = (names - {target}).pop()
        if rel == "many_to_one":
            # many ``declarer`` rows per one ``target`` row → target duplicates.
            duplicated.setdefault(target, (declarer, rel))
        elif rel == "one_to_many":
            # one ``declarer`` row per many ``target`` rows → declarer duplicates.
            duplicated.setdefault(declarer, (target, rel))
    if not duplicated:
        return
    for cube, m in measure_fields:
        if m.agg in _FAN_OUT_SENSITIVE_AGGS and cube.name in duplicated:
            other, rel = duplicated[cube.name]
            raise CompileError(
                f"Measure {cube.name}.{m.name} ({m.agg}) fans out: the "
                f"{rel} join between {cube.name!r} and {other!r} duplicates "
                f"{cube.name!r}'s rows, so {m.agg.upper()} would over-count. "
                f"Aggregate it without traversing that join, or pre-aggregate "
                f"{cube.name!r} to the join grain."
            )


def _pick_single_dialect(touched: list[Cube]) -> Dialect:
    """Single-backend gate. Cross-backend queries route to
    :func:`semql.compile_federated_query` — non-federated compile is
    one-backend-only."""
    backends = {c.backend for c in touched}
    if len(backends) > 1:
        backend_names = sorted(b.value for b in backends)
        raise CrossDialectError(
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
        dialects: dict[Dialect, DialectStrategy] | None,
        views: dict[str, View] | None,
        viewer: AuthContext | None,
        policy: PolicyFn | None,
        scope_fns: dict[str, ScopeFn] | None,
        allow_unbounded_ungrouped: bool,
        plan: LogicalPlan | None = None,
    ) -> None:
        # ``plan`` — a prebuilt :class:`LogicalPlan` the env must *trust*
        # verbatim instead of re-lowering ``q`` itself.  ``compile_plan``
        # passes the caller's (possibly transformed) plan here: a scan
        # rewritten to another physical table, a join dropped by the
        # federation split-point, a predicate pushed down.  When given,
        # the rollup / partition *plan* transforms are the caller's job
        # — the env does not re-run ``to_logical_plan`` and does not
        # re-pick a rollup (which would re-derive a fresh, untransformed
        # plan and discard the caller's work — the original A1 bug).
        # ``q`` must still be a faithful description of the plan (the
        # resolution caches + pre-flight checks read it); ``compile_plan``
        # reconstructs it from the plan to guarantee that.
        self._prebuilt_plan = plan
        _validate_query_invariants(q, allow_unbounded_ungrouped=allow_unbounded_ungrouped)

        # Rollup routing — check before resolution. When a rollup
        # covers the query, the plan→plan transform rewrites the
        # matched Scan to point at the rollup's physical_table.  For
        # now the catalog is also rewritten to the synthetic Cube so
        # the existing emission path (which reads from ``self.catalog``)
        # continues to work transparently.  A future refactor can drop
        # the catalog rewrite once emission reads scans from
        # ``self.plan.scans``.  The picked name is surfaced on
        # ``CompiledQuery.applied_rollup``.
        self.applied_rollup: str | None = None
        picked = None if plan is not None else pick_rollup(q, catalog)
        if picked is not None:
            rollup_cube, rollup = picked
            catalog = apply_rollup(catalog, rollup_cube, rollup)
            self.applied_rollup = rollup.name

        # Time-partitioned source routing (#48). For each cube with
        # ``physical_sources``, intersect the query's TimeWindow.range
        # with each source's range and stash the matches here. The
        # _from_clause_stage reads this to dispatch the partitioned
        # emit path; ``CompiledQuery.physical_sources_hit`` exposes
        # the matched names for observability.
        self.physical_sources_matched: dict[str, tuple[str, ...]] = {}

        self.q = q
        self.catalog = catalog
        self.group_by_alias = group_by_alias
        self.having_alias = having_alias
        self.dialects = dialects
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

        _check_lifecycle(self.touched)
        _check_viewer_authorization(self.touched, viewer, policy)
        _check_field_visibility(q, catalog, viewer)
        _check_required_filters(self.touched, q)
        _check_alias_uniqueness(self.touched)

        self.backend = _pick_single_dialect(self.touched)
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

        # LogicalPlan — lower the SemanticQuery to the IR.  This is the
        # single source of truth for emission: every post-build read
        # (scans, joins, project, aggregate, time-window granularity,
        # order, limit, filters) goes through ``self.plan`` rather
        # than the spec tree.  The join graph, projection columns, and
        # aggregate / time / order / limit fields are all derived
        # from the plan below — the legacy ``cubes_in_from`` /
        # ``join_edges`` / ``dim_fields`` / ``measure_fields`` /
        # ``time_dim`` slots survive as derived caches so downstream
        # emission helpers can read them without re-walking the plan
        # repeatedly.
        from semql.logical import apply_rollup_to_plan, to_logical_plan

        if plan is not None:
            # Trust the caller's plan verbatim.  No re-lowering, no
            # rollup transform — any plan→plan rewrite (rollup /
            # partition routing, split-point join drops, predicate
            # pushdown) the caller applied is load-bearing and must
            # survive to emission.
            self.plan = plan
        else:
            self.plan = to_logical_plan(q, catalog, views=self.views_map, resolved=resolved)

            # Apply the plan→plan rollup transform if a rollup was
            # picked.  The catalog is also rewritten (legacy path) so
            # backends that read field SQLs from the catalog still see
            # the rolled-up version; emission reads field SQLs from
            # ``self.plan`` (via ``ColumnRef.field.sql``) so the rewrite
            # is no longer load-bearing — it stays for now so the rest
            # of the pipeline (auth wrappers, ``_collect_derived_sources``)
            # keeps working transparently.
            if picked is not None:
                rollup_cube, rollup = picked
                rollup_plan = apply_rollup_to_plan(self.plan, rollup_cube, rollup)
                # The plan→plan transform produces a fresh logical cube;
                # we still rewrite the catalog so the auth / tenancy
                # code paths see the rolled-up shape.
                catalog = apply_rollup(catalog, rollup_cube, rollup)
                self.applied_rollup = rollup.name
                self.plan = rollup_plan

        # Time-partitioned source routing (#48). For each cube in the
        # join graph that declares ``physical_sources``, intersect
        # the plan's ``time_window.range`` with each source's range
        # and stash the matched names. ``_from_clause_stage`` reads
        # this to dispatch the partitioned emit path.
        from semql.partition import select_physical_sources

        for s in self.plan.scans:
            c = s.cube
            if c.physical_sources:
                matched = select_physical_sources(c, self.plan.time_window)
                self.physical_sources_matched[c.name] = tuple(s.name for s in matched)

        # Derive the join-graph view from the plan. The plan's
        # ``scans`` is the list of cubes the FROM clause sees
        # (root first, joins appended in BFS order); ``joins`` is
        # the list of edges.  The legacy ``cubes_in_from`` /
        # ``join_edges`` fields are kept as derived caches — same
        # shape as before so the rest of the emission code reads
        # them transparently.
        self.cubes_in_from: list[Cube] = [s.cube for s in self.plan.scans]
        self.join_edges: list[tuple[Cube, Cube, Any]] = [
            (j.left, j.right, j.model) for j in self.plan.joins
        ]
        self.root: Cube = self.cubes_in_from[0]

        # Now that the query's join graph is resolved, refuse additive
        # measures that the joins would fan out (silently inflated SUM /
        # COUNT). Needs the edges, so it runs here rather than in the
        # touched-cube prelude above.
        _check_fan_out(self.measure_fields, self.join_edges)

        self.cube_aliases: dict[str, str] = {c.name: c.alias for c in self.cubes_in_from}
        self.dialect = dialect_for(self.backend, dialects)
        self.sqlglot_dialect = sqlglot_dialect_for(self.backend)

        # Param binder state. Memoise ``(value, dim_type) → placeholder``
        # so a filter value referenced in both compare-mode CTEs binds
        # once — the database round-trip stays cheap and the planner's
        # intent (one filter, one bound value) is preserved.
        self.params: dict[str, Any] = {}
        self._binds: dict[tuple[Any, str], str] = {}

        # Output-column allocation. Collision-prefix any name that
        # appears more than once across dims + measures so each output
        # column is unique. Shared with the plan's projection via
        # ``output_column_collisions`` / ``output_alias`` so the two can't
        # disagree (review B1: the convention used to exist twice).
        self._collisions: set[str] = output_column_collisions(
            [d.name for _, d in self.dim_fields],
            [m.name for _, m in self.measure_fields],
        )
        self.dim_col_names: list[str] = [self._col_name(c, d.name) for c, d in self.dim_fields]
        self.measure_col_names: list[str] = [
            self._col_name(c, m.name) for c, m in self.measure_fields
        ]
        # I14 placeholder — populated after the time-block below,
        # once ``self.time_col_name`` is set.
        self.alias_map: dict[str, str] = {}

        # Time-block: read granularity from the plan, not the spec tree.
        # The plan's ``time_window`` is the lowered representation of
        # ``q.time_dimension`` — same value, single source of truth.
        plan_time_window = self.plan.time_window
        self.has_time_breakdown: bool = (
            self.time_cube is not None
            and self.time_dim is not None
            and plan_time_window is not None
            and plan_time_window.granularity is not None
        )
        self.time_col_name: str | None = None
        if self.has_time_breakdown:
            assert self.time_dim is not None and plan_time_window is not None
            granularity = plan_time_window.granularity
            assert granularity is not None
            self.time_col_name = f"{self.time_dim.name}_{granularity}"

        # I14 — Output column aliases. Validate every alias (resolves
        # to a declared field; key doesn't collide with an existing
        # output column). Build a ``col_name -> alias_key`` map so
        # projection / order / having can substitute the alias key
        # for the original column name. Lives after the time-block
        # so ``self.time_col_name`` is available.
        if q.aliases:
            existing_cols: set[str] = set(self.dim_col_names) | set(self.measure_col_names)
            if self.time_col_name is not None:
                existing_cols.add(self.time_col_name)
            for alias_key, alias_ref in q.aliases.items():
                # Resolve the alias target; reject unknown fields.
                try:
                    ref_cube, ref_field = _resolve_field(alias_ref, catalog)
                except CompileError:
                    raise CompileError(
                        f"aliases[{alias_key!r}] = {alias_ref!r}: "
                        f"field does not exist on any touched cube."
                    ) from None
                # Reject collisions with existing output column names.
                if alias_key in existing_cols:
                    raise CompileError(
                        f"aliases[{alias_key!r}]: alias key collides with "
                        f"an existing output column. Pick a distinct name."
                    )
                # The original col_name for this field is what
                # projection emits; we substitute the alias key in
                # its place. ``_col_name`` returns either the bare
                # name or ``{cube}_{name}`` for collisions.
                original_col = self._col_name(ref_cube, ref_field.name)
                # If the alias target is a measure / dim that's
                # already in the query, replace its column name with
                # the alias key. If the target is a measure / dim
                # NOT in the query, we'd need to add it — out of
                # scope for I14 (keep the contract simple: aliases
                # only re-label already-selected fields).
                if original_col not in existing_cols:
                    raise CompileError(
                        f"aliases[{alias_key!r}] = {alias_ref!r}: target "
                        "field is not selected in this query. I14 "
                        "aliases re-label existing outputs only."
                    )
                self.alias_map[original_col] = alias_key
                existing_cols.add(alias_key)

    # ------------------------------------------------------------------
    # Helpers — were inline closures inside ``compile_query`` before.
    # ------------------------------------------------------------------

    def _col_name(self, cube: Cube, field_name: str) -> str:
        return output_alias(cube.name, field_name, self._collisions)

    def field_is_masked(self, field: Measure | Dimension | TimeDimension) -> bool:
        """A1 mask gate — viewer has any role in ``field.mask_roles``.

        The field-hide gate (a different error path) already filters
        fields whose ``required_roles`` the viewer doesn't intersect.
        If we got here, the viewer has the required role; we just
        check whether they also have a mask role.
        """
        if self.viewer is None:
            return False
        # ``TimeDimension`` doesn't carry ``mask_roles`` — only
        # ``Measure`` and ``Dimension`` do.  Return early for the
        # time-dim case (the projection's ``kind="time"`` branch
        # passes the source ``TimeDimension`` through here so the
        # bucketed column participates in the same mask check).
        mask_roles = getattr(field, "mask_roles", None)
        if not mask_roles:
            return False
        return any(r in self.viewer.roles for r in mask_roles)

    def _masked_field_expr(
        self,
        field: Measure | Dimension | TimeDimension,
        default_expr: exp.Expression,
        col_name: str,
    ) -> exp.Expression:
        """A1 mask substitution — return the field's SQL or a constant.

        ``None`` mask value → ``CAST(NULL AS <inferred_type>)``.
        String mask value → literal SQL (caller is responsible for
        proper SQL syntax — e.g. ``"'REDACTED'"`` with quotes).
        """
        if not self.field_is_masked(field):
            return default_expr
        # ``TimeDimension`` doesn't carry ``mask_value`` — fall
        # through to the truncation expression (the time bucket
        # isn't itself sensitive; the dim it slices is).
        mask_value = getattr(field, "mask_value", None)
        if mask_value is not None:
            from sqlglot import exp

            return exp.Literal.string(mask_value.strip("'\""))
        # Default to a NULL cast whose type matches the column's
        # storage. For measures we infer from ``agg``; for dimensions
        # we read the declared ``type``; fall back to plain NULL.
        from sqlglot import exp

        if isinstance(field, Measure):
            storage = _infer_measure_storage_type(field.agg)
        else:
            storage = field.type
        return exp.Cast(this=exp.Null(), to=exp.DataType.build(storage))

    def bind(self, value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        """Allocate (or reuse) a parameter placeholder for ``value``."""
        key = (value, dim_type)
        if key in self._binds:
            return self.dialect.placeholder(self._binds[key], dim_type)
        name = f"p{len(self.params)}"
        self.params[name] = value
        self._binds[key] = name
        return self.dialect.placeholder(name, dim_type)

    def resolve_in_ctx(self, sql: str) -> str:
        """Apply ``{alias}`` / ``{ctx.X}`` substitution to a raw SQL fragment."""
        return _resolve_sql(sql, self.cube_aliases, self.ctx)

    def parse(self, sql: str) -> exp.Expression:
        """Resolve + parse a SQL fragment in the env's dialect."""
        return _parse_fragment(self.resolve_in_ctx(sql), self.sqlglot_dialect)

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
                dialect=self.sqlglot_dialect, normalize_functions=False
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
            predicates.append(_parse_fragment(resolved_sql, self.sqlglot_dialect))

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
                        _parse_fragment(
                            self._resolve_security_sql(cube, pred.sql), self.sqlglot_dialect
                        )
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

        # Percentile family routes through the dialect because the
        # SQL shape varies per dialect (PERCENTILE_CONT WITHIN GROUP
        # on PG/DuckDB/Snowflake; APPROX_QUANTILES[OFFSET] on BigQuery;
        # quantile(q)(expr) on ClickHouse). Plain aggs use the
        # dialect-agnostic exp.Sum / Count / etc. nodes via _agg_node.
        if m.agg in _PERCENTILE_AGGS:
            q_val = _PERCENTILE_AGGS[m.agg]
            agg: exp.Expression = self.dialect.emit_percentile(q_val, inner)
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
        scope) wrap each source via ``wrap_for_tenancy``.

        The root + join targets are read from
        ``self.plan.scans`` / ``self.plan.joins`` (LOGb — the plan
        is the single source of truth).  The legacy
        derived-cube / derived-edge slots on the env survive for
        any helpers that still reach past the plan; emission here
        never touches them.

        For cubes with ``physical_sources`` (#48), the per-source
        match list was computed at env-init; the dialect's regular
        ``emit_source`` is bypassed and the partitioned emit path
        (a single subquery over the matched sources, all aliased
        to ``cube.alias``) takes its place. ``wrap_for_tenancy``
        then wraps that single subquery, so the auth / tenancy /
        scope predicates apply to the union as a whole."""
        from semql.partition import emit_physical_sources

        plan_scans = self.plan.scans
        plan_joins = self.plan.joins
        root = plan_scans[0].cube

        if root.physical_sources:
            matched = self._matched_physical_sources_for(root)
            # Query range outside every source — emit a
            # zero-row subquery aliased to the cube's alias.
            # The outer query's predicates still apply, so the
            # result is empty by construction.
            source: exp.Expression = (
                emit_physical_sources(root, matched) if matched else _empty_source_for(root)
            )
        else:
            source = self.dialect.emit_source(root, self.catalog, self.resolve_in_ctx)
        sel = sel.from_(self.wrap_for_tenancy(root, source))
        for plan_join in plan_joins:
            tgt = plan_join.right
            j = plan_join.model
            tgt_dialect = dialect_for(tgt.backend, self.dialects)
            if tgt.physical_sources:
                matched_t = self._matched_physical_sources_for(tgt)
                target_source: exp.Expression = (
                    emit_physical_sources(tgt, matched_t) if matched_t else _empty_source_for(tgt)
                )
            else:
                target_source = tgt_dialect.emit_source(tgt, self.catalog, self.resolve_in_ctx)
            sel = sel.join(
                self.wrap_for_tenancy(tgt, target_source),
                on=self.parse(j.on),
                join_type=plan_join.kind,
            )
        return sel

    def _matched_physical_sources_for(self, cube: Cube) -> list[TimePartitionedSource]:
        """Return the list of ``TimePartitionedSource`` objects that
        matched for ``cube`` at env-init time. The list preserves
        declaration order so the emitted SQL is stable."""

        # Re-derive the list (cheap) — the env stores names, not
        # the objects themselves, because the same source set
        # shouldn't be re-validated on every emit call.
        matched_names = self.physical_sources_matched.get(cube.name, ())
        return [s for s in cube.physical_sources if s.name in matched_names]

    def all_matched_physical_source_names(self) -> tuple[str, ...]:
        """Flat tuple of every physical source that matched across
        all touched cubes, in cube-then-source declaration order.
        Empty for catalogs that don't use ``physical_sources``."""
        out: list[str] = []
        for cube in self.cubes_in_from:
            if cube.physical_sources:
                out.extend(self.physical_sources_matched.get(cube.name, ()))
        return tuple(out)

    def _projection_stage(self, sel: exp.Select) -> exp.Select:
        """Emit the SELECT list. Source: ``self.plan.project.columns`` —
        one ``ColumnRef`` per output column, carrying the resolved
        field object (``Measure`` / ``Dimension`` / ``TimeDimension``)
        and the alias.

        The legacy projection-derived slots on the env survive for
        callers (e.g. ``_apply_mask_metadata``) but emission here
        never reads them — the plan is the single source of truth
        for the projection.
        """
        # Cache the lookup so we don't re-walk the ColumnRef list
        # per column.  The same dim/measure field list still drives
        # the collision-prefix table populated at env-init.
        col_by_name: dict[str, str] = {}
        for col_name in self.dim_col_names:
            col_by_name[col_name] = "dimension"
        for col_name in self.measure_col_names:
            col_by_name[col_name] = "measure"
        if self.time_col_name is not None:
            col_by_name[self.time_col_name] = "time"

        measure_count = 0
        for col in self.plan.project.columns:
            col_name = col.alias
            out_name = self.alias_map.get(col_name, col_name)
            if col.kind == "dimension":
                assert col.field is not None and isinstance(col.field, Dimension)
                expr = self._masked_field_expr(col.field, self.parse(col.field.sql), col_name)
            elif col.kind == "time":
                assert col.field is not None and isinstance(col.field, TimeDimension)
                assert self.plan.aggregate is not None
                assert self.plan.aggregate.time is not None
                granularity = self.plan.aggregate.time.granularity
                assert granularity is not None
                expr = self._masked_field_expr(
                    col.field,
                    self.dialect.trunc(granularity, self.parse(col.field.sql)),
                    col_name,
                )
            elif col.kind == "measure":
                assert col.field is not None and isinstance(col.field, Measure)
                measure_count += 1
                # ``build_measure_expr`` needs (cube, measure) — both
                # are on the ColumnRef.
                expr = self._masked_field_expr(
                    col.field, self.build_measure_expr(col.cube, col.field), col_name
                )
            else:  # "computed" — derived / inline measure, deferred to ``_emit_simple_query``
                continue
            sel = sel.select(exp.alias_(expr, out_name))

        if self.plan.aggregate is not None and measure_count == 0:
            # Row-listing mode with no measures — DISTINCT collapses
            # the duplicate rows the join produced.  ``plan.aggregate
            # is None`` is the ungrouped case; ``measure_count == 0``
            # with an aggregate is the "SELECT DISTINCT dim" pattern.
            sel.set("distinct", exp.Distinct())
        return sel

    def _predicate_stage(self, sel: exp.Select, time_range: tuple[str, str]) -> exp.Select:
        """AND-compose every WHERE-clause source: per-cube
        ``base_predicate`` (excluding META cubes), the plan's
        ``Predicate`` nodes (carrying the spec tree from ``q.filters``
        and ``q.where``), named segments, and the time-window range
        bounds.

        The plan's ``Predicate`` nodes are the single source of truth
        for the user-supplied predicates — read from there so any
        plan-level transform (CNF, pushdown) is reflected in
        emission without a separate code path.
        """
        where_terms: list[exp.Expression] = []

        for cube_w in self.cubes_in_from:
            if cube_w.base_predicate and cube_w.backend is not Dialect.META:
                where_terms.append(self.parse(cube_w.base_predicate))

        for pred in self.plan.filters:
            where_terms.append(self._predicate_term(pred.expr))

        for _seg_cube, segment in self.segment_resolutions:
            where_terms.append(self.parse(segment.sql))

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

    def _predicate_term(self, expr: BoolExpr | Filter) -> exp.Expression:
        """Render one plan Predicate into a SQL predicate.

        Flat ``Filter`` leaves reuse ``_filter_term`` (which knows how
        to bind values).  ``BoolExpr`` trees reuse ``_compile_where_tree``
        so the AND/OR/NOT combinators emit identically to the
        pre-plan path.
        """

        if isinstance(expr, Filter):
            # Look up the field for the leaf.  ``filter_resolutions``
            # keys by Filter object identity; fall back to
            # ``where_leaf_resolutions`` for leaves in the where tree.
            fld = self._lookup_filter_field(expr)
            return self._filter_term(expr, fld)

        def _leaf_to_node(leaf: Filter) -> exp.Expression:
            fld = self._lookup_filter_field(leaf)
            return self._filter_term(leaf, fld)

        return _compile_where_tree(expr, _leaf_to_node)

    def _lookup_filter_field(self, leaf: Filter) -> Dimension | Measure | TimeDimension | Segment:
        """Find the field for a filter leaf, considering both the
        flat ``filters`` list and the where-tree leaves.

        Keyed by the leaf's qualified ``dimension``, not object identity:
        the resolved field depends only on which ``cube.field`` the leaf
        names, so a leaf an IR transform *copied* (federation split-point,
        CNF routing) resolves the same as the original (review B6)."""
        for f, _cube, fld in self.filter_resolutions:
            if f.dimension == leaf.dimension:
                return fld
        pair = self.where_leaf_resolutions.get(leaf.dimension)
        if pair is not None:
            return pair[1]
        raise CompileError(
            f"Filter leaf {leaf.dimension!r} could not be resolved to a "
            "catalog field.  The plan recorded a Predicate node whose "
            "field is not in the resolution results."
        )

    def _filter_term(
        self,
        f: Filter,
        fld: Dimension | Measure | TimeDimension | Segment,
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
        return _filter_node(f, fld_node, fld_type, self.dialect, self.bind)

    def _group_by_stage(self, sel: exp.Select) -> exp.Select:
        """Emit GROUP BY from the plan's ``Aggregate`` node.

        ``plan.aggregate is None`` is the row-listing case (no
        GROUP BY).  Otherwise, group by every projected
        ``kind="dimension"`` ColumnRef, then by the time bucket
        when the plan carries a ``kind="time"`` ColumnRef.
        ``group_by_alias`` controls whether GROUP BY references
        the SELECT alias or repeats the resolved expression.

        Source: ``self.plan.aggregate`` — the plan is the single
        source of truth for the aggregation shape.
        """
        aggregate = self.plan.aggregate
        if aggregate is None:
            return sel

        # Group by the SELECT alias when configured.  The plan's
        # ``Project.columns`` is in the same order the SELECT list
        # was emitted, so a parallel walk over the dim columns
        # matches the SELECT list.
        dim_idx = 0
        for col in self.plan.project.columns:
            if col.kind == "dimension":
                if self.group_by_alias:
                    sel = sel.group_by(exp.column(self.dim_col_names[dim_idx]))
                else:
                    assert col.field is not None
                    sel = sel.group_by(self.parse(col.field.sql))
                dim_idx += 1

        if aggregate.time is not None and self.time_col_name is not None:
            assert self.plan.time_window is not None
            granularity = self.plan.time_window.granularity
            assert granularity is not None
            if self.group_by_alias:
                sel = sel.group_by(exp.column(self.time_col_name))
            else:
                # Find the time-dim field for the trunc source.
                td = next(
                    (
                        c.field
                        for c in self.plan.project.columns
                        if c.kind == "time" and c.field is not None
                    ),
                    None,
                )
                if td is not None:
                    sel = sel.group_by(self.dialect.trunc(granularity, self.parse(td.sql)))
        return sel

    def emit(self) -> CompiledQuery:
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


def _emit_compare_query(env: _CompileEnv) -> CompiledQuery:
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

    # Ranges come from the plan's CompareSplit — the single place the
    # prior-range math runs (``to_logical_plan``). The emitter trusts the
    # plan rather than recomputing from ``q.compare`` (review B1: the math
    # used to be computed twice and the plan node was write-only).
    assert env.plan.compare is not None
    current_range = env.plan.compare.current_range
    prior_range = env.plan.compare.prior_range

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

    for ref, direction in env.plan.order.keys:
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
        outer = outer.having(_filter_node(hf, exp.column(col), "number", env.dialect, env.bind))

    if env.plan.limit.limit is not None:
        outer = outer.limit(int(env.plan.limit.limit))
    if env.plan.limit.offset is not None and env.plan.limit.offset > 0:
        outer = outer.offset(int(env.plan.limit.offset))

    outer = _apply_with_clause(
        outer, _collect_hoisted_ctes(env.touched, env.resolve_in_ctx), env.backend
    )
    sql = outer.sql(dialect=env.sqlglot_dialect, pretty=False, normalize_functions=False)
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
    cm = _apply_mask_metadata(cm, env)
    return CompiledQuery(
        backend=env.backend,
        sql=sql,
        params=env.params,
        columns=outer_columns,
        column_meta=cm,
        touched_cube_names=[c.name for c in env.touched],
        derived_sources=_collect_derived_sources(env.touched, env.resolve_in_ctx),
        applied_rollup=env.applied_rollup,
        physical_sources_hit=env.all_matched_physical_source_names(),
    )


def _emit_simple_query(env: _CompileEnv) -> CompiledQuery:
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

    assert env.plan.time_window is None or env.time_dim is not None
    time_range_for_query: tuple[str, str] | None = (
        env.plan.time_window.range if env.plan.time_window is not None else None
    )
    select_node = env.build_inner(time_range_for_query or ("", ""))

    columns: list[str] = list(dim_col_names)
    if env.has_time_breakdown and time_col_name is not None:
        columns.append(time_col_name)
    columns.extend(measure_col_names)
    # I14: substitute alias keys for original column names in the
    # output column list. order_by / having also use the alias key
    # because the user wrote it that way — the SQL was already
    # rendered with the alias key in projection.
    if env.alias_map:
        columns = [env.alias_map.get(c, c) for c in columns]

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

    # I14: HAVING may reference an alias key (e.g. ``net``) instead
    # of the underlying measure name. Build a reverse map so we can
    # substitute the canonical measure name before lookup.
    alias_to_measure: dict[str, str] = {}
    for alias_key, alias_ref in q.aliases.items():
        # ``alias_ref`` is ``cube.field`` — extract the field name.
        if "." in alias_ref:
            alias_to_measure[alias_key] = alias_ref.rsplit(".", 1)[-1]

    for hf in q.having:
        if hf.dimension.startswith("compare."):
            raise CompileError(
                f"HAVING {hf.dimension!r}: ``compare.<measure>.<facet>`` "
                "references are only valid when the query sets "
                "``compare=CompareWindow(...)``."
            )
        lookup_name = hf.dimension
        # I14: alias key → underlying measure name.
        if lookup_name in alias_to_measure:
            lookup_name = alias_to_measure[lookup_name]
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
            _filter_node(hf, target_node, "number", env.dialect, env.bind)
        )

    # Time spine — wrap the aggregation in a CTE and LEFT JOIN a
    # spine CTE so every bucket in [start, end) gets a row.
    fill_value: int | None = (
        env.plan.time_window.fill_nulls_with if env.plan.time_window is not None else None
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
        granularity_val = (
            env.plan.time_window.granularity if env.plan.time_window is not None else None
        )
        assert granularity_val is not None
        start_ph = env.bind(time_range_for_query[0], "time")
        end_ph = env.bind(time_range_for_query[1], "time")
        spine_inner = env.dialect.emit_time_spine(granularity_val, start_ph, end_ph, time_col_name)

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

    for ref, direction in env.plan.order.keys:
        if ref.startswith("compare."):
            raise CompileError(
                f"ORDER BY {ref!r}: ``compare.<measure>.<facet>`` "
                "references are only valid when the query sets "
                "``compare=CompareWindow(...)``."
            )
        # I14: alias key — already in the columns list, so the
        # ``ref in columns`` branch below matches.
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

    if env.plan.limit.limit is not None:
        select_node = select_node.limit(int(env.plan.limit.limit))
    if env.plan.limit.offset is not None and env.plan.limit.offset > 0:
        select_node = select_node.offset(int(env.plan.limit.offset))

    select_node = _apply_with_clause(
        select_node,
        _collect_hoisted_ctes(env.touched, env.resolve_in_ctx),
        env.backend,
    )
    sql = select_node.sql(dialect=env.sqlglot_dialect, pretty=False, normalize_functions=False)
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
    cm = _apply_mask_metadata(cm, env)
    return CompiledQuery(
        backend=env.backend,
        sql=sql,
        params=env.params,
        columns=columns,
        column_meta=cm,
        touched_cube_names=[c.name for c in env.touched],
        derived_sources=_collect_derived_sources(env.touched, env.resolve_in_ctx),
        applied_rollup=env.applied_rollup,
        physical_sources_hit=env.all_matched_physical_source_names(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def explain_plan(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    viewer: AuthContext | None = None,
) -> LogicalPlan:
    """Return the :class:`LogicalPlan` the compiler will emit for ``q``.

    Public companion to :func:`compile_query` for callers that want
    the plan-shape surface (debugging, MCP ``explain`` tool, plan
    visualisation) without the cost of building the sqlglot AST.

    Same diagnostic surface as ``compile_query``: lifecycle checks,
    authorisation, required-filters, cross-backend validation all
    run; the same resolution + join-graph + rollup routing happens.
    The plan is the IR a future emission-from-plan refactor will
    consume directly; today it's a printable inspection surface.
    """
    env = _CompileEnv(
        q,
        catalog,
        context=context,
        group_by_alias=True,
        having_alias=False,
        dialects=None,
        views=None,
        viewer=viewer,
        policy=None,
        scope_fns=None,
        allow_unbounded_ungrouped=False,
    )
    return env.plan


def compile_plan(
    plan: LogicalPlan,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    dialects: dict[Dialect, DialectStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    _allow_unbounded_ungrouped: bool = False,
) -> CompiledQuery:
    """Compile a :class:`LogicalPlan` directly to a :class:`CompiledQuery`.

    LOGb entry point — the plan is the single source of truth for
    emission.  Used by:

    - The federation split-point
      (:func:`semql.federate.compile_federated_query` calls
      :func:`partition_scans` to derive per-backend plans, then this
      function compiles each).
    - Callers that want the MCP ``explain``-then-emit flow
      (build a plan, inspect it, then compile to SQL).
    - Round-trip tests that precompute a plan and assert the
      emitted SQL matches the spec-tree path
      (see ``test_logb_ii_plan_driven_compile``).

    The plan is *trusted verbatim*: ``_CompileEnv`` emits from this
    plan and does not re-lower it.  A matching ``SemanticQuery`` is
    re-derived from the plan's fields only to run the pre-flight
    checks (``_validate_query_invariants``, ``_check_lifecycle``,
    auth / visibility) and to populate the resolution caches — it is
    *not* re-planned, so any plan→plan transform the caller applied
    (a scan rewritten to another physical table, a join dropped by
    the federation split-point, a predicate pushed down) survives to
    emission.  For an untransformed plan the end-to-end SQL is
    byte-identical to :func:`compile_query` on the equivalent query —
    the plan is a strict intermediate representation; the spec-tree
    path and the plan path agree exactly.
    """
    from semql.spec import SemanticQuery

    # Re-derive the SemanticQuery from the plan.  The plan is
    # frozen and the schema is small; a structural rebuild keeps
    # the round-trip lossless and avoids a side channel for
    # "the spec was X but the plan lowered to Y" mismatches.
    #
    # Every field that flows query -> plan must be reconstructed here,
    # because ``_CompileEnv`` re-plans this query: anything not copied
    # back is silently dropped from the emitted SQL (architecture
    # review A1).  Measures / dimensions come from the ``Aggregate``
    # node verbatim (``group_by`` / ``measures`` are ``list(q.*)``);
    # an ungrouped row-listing has no aggregate, so its dimensions are
    # read off the projection instead.
    ungrouped = plan.aggregate is None
    if plan.aggregate is not None:
        dim_refs: list[str] = list(plan.aggregate.group_by)
        measure_refs: list[str] = list(plan.aggregate.measures)
        derived_measures: list[InlineDerived] = [
            d for d in plan.aggregate.derived if isinstance(d, InlineDerived)
        ]
    else:
        dim_refs = [
            f"{col.cube.name}.{col.field.name}"
            for col in plan.project.columns
            if col.kind == "dimension" and col.field is not None
        ]
        measure_refs = []
        derived_measures = []

    # Predicates: ``plan.filters`` merged ``q.filters`` (flat ``Filter``
    # leaves) and ``q.where`` (a ``BoolExpr`` tree, always last) through
    # the CNF pre-pass.  Split them back by type — ``to_cnf`` is
    # idempotent on CNF input, so re-planning rebuilds an identical
    # ``plan.filters`` list and the WHERE clause is byte-identical.
    flat_filters: list[Filter] = []
    where_terms: list[BoolExpr | Filter] = []
    for pred in plan.filters:
        if isinstance(pred.expr, Filter):
            flat_filters.append(pred.expr)
        else:
            where_terms.append(pred.expr)
    if not where_terms:
        where_expr: BoolExpr | None = None
    elif len(where_terms) == 1 and isinstance(where_terms[0], BoolExpr):
        where_expr = where_terms[0]
    else:
        where_expr = BoolExpr(op="and", children=list(where_terms))

    # LEFT-joined cubes are the right-hand side of any join the plan
    # marked ``kind="left"`` (the spec field is consumed as a set, so
    # order is irrelevant).
    left_joins = [j.right.name for j in plan.joins if j.kind == "left"]

    # Rebuild the compare-mode CompareWindow from the plan. The
    # plan stores the pre-computed current / prior ranges;
    # we surface them as a fresh ``CompareWindow`` so the
    # emit path's compare-mode shape checks match.
    from semql.spec import CompareWindow

    compare_window: CompareWindow | None = None
    if plan.compare is not None:
        compare_window = CompareWindow(
            mode="explicit",
            range=plan.compare.prior_range,
        )

    q = SemanticQuery(
        measures=measure_refs,
        dimensions=dim_refs,
        filters=flat_filters,
        where=where_expr,
        having=list(plan.having),
        segments=list(plan.segments),
        derived_measures=derived_measures,
        left_joins=left_joins,
        aliases=dict(plan.aliases),
        ungrouped=ungrouped,
        time_dimension=plan.time_window,
        compare=compare_window,
        order=list(plan.order.keys),
        limit=plan.limit.limit,
        offset=plan.limit.offset,
    )
    env = _CompileEnv(
        q,
        catalog,
        context=context,
        group_by_alias=group_by_alias,
        having_alias=having_alias,
        dialects=dialects,
        views=views,
        viewer=viewer,
        policy=policy,
        scope_fns=scope_fns,
        allow_unbounded_ungrouped=_allow_unbounded_ungrouped,
        plan=plan,
    )
    return env.emit()


def compile_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    dialects: dict[Dialect, DialectStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    _allow_unbounded_ungrouped: bool = False,
) -> CompiledQuery:
    """Compile a SemanticQuery to a CompiledQuery bundle.

    ``catalog`` — dict of cube name → Cube (build from ``Catalog.as_dict()``).
    ``context`` — optional string substitutions applied to ``{key}``
        placeholders in table names and SQL expressions
        (e.g. ``{"schema": "mydb"}``).
    ``group_by_alias`` — when True (default), GROUP BY references the
        SELECT output alias. Set False to repeat the resolved expression.
    ``having_alias`` — when False (default), HAVING repeats the aggregate
        expression. Set True only when you control the backend.
    ``dialects`` — optional per-backend dialect overrides. Pass a
        ``RecordingDialect`` for tests; pass a custom Snowflake / BigQuery
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
        dialects=dialects,
        views=views,
        viewer=viewer,
        policy=policy,
        scope_fns=scope_fns,
        allow_unbounded_ungrouped=_allow_unbounded_ungrouped,
    )
    return env.emit()


__all__ = [
    "CompiledQuery",
    "CompileError",
    "CrossDialectError",
    "FilterTypeError",
    "JoinPathError",
    "MAX_UNGROUPED_ROWS",
    "PhaseDeferredError",
    "PlaceholderError",
    "UnknownIdentifierError",
    "compile_plan",
    "compile_query",
    "explain_plan",
]
