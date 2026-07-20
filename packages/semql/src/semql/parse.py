# pyright: reportPrivateImportUsage=false, reportUnusedImport=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false
#
# DIALECT CONVENTION: parsing reads SQL under a target
# sqlglot dialect (``dialect_for`` in ``dialect.py``); per-backend specifics
# live behind ``DialectStrategy`` in ``backend.py``. See the full convention
# block at the top of ``backend.py``.
"""SQL → SemanticQuery parser.

Converts a SQL-like statement (typically emitted by an LLM agent)
into a :class:`~semql.spec.SemanticQuery`. The output goes through
the existing compile pipeline — same authorization, row-level
scope, and prompt rendering as a hand-written query.

Pure function: no I/O, no globals. ``catalog`` is optional; without
it, references are collected as bare strings (the caller can
resolve them later or against their own catalog). With it,
references are validated against the catalog's cubes and fields.

Spec at ``docs/specs/sql-parser.md``. Stages:

  1. Tokenize + parse via sqlglot (``sqlglot.parse_one``).
  2. Walk the AST, projecting columns / aggregates / WHERE / etc.
  3. Resolve references against ``catalog`` if provided.
  4. Build ``SemanticQuery`` (CNF normalise happens later in the
     plan).
  5. Collect diagnostics into ``parse_errors`` / ``parse_warnings``.

The parser is intentionally narrow — it handles the well-trodden
SQL shapes an LLM emits (``SELECT cols FROM cube WHERE ...``),
not the full SQL grammar. Anything outside the supported set lands
in ``parse_errors`` with a clear message.

Multi-cube JOINs — the Malloy-style contract
---------------------------------------------
A ``JOIN`` in this SQL is a *semantic directive*, not a row-level
join. It declares which cubes a query touches; the **catalog** is the
single source of truth for how those cubes relate. Concretely:

- The SQL ``ON`` clause is **not read**. The compiler derives the
  actual join from ``Cube.joins`` (identifier resolution → join-graph
  BFS), exactly as it does for a hand-built ``SemanticQuery`` that
  simply references fields from two cubes. We only verify the catalog
  *does* relate the cubes (``_cubes_joinable``) and refuse otherwise —
  the parser never invents a join the catalog doesn't declare.
- Table aliases resolve columns to cubes: ``FROM orders o JOIN
  customers c`` makes ``o.revenue`` → ``orders.revenue`` and ``c.name``
  → ``customers.name``. An unqualified column resolves to the one cube
  that declares it, or is rejected as ambiguous.

Why ``ON`` is ignored, and the caveat: a flat ``a JOIN b`` executed
literally fans out — summing a measure across a one-to-many join
multiplies it by the match count. Semantic layers (Malloy, Cube,
LookML) avoid this by treating the join as a modelled relationship and
aggregating safely, *not* by trusting the query's literal join. SemQL
follows that model: the literal SQL semantics of a flat JOIN are NOT
preserved — the catalog's relationship + the compiler's fan-out guard
(``_check_fan_out``) define the meaning. If you want literal fidelity,
the equivalent faithful SQL is aggregate-then-join on the conformed
key (two grouped subqueries joined on the shared dimension), which is
also what a cross-backend ``FederatedPlan`` emits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

import sqlglot
from sqlglot import exp

from semql.errors import SemQLError
from semql.model import Cube, GranularityLiteral
from semql.refs import local_name
from semql.spec import (
    BoolExpr,
    CompareWindow,
    Filter,
    SemanticQuery,
    TimeWindow,
)

# Aggregate function names that mark a column as a measure (vs dimension).
_AGG_FUNCS = frozenset(
    {"SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT", "MEDIAN", "PERCENTILE"}
)

# Valid time-bucket grains for a ``DATE_TRUNC('<grain>', dim)`` projection —
# the same literal set the ``TimeWindow.granularity`` field accepts.
_GRAINS = frozenset(get_args(GranularityLiteral))

# Logical negation of a filter op, for unwrapping a wrapping ``NOT``.
# ``contains`` (LIKE) has no negated counterpart in the FilterOp set, so
# it's intentionally absent — a negated LIKE is left as-is rather than
# silently inverted to the wrong op.
_NEGATE_OP: dict[str, str] = {
    "eq": "neq",
    "neq": "eq",
    "gt": "lte",
    "gte": "lt",
    "lt": "gte",
    "lte": "gt",
    "in": "not_in",
    "not_in": "in",
    "is_null": "not_null",
    "not_null": "is_null",
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParserDecision:
    """The result of parsing a SQL-like statement.

    ``query`` is the SemanticQuery; callers can pass it directly to
    ``compile_query``. ``original_statement`` is the SQL string for
    reference (audit, retry, prompt). ``parse_warnings`` are
    non-fatal (e.g. unknown field in lenient mode). ``parse_errors``
    are fatal in strict mode (the function raises) but tolerated
    in lenient mode.
    """

    query: SemanticQuery
    original_statement: str
    parse_warnings: tuple[str, ...] = ()
    parse_errors: tuple[str, ...] = ()
    resolved_references: dict[str, str] = field(default_factory=dict)


class ParseError(SemQLError):
    """Raised by :func:`parse_sql_statement` in strict mode on
    unsupported SQL or unknown references."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _op_from_sql(op: exp.Expression) -> str | None:
    """Map a sqlglot comparison op to a ``FilterOp`` literal."""
    if isinstance(op, exp.In):
        # ``NOT IN`` — sqlglot doesn't have a separate ``NotIn`` class;
        # the ``not_`` flag is on the ``In`` instance.
        if op.args.get("not"):
            return "not_in"
        return "in"
    mapping = {
        exp.EQ: "eq",
        exp.NEQ: "neq",
        exp.GT: "gt",
        exp.GTE: "gte",
        exp.LT: "lt",
        exp.LTE: "lte",
        exp.Like: "contains",
        exp.Is: None,  # special — see _unpack_is
    }
    return mapping.get(type(op))  # type: ignore[arg-type]


def _unpack_is(op: exp.Is) -> tuple[str | None, list[object]]:
    """``IS NULL`` / ``IS NOT NULL`` — return (op, values)."""
    inner = op.expression
    is_null = isinstance(inner, exp.Null)
    is_not = op.args.get("not") or False
    if is_null and not is_not:
        return "is_null", []
    if is_null and is_not:
        return "not_null", []
    return None, []


def _values_from_sql(node: exp.Expression | list[object] | None) -> list[object]:
    """Extract a flat list of literal values from a SQL expression.

    Handles three shapes:
      - a single ``Literal`` — one value.
      - an ``In`` / ``Tuple`` node — multiple values via ``.expressions``.
      - a list of expressions — flattened.
    """
    if node is None:
        return []
    if isinstance(node, list):
        out: list[Any] = []
        for e in node:
            out.extend(_values_from_sql(cast(exp.Expression, e)))
        return out
    if isinstance(node, exp.Tuple):
        return _values_from_sql(node.expressions)
    if isinstance(node, exp.Literal):
        return [node.this]
    return [_literal_value(node)]


def _literal_value(node: exp.Expression) -> str | int | float | bool | None:
    """Best-effort scalar conversion from a sqlglot literal."""
    if isinstance(node, exp.Boolean):
        # ``WHERE is_paid = true`` — sqlglot models the keyword as a
        # Boolean node carrying a Python bool, not a Literal. Without this
        # it stringifies to ``'TRUE'`` and a bool dimension filter rejects
        # the non-bool value.
        return bool(node.this)
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Literal):
        v: Any = node.this
        if node.is_string:
            return cast(str, v)
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
        if v in ("TRUE", "true", "True"):
            return True
        if v in ("FALSE", "false", "False"):
            return False
        return cast(str | int | float | bool, v)
    return str(node.sql())


def _agg_func_name(func: exp.Expression) -> str | None:
    """Return the uppercase function name if ``func`` is an aggregate."""
    if isinstance(func, exp.Anonymous):
        return func.name.upper() if func.name else None
    if isinstance(func, exp.AggFunc):
        # Covers every dedicated aggregate node — Sum, Count, Avg, Min,
        # Max, Median, Stddev, … — not just a hand-listed few. The SQL
        # function is only a *marker* that the column is a measure; the
        # catalog measure's ``agg`` decides the rendered aggregate.
        return func.sql_name().upper()
    return None


def _count_measure_ref(catalog: dict[str, Cube] | None, ctx: ResolveCtx) -> str | None:
    """Resolve ``COUNT(*)`` to a qualified count-measure ref.

    A ``COUNT(*)`` projection has no inner column, so it can't be matched
    by name like ``SUM(amount)``. Find the cube measure declared as a row
    count (``agg="count"``, ``sql="*"``) among the query's cubes:

    - exactly one participating cube declares one → use it;
    - more than one (a JOIN where both sides count) → ambiguous, return
      ``None`` so the caller surfaces it rather than guessing a grain;
    - none found / uncatalogued → fall back to ``<cube>.count`` for the
      single-cube case (the validation pass flags it if absent)."""
    if catalog is not None:
        owners: list[str] = []
        for name in ctx.cubes:
            cube = catalog.get(name)
            if cube is None:
                continue
            for m in cube.measures:
                if m.agg == "count" and m.sql.strip() == "*":
                    owners.append(f"{name}.{m.name}")
                    break
        if len(owners) == 1:
            return owners[0]
        if len(owners) > 1:
            return None  # ambiguous across joined cubes
    if ctx.default_cube is not None:
        return f"{ctx.default_cube}.count"
    return None


def _parse_date_trunc(node: exp.DateTrunc, ctx: ResolveCtx) -> tuple[str | None, str | None]:
    """Unpack a ``DATE_TRUNC('<grain>', dim)`` projection.

    Returns ``(grain, dim_ref)`` — the lower-cased grain literal and the
    resolved ``cube.field`` ref of the truncated column. Either element is
    ``None`` when it can't be determined (unknown grain, unresolvable
    column); the caller surfaces the diagnostic."""
    unit = node.unit
    grain: str | None = None
    if isinstance(unit, (exp.Literal, exp.Var)):
        grain = str(unit.this).lower()
    if grain is not None and grain not in _GRAINS:
        grain = None
    dim_ref = ctx.resolve(node.this) if node.this is not None else None
    return grain, dim_ref


def _inside_subquery(node: exp.Expression, root: exp.Expression) -> bool:
    """True if ``node`` sits inside a nested SELECT/subquery relative to
    ``root``.

    ``Expression.walk()`` descends into subqueries, so an aggregate that
    belongs to a scalar subquery would otherwise be attributed to the outer
    query. Walking up from ``node`` to ``root``, a ``Subquery`` or ``Select``
    boundary means the node is nested, not top-level."""
    if node is root:
        # The projection node itself; its parent is the outer SELECT, which
        # must not count as a boundary.
        return False
    parent = node.parent
    while parent is not None and parent is not root:
        if isinstance(parent, (exp.Subquery, exp.Select)):
            return True
        parent = parent.parent
    return False


def _collect_aggregates(expr: exp.Expression) -> list[exp.AggFunc | exp.Anonymous]:
    """Find aggregate-function call nodes at the top level of ``expr``.

    Aggregates nested inside a subquery are excluded — they belong to
    that subquery, not the projection being classified."""
    out: list[Any] = []
    for node in expr.walk():
        # ``cast`` works around sqlglot's typing: the ``AggFunc`` base
        # isn't seen as ``exp.Expression`` by mypy though concrete nodes are.
        n = cast(exp.Expression, node)
        if isinstance(node, exp.AggFunc) and not _inside_subquery(n, expr):
            # Any dedicated aggregate node is a measure marker.
            out.append(node)
        elif (
            isinstance(node, exp.Anonymous)
            and _agg_func_name(node) in _AGG_FUNCS
            and not _inside_subquery(n, expr)
        ):
            # Functions sqlglot doesn't model (e.g. PERCENTILE) arrive as
            # Anonymous — gate those on the known-aggregate name set so a
            # scalar function isn't mistaken for a measure.
            out.append(node)
    return out


def _cube_has_field(cube: Cube | None, name: str) -> bool:
    if cube is None:
        return False
    return any(
        f.name == name
        for f in (*cube.measures, *cube.dimensions, *cube.time_dimensions, *cube.segments)
    )


@dataclass(frozen=True)
class ResolveCtx:
    """Resolves SQL column references to qualified ``cube.field`` refs.

    A query's FROM / JOIN tables map table aliases — and bare cube names
    — to cube names. ``resolve`` turns a column node into a qualified
    ref three ways, in order:

    1. by its table qualifier — ``o.region`` → ``orders.region``;
    2. when unqualified and the query touches a single cube, by that
       cube — ``region`` → ``orders.region``;
    3. when unqualified in a multi-cube query, by the *one* cube that
       declares the field. If two cubes declare it, it's ambiguous and
       resolves to ``None`` (the caller surfaces it).

    With no cube context at all (no FROM, no catalog), the bare name is
    returned unchanged — an uncatalogued best-effort parse.
    """

    alias_to_cube: dict[str, str]
    cubes: tuple[str, ...]
    catalog: dict[str, Cube] | None

    @property
    def default_cube(self) -> str | None:
        return self.cubes[0] if len(self.cubes) == 1 else None

    def resolve(self, col: exp.Expression) -> str | None:
        name = _column_name(col)
        if name is None:
            return None
        qualifier = col.table if isinstance(col, exp.Column) else ""
        if qualifier:
            cube = self.alias_to_cube.get(qualifier, qualifier)
            return f"{cube}.{name}"
        if self.default_cube is not None:
            return f"{self.default_cube}.{name}"
        if self.catalog is not None and self.cubes:
            owners = [c for c in self.cubes if _cube_has_field(self.catalog.get(c), name)]
            if len(owners) == 1:
                return f"{owners[0]}.{name}"
            return None  # multi-cube ambiguous (or unknown) — caller surfaces it
        if not self.cubes:
            return name  # uncatalogued / no FROM — bare best-effort
        return None


def _build_resolve_ctx(ast: exp.Select, catalog: dict[str, Cube] | None) -> ResolveCtx:
    """Map FROM / JOIN table aliases (and cube names) to cube names."""
    alias_to_cube: dict[str, str] = {}
    cubes: list[str] = []
    tables: list[exp.Table] = []
    from_node = ast.find(exp.From)
    if from_node is not None:
        tables.extend(from_node.find_all(exp.Table))
    for join in ast.args.get("joins") or []:
        tables.extend(join.find_all(exp.Table))
    for t in tables:
        cube = t.name
        if not cube:
            continue
        if cube not in cubes:
            cubes.append(cube)
        alias_to_cube[cube] = cube
        if t.alias:
            alias_to_cube[t.alias] = cube
    return ResolveCtx(alias_to_cube=alias_to_cube, cubes=tuple(cubes), catalog=catalog)


def _cubes_joinable(catalog: dict[str, Cube], cubes: Sequence[str]) -> bool:
    """True if every named cube sits in one connected component of the
    catalog's (undirected) join graph.

    The JOIN keyword in semantic SQL declares *participation*, not a row
    predicate — the catalog is the source of truth for how cubes relate.
    So we don't honour the SQL ``ON`` clause; we only verify the catalog
    actually relates the cubes, and refuse if it doesn't (rather than
    invent a join)."""
    present = [c for c in cubes if c in catalog]
    if len(present) <= 1:
        return True
    adj: dict[str, set[str]] = {}
    for name, cube in catalog.items():
        for j in cube.joins:
            adj.setdefault(name, set()).add(j.to)
            adj.setdefault(j.to, set()).add(name)
    seen = {present[0]}
    stack = [present[0]]
    while stack:
        cur = stack.pop()
        for nb in adj.get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return all(c in seen for c in present)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_sql_statement(
    statement: str,
    catalog: dict[str, Cube] | None = None,
    *,
    strict: bool = True,
) -> ParserDecision:
    """Parse a SQL-like statement into a ``SemanticQuery``.

    ``catalog``: optional ``{cube_name: Cube}`` dict. When provided,
    references are validated; unknown cube / field produces
    ``parse_errors``. When ``None``, references are collected as
    bare strings.

    ``strict`` (default ``True``): raise :class:`ParseError` on any
    parse error or unknown reference. When ``False``, errors
    accumulate on ``ParserDecision.parse_errors`` and the function
    returns a best-effort query.
    """
    errors: list[str] = []
    warnings: list[str] = []
    resolved_refs: dict[str, str] = {}

    try:
        ast = sqlglot.parse_one(statement)
    except sqlglot.errors.ParseError as exc:
        msg = f"SQL is unparseable: {exc}"
        if strict:
            raise ParseError(msg) from exc
        errors.append(msg)
        return ParserDecision(
            query=SemanticQuery(),
            original_statement=statement,
            parse_errors=tuple(errors),
            resolved_references=resolved_refs,
        )

    if not isinstance(ast, exp.Select):
        msg = f"Only SELECT statements are supported; got {type(ast).__name__}."
        if strict:
            raise ParseError(msg)
        errors.append(msg)
        return ParserDecision(
            query=SemanticQuery(),
            original_statement=statement,
            parse_errors=tuple(errors),
            resolved_references=resolved_refs,
        )

    ctx = _build_resolve_ctx(ast, catalog)
    if not ctx.cubes and catalog:
        msg = "Could not determine cube name from FROM clause."
        if strict:
            raise ParseError(msg)
        errors.append(msg)

    if catalog is not None:
        for cube in ctx.cubes:
            if cube not in catalog:
                msg = f"Unknown cube: {cube!r}. Known cubes: {sorted(catalog)}."
                if strict:
                    raise ParseError(msg)
                errors.append(msg)
        # Malloy-style: a JOIN declares which cubes participate; the
        # catalog is the source of truth for how they relate. We do not
        # read the SQL ``ON`` clause — we only require the catalog to
        # actually connect the cubes, and refuse rather than invent a join.
        if not _cubes_joinable(catalog, ctx.cubes):
            msg = (
                f"Cubes {sorted(ctx.cubes)!r} are not related in the catalog: "
                "no join path connects them. Add a catalog join or query them "
                "separately — the SQL JOIN's ON clause is not used to invent one."
            )
            if strict:
                raise ParseError(msg)
            errors.append(msg)

    measures: list[str] = []
    dimensions: list[str] = []
    aliases: dict[str, str] = {}
    filters: list[Filter] = []
    where: BoolExpr | None = None
    having: list[Filter] = []
    order: list[tuple[str, Literal["asc", "desc"]]] = []
    limit: int | None = None
    offset: int | None = None
    time_dim: TimeWindow | None = None
    compare: CompareWindow | None = None
    # A ``DATE_TRUNC('<grain>', dim)`` projection carries the time bucket;
    # every bucket found is stashed here and attached to the ``BETWEEN``-
    # derived ``TimeWindow`` once the WHERE clause is parsed. Kept as a list
    # (not overwritten in place) so a second bucket can't silently discard
    # an already-valid one.
    select_grains: list[tuple[str, str]] = []

    # --- SELECT list ---
    select_expressions = ast.expressions or []
    for raw_expr in select_expressions:
        # ``SUM(revenue) AS rev`` / ``region AS r`` — capture the output
        # alias, then unwrap to classify the underlying expression.
        output_alias: str | None = None
        expr = raw_expr
        if isinstance(expr, exp.Alias):
            output_alias = expr.output_name or None
            expr = expr.this
        if isinstance(expr, exp.Star):
            # SELECT * — every dimension implicitly. We don't emit
            # anything; the compile pipeline will infer.
            continue
        if isinstance(expr, exp.DateTrunc):
            # ``DATE_TRUNC('<grain>', dim)`` is the time-bucket projection.
            # It is not a plain dimension — the grain rides on the query's
            # ``time_dimension``, attached below once the BETWEEN is parsed.
            grain, dim_ref = _parse_date_trunc(expr, ctx)
            if dim_ref is None:
                errors.append(f"Could not resolve the DATE_TRUNC column in {expr.sql()!r}.")
            elif grain is None:
                errors.append(
                    f"Unsupported DATE_TRUNC grain in {expr.sql()!r}; "
                    f"expected one of {sorted(_GRAINS)}."
                )
            else:
                resolved_refs[local_name(dim_ref)] = dim_ref
                select_grains.append((dim_ref, grain))
            continue
        if expr.find(exp.Select) is not None:
            # A subquery projection (scalar subquery) is unsupported — its
            # inner aggregates are NOT outer-query measures. Surface it
            # rather than silently dropping the projection.
            errors.append(
                f"Unsupported SELECT projection {expr.sql()!r}: subqueries are not supported."
            )
            continue
        agg_nodes = _collect_aggregates(expr)
        if agg_nodes:
            # One top-level aggregate → one measure. Several in a single
            # projection means a composed expression (e.g. SUM(a)/SUM(b));
            # only record an output alias when the projection is exactly
            # one aggregate, so a derived expression can't mislabel a column.
            single = len(agg_nodes) == 1
            for agg in agg_nodes:
                inner = agg.this if hasattr(agg, "this") else None
                # Resolve to a qualified ``cube.field`` ref — the form the
                # compiler requires (bare refs raise a resolution error).
                ref = ctx.resolve(inner) if inner is not None else None
                if ref is None and isinstance(agg, exp.Count):
                    # COUNT(*) → the row-count measure of the participating cube.
                    ref = _count_measure_ref(catalog, ctx)
                if ref:
                    field_name = local_name(ref)
                    resolved_refs[field_name] = ref
                    if ref not in measures:
                        measures.append(ref)
                    if single and output_alias and output_alias != field_name:
                        aliases[output_alias] = ref
        else:
            # Plain column — dimension.
            ref = ctx.resolve(expr)
            if ref:
                field_name = local_name(ref)
                resolved_refs[field_name] = ref
                if ref not in dimensions:
                    dimensions.append(ref)
                if output_alias and output_alias != field_name:
                    aliases[output_alias] = ref
            elif _column_name(expr) is not None and ctx.cubes:
                errors.append(
                    f"Ambiguous or unknown column {_column_name(expr)!r} in SELECT: "
                    "qualify it as cube.field."
                )

    # --- WHERE / HAVING / GROUP BY / ORDER BY / LIMIT / OFFSET ---
    where_tree = ast.args.get("where")
    if where_tree is not None:
        time_dim, filters, where = _parse_where(where_tree, ctx, resolved_refs)

    # A ``DATE_TRUNC`` bucket in SELECT sets the granularity on the time
    # window the ``BETWEEN`` produced. The two must name the same dimension;
    # a bucket with no matching window is flagged, not silently dropped. More
    # than one bucket in a single SELECT is rejected outright — only one
    # ``TimeWindow.granularity`` slot exists, so silently keeping the last
    # one would discard an earlier, possibly-valid bucket without a trace.
    if len(select_grains) > 1:
        errors.append(
            "Multiple DATE_TRUNC bucket projections in SELECT "
            f"({select_grains!r}); only one time-bucket projection is supported."
        )
    elif len(select_grains) == 1:
        grain_dim, grain = select_grains[0]
        if time_dim is not None and time_dim.dimension == grain_dim:
            time_dim = time_dim.model_copy(update={"granularity": grain})
        elif time_dim is None:
            errors.append(
                f"DATE_TRUNC({grain!r}, {grain_dim!r}) in SELECT needs a matching "
                f"BETWEEN window on {grain_dim!r} to set the time granularity."
            )
        else:
            errors.append(
                f"DATE_TRUNC dimension {grain_dim!r} does not match the "
                f"BETWEEN window dimension {time_dim.dimension!r}."
            )

    having_node = ast.args.get("having")
    if having_node is not None:
        having = _parse_having(having_node, ctx, resolved_refs, errors)

    order_node = ast.args.get("order")
    if order_node is not None:
        # sqlglot stores the order clauses as a list of ``Ordered``
        # nodes on ``order_node.expressions``.
        order = _parse_order(order_node, ctx, resolved_refs, set(aliases))

    limit_node = ast.args.get("limit")
    if limit_node is not None:
        # sqlglot parses LIMIT as exp.Limit with expression.
        lit = _literal_value(limit_node.expression)
        if isinstance(lit, int):
            limit = lit
        else:
            errors.append(f"LIMIT must be an integer; got {lit!r}")

    offset_node = ast.args.get("offset")
    if offset_node is not None:
        lit = _literal_value(offset_node.expression)
        if isinstance(lit, int):
            offset = lit
        else:
            errors.append(f"OFFSET must be an integer; got {lit!r}")

    # --- COMPARE TO prior_period ---
    compare_node = _extract_compare(ast)
    if compare_node is not None:
        compare = compare_node

    # --- Catalog validation pass (post-parse) ---
    if catalog is not None and ctx.cubes:
        _validate_refs(
            catalog=catalog,
            measures=measures,
            dimensions=dimensions,
            filters=filters,
            having=having,
            time_dim=time_dim,
            errors=errors,
        )

    query = SemanticQuery(
        measures=measures,
        dimensions=dimensions,
        aliases=aliases,
        filters=filters,
        where=where,
        time_dimension=time_dim,
        compare=compare,
        having=having,
        order=order,
        limit=limit,
        offset=offset,
    )

    if strict and errors:
        raise ParseError(f"Parse errors: {'; '.join(errors)}")

    return ParserDecision(
        query=query,
        original_statement=statement,
        parse_warnings=tuple(warnings),
        parse_errors=tuple(errors),
        resolved_references=resolved_refs,
    )


# ---------------------------------------------------------------------------
# Internal walkers
# ---------------------------------------------------------------------------


def _column_name(node: exp.Expression) -> str | None:
    """Return the bare column name from a Column / Identifier / Star node."""
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    if isinstance(node, exp.Star):
        return None
    return None


def _parse_where(
    where_tree: exp.Where,
    ctx: ResolveCtx,
    resolved: dict[str, str],
) -> tuple[TimeWindow | None, list[Filter], BoolExpr | None]:
    """Split a WHERE clause into time-window, flat filters, and a where-tree.

    A simple ``a = b AND c = d`` is implicit-AND — flat filters.
    A parenthetical OR becomes a BoolExpr. ``BETWEEN`` over a
    single field becomes a TimeWindow.
    """
    predicate = where_tree.this
    flat: list[Filter] = []
    tree: BoolExpr | None = None
    time_dim: TimeWindow | None = None

    if isinstance(predicate, exp.And):
        for child in _bool_children(predicate):
            td, fs, wt = _classify_predicate(child, ctx, resolved)
            if td is not None:
                time_dim = td
            if wt is not None:
                # OR-containing subtree — promote to a where-tree.
                tree = _merge_where_tree(tree, wt)
            flat.extend(fs)
        return time_dim, flat, tree

    td, fs, wt = _classify_predicate(predicate, ctx, resolved)
    return td, fs, wt


def _classify_predicate(
    pred: exp.Expression,
    ctx: ResolveCtx,
    resolved: dict[str, str],
) -> tuple[TimeWindow | None, list[Filter], BoolExpr | None]:
    """Classify one predicate: time-window, flat filter, or where-tree node."""
    if isinstance(pred, exp.Between):
        # ``dim BETWEEN low AND high`` → TimeWindow.
        ref = ctx.resolve(pred.this)
        if ref is None:
            return None, [], None
        low = _literal_value(pred.args["low"])
        high = _literal_value(pred.args["high"])
        resolved[local_name(ref)] = ref
        td = TimeWindow(dimension=ref, range=(str(low), str(high)))
        return td, [], None
    if isinstance(pred, exp.Paren):
        # Strip parens, recurse.
        return _classify_predicate(pred.this, ctx, resolved)
    if isinstance(pred, (exp.Or, exp.And)):
        if isinstance(pred, exp.Or):
            # An OR anywhere in the WHERE becomes a where-tree.
            return None, [], _build_bool_tree(pred, ctx, resolved)
        # AND of mixed predicates — recurse and assemble.
        flat: list[Filter] = []
        tree: BoolExpr | None = None
        time_dim: TimeWindow | None = None
        for c in _bool_children(pred):
            time_dim, tree, flat = _merge_classified(
                time_dim, tree, flat, _classify_predicate(c, ctx, resolved)
            )
        return time_dim, flat, tree
    # ``NOT (...)`` — sqlglot wraps ``NOT IN`` and ``IS NOT NULL`` as an
    # ``exp.Not`` around the inner comparison (the ``not`` flag is NOT set
    # on the In/Is node itself). Unwrap and negate, or the predicate is
    # silently dropped — see ``_NEGATE_OP``.
    negate = False
    op_node = pred
    if isinstance(op_node, exp.Not):
        op_node = op_node.this
        negate = True
        if isinstance(op_node, exp.Paren):
            op_node = op_node.this
    # Plain comparison.
    if isinstance(
        op_node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like, exp.Is)
    ):
        return _comparison_to_filter(op_node, ctx, resolved, negate=negate)
    return None, [], None


def _comparison_to_filter(
    op_node: exp.Expression,
    ctx: ResolveCtx,
    resolved: dict[str, str],
    *,
    negate: bool = False,
) -> tuple[TimeWindow | None, list[Filter], BoolExpr | None]:
    """Convert a comparison node to a Filter (or zero filters).

    ``negate`` flips the op (``in`` → ``not_in``, ``is_null`` →
    ``not_null``, etc.) when the node came from a wrapping ``NOT``."""
    if isinstance(op_node, exp.Is):
        op_str, values = _unpack_is(op_node)
    else:
        op_str = _op_from_sql(op_node)
        if op_str is None:
            return None, [], None
        if op_str in ("in", "not_in"):
            # sqlglot stores IN's list as ``.expressions`` (a list of
            # Literals), not ``.expression`` (which is None for IN).
            values = _values_from_sql(op_node.expressions)
        elif op_str == "contains":
            # LIKE — extract the pattern.
            values = _values_from_sql(op_node.expression)
        else:
            values = [_literal_value(op_node.expression)]
    # Comparison nodes have the column on ``.left``; IN has it on
    # ``.this``.
    col_node = op_node.left if hasattr(op_node, "left") else op_node.this
    # HAVING ``SUM(revenue) > 1000`` — the LHS is an aggregate wrapping
    # the measure column. Unwrap it so the filter targets the measure;
    # otherwise the predicate has no column name and is dropped.
    agg_wrapper: exp.AggFunc | exp.Anonymous | None = None
    if isinstance(col_node, (exp.AggFunc, exp.Anonymous)):
        agg_wrapper = col_node
        col_node = col_node.this
    dim = ctx.resolve(col_node) if col_node is not None else None
    if dim is None and isinstance(agg_wrapper, exp.Count):
        # HAVING ``COUNT(*) > N`` — the inner is ``*``, not a column, so
        # it resolves to the cube's row-count measure, exactly like a
        # ``SELECT COUNT(*)`` projection. Without this the predicate is
        # silently dropped.
        dim = _count_measure_ref(ctx.catalog, ctx)
    if dim is None:
        return None, [], None
    resolved[local_name(dim)] = dim
    if op_str is None:
        return None, [], None
    if negate:
        # A wrapping ``NOT`` flips the op. ``contains`` has no negated
        # form (no ``not_contains`` in FilterOp), so it's left unchanged.
        op_str = _NEGATE_OP.get(op_str, op_str)
    return (
        None,
        [
            Filter(
                dimension=dim,
                op=cast(
                    Literal[
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
                    ],
                    op_str,
                ),
                values=cast(list[str | int | float | bool], values),
            )
        ],
        None,
    )


def _bool_children(pred: exp.Expression) -> list[exp.Expression]:
    """Return the children of a binary boolean op (AND / OR) as a list.

    sqlglot models AND/OR as binary trees (``this`` / ``expression``),
    not n-ary. Walk the tree to flatten to a list — left-leaning
    recursion so we preserve left-to-right order.
    """
    if isinstance(pred, (exp.And, exp.Or)):
        return _bool_children(pred.this) + _bool_children(pred.expression)
    return [pred]


def _build_bool_tree(
    pred: exp.Expression,
    ctx: ResolveCtx,
    resolved: dict[str, str],
) -> BoolExpr:
    """Build a BoolExpr from a sqlglot OR / AND subtree."""
    if isinstance(pred, exp.Or):
        op = "or"
    elif isinstance(pred, exp.And):
        op = "and"
    else:
        # Leaf: wrap in a single-child tree for consistency.
        op = "or"
        pred = exp.Or(this=pred, expression=pred.copy())
    children: list[BoolExpr | Filter] = []
    for c in _bool_children(pred):
        if isinstance(c, (exp.Or, exp.And)):
            children.append(_build_bool_tree(c, ctx, resolved))
        else:
            _, fs, _ = _classify_predicate(c, ctx, resolved)
            if fs:
                children.extend(fs)
    if len(children) < 2:
        # BoolExpr requires >= 2 children; if we couldn't classify
        # at least 2, return a single filter unwrapped (caller
        # re-classifies).
        if children:
            return children[0]  # type: ignore[return-value]
        raise ValueError("Cannot build BoolExpr with empty children.")
    return BoolExpr(op=cast(Literal["and", "or", "not"], op), children=children)


def _merge_classified(
    time_dim: TimeWindow | None,
    tree: BoolExpr | None,
    flat: list[Filter],
    classified: tuple[TimeWindow | None, list[Filter], BoolExpr | None],
) -> tuple[TimeWindow | None, BoolExpr | None, list[Filter]]:
    """Merge one classified predicate into the running accumulator.

    The standalone helper exists to give mypy a fresh scope — the
    inlined destructure version tripped mypy's loop-narrowing on
    re-assignment across iterations.
    """
    td, fs, wt = classified
    new_time: TimeWindow | None = time_dim
    if td is not None:
        new_time = td
    new_tree: BoolExpr | None = tree
    if wt is not None:
        new_tree = _merge_where_tree(tree, wt)
    new_flat: list[Filter] = list(flat)
    new_flat.extend(fs)
    return new_time, new_tree, new_flat


def _merge_where_tree(
    existing: BoolExpr | None,
    new: BoolExpr,
) -> BoolExpr:
    """AND-compose a new BoolExpr subtree into an existing one."""
    if existing is None:
        return new
    if existing.op == "and":
        return BoolExpr(op="and", children=[*existing.children, new])
    return BoolExpr(op="and", children=[existing, new])


def _parse_having(
    having_node: exp.Having,
    ctx: ResolveCtx,
    resolved: dict[str, str],
    errors: list[str],
) -> list[Filter]:
    out: list[Filter] = []
    pred = having_node.this
    if isinstance(pred, exp.Or) or pred.find(exp.Or) is not None:
        # ``having`` is an implicit-AND ``list[Filter]`` — it has no OR
        # representation (unlike WHERE, which gets a BoolExpr tree). Refuse
        # rather than silently flatten ``a OR b`` into ``a AND b``.
        errors.append(
            "HAVING does not support OR — it is an implicit-AND list of "
            "measure filters. Split the query or move the condition to WHERE."
        )
        return out
    if isinstance(pred, exp.And):
        for c in _bool_children(pred):
            _, fs, _ = _classify_predicate(c, ctx, resolved)
            out.extend(fs)
    else:
        _, fs, _ = _classify_predicate(pred, ctx, resolved)
        out.extend(fs)
    return out


def _parse_order(
    order_node: exp.Order,
    ctx: ResolveCtx,
    resolved: dict[str, str],
    alias_keys: set[str],
) -> list[tuple[str, Literal["asc", "desc"]]]:
    out: list[tuple[str, Literal["asc", "desc"]]] = []
    for o in order_node.expressions or []:
        # Each entry is an ``Ordered`` node with ``.this`` (the
        # column or aggregate) and ``.args['desc']`` (True/False).
        col = o.this
        direction: Literal["asc", "desc"] = "desc" if o.args.get("desc") else "asc"
        if isinstance(col, exp.Column):
            if col.name in alias_keys and not col.table:
                # ORDER BY a SELECT alias — emit the bare alias key; the
                # compiler resolves it against the output columns. Do NOT
                # qualify it (``orders.rev``) — that field doesn't exist.
                out.append((col.name, direction))
                continue
            ref = ctx.resolve(col)
            if ref is not None:
                resolved[local_name(ref)] = ref
                out.append((ref, direction))
        elif isinstance(col, (exp.AggFunc, exp.Anonymous)):
            # ORDER BY SUM(amount) DESC — strip the aggregate wrapper
            # and resolve the inner column to its measure ref.
            inner = getattr(col, "this", None)
            ref = ctx.resolve(inner) if inner is not None else None
            if ref is None and isinstance(col, exp.Count):
                # ORDER BY COUNT(*) — inner is ``*``; use the row-count measure.
                ref = _count_measure_ref(ctx.catalog, ctx)
            if ref is not None:
                resolved[local_name(ref)] = ref
                out.append((ref, direction))
    return out


def _extract_compare(ast: exp.Select) -> CompareWindow | None:
    """Look for ``COMPARE TO prior_period`` or ``COMPARE TO explicit`` hint.

    sqlglot doesn't parse ``COMPARE TO`` — we treat it as a comment
    in the SQL or detect it in a trailing token. For now: the
    parser recognizes a magic trailing comment ``-- COMPARE: prior_period``
    on the SQL string and surfaces a CompareWindow.
    """
    # Walk the AST for a ``/* COMPARE ... */`` hint comment.
    hints: list[object] = []
    for node in ast.walk():
        if isinstance(node, exp.Hint):
            hints.append(node.sql())
    for h in hints:
        h_str = h if isinstance(h, str) else ""
        if "PRIOR_PERIOD" in h_str.upper():
            return CompareWindow(mode="previous_period")
    return None


def _validate_refs(
    catalog: dict[str, Cube],
    measures: list[str],
    dimensions: list[str],
    filters: list[Filter],
    having: list[Filter],
    time_dim: TimeWindow | None,
    errors: list[str],
) -> None:
    """Check that every referenced ``cube.field`` exists on its cube.

    Refs are qualified, so each is validated against the cube it names —
    correct for multi-cube JOIN queries, not just single-cube ones.
    Unqualified refs (uncatalogued best-effort) are skipped."""

    def check(ref: str, clause: str) -> None:
        if "." not in ref:
            return
        cube_name, _, field = ref.rpartition(".")
        cube = catalog.get(cube_name)
        if cube is None:
            return  # unknown cube already flagged in the FROM pass
        if not _cube_has_field(cube, field):
            errors.append(f"Unknown field {ref!r} in {clause}.")

    for m in measures:
        check(m, "SELECT")
    for d in dimensions:
        check(d, "SELECT")
    for f in filters:
        check(f.dimension, "WHERE")
    for h in having:
        check(h.dimension, "HAVING")
    if time_dim is not None:
        check(time_dim.dimension, "WHERE (BETWEEN)")


__all__ = ["ParserDecision", "ParseError", "parse_sql_statement"]
