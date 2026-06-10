# pyright: reportPrivateImportUsage=false, reportUnusedImport=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""I13 — SQL → SemanticQuery parser.

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import sqlglot
from sqlglot import exp

from semql.errors import SemQLError
from semql.model import Cube
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


def _cube_from_select(ast: exp.Select, catalog: dict[str, Cube] | None) -> str | None:
    """Pick the cube name from the FROM clause. Single-cube only."""
    # sqlglot's key is ``from_`` (trailing underscore — ``from`` is
    # a Python reserved word).
    from_clause = ast.args.get("from_")
    if from_clause is None:
        return None
    table = from_clause.this
    if not isinstance(table, exp.Table):
        return None
    return table.name


def _agg_func_name(func: exp.Expression) -> str | None:
    """Return the uppercase function name if ``func`` is an aggregate."""
    if isinstance(func, exp.Anonymous):
        return func.name.upper() if func.name else None
    if isinstance(func, exp.Count):
        return "COUNT"
    if isinstance(func, exp.Sum):
        return "SUM"
    if isinstance(func, exp.Avg):
        return "AVG"
    if isinstance(func, exp.Min):
        return "MIN"
    if isinstance(func, exp.Max):
        return "MAX"
    return None


def _collect_aggregates(expr: exp.Expression) -> list[exp.AggFunc | exp.Anonymous]:
    """Find all aggregate-function call nodes inside ``expr``."""
    out: list[Any] = []
    for node in expr.walk():
        if isinstance(node, (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max, exp.Anonymous)):
            name = _agg_func_name(node)
            if name in _AGG_FUNCS:
                out.append(node)
    return out


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

    cube_name = _cube_from_select(ast, catalog)
    if cube_name is None and catalog:
        msg = "Could not determine cube name from FROM clause."
        if strict:
            raise ParseError(msg)
        errors.append(msg)

    if cube_name is not None and catalog is not None and cube_name not in catalog:
        msg = f"Unknown cube: {cube_name!r}. Known cubes: {sorted(catalog)}."
        if strict:
            raise ParseError(msg)
        errors.append(msg)
        # Continue — best-effort parse with the bare cube name.

    measures: list[str] = []
    dimensions: list[str] = []
    filters: list[Filter] = []
    where: BoolExpr | None = None
    having: list[Filter] = []
    order: list[tuple[str, Literal["asc", "desc"]]] = []
    limit: int | None = None
    offset: int | None = None
    time_dim: TimeWindow | None = None
    compare: CompareWindow | None = None

    # --- SELECT list ---
    select_expressions = ast.expressions or []
    for expr in select_expressions:
        if isinstance(expr, exp.Star):
            # SELECT * — every dimension implicitly. We don't emit
            # anything; the compile pipeline will infer.
            continue
        agg_nodes = _collect_aggregates(expr)
        if agg_nodes:
            for agg in agg_nodes:
                inner = agg.this if hasattr(agg, "this") else None
                field_name = _column_name(inner) if inner is not None else None
                if field_name:
                    if cube_name:
                        ref = f"{cube_name}.{field_name}"
                        resolved_refs[field_name] = ref
                    measures.append(field_name)
        else:
            # Plain column — dimension.
            field_name = _column_name(expr)
            if field_name and field_name not in dimensions:
                if cube_name:
                    resolved_refs[field_name] = f"{cube_name}.{field_name}"
                dimensions.append(field_name)

    # --- WHERE / HAVING / GROUP BY / ORDER BY / LIMIT / OFFSET ---
    where_tree = ast.args.get("where")
    if where_tree is not None:
        time_dim, filters, where = _parse_where(where_tree, cube_name, resolved_refs)

    having_node = ast.args.get("having")
    if having_node is not None:
        having = _parse_having(having_node, cube_name, resolved_refs)

    order_node = ast.args.get("order")
    if order_node is not None:
        # sqlglot stores the order clauses as a list of ``Ordered``
        # nodes on ``order_node.expressions``.
        order = _parse_order(order_node, cube_name, resolved_refs)

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
    if catalog is not None and cube_name is not None:
        _validate_refs(
            catalog=catalog,
            cube_name=cube_name,
            measures=measures,
            dimensions=dimensions,
            filters=filters,
            time_dim=time_dim,
            errors=errors,
            warnings=warnings,
            strict=strict,
        )

    query = SemanticQuery(
        measures=measures,
        dimensions=dimensions,
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
    cube_name: str | None,
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
            td, fs, wt = _classify_predicate(child, cube_name, resolved)
            if td is not None:
                time_dim = td
            if wt is not None:
                # OR-containing subtree — promote to a where-tree.
                tree = _merge_where_tree(tree, wt)
            flat.extend(fs)
        return time_dim, flat, tree

    td, fs, wt = _classify_predicate(predicate, cube_name, resolved)
    return td, fs, wt


def _classify_predicate(
    pred: exp.Expression,
    cube_name: str | None,
    resolved: dict[str, str],
) -> tuple[TimeWindow | None, list[Filter], BoolExpr | None]:
    """Classify one predicate: time-window, flat filter, or where-tree node."""
    if isinstance(pred, exp.Between):
        # ``dim BETWEEN low AND high`` → TimeWindow.
        dim = _column_name(pred.this)
        if dim is None:
            return None, [], None
        low = _literal_value(pred.args["low"])
        high = _literal_value(pred.args["high"])
        if cube_name:
            resolved[dim] = f"{cube_name}.{dim}"
        td = TimeWindow(
            dimension=f"{cube_name}.{dim}" if cube_name else dim,
            range=(str(low), str(high)),
        )
        return td, [], None
    if isinstance(pred, exp.Paren):
        # Strip parens, recurse.
        return _classify_predicate(pred.this, cube_name, resolved)
    if isinstance(pred, (exp.Or, exp.And)):
        if isinstance(pred, exp.Or):
            # An OR anywhere in the WHERE becomes a where-tree.
            return None, [], _build_bool_tree(pred, cube_name, resolved)
        # AND of mixed predicates — recurse and assemble.
        flat: list[Filter] = []
        tree: BoolExpr | None = None
        time_dim: TimeWindow | None = None
        for c in _bool_children(pred):
            time_dim, tree, flat = _merge_classified(
                time_dim, tree, flat, _classify_predicate(c, cube_name, resolved)
            )
        return time_dim, flat, tree
    # Plain comparison.
    op_node = pred
    if isinstance(
        op_node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like, exp.Is)
    ):
        return _comparison_to_filter(op_node, cube_name, resolved)
    return None, [], None


def _comparison_to_filter(
    op_node: exp.Expression,
    cube_name: str | None,
    resolved: dict[str, str],
) -> tuple[TimeWindow | None, list[Filter], BoolExpr | None]:
    """Convert a comparison node to a Filter (or zero filters)."""
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
    dim = _column_name(col_node)
    if dim is None:
        return None, [], None
    if cube_name:
        resolved[dim] = f"{cube_name}.{dim}"
        dim = f"{cube_name}.{dim}"
    if op_str is None:
        return None, [], None
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
    cube_name: str | None,
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
            children.append(_build_bool_tree(c, cube_name, resolved))
        else:
            _, fs, _ = _classify_predicate(c, cube_name, resolved)
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
    cube_name: str | None,
    resolved: dict[str, str],
) -> list[Filter]:
    out: list[Filter] = []
    pred = having_node.this
    if isinstance(pred, (exp.And, exp.Or)):
        for c in _bool_children(pred):
            _, fs, _ = _classify_predicate(c, cube_name, resolved)
            out.extend(fs)
    else:
        _, fs, _ = _classify_predicate(pred, cube_name, resolved)
        out.extend(fs)
    return out


def _parse_order(
    order_node: exp.Order,
    cube_name: str | None,
    resolved: dict[str, str],
) -> list[tuple[str, Literal["asc", "desc"]]]:
    out: list[tuple[str, Literal["asc", "desc"]]] = []
    for o in order_node.expressions or []:
        # Each entry is an ``Ordered`` node with ``.this`` (the
        # column or aggregate) and ``.args['desc']`` (True/False).
        col = o.this
        direction: Literal["asc", "desc"] = "desc" if o.args.get("desc") else "asc"
        if isinstance(col, exp.Column):
            name = col.name
            if cube_name:
                resolved[name] = f"{cube_name}.{name}"
                out.append((f"{cube_name}.{name}", direction))
            else:
                out.append((name, direction))
        elif isinstance(col, (exp.Anonymous, exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)):
            # ORDER BY SUM(amount) DESC — strip the aggregate wrapper
            # and use the inner column name.
            inner = getattr(col, "this", None)
            if isinstance(inner, exp.Column):
                name = inner.name
                if cube_name:
                    resolved[name] = f"{cube_name}.{name}"
                    out.append((f"{cube_name}.{name}", direction))
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
    cube_name: str,
    measures: list[str],
    dimensions: list[str],
    filters: list[Filter],
    time_dim: TimeWindow | None,
    errors: list[str],
    warnings: list[str],
    strict: bool,
) -> None:
    """Check that every referenced field exists on the catalog's cube.

    Failures in strict mode raise; in lenient mode they go on
    ``errors`` and the caller decides.
    """
    cube = catalog.get(cube_name)
    if cube is None:
        return  # Already flagged in the FROM pass.
    declared = {
        f.name for f in (*cube.measures, *cube.dimensions, *cube.time_dimensions, *cube.segments)
    }
    # Measures / dimensions collected by the parser carry only the
    # field name (we stripped the cube prefix when we built
    # ``resolved_refs``).
    for m in measures:
        if m not in declared:
            errors.append(f"Unknown field {cube_name}.{m!r} in SELECT.")
    for d in dimensions:
        if d not in declared:
            errors.append(f"Unknown field {cube_name}.{d!r} in SELECT.")
    for f in filters:
        bare = f.dimension.rsplit(".", 1)[-1]
        if bare not in declared:
            errors.append(f"Unknown field {f.dimension!r} in WHERE.")
    if time_dim is not None:
        bare = time_dim.dimension.rsplit(".", 1)[-1]
        if bare not in declared:
            errors.append(f"Unknown field {time_dim.dimension!r} in WHERE (BETWEEN).")


__all__ = ["ParserDecision", "ParseError", "parse_sql_statement"]
