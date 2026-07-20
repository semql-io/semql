# pyright: reportUnknownMemberType=false
"""``SemanticQuery`` → semantic-SQL serializer — the inverse of
:func:`semql.parse.parse_sql_statement`.

``query_to_sql`` renders a :class:`~semql.spec.SemanticQuery` back into the
narrow "semantic SQL" dialect the parser accepts (identifiers are *catalog*
names — ``orders.revenue`` — not physical columns). The design contract is a
**round-trip**::

    parse_sql_statement(query_to_sql(q, catalog), catalog).query == q

for every query shape the parser supports.

Round-trip equality is exact model ``==`` on the re-parsed query. It holds for
*parser-canonical* queries — the shape the parser itself emits — because the
serializer mirrors the parser's own normalisation:

- A top-level conjunction lives in ``filters`` (flat implicit-AND), never as a
  top-level ``and`` ``BoolExpr``; the parser folds ``a AND b`` into flat
  filters, so ``where`` carries only OR/NOT structure (its root ``and`` nodes,
  if any, sit above ``or`` children — e.g. ``(a OR b) AND (c OR d)``).
- ``aliases`` is assumed to hold at most one alias per referenced field (the
  parser records one ``AS`` per projected column).

The SQL aggregate wrapping a measure is only a *marker* that the column is a
measure — the parser re-derives the real aggregate from the catalog. So the
wrapper the serializer emits need not match the measure's ``agg`` for the
round-trip to hold; with a ``catalog`` it emits the true aggregate anyway
(``SUM`` / ``AVG`` / …, ``COUNT(*)`` for a row-count measure) for readable
output, and falls back to a generic ``SUM(<ref>)`` marker without one.

Features with no surface in the parser's grammar are **flagged** (an
:class:`UnparseError` is raised) rather than silently dropped: ``segments``,
``ungrouped``, ``derived_measures``, ``semi_joins``, ``left_joins``,
``compare`` mode ``'explicit'``, and ``time_dimension.fill_nulls_with``.
"""

from __future__ import annotations

from semql.errors import SemQLError
from semql.model import AggLiteral, Cube
from semql.refs import cube_of, is_qualified
from semql.spec import (
    BoolExpr,
    CompareWindow,
    Filter,
    SemanticQuery,
    TimeWindow,
)

__all__ = ["UnparseError", "query_to_sql"]


class UnparseError(SemQLError):
    """Raised by :func:`query_to_sql` when a query uses a feature that has
    no representation in the semantic-SQL grammar the parser accepts."""


# Measure ``agg`` → the SQL aggregate function used as a measure *marker*.
# Aggregates with no clean parser-recognised function (``ratio``, the ``pNN``
# percentiles) fall back to ``SUM`` — the wrapper is only a marker, so this
# does not change what the round-trip recovers.
_AGG_SQL: dict[AggLiteral, str] = {
    "sum": "SUM",
    "count": "COUNT",
    "count_distinct": "COUNT",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
}

# FilterOp → (SQL operator template). ``{v}`` is the rendered value(s).
_SCALAR_OP_SQL: dict[str, str] = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "contains": "LIKE",
}


def query_to_sql(query: SemanticQuery, catalog: dict[str, Cube] | None = None) -> str:
    """Render ``query`` as semantic SQL the parser round-trips.

    ``catalog`` is optional. With it, each measure is wrapped in its true
    aggregate (``SUM`` / ``COUNT`` / ``AVG`` / …) and row-count measures emit
    ``COUNT(*)``; without it, every measure gets a generic ``SUM(<ref>)``
    marker that still round-trips (the parser re-derives the aggregate from
    the catalog it is given at parse time).

    Raises :class:`UnparseError` for query features with no SQL surface (see
    the module docstring).
    """
    _reject_unsupported(query)
    return _Serializer(query, catalog).render()


def _reject_unsupported(query: SemanticQuery) -> None:
    """Flag query features the semantic-SQL grammar cannot express."""
    unsupported: list[str] = []
    if query.segments:
        unsupported.append("segments")
    if query.ungrouped:
        unsupported.append("ungrouped")
    if query.derived_measures:
        unsupported.append("derived_measures")
    if query.semi_joins:
        unsupported.append("semi_joins")
    if query.left_joins:
        unsupported.append("left_joins")
    if query.compare is not None and query.compare.mode == "explicit":
        unsupported.append("compare(mode='explicit')")
    if query.time_dimension is not None and query.time_dimension.fill_nulls_with is not None:
        unsupported.append("time_dimension.fill_nulls_with")
    if unsupported:
        raise UnparseError(
            "SemanticQuery uses features with no semantic-SQL representation: "
            f"{', '.join(unsupported)}. These cannot be serialized to SQL that "
            "round-trips through parse_sql_statement."
        )


class _Serializer:
    """Renders one query. Holds the participating-cube set so ``COUNT(*)``
    disambiguation and the FROM/JOIN clause share a single source of truth."""

    def __init__(self, query: SemanticQuery, catalog: dict[str, Cube] | None) -> None:
        self.q = query
        self.catalog = catalog
        self.cubes = self._participating_cubes()

    # -- reference / cube bookkeeping --------------------------------------

    def _all_refs(self) -> list[str]:
        """Every qualified field ref the query names, in a stable order — the
        order determines which cube heads the FROM clause."""
        q = self.q
        refs: list[str] = [*q.dimensions]
        if q.time_dimension is not None:
            refs.append(q.time_dimension.dimension)
        refs.extend(q.measures)
        for f in q.filters:
            refs.append(f.dimension)
        if q.where is not None:
            refs.extend(_boolexpr_dims(q.where))
        for h in q.having:
            refs.append(h.dimension)
        for field_ref in q.aliases.values():
            refs.append(field_ref)
        for field_name, _direction in q.order:
            if is_qualified(field_name):
                refs.append(field_name)
        return refs

    def _participating_cubes(self) -> list[str]:
        """Unique cube names, first-appearance order. Raises if a ref is not
        qualified (the parser only accepts ``cube.field``)."""
        seen: list[str] = []
        for ref in self._all_refs():
            if not is_qualified(ref):
                raise UnparseError(
                    f"Cannot serialize unqualified reference {ref!r}: the "
                    "semantic-SQL parser resolves only 'cube.field' refs."
                )
            cube = cube_of(ref)
            if cube not in seen:
                seen.append(cube)
        return seen

    def _row_count_cube(self, ref: str) -> str | None:
        """The cube of ``ref`` if ``ref`` is *the* catalog row-count measure
        (``agg='count'``, ``sql='*'``) on its cube, else ``None``.

        A cube declaring more than one row-count measure makes bare
        ``COUNT(*)`` ambiguous — the parser can't tell which one it means
        (it always resolves to the first declared), so ``ref`` must be the
        cube's *sole* row-count measure to qualify."""
        if self.catalog is None:
            return None
        cube_name = cube_of(ref)
        cube = self.catalog.get(cube_name)
        if cube is None:
            return None
        field = ref.split(".", 1)[1]
        count_measures = [
            m.name for m in cube.measures if m.agg == "count" and m.sql.strip() == "*"
        ]
        if field in count_measures and len(count_measures) == 1:
            return cube_name
        return None

    def _cubes_with_row_count(self) -> set[str]:
        """Participating cubes that declare a row-count measure — used to
        decide whether ``COUNT(*)`` is unambiguous for the parser."""
        out: set[str] = set()
        if self.catalog is None:
            return out
        for cube_name in self.cubes:
            cube = self.catalog.get(cube_name)
            if cube is None:
                continue
            if any(m.agg == "count" and m.sql.strip() == "*" for m in cube.measures):
                out.add(cube_name)
        return out

    # -- measure rendering -------------------------------------------------

    def _measure_expr(self, ref: str) -> str:
        """The aggregate-wrapped SQL for a measure ref: ``COUNT(*)`` for an
        unambiguous row-count measure, else ``<AGG>(<ref>)``."""
        row_count_cube = self._row_count_cube(ref)
        if row_count_cube is not None and self._cubes_with_row_count() == {row_count_cube}:
            # ``COUNT(*)`` resolves back to this cube's row-count measure only
            # when it's the single participating cube that has one; otherwise
            # the parser can't tell which cube it means, so fall through to the
            # explicit ``COUNT(<ref>)`` form below.
            return "COUNT(*)"
        func = "SUM"
        if self.catalog is not None:
            cube = self.catalog.get(cube_of(ref))
            field = ref.split(".", 1)[1]
            if cube is not None:
                for m in cube.measures:
                    if m.name == field:
                        func = _AGG_SQL.get(m.agg, "SUM")
                        break
        return f"{func}({ref})"

    # -- clause rendering --------------------------------------------------

    def _from_clause(self) -> str:
        if not self.cubes:
            raise UnparseError(
                "Cannot serialize a query that references no catalog fields: "
                "there is no cube to put in the FROM clause."
            )
        head, *rest = self.cubes
        # ``ON TRUE`` keeps the emitted SQL syntactically valid; the parser
        # ignores the ON clause (the catalog is the source of truth for joins).
        joins = "".join(f" JOIN {c} ON TRUE" for c in rest)
        return f"FROM {head}{joins}"

    def _select_list(self) -> str:
        q = self.q
        alias_for = _reverse_aliases(q.aliases)
        items: list[str] = []
        for dim in q.dimensions:
            items.append(_with_alias(dim, alias_for.get(dim)))
        td = q.time_dimension
        if td is not None and td.granularity is not None:
            # The time bucket is a projection, not a plain dimension; its grain
            # rides on ``time_dimension`` and is recovered from this DATE_TRUNC.
            items.append(f"DATE_TRUNC('{td.granularity}', {td.dimension})")
        for measure in q.measures:
            items.append(_with_alias(self._measure_expr(measure), alias_for.get(measure)))
        if not items:
            # A query with only a time filter and no projections still needs a
            # SELECT target; the parser treats ``*`` as "infer", emitting nothing.
            return "*"
        return ", ".join(items)

    def _where_clause(self) -> str | None:
        q = self.q
        terms: list[str] = [_render_filter(f) for f in q.filters]
        if q.time_dimension is not None:
            terms.append(_render_between(q.time_dimension))
        if q.where is not None:
            terms.append(_render_bool(q.where))
        if not terms:
            return None
        return "WHERE " + " AND ".join(terms)

    def _group_by_clause(self) -> str | None:
        """GROUP BY over the grouping keys — only when the query aggregates
        (has measures). The parser ignores GROUP BY (it derives grouping from
        dimensions vs measures), so this is for valid, readable output."""
        q = self.q
        if not q.measures:
            return None
        keys: list[str] = [*q.dimensions]
        td = q.time_dimension
        if td is not None and td.granularity is not None:
            keys.append(f"DATE_TRUNC('{td.granularity}', {td.dimension})")
        if not keys:
            return None
        return "GROUP BY " + ", ".join(keys)

    def _having_clause(self) -> str | None:
        q = self.q
        if not q.having:
            return None
        # HAVING predicates target measures; render the LHS with the same
        # aggregate marker the SELECT uses so the parser unwraps it to the
        # measure ref.
        terms = [_render_filter(h, lhs=self._measure_expr(h.dimension)) for h in q.having]
        return "HAVING " + " AND ".join(terms)

    def _order_by_clause(self) -> str | None:
        q = self.q
        if not q.order:
            return None
        # ``field`` is either an output-alias key (bare — emit as-is) or a
        # qualified ref (emit the ref; the parser resolves it back).
        parts = [f"{field} {direction.upper()}" for field, direction in q.order]
        return "ORDER BY " + ", ".join(parts)

    def _limit_offset_clause(self) -> str | None:
        q = self.q
        parts: list[str] = []
        if q.limit is not None:
            parts.append(f"LIMIT {q.limit}")
        if q.offset is not None:
            parts.append(f"OFFSET {q.offset}")
        return " ".join(parts) if parts else None

    def _compare_hint(self) -> str:
        """The ``/*+ COMPARE prior_period */`` hint for previous-period
        compare. ``explicit`` mode is rejected upstream in ``_reject_unsupported``."""
        if isinstance(self.q.compare, CompareWindow) and self.q.compare.mode == "previous_period":
            return "/*+ COMPARE prior_period */ "
        return ""

    def render(self) -> str:
        select = f"SELECT {self._compare_hint()}{self._select_list()}"
        clauses: list[str] = [select, self._from_clause()]
        for clause in (
            self._where_clause(),
            self._group_by_clause(),
            self._having_clause(),
            self._order_by_clause(),
            self._limit_offset_clause(),
        ):
            if clause is not None:
                clauses.append(clause)
        return " ".join(clauses)


# ---------------------------------------------------------------------------
# Free helpers (no query state)
# ---------------------------------------------------------------------------


def _reverse_aliases(aliases: dict[str, str]) -> dict[str, str]:
    """Map each referenced field to one output alias.

    ``aliases`` is ``{output_name: qualified_ref}``; the serializer needs the
    reverse. Parser-canonical queries hold at most one alias per ref, so last
    write wins on the rare many-to-one case."""
    out: dict[str, str] = {}
    for alias, ref in aliases.items():
        out[ref] = alias
    return out


def _with_alias(expr: str, alias: str | None) -> str:
    return f"{expr} AS {alias}" if alias is not None else expr


def _boolexpr_dims(node: BoolExpr | Filter) -> list[str]:
    """Every dimension ref mentioned in a BoolExpr tree (for cube discovery)."""
    if isinstance(node, Filter):
        return [node.dimension]
    out: list[str] = []
    for child in node.children:
        out.extend(_boolexpr_dims(child))
    return out


def _render_value(value: str | int | float | bool) -> str:
    """Render a scalar literal as SQL. ``bool`` is checked before ``int``
    because ``bool`` is an ``int`` subclass — a paid flag must emit ``TRUE``,
    not ``1``."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _render_filter(f: Filter, *, lhs: str | None = None) -> str:
    """Render one ``Filter`` as a SQL predicate. ``lhs`` overrides the
    left-hand side (used for HAVING, where the LHS is an aggregate marker)."""
    left = lhs if lhs is not None else f.dimension
    if f.op == "is_null":
        return f"{left} IS NULL"
    if f.op == "not_null":
        return f"{left} IS NOT NULL"
    if f.op in ("in", "not_in"):
        rendered = ", ".join(_render_value(v) for v in f.values)
        keyword = "IN" if f.op == "in" else "NOT IN"
        return f"{left} {keyword} ({rendered})"
    op_sql = _SCALAR_OP_SQL[f.op]
    return f"{left} {op_sql} {_render_value(f.values[0])}"


def _render_between(td: TimeWindow) -> str:
    start, end = td.range
    return f"{td.dimension} BETWEEN {_render_value(start)} AND {_render_value(end)}"


def _render_bool(node: BoolExpr | Filter) -> str:
    """Render a BoolExpr tree, parenthesizing compound nodes so the parser
    reconstructs the same structure."""
    if isinstance(node, Filter):
        return _render_filter(node)
    if node.op == "not":
        return f"NOT ({_render_bool(node.children[0])})"
    sep = " AND " if node.op == "and" else " OR "
    inner = sep.join(_render_bool(c) for c in node.children)
    return f"({inner})"
