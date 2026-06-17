# pyright: reportPrivateImportUsage=false
"""Render a :class:`semql.federate.MergeSpec` into DuckDB merge SQL.

The merge step runs in-process DuckDB over per-fragment result sets
materialised as ``frag_0`` / ``frag_1`` / … tables. This module turns
the *structured* :class:`MergeSpec` — the authoritative federation plan
emitted by :func:`semql.compile_federated_query` — into the ``(sql,
params)`` that produces the final shape.

Pure and sans-io: no database handle, no row data, just spec → SQL. The
core compiler stays dialect-agnostic (it emits only the spec); the
DuckDB dialect knowledge lives here, beside the executor that runs it.

Cross-partition filter and HAVING values bind as ``$m{n}`` placeholders,
never inlined — the same bind-never-inline invariant the fragments hold.
"""

from __future__ import annotations

from collections.abc import Callable

from semql.federate import (
    DimensionOutput,
    FragmentColumn,
    MeasureOutput,
    MergeSpec,
)
from semql.refs import field_of, local_name
from sqlglot import exp

# Structural mirror of ``semql.federate``'s resolved cross-partition
# clause: each literal is ``(negated, fragment_index, column_name, op,
# values)``; a clause is an OR of literals. Declared locally rather than
# importing the core's private alias — the values arrive as plain tuples.
_ResolvedCrossClause = tuple[tuple[bool, int, str, str, tuple[object, ...]], ...]

# Quantile for each percentile aggregate — mirrors the core federation
# emitter so the rendered ``PERCENTILE_CONT`` matches byte-for-byte.
_PERCENTILES: dict[str, float] = {"median": 0.5, "p75": 0.75, "p90": 0.90, "p95": 0.95}

# Concrete comparison-node classes keyed by Filter op. ``Callable[...,
# exp.Expression]`` (not ``type[exp.Binary]``) because sqlglot's
# ``Binary`` is an *ancestor* of ``Expression``, not a subtype.
_BINARY_OPS: dict[str, Callable[..., exp.Expression]] = {
    "eq": exp.EQ,
    "neq": exp.NEQ,
    "gt": exp.GT,
    "gte": exp.GTE,
    "lt": exp.LT,
    "lte": exp.LTE,
}


class _Binder:
    """Bind merge values as DuckDB ``$m{n}`` named parameters, in
    emission order — identical naming to the core emitter so a rendered
    plan's params match the spec's recorded values."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def placeholder(self, value: object) -> exp.Placeholder:
        name = f"m{len(self.params)}"
        self.params[name] = value
        return exp.Placeholder(this=name)


def _frag_col(c: FragmentColumn) -> exp.Column:
    """Quoted ``"f{idx}"."col"`` reference into a fragment's result set."""
    return exp.column(c.column_name, table=f"f{c.fragment_index}", quoted=True)


def _aliased(value: exp.Expression, name: str) -> exp.Alias:
    """``value AS "name"`` with a quoted output identifier."""
    return exp.Alias(this=value, alias=exp.to_identifier(name, quoted=True))


def _simple_agg(agg: str, src: FragmentColumn) -> exp.Expression:
    """Merge-side aggregate over a single fragment column. An empty
    ``column_name`` is the count-star sentinel → ``COUNT(*)``."""
    if src.column_name == "":
        return exp.Count(this=exp.Star())
    ref = _frag_col(src)
    if agg == "sum":
        return exp.Sum(this=ref)
    if agg == "count":
        return exp.Count(this=ref)
    if agg == "count_distinct":
        return exp.Count(this=exp.Distinct(expressions=[ref]))
    if agg == "avg":
        return exp.Avg(this=ref)
    if agg == "min":
        return exp.Min(this=ref)
    if agg == "max":
        return exp.Max(this=ref)
    if agg in _PERCENTILES:
        # Renders as DuckDB's QUANTILE_CONT(col, q ORDER BY col).
        return exp.WithinGroup(
            this=exp.Anonymous(
                this="PERCENTILE_CONT", expressions=[exp.Literal.number(_PERCENTILES[agg])]
            ),
            expression=exp.Order(expressions=[exp.Ordered(this=ref)]),
        )
    raise ValueError(f"merge agg {agg!r} has no DuckDB recipe.")


def _nullif_div(num: exp.Expression, den: exp.Expression) -> exp.Expression:
    """``num / NULLIF(den, 0)`` — the guarded division shared by
    avg-recomposition and ratio measures."""
    return exp.Div(
        this=num,
        expression=exp.Anonymous(this="NULLIF", expressions=[den, exp.Literal.number(0)]),
    )


def _measure_expr(m: MeasureOutput) -> exp.Expression:
    """Reconstruct a measure's merge-side expression from its recipe."""
    if m.merge_agg == "passthrough":
        assert m.source is not None
        return _frag_col(m.source)
    if m.merge_agg == "avg_recomposed":
        assert m.sum_source is not None and m.count_source is not None
        return _nullif_div(
            exp.Sum(this=_frag_col(m.sum_source)), exp.Sum(this=_frag_col(m.count_source))
        )
    if m.merge_agg == "ratio":
        assert m.numerator is not None and m.denominator is not None
        assert m.numerator_agg is not None and m.denominator_agg is not None
        return _nullif_div(
            _simple_agg(m.numerator_agg, m.numerator),
            _simple_agg(m.denominator_agg, m.denominator),
        )
    assert m.source is not None
    return _simple_agg(m.merge_agg, m.source)


def _dimension_expr(d: DimensionOutput) -> exp.Expression:
    """A dimension's merge-side value: a ``date_trunc`` bucket when the
    merge buckets it (``time_grain`` set), else a straight passthrough of
    its single fragment source."""
    src = d.sources[0]
    if d.time_grain is not None:
        return exp.Anonymous(
            this="date_trunc", expressions=[exp.Literal.string(d.time_grain), _frag_col(src)]
        )
    return _frag_col(src)


def _filter_predicate(
    col: exp.Expression, op: str, values: tuple[object, ...], binder: _Binder
) -> exp.Expression:
    """Build a DuckDB predicate for ``op``/``values`` over ``col``,
    binding every comparison value as a ``$m`` parameter."""
    vals = list(values)
    if op in _BINARY_OPS:
        return _BINARY_OPS[op](this=col, expression=binder.placeholder(vals[0]))
    if op == "in":
        return exp.In(this=col, expressions=[binder.placeholder(v) for v in vals])
    if op == "not_in":
        return exp.Not(this=exp.In(this=col, expressions=[binder.placeholder(v) for v in vals]))
    if op == "is_null":
        return exp.Is(this=col, expression=exp.Null())
    if op == "not_null":
        return exp.Not(this=exp.Is(this=col, expression=exp.Null()))
    if op == "contains":
        return exp.ILike(this=col, expression=binder.placeholder("%" + str(vals[0]) + "%"))
    raise ValueError(f"merge filter op {op!r} unsupported.")


def _cross_clause(clause: _ResolvedCrossClause, binder: _Binder) -> exp.Expression:
    """One resolved cross-partition clause → an OR of (optionally
    negated) predicate nodes, read straight off fragment coordinates."""
    lits: list[exp.Expression] = []
    for negated, frag_idx, col_name, op, values in clause:
        pred = _filter_predicate(
            exp.column(col_name, table=f"f{frag_idx}", quoted=True), op, values, binder
        )
        lits.append(exp.Not(this=pred) if negated else pred)
    clause_expr = lits[0]
    for lit in lits[1:]:
        clause_expr = exp.Or(this=clause_expr, expression=lit)
    return clause_expr


def _having_term(
    alias: str, op: str, values: tuple[object, ...], binder: _Binder
) -> exp.Expression:
    if op not in _BINARY_OPS:
        raise ValueError(f"merge HAVING op {op!r} unsupported.")
    return _BINARY_OPS[op](
        this=exp.column(alias, quoted=True), expression=binder.placeholder(values[0])
    )


def render_merge_sql(spec: MergeSpec) -> tuple[str, dict[str, object]]:
    """Render ``spec`` into ``(duckdb_sql, params)``.

    Fragments are aliased ``f0``, ``f1``, …; the primary fragment is the
    FROM target and bridges become ordered joins, each honouring its
    ``join_kind`` (``left`` lookup, ``inner`` filter-restriction). The SELECT
    list is
    dimensions (the time bucket among them), then measures — so the
    positional GROUP BY lines up. An all-passthrough spec (the degenerate
    single-backend plan) renders as a plain projection with no GROUP BY.
    """
    binder = _Binder()
    primary = spec.primary_index

    select_list: list[exp.Expression] = [
        _aliased(_dimension_expr(d), d.output_name) for d in spec.dimensions
    ]
    select_list += [_aliased(_measure_expr(m), m.output_name) for m in spec.measures]

    select = exp.select(*select_list).from_(
        exp.alias_(exp.to_table(f"frag_{primary}"), f"f{primary}")
    )
    join_type_for = {"left": "LEFT", "inner": "INNER", "full_outer": "FULL OUTER"}
    for b in spec.bridges:
        new_idx = b.right.fragment_index
        select = select.join(
            exp.alias_(exp.to_table(f"frag_{new_idx}"), f"f{new_idx}"),
            on=exp.EQ(this=_frag_col(b.left), expression=_frag_col(b.right)),
            join_type=join_type_for[getattr(b, "join_kind", "left")],
        )

    for clause in spec.cross_partition_clauses:
        select = select.where(_cross_clause(clause, binder))

    # Aggregate only when a measure actually re-aggregates; an
    # all-passthrough spec is a plain projection (no GROUP BY).
    aggregating = any(m.merge_agg != "passthrough" for m in spec.measures)
    n_group = len(spec.dimensions)
    if aggregating and n_group > 0:
        select = select.group_by(*(exp.Literal.number(i + 1) for i in range(n_group)))

    selected = {m.output_name for m in spec.measures}
    for hf in spec.having:
        alias = field_of(hf.dimension)
        if alias not in selected:
            raise ValueError(f"HAVING references {hf.dimension!r}, not a selected measure.")
        select = select.having(_having_term(alias, hf.op, tuple(hf.values), binder))

    for ref, direction in spec.order_by:
        alias = local_name(ref)
        select = select.order_by(
            exp.Ordered(this=exp.column(alias, quoted=True), desc=direction == "desc")
        )

    if spec.limit is not None:
        select = select.limit(int(spec.limit))
    if spec.offset is not None and spec.offset > 0:
        select = select.offset(int(spec.offset))

    return select.sql(dialect="duckdb"), binder.params


__all__ = ["render_merge_sql"]
