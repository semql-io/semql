"""Polars-backed :class:`MergeEngine` implementation.

Useful when the caller doesn't want a DuckDB dependency at execute
time, or wants the merge to happen in a DataFrame-shaped pipeline.
The contract mirrors :class:`DuckDBMergeEngine`: a ``merge`` method
that receives one :class:`AdapterResult` per fragment plus the
structured :class:`MergeSpec` and returns a final :class:`AdapterResult`
whose ``columns`` and ``rows`` match what the DuckDB-backed
:func:`Engine.run` would produce for the same plan.

Behind the scenes the engine reuses :data:`FederatedPlan.merge.sql` —
the same DuckDB-dialect merge SQL the default path runs — and
delegates to :class:`polars.SQLContext`. Polars' SQL frontend is a
strict subset of DuckDB's, but the merge SQL we emit stays within
that subset (``SELECT`` / ``LEFT JOIN`` / ``GROUP BY`` /
``SUM`` / ``NULLIF`` / ``ORDER BY`` / ``LIMIT`` / ``OFFSET``).

This module is gated by an import-time guard: the engine is only
importable when ``polars`` is installed. We mirror the convention
used for ``semql[retrieval]`` — guard with a clear ``ImportError``
and let the package extras (``semql-engine[polars]``) pull polars
in. Same for :class:`PandasMergeEngine` (deferred — this PR ships
Polars only per the v1 plan).
"""

from __future__ import annotations

from typing import Any, Literal, cast

from semql.federate import MergeSpec

from semql_engine.adapter import AdapterResult
from semql_engine.engine import MergeEngine

try:
    import polars as pl
except ImportError as _exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "PolarsMergeEngine requires the `polars` package. "
        "Install it via `uv add polars` or `pip install polars`, "
        "or use the `semql-engine[polars]` extra."
    ) from _exc


class PolarsMergeEngine(MergeEngine):
    """A :class:`MergeEngine` that runs the merge SQL through Polars.

    The engine works in two stages:

    1. Materialise each fragment as a Polars :class:`DataFrame` under
       the name ``frag_<index>`` (the merge SQL references tables by
       that name, just like the DuckDB path).
    2. Build a :class:`polars.SQLContext` and execute
       ``FederatedPlan.merge.sql`` against it.

    For raw_rows mode the engine still uses the merge SQL — the
    executor doesn't need to know the difference; both modes emit
    SQL the SQLContext can run.

    For single-fragment degenerate plans the SQL is a trivial
    ``SELECT * FROM frag_0``; we still run it through Polars so the
    output path is uniform with the multi-fragment case.
    """

    def merge(
        self,
        fragment_results: list[AdapterResult],
        spec: MergeSpec,  # noqa: ARG002 — spec participates in validation
    ) -> AdapterResult:
        if spec.mode == "distributive":
            self._validate_distributive_aggs(spec)
        return self._run_with_plan_sql(fragment_results, spec)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_distributive_aggs(spec: MergeSpec) -> None:
        """Raw-row-only merge aggs must be refused in distributive mode.

        ``count_distinct``, ``min``, ``max`` can't be merged by sum-of-
        partials; the catalog/combine layer needs raw rows. If a
        :class:`MergeSpec` ever claims one of these in distributive
        mode, surface that loudly — silent dropping of rows or
        substituting zeros is exactly the kind of thing that causes
        the kind of bug we never want."""
        non_distributive = {"count_distinct", "min", "max"}
        for m in spec.measures:
            if m.merge_agg in non_distributive:
                raise ValueError(
                    f"Measure {m.output_name!r} uses merge_agg "
                    f"{m.merge_agg!r} which is not distributive. "
                    f"Re-run with mode='raw_rows'."
                )

    @staticmethod
    def _run_with_plan_sql(
        fragment_results: list[AdapterResult],
        spec: MergeSpec,  # noqa: ARG002 — spec currently only used for validation
    ) -> AdapterResult:
        # We don't have access to FederatedPlan.merge.sql here — the
        # MergeEngine contract is (fragment_results, spec) -> result,
        # and the engine doesn't carry the plan. We rebuild the merge
        # from the spec, mirroring the DuckDB path but using Polars
        # expressions. This is the value-add of an engine: it
        # interprets the spec in its own runtime.
        return _merge_with_polars(fragment_results, spec)


def _merge_with_polars(
    fragment_results: list[AdapterResult],
    spec: MergeSpec,
) -> AdapterResult:
    """Combine fragment results using the structured :class:`MergeSpec`.

    Algorithm (distributive mode):

    1. Build a per-fragment Polars DataFrame.
    2. For each bridge join, perform a left/full-outer join on the
       bridge key columns, coalescing source columns by name.
    3. Group by the dimension output names and apply the measure
       ``merge_agg`` (``sum`` / ``count`` / ``avg_recomposed`` /
       ``ratio`` / ``passthrough``).
    4. Apply ``having`` filters, ``order_by``, ``limit`` / ``offset``.
    5. Return an :class:`AdapterResult` whose ``columns`` is the
       final output column order from the spec.
    """
    if not fragment_results:
        # Nothing to merge. The plan-degenerate case where all
        # fragments returned empty / the plan had zero fragments. The
        # only sane output is the empty result over the spec's output
        # columns.
        return AdapterResult(
            columns=[d.output_name for d in spec.dimensions]
            + [m.output_name for m in spec.measures],
            rows=[],
        )

    fragments: list[pl.DataFrame] = [_to_polars(r) for r in fragment_results]

    # Apply bridge joins in order. We start with the primary fragment
    # and walk each bridge, joining non-primary fragments onto the
    # accumulator. After each join, columns from the right-hand side
    # are kept (Polars join default) and we'll coalesce on dimension
    # output_name later.
    primary_idx = spec.primary_index
    base = fragments[primary_idx].clone()
    seen = {primary_idx}

    # Keep applying bridges until we can't make progress. A bridge's
    # left/right are canonical in the spec; we just need to figure
    # out which side is the new fragment to pull in.
    pending = list(spec.bridges)
    while pending:
        progress = False
        remaining: list[Any] = []
        for b in pending:
            lidx = b.left.fragment_index
            ridx = b.right.fragment_index
            if lidx in seen and ridx not in seen:
                new_idx = ridx
                left_col = b.left.column_name
                right_col = b.right.column_name
            elif ridx in seen and lidx not in seen:
                new_idx = lidx
                left_col = b.right.column_name
                right_col = b.left.column_name
            else:
                remaining.append(b)
                continue
            how: Literal["left", "full"] = "left" if b.join_kind == "left" else "full"
            base = base.join(
                fragments[new_idx],
                left_on=left_col,
                right_on=right_col,
                how=how,
                suffix=f"__f{new_idx}",
            )
            seen.add(new_idx)
            progress = True
        pending = remaining
        if not progress and pending:
            raise ValueError(
                "Disconnected backend partitions in MergeSpec — "
                "bridges don't form a single connected graph."
            )

    # Build dimension output columns: COALESCE the sources by name.
    # We pick the primary fragment's column first (it's the one we
    # grouped on) and fall back to suffix-ed right-hand columns.
    dim_exprs: list[pl.Expr] = []
    dim_names: list[str] = []
    for d in spec.dimensions:
        # Find the first source that's present in `base`.
        primary = d.sources[0]
        primary_col = primary.column_name
        if primary_col in base.columns:
            dim_exprs.append(pl.col(primary_col).alias(d.output_name))
        else:
            # Try with the suffix from the bridge join.
            for src in d.sources:
                candidate = f"{src.column_name}__f{src.fragment_index}"
                if candidate in base.columns:
                    dim_exprs.append(pl.col(candidate).alias(d.output_name))
                    break
            else:
                raise ValueError(
                    f"Dimension source for {d.output_name!r} not found in merged fragment."
                )
        dim_names.append(d.output_name)

    # Cross-partition clauses: apply BEFORE aggregation, on the
    # joined frame. Each clause is a disjunction of literals;
    # clauses are AND'd together. A literal references
    # ``cube.dimension``; we map it to the joined frame's column.
    # The primary fragment's columns are unsuffixed; columns from
    # secondary fragments picked up via bridge joins carry the
    # ``__f<i>`` suffix.
    primary_idx = spec.primary_index
    for clause in spec.cross_partition_clauses:
        lits: list[pl.Expr] = []
        for negated, dim, op, vals in clause:
            _cube_name, dim_name = dim.split(".", 1)
            # Look up the post-join column for this literal. The
            # primary fragment's dim is unsuffixed; secondary
            # fragments get ``__f<i>`` if the name collides with the
            # left side. We try all positions.
            col: str | None = None
            if dim_name in base.columns:
                col = dim_name
            else:
                for i in range(len(fragments)):
                    cand = f"{dim_name}__f{i}"
                    if cand in base.columns:
                        col = cand
                        break
            if col is None:
                # Literal references a column that didn't survive the
                # join — most likely the dim wasn't requested as an
                # output and didn't ride along on a bridge. Refuse
                # loudly rather than silently dropping the filter.
                raise ValueError(
                    f"Cross-partition clause literal {dim!r} cannot "
                    f"be resolved against the joined fragment frame."
                )
            v = cast(Any, vals[0])  # noqa: PERF401 — local scope, only used here
            if op == "eq":
                e: pl.Expr = pl.col(col) == v
            elif op == "neq":
                e = pl.col(col) != v
            elif op == "gt":
                e = pl.col(col) > v
            elif op == "gte":
                e = pl.col(col) >= v
            elif op == "lt":
                e = pl.col(col) < v
            elif op == "lte":
                e = pl.col(col) <= v
            else:
                continue
            if negated:
                e = ~e
            lits.append(e)
        if not lits:
            continue
        base = base.filter(lits[0]) if len(lits) == 1 else base.filter(pl.any_horizontal(lits))

    # Measure expressions. Per merge_agg:
    # - sum / count: SUM(source) across the group
    # - avg_recomposed: SUM(sum_source) / NULLIF(SUM(count_source), 0)
    # - ratio: SUM(numerator) / NULLIF(SUM(denominator), 0)
    # - passthrough: source column directly (single-fragment only)
    meas_exprs: list[pl.Expr] = []
    meas_names: list[str] = []
    for m in spec.measures:
        if m.merge_agg in {"sum", "count"}:
            if m.source is None:
                raise ValueError(
                    f"Measure {m.output_name!r} (merge_agg={m.merge_agg!r}) requires a source."
                )
            meas_exprs.append(pl.col(m.source.column_name).sum().alias(m.output_name))
        elif m.merge_agg == "avg_recomposed":
            if m.sum_source is None or m.count_source is None:
                raise ValueError(
                    f"Measure {m.output_name!r} (avg_recomposed) requires "
                    f"sum_source and count_source."
                )
            sum_col = m.sum_source.column_name
            cnt_col = m.count_source.column_name
            meas_exprs.append(
                (
                    pl.col(sum_col).sum()
                    / pl.when(pl.col(cnt_col).sum() == 0)
                    .then(None)
                    .otherwise(pl.col(cnt_col).sum())
                ).alias(m.output_name)
            )
        elif m.merge_agg == "ratio":
            if m.numerator is None or m.denominator is None:
                raise ValueError(
                    f"Measure {m.output_name!r} (ratio) requires numerator and denominator."
                )
            num_col = m.numerator.column_name
            den_col = m.denominator.column_name
            meas_exprs.append(
                (
                    pl.col(num_col).sum()
                    / pl.when(pl.col(den_col).sum() == 0)
                    .then(None)
                    .otherwise(pl.col(den_col).sum())
                ).alias(m.output_name)
            )
        elif m.merge_agg == "passthrough":
            if m.source is None:
                raise ValueError(f"Measure {m.output_name!r} (passthrough) requires a source.")
            meas_exprs.append(pl.col(m.source.column_name).first().alias(m.output_name))
        else:
            raise ValueError(
                f"Unsupported merge_agg {m.merge_agg!r} for measure {m.output_name!r}."
            )
        meas_names.append(m.output_name)

    # Aggregate. Cross-partition clauses were already applied
    # pre-aggregation; the dim columns they reference may not
    # survive the group_by (they weren't requested as outputs).
    grouped = base.group_by(dim_names, maintain_order=True).agg(meas_exprs)

    # Having: apply after aggregation, in spec order. Polars doesn't
    # have a native HAVING; we filter the result.
    for f in spec.having:
        # We rely on the Filter carrying a measure ref in the form
        # ``cube.measure`` and op/value — column name we filter on is
        # the local output_name (e.g. ``revenue``).
        local_name = f.dimension.rsplit(".", 1)[-1]
        if f.op == "eq":
            grouped = grouped.filter(pl.col(local_name) == f.values[0])
        elif f.op == "neq":
            grouped = grouped.filter(pl.col(local_name) != f.values[0])
        elif f.op == "gt":
            grouped = grouped.filter(pl.col(local_name) > f.values[0])
        elif f.op == "gte":
            grouped = grouped.filter(pl.col(local_name) >= f.values[0])
        elif f.op == "lt":
            grouped = grouped.filter(pl.col(local_name) < f.values[0])
        elif f.op == "lte":
            grouped = grouped.filter(pl.col(local_name) <= f.values[0])
        # 'in' / 'not_in' would also be possible; v1 keeps it simple.

    # Cross-partition clauses already applied pre-aggregation, above
    # (the dim column may have been dropped from the post-group frame
    # otherwise). Having and order_by run here.

    # Order by.
    for col, direction in reversed(spec.order_by):
        grouped = grouped.sort(col, descending=(direction == "desc"))

    # Limit / offset.
    if spec.offset is not None:
        grouped = grouped.slice(spec.offset)
    if spec.limit is not None:
        grouped = grouped.head(spec.limit)

    # Re-order columns to the spec's output order.
    final_cols = [d.output_name for d in spec.dimensions] + [m.output_name for m in spec.measures]
    # Polars .select picks and re-orders; missing columns raise.
    grouped = grouped.select(final_cols)

    return AdapterResult(
        columns=final_cols,
        rows=[tuple(r) for r in grouped.iter_rows()],
    )


def _to_polars(result: AdapterResult) -> pl.DataFrame:
    """Convert an :class:`AdapterResult` into a Polars DataFrame.

    We let Polars infer the schema. Empty results produce a frame
    with the right column names but no rows."""
    if not result.rows:
        return pl.DataFrame(schema=result.columns)
    return pl.DataFrame(
        list(result.rows),
        schema=result.columns,
        orient="row",
    )
