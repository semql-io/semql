# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false
"""Time-partitioned source routing (#48).

A cube with ``physical_sources`` declares N physical tables, each tied
to a half-open time range on a routing time dimension. The compiler
intersects the query's ``TimeWindow.range`` with each source's range
and ``UNION ALL``s the matches.

Two functions:

- :func:`select_physical_sources` decides which physical sources (if
  any) cover the query's time range. Half-open intervals: a source
  with ``range_end="2024-01-01"`` covers rows strictly before
  ``2024-01-01``. A query without a time window matches every source.

- :func:`emit_physical_sources` builds the FROM-clause expression for
  a partitioned cube. For each matched source it emits a per-source
  subquery that projects the cube's logical fields from the source's
  physical columns (``SELECT <physical> AS <logical>, ... FROM
  <table>``), then unions them and aliases the result to
  ``cube.alias``. The auth / tenancy / scope wrappers apply to the
  union as a whole, the same way they apply to a single physical
  source.

The integration point in ``compile_query`` calls
:func:`select_physical_sources` and stashes the matches on the
``_CompileEnv``; ``_from_clause_stage`` then dispatches the dialect's
``emit_source`` to the partitioned emit path when the cube has
``physical_sources``. The matched names are surfaced on
``CompiledQuery.physical_sources_hit`` for observability.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

from semql.backend import _ident
from semql.model import Cube, TimePartitionedSource
from semql.spec import TimeWindow


def select_physical_sources(
    cube: Cube,
    time_window: TimeWindow | None,
) -> list[TimePartitionedSource]:
    """Return the sources whose half-open range intersects the
    query's ``TimeWindow.range``.

    No time window → every source matches. A source with a range
    that doesn't intersect the query's range is skipped. Two
    sources MAY overlap; the caller is responsible for
    deduplicating rows if the overlap is non-empty. Returns the
    sources in declaration order so the emitted SQL is stable."""
    if not cube.physical_sources:
        return []

    if time_window is None:
        return list(cube.physical_sources)

    q_start, q_end = time_window.range
    matched: list[TimePartitionedSource] = []
    for src in cube.physical_sources:
        if _ranges_intersect(
            (q_start, q_end),
            (src.range_start, src.range_end),
        ):
            matched.append(src)
    return matched


def _ranges_intersect(
    query: tuple[str | None, str | None],
    src: tuple[str | None, str | None],
) -> bool:
    """Half-open interval intersection. ``None`` is open-ended.

    Touching ranges (``a_end == b_start``) do NOT intersect — the
    boundary belongs to the source with the matching inclusive
    lower bound, not both."""
    q_lo, q_hi = query
    s_lo, s_hi = src
    if q_lo is not None and s_hi is not None and not q_lo < s_hi:
        return False
    return not (q_hi is not None and s_lo is not None and not s_lo < q_hi)


def _projected_field_list(
    cube: Cube,
    src: TimePartitionedSource,
) -> list[exp.Expression]:
    """Build the ``SELECT <expr> AS <logical>, ...`` list for one
    source CTE.

    Every measure / dimension / time dimension on the cube is
    projected. ``column_renames`` maps a logical field name to
    this source's physical column expression; missing renames
    fall back to the logical name verbatim (the source's
    physical column has the same name as the cube's logical
    field). The projection is aliased to the logical name so
    the outer query sees a uniform column vocabulary."""
    projections: list[exp.Expression] = []
    for d in cube.dimensions:
        physical = src.column_renames.get(d.name, d.name)
        projections.append(_project(physical, d.name))
    for td in cube.time_dimensions:
        physical = src.column_renames.get(td.name, td.name)
        projections.append(_project(physical, td.name))
    for m in cube.measures:
        # Measures are aggregations; their underlying column is
        # referenced in ``m.sql`` with the cube alias, which
        # doesn't bind inside the source subquery. We treat the
        # measure's column reference the same way — a bare
        # identifier matching the source's physical column, or a
        # rename if the source calls it something different.
        physical = src.column_renames.get(m.name, _column_from_sql(m.sql))
        projections.append(_project(physical, m.name))
    return projections


def _project(physical: str, logical: str) -> exp.Expression:
    """Parse ``physical`` as a column reference and alias it to
    ``logical``. A simple identifier is the common case; more
    complex expressions (computed columns) fall through
    unchanged.

    The ``type: ignore[return-value]`` follows the existing project
    convention in :mod:`semql.compile` (``_parse_fragment``): the
    sqlglot ``parse_one`` stub is typed as returning the broad
    ``Expr`` superclass, but the dialect-tagged branch we pass
    always returns a concrete ``Expression`` at runtime."""
    parsed = sqlglot.parse_one(physical, read="postgres")
    if isinstance(parsed, exp.Column):
        return exp.alias_(parsed, logical, quoted=False)  # type: ignore[return-value]
    return exp.alias_(parsed, logical, quoted=parsed.is_string)  # type: ignore[return-value]


def _column_from_sql(sql: str) -> str:
    """Extract a bare column name from a fragment like
    ``{o}.amount``. The ``{o}`` placeholder is the cube's
    alias; on the source side we just want the column name.
    Strips the alias prefix and any whitespace."""
    cleaned = sql.replace("{o}.", "").replace("{", "").replace("}", "").strip()
    return cleaned


def _table_node(table: str) -> exp.Table:
    parts = table.split(".", 1)
    if len(parts) == 2:
        return exp.Table(this=_ident(parts[1]), db=_ident(parts[0]))
    return exp.Table(this=_ident(parts[0]))


def _per_source_subquery(
    cube: Cube,
    src: TimePartitionedSource,
) -> exp.Select:
    """Build a ``SELECT ... FROM <src.table>`` AST node for one
    source. The source's column renames are baked into the
    projection list; the rest is straightforward."""
    sel = exp.Select().from_(_table_node(src.table))
    for proj in _projected_field_list(cube, src):
        sel = sel.select(proj)
    return sel


def emit_physical_sources(
    cube: Cube,
    matched: list[TimePartitionedSource],
) -> exp.Subquery:
    """Build the FROM-clause expression for a partitioned cube.

    Emits ``(<union>) AS <cube.alias>`` — a single aliased subquery
    the outer query treats as the cube's row source. The
    auth / tenancy / scope wrappers in ``wrap_for_tenancy`` apply
    to this subquery as a whole, preserving the
    bypass-proof-by-isolation contract.

    A single matched source is emitted as a single subquery, not
    a one-armed ``UNION ALL`` — the outer query sees a
    single-table row source, which sqlglot's renderer emits
    without a redundant ``SELECT * FROM <sub>`` wrapper."""
    if not matched:
        # Should be impossible — the compile path is gated on
        # ``select_physical_sources`` returning non-empty. Raise
        # loudly rather than silently emit a broken query.
        raise ValueError(
            f"emit_physical_sources called with no matched sources for "
            f"cube {cube.name!r}; this is a bug in the compile path."
        )

    if len(matched) == 1:
        sel = _per_source_subquery(cube, matched[0])
    else:
        parts: list[exp.Query] = [_per_source_subquery(cube, src) for src in matched]
        union: exp.Query = parts[0]
        for p in parts[1:]:
            union = exp.Union(this=union, expression=p, distinct=False)
        sel = exp.Select().from_(union).select(exp.Star())

    return exp.Subquery(
        this=sel,
        alias=exp.TableAlias(this=exp.to_identifier(cube.alias)),
    )
