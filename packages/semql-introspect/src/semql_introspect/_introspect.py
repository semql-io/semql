"""Orchestrator — turn a :class:`SchemaProbe` into a list of ``Cube``s.

Reads tables / FKs / PKs from the probe, classifies each column via
:mod:`._heuristics`, and assembles ``Cube`` / ``Measure`` / ``Dimension``
/ ``TimeDimension`` instances. The result is a pure
:class:`semql.model.Cube` list — ready to wrap in a
:class:`semql.Catalog` or to round-trip through the emitter.

Per-column heuristic reasons (the ``# TODO: review`` hints) ride along
on a parallel :class:`HeuristicAnnotation` list, indexed by
``(cube_name, field_name)``. The emitter renders them as inline
comments; programmatic callers can ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass

from semql.model import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    TimeDimension,
)

from semql_introspect._heuristics import classify_column
from semql_introspect._probe import SchemaProbe


def _sql_quote(name: str) -> str:
    """Wrap ``name`` in SQL double-quoted identifier syntax.

    Column names copied from information_schema into generated SQL fragments
    must be quoted to survive names with spaces, reserved words, or hostile
    content (SEMQL-DISC-INTROSPECT-GENERATED-SQL-002).
    """
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class HeuristicAnnotation:
    """One ``# TODO: review`` hint surfaced by the introspector."""

    cube: str
    field: str
    reason: str


@dataclass(frozen=True)
class IntrospectionResult:
    """Cubes + the heuristic-guess annotations parallel to them.

    The cubes alone can be plugged straight into a ``Catalog``; the
    annotations are presentation metadata for emitters / reports."""

    cubes: list[Cube]
    annotations: list[HeuristicAnnotation]


def _alias_for(table: str) -> str:
    """Single-or-two-letter alias derived from the table name.

    semql cube aliases must match ``[a-z_][a-z0-9_]*``. The introspect
    convention: take the initial letters of ``_``-separated tokens
    (``user_events`` → ``ue``); fall back to the first letter for
    single-token names (``orders`` → ``o``). Aliases get
    deduplicated at the catalog assembly step."""
    tokens = [t for t in table.lower().split("_") if t]
    if not tokens:
        return "t"
    if len(tokens) == 1:
        return tokens[0][0]
    return "".join(t[0] for t in tokens)


def _dedupe_aliases(cubes: list[Cube]) -> list[Cube]:
    """Ensure every cube has a unique alias.

    Two ``user_events`` / ``user_eligibility`` tables both alias to
    ``ue`` by default — append a suffix to all but the first
    collider."""
    seen: dict[str, int] = {}
    out: list[Cube] = []
    for cube in cubes:
        alias = cube.alias
        n = seen.get(alias, 0)
        if n > 0:
            new_alias = f"{alias}{n + 1}"
            seen[alias] = n + 1
            cube = cube.model_copy(update={"alias": new_alias})
        else:
            seen[alias] = 1
        out.append(cube)
    return out


def introspect(probe: SchemaProbe, *, dialect: Dialect) -> IntrospectionResult:
    """Build a catalog from a probe + a target backend.

    ``backend`` is the semql ``Dialect`` enum tag stamped onto every
    emitted cube — the introspector doesn't try to detect the dialect
    itself because a single probe shape often spans multiple backends
    (ANSI ``information_schema`` works against PG, DuckDB, and
    Snowflake)."""
    tables = probe.list_tables()
    fks = probe.list_foreign_keys()
    pks = probe.list_primary_keys()

    table_names = {t.name for t in tables}

    # ``{from_table.from_column → to_table}`` so the per-column loop
    # knows which dim should carry ``foreign_key=``.
    fk_by_source: dict[tuple[str, str], str] = {
        (fk.from_table, fk.from_column): fk.to_table for fk in fks if fk.to_table in table_names
    }

    cubes: list[Cube] = []
    annotations: list[HeuristicAnnotation] = []

    for tbl in tables:
        alias = _alias_for(tbl.name)
        pk_col = pks.get(tbl.name)
        measures: list[Measure] = []
        dimensions: list[Dimension] = []
        time_dims: list[TimeDimension] = []
        cube_pk: str | None = None

        for col in tbl.columns:
            is_fk = (tbl.name, col.name) in fk_by_source
            is_pk = pk_col == col.name
            cls = classify_column(col, is_fk=is_fk, is_pk=is_pk)
            sql = f"{{{alias}}}.{_sql_quote(col.name)}"

            if cls.kind == "time_dimension":
                time_dims.append(TimeDimension(name=col.name, sql=sql))
            elif cls.kind == "measure_sum":
                measures.append(Measure(name=col.name, sql=sql, agg="sum"))
                if cls.heuristic_reason:
                    annotations.append(
                        HeuristicAnnotation(
                            cube=tbl.name, field=col.name, reason=cls.heuristic_reason
                        )
                    )
            elif cls.kind == "measure_count_distinct":
                # Distinct-count measure named ``distinct_<col>`` so the
                # output catalog reads cleanly — ``orders.distinct_customer_id``.
                m_name = f"distinct_{col.name}"
                measures.append(Measure(name=m_name, sql=sql, agg="count_distinct"))
                if cls.heuristic_reason:
                    annotations.append(
                        HeuristicAnnotation(
                            cube=tbl.name, field=m_name, reason=cls.heuristic_reason
                        )
                    )
            else:
                # Plain dimension. FK targets get ``foreign_key=...``;
                # everything else is just a typed dimension.
                assert cls.dim_type is not None  # classifier invariant
                fk_target = fk_by_source.get((tbl.name, col.name))
                dimensions.append(
                    Dimension(
                        name=col.name,
                        sql=sql,
                        type=cls.dim_type,
                        foreign_key=fk_target,
                    )
                )
                if is_pk:
                    cube_pk = col.name

        # Auto-derived joins: for every FK source on this table whose
        # target also got introspected, emit a ``many_to_one`` Join.
        # The semql catalog will further auto-derive from
        # ``Dimension.foreign_key`` at construction time, but spelling
        # the joins out in the generated source makes them visible at
        # review.
        joins: list[Join] = []
        for fk in fks:
            if fk.from_table != tbl.name or fk.to_table not in table_names:
                continue
            target_alias = _alias_for(fk.to_table)
            joins.append(
                Join(
                    to=fk.to_table,
                    relationship="many_to_one",
                    on=(
                        f"{{{alias}}}.{_sql_quote(fk.from_column)}"
                        f" = {{{target_alias}}}.{_sql_quote(fk.to_column)}"
                    ),
                )
            )

        cubes.append(
            Cube(
                name=tbl.name,
                dialect=dialect,
                table=tbl.name,
                alias=alias,
                primary_key=cube_pk,
                measures=measures,
                dimensions=dimensions,
                time_dimensions=time_dims,
                joins=joins,
            )
        )

    cubes = _dedupe_aliases(cubes)
    return IntrospectionResult(cubes=cubes, annotations=annotations)


__all__ = ["HeuristicAnnotation", "IntrospectionResult", "introspect"]
