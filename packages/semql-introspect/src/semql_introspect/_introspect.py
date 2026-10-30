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
from enum import StrEnum

from semql.catalog import SourceProvenance
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


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    INFERRED = "inferred"
    REPORT_ONLY = "report_only"


@dataclass(frozen=True)
class MappingFact:
    source_kind: str
    source_columns: tuple[str, ...]
    status: MappingStatus
    detail: str


@dataclass(frozen=True)
class TableMappingReport:
    table: str
    source_primary_key: tuple[str, ...]
    mapped_primary_key: str | None
    unsupported_primary_key_components: tuple[str, ...]
    unsupported_foreign_keys: tuple[str, ...]
    mapping_status: MappingStatus
    facts: tuple[MappingFact, ...]


@dataclass(frozen=True)
class IntrospectionResult:
    """Catalog cubes with heuristic annotations and loss-aware reports."""

    cubes: list[Cube]
    annotations: list[HeuristicAnnotation]
    table_reports: list[TableMappingReport]
    source_provenance: SourceProvenance | None = None


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


def introspect(probe: SchemaProbe, *, dialect: Dialect) -> IntrospectionResult:
    """Build a catalog from a probe + a target backend.

    ``backend`` is the semql ``Dialect`` enum tag stamped onto every
    emitted cube — the introspector doesn't try to detect the dialect
    itself because a single probe shape often spans multiple backends
    (ANSI ``information_schema`` works against PG, DuckDB, and
    Snowflake)."""
    tables = sorted(probe.list_tables(), key=lambda table: table.name)
    fks = sorted(
        probe.list_foreign_keys(), key=lambda fk: (fk.from_table, fk.to_table, fk.constraint_name)
    )
    pks = probe.list_primary_keys()
    table_names = {table.name for table in tables}
    aliases: dict[str, str] = {}
    used_aliases: set[str] = set()
    for table in tables:
        base = _alias_for(table.name)
        alias = base
        suffix = 2
        while alias in used_aliases:
            alias = f"{base}{suffix}"
            suffix += 1
        aliases[table.name] = alias
        used_aliases.add(alias)
    fk_columns = {(fk.from_table, column) for fk in fks for column in fk.from_columns}
    fk_targets: dict[tuple[str, str], set[str]] = {}
    for fk in fks:
        if (
            fk.to_table in table_names
            and len(fk.from_columns) == 1
            and fk.to_columns == pks.get(fk.to_table, ())
            and len(fk.to_columns) == 1
        ):
            for column in fk.from_columns:
                fk_targets.setdefault((fk.from_table, column), set()).add(fk.to_table)
    fk_by_source = {
        key: next(iter(targets)) for key, targets in fk_targets.items() if len(targets) == 1
    }
    cubes: list[Cube] = []
    annotations: list[HeuristicAnnotation] = []
    reports: list[TableMappingReport] = []
    for tbl in tables:
        alias = aliases[tbl.name]
        pk_columns = pks.get(tbl.name, ())
        single_pk = pk_columns[0] if len(pk_columns) == 1 else None
        measures: list[Measure] = []
        dimensions: list[Dimension] = []
        time_dims: list[TimeDimension] = []
        cube_pk: str | None = None
        for col in tbl.columns:
            is_fk = (tbl.name, col.name) in fk_columns
            is_pk = col.name in pk_columns
            cls = classify_column(col, is_fk=is_fk, is_pk=is_pk)
            sql = f"{{{alias}}}.{_sql_quote(col.name)}"
            if cls.kind == "time_dimension":
                time_dims.append(TimeDimension(name=col.name, sql=sql))
            elif cls.kind == "measure_sum":
                measures.append(Measure(name=col.name, sql=sql, agg="sum"))
                if cls.heuristic_reason:
                    annotations.append(
                        HeuristicAnnotation(tbl.name, col.name, cls.heuristic_reason)
                    )
            elif cls.kind == "measure_count_distinct":
                name = f"distinct_{col.name}"
                measures.append(Measure(name=name, sql=sql, agg="count_distinct"))
                if cls.heuristic_reason:
                    annotations.append(HeuristicAnnotation(tbl.name, name, cls.heuristic_reason))
            else:
                assert cls.dim_type is not None
                dimensions.append(
                    Dimension(
                        name=col.name,
                        sql=sql,
                        type=cls.dim_type,
                        foreign_key=fk_by_source.get((tbl.name, col.name)),
                    )
                )
                if col.name == single_pk:
                    cube_pk = col.name
        joins: list[Join] = []
        table_fks = [fk for fk in fks if fk.from_table == tbl.name]
        unsupported_fks: list[str] = [
            f"{column}: multiple target tables {tuple(sorted(targets))} "
            "cannot fit Dimension.foreign_key"
            for (source_table, column), targets in sorted(fk_targets.items())
            if source_table == tbl.name and len(targets) > 1
        ]
        for fk in table_fks:
            if fk.to_table not in table_names:
                unsupported_fks.append(
                    f"{fk.constraint_name}: {fk.from_table}{fk.from_columns} -> "
                    f"{fk.to_table}{fk.to_columns}; target table not introspected"
                )
                continue
            target_alias = aliases[fk.to_table]
            predicate = " AND ".join(
                f"{{{alias}}}.{_sql_quote(source)} = {{{target_alias}}}.{_sql_quote(target)}"
                for source, target in zip(fk.from_columns, fk.to_columns, strict=True)
            )
            joins.append(Join(to=fk.to_table, relationship="many_to_one", on=predicate))
        pk_loss = tuple(column for column in pk_columns if column != single_pk)
        mapping_status = (
            MappingStatus.REPORT_ONLY
            if pk_loss or unsupported_fks
            else MappingStatus.INFERRED
            if any(a.cube == tbl.name for a in annotations)
            else MappingStatus.MAPPED
        )
        facts: list[MappingFact] = []
        if pk_columns:
            facts.append(
                MappingFact(
                    source_kind="primary_key",
                    source_columns=tuple(pk_columns),
                    status=MappingStatus.REPORT_ONLY if pk_loss else MappingStatus.MAPPED,
                    detail=(
                        "Composite primary-key structure is retained in the report; "
                        "Cube.primary_key is scalar."
                        if pk_loss
                        else "Mapped to Cube.primary_key."
                    ),
                )
            )
        for fk in table_fks:
            facts.append(
                MappingFact(
                    source_kind="foreign_key",
                    source_columns=fk.from_columns,
                    status=(
                        MappingStatus.REPORT_ONLY
                        if fk.to_table not in table_names
                        else MappingStatus.MAPPED
                    ),
                    detail=(
                        f"{fk.constraint_name}: {fk.from_table}{fk.from_columns} -> "
                        f"{fk.to_table}{fk.to_columns}"
                        + (
                            "; target table not introspected"
                            if fk.to_table not in table_names
                            else ""
                        )
                    ),
                )
            )
        facts.extend(
            MappingFact(
                "column_heuristic", (annotation.field,), MappingStatus.INFERRED, annotation.reason
            )
            for annotation in annotations
            if annotation.cube == tbl.name
        )
        reports.append(
            TableMappingReport(
                table=tbl.name,
                source_primary_key=tuple(pk_columns),
                mapped_primary_key=cube_pk,
                unsupported_primary_key_components=pk_loss,
                unsupported_foreign_keys=tuple(unsupported_fks),
                mapping_status=mapping_status,
                facts=tuple(facts),
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
    return IntrospectionResult(
        cubes=cubes,
        annotations=annotations,
        table_reports=reports,
        source_provenance=getattr(probe, "source_provenance", None),
    )


__all__ = [
    "HeuristicAnnotation",
    "IntrospectionResult",
    "MappingFact",
    "MappingStatus",
    "TableMappingReport",
    "introspect",
]
