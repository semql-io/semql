"""Schema probes — read tables / columns / FKs from a live database.

The probe abstracts the dialect-specific information_schema layout:
PG / DuckDB / Snowflake mostly share the ANSI ``information_schema``
shape; ClickHouse uses ``system.columns`` + ``system.tables``; BigQuery
adds a ``project_id.dataset`` prefix. v1 ships
:class:`InformationSchemaProbe` which covers the ANSI dialects;
non-ANSI dialects can implement :class:`SchemaProbe` directly without
touching the rest of the package.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _assert_safe_identifier(value: str, label: str) -> None:
    """Raise ValueError if ``value`` is not a conservative SQL identifier.

    Prevents SQL injection via information_schema string comparisons
    (SEMQL-DISC-PROBE-SQLI-001). We validate rather than parameterise
    because the DB-API parameterisation format varies by backend.
    """
    if not _SAFE_IDENT_RE.match(value):
        raise ValueError(
            f"Unsafe {label}: {value!r} contains characters that are not "
            "allowed in an identifier. Only letters, digits, underscores, "
            "and $ (after the first character) are permitted."
        )


@dataclass(frozen=True)
class ColumnInfo:
    """One column on a table as returned by a probe.

    ``data_type`` is the raw SQL type string the database returned
    (``"integer"``, ``"timestamp without time zone"``, etc.). Heuristics
    walk this to decide field kind / dimension type.
    """

    name: str
    data_type: str
    is_nullable: bool


@dataclass(frozen=True)
class ForeignKeyInfo:
    """One FK edge — ``from_table.from_column → to_table.to_column``."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class TableInfo:
    """One table the catalog will materialise as a ``Cube``."""

    name: str
    columns: tuple[ColumnInfo, ...]


class SchemaProbe(Protocol):
    """Read-only dialect adapter for introspection.

    Implementations should return tables / columns / foreign keys
    scoped to whatever schema the caller asked about — the orchestrator
    treats the result as authoritative and doesn't filter again."""

    def list_tables(self) -> list[TableInfo]: ...

    def list_foreign_keys(self) -> list[ForeignKeyInfo]: ...

    def list_primary_keys(self) -> dict[str, str]:
        """``{table_name: primary_key_column}``.

        Probes that can't recover primary keys (some warehouses don't
        ship constraints) should return ``{}`` — heuristics will fall
        back to "first column named ``id``" detection."""
        ...


class InformationSchemaProbe:
    """ANSI ``information_schema`` probe.

    Works against any database that ships the standard
    ``information_schema.columns`` / ``information_schema.table_constraints``
    layout — Postgres, DuckDB, Snowflake (mostly), and SQL Server. The
    schema argument scopes ``table_schema``; pass it explicitly so
    catalog dumps don't accidentally span system schemas.
    """

    def __init__(
        self,
        connection: Any,  # noqa: ANN401 — any DB-API 2.0 conn
        *,
        schema: str,
        include_tables: Iterable[str] | None = None,
        exclude_tables: Iterable[str] | None = None,
    ) -> None:
        _assert_safe_identifier(schema, "schema")
        self._conn = connection
        self._schema = schema
        self._include = set(include_tables) if include_tables else None
        self._exclude = set(exclude_tables or ())

    def list_tables(self) -> list[TableInfo]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{self._schema}' "
                "AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )
            table_names = [row[0] for row in cur.fetchall()]
        finally:
            cur.close()
        if self._include is not None:
            table_names = [t for t in table_names if t in self._include]
        table_names = [t for t in table_names if t not in self._exclude]

        out: list[TableInfo] = []
        for tbl in table_names:
            out.append(TableInfo(name=tbl, columns=self._columns_for(tbl)))
        return out

    def _columns_for(self, table: str) -> tuple[ColumnInfo, ...]:
        _assert_safe_identifier(table, "table")
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                f"WHERE table_schema = '{self._schema}' "
                f"AND table_name = '{table}' "
                "ORDER BY ordinal_position"
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        return tuple(
            ColumnInfo(
                name=row[0],
                data_type=str(row[1]),
                is_nullable=(str(row[2]).upper() == "YES"),
            )
            for row in rows
        )

    def list_foreign_keys(self) -> list[ForeignKeyInfo]:
        # ``information_schema.constraint_column_usage`` differs per
        # backend: Postgres returns the referenced (target) table for
        # FK constraints, DuckDB returns the source table. The more
        # portable shape is ``referential_constraints`` → the PK
        # constraint on the referenced side, joined via ``key_column_usage``
        # on both ends with matching ordinal positions for composite
        # keys. Works on PG / DuckDB / Snowflake unchanged.
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT kcu_from.table_name AS from_table, "
                "       kcu_from.column_name AS from_column, "
                "       kcu_to.table_name AS to_table, "
                "       kcu_to.column_name AS to_column "
                "FROM information_schema.referential_constraints rc "
                "JOIN information_schema.key_column_usage kcu_from "
                "  ON rc.constraint_name = kcu_from.constraint_name "
                " AND rc.constraint_schema = kcu_from.constraint_schema "
                "JOIN information_schema.key_column_usage kcu_to "
                "  ON rc.unique_constraint_name = kcu_to.constraint_name "
                " AND rc.unique_constraint_schema = kcu_to.constraint_schema "
                " AND kcu_from.ordinal_position = kcu_to.ordinal_position "
                f"WHERE rc.constraint_schema = '{self._schema}' "
                "ORDER BY kcu_from.table_name, kcu_from.ordinal_position"
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        return [
            ForeignKeyInfo(
                from_table=row[0],
                from_column=row[1],
                to_table=row[2],
                to_column=row[3],
            )
            for row in rows
        ]

    def list_primary_keys(self) -> dict[str, str]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT kcu.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                " AND tc.table_schema = kcu.table_schema "
                "WHERE tc.constraint_type = 'PRIMARY KEY' "
                f"  AND tc.table_schema = '{self._schema}' "
                "ORDER BY kcu.table_name, kcu.ordinal_position"
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        # If a PK is composite, the first column wins — the catalog
        # model only supports a single ``primary_key`` per cube. A
        # ``# TODO: review`` will surface composite-key tables anyway.
        out: dict[str, str] = {}
        for row in rows:
            out.setdefault(str(row[0]), str(row[1]))
        return out


__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "InformationSchemaProbe",
    "SchemaProbe",
    "TableInfo",
]
