"""Bootstrap a semql Catalog from a live database.

Top-level entrypoints — ``introspect_catalog`` returns
``list[Cube]`` for programmatic callers, ``introspect_to_python``
returns an importable Python source string for review-then-commit
workflows. Both layer over :class:`semql_introspect.InformationSchemaProbe`
or a caller-supplied :class:`SchemaProbe` for non-ANSI dialects.
"""

from __future__ import annotations

from typing import Any

from semql.model import Cube, Dialect

from semql_introspect._emit import emit_python
from semql_introspect._introspect import (
    HeuristicAnnotation,
    IntrospectionResult,
    introspect,
)
from semql_introspect._probe import (
    ColumnInfo,
    ForeignKeyInfo,
    InformationSchemaProbe,
    SchemaProbe,
    TableInfo,
)


def introspect_catalog(
    connection: Any,  # noqa: ANN401 — any DB-API 2.0 conn
    *,
    backend: Dialect,
    schema: str,
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
) -> list[Cube]:
    """Convenience wrapper that builds the ANSI probe + runs ``introspect``.

    Returns the cube list only; for the parallel heuristic annotations
    call :func:`introspect_to_result` instead. Pass a custom
    :class:`SchemaProbe` plus :func:`introspect` directly for non-ANSI
    dialects (ClickHouse, BigQuery)."""
    return introspect_to_result(
        connection,
        backend=backend,
        schema=schema,
        include_tables=include_tables,
        exclude_tables=exclude_tables,
    ).cubes


def introspect_to_result(
    connection: Any,  # noqa: ANN401
    *,
    backend: Dialect,
    schema: str,
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
) -> IntrospectionResult:
    """Full result envelope (cubes + heuristic annotations)."""
    probe = InformationSchemaProbe(
        connection,
        schema=schema,
        include_tables=include_tables,
        exclude_tables=exclude_tables,
    )
    return introspect(probe, backend=backend)


def introspect_to_python(
    connection: Any,  # noqa: ANN401
    *,
    backend: Dialect,
    schema: str,
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
    header: str | None = None,
) -> str:
    """Return a self-contained Python module string with the inferred cubes."""
    result = introspect_to_result(
        connection,
        backend=backend,
        schema=schema,
        include_tables=include_tables,
        exclude_tables=exclude_tables,
    )
    return emit_python(result, header=header)


__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "HeuristicAnnotation",
    "InformationSchemaProbe",
    "IntrospectionResult",
    "SchemaProbe",
    "TableInfo",
    "emit_python",
    "introspect",
    "introspect_catalog",
    "introspect_to_python",
    "introspect_to_result",
]
