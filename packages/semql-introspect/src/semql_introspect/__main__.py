"""CLI entrypoint — ``python -m semql_introspect`` / ``semql-introspect``.

Connects to a database via the matching DB-API driver, runs the
introspector, and writes the emitted Python to stdout. Driver imports
happen lazily so a user introspecting DuckDB doesn't need ``psycopg``
installed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, cast

from semql.model import Dialect

from semql_introspect import introspect_to_python

_BACKENDS_BY_NAME = {b.value: b for b in Dialect}


def _connect(backend_name: str, conn_string: str) -> Any:  # noqa: ANN401
    """Lazy-import a DB-API driver and open a connection.

    Drivers are imported inline so the CLI launches even when only
    one backend's driver is installed. Connection-string syntax is
    driver-native (DSNs for psycopg, file paths for DuckDB)."""
    if backend_name == "postgres":
        # psycopg lacks py.typed; cast the module so attribute access yields Any.
        import psycopg  # type: ignore[import-not-found]

        return cast(Any, psycopg).connect(conn_string)
    if backend_name == "duckdb":
        import duckdb

        return cast(Any, duckdb).connect(conn_string)
    if backend_name == "snowflake":
        # snowflake-connector-python lacks py.typed.
        from snowflake import connector as sf_connector  # type: ignore[import-not-found]

        # Snowflake takes kwargs; expect a key=value;... string.
        kwargs: dict[str, str] = {}
        for pair in conn_string.split(";"):
            if not pair:
                continue
            k, _, v = pair.partition("=")
            kwargs[k.strip()] = v.strip()
        return cast(Any, sf_connector).connect(**kwargs)
    raise SystemExit(
        f"semql-introspect: no driver wired up for backend {backend_name!r}. "
        "Supported via the CLI: postgres, duckdb, snowflake. For other "
        "backends, call ``introspect_to_python`` from Python with a "
        "connection you opened yourself."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="semql-introspect",
        description=(
            "Generate a semql cube catalog from a live database. "
            "Reads Information Schema, applies heuristic measure / "
            "dimension inference, and emits Python."
        ),
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=sorted(_BACKENDS_BY_NAME),
        help="semql Dialect tag stamped onto every emitted cube.",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="information_schema table_schema to scan.",
    )
    parser.add_argument(
        "--conn",
        required=True,
        help=(
            "Driver-native connection string (DSN for psycopg, file "
            "path for DuckDB, key=value;... for Snowflake)."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="TABLE",
        help="Only introspect these tables (repeat for multiple).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="TABLE",
        help="Skip these tables (repeat for multiple).",
    )
    parser.add_argument(
        "--header",
        default=None,
        help="Custom docstring for the emitted module (defaults to a TODO-review reminder).",
    )
    args = parser.parse_args(argv)

    backend = _BACKENDS_BY_NAME[args.backend]
    connection = _connect(args.backend, args.conn)
    src = introspect_to_python(
        connection,
        backend=backend,
        schema=args.schema,
        include_tables=args.include,
        exclude_tables=args.exclude,
        header=args.header,
    )
    sys.stdout.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
