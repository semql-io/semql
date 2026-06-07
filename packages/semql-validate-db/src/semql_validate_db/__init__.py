"""Pre-deploy drift checker for semql catalogues.

``validate_against_db(catalog, connection=...)`` runs cheap probe
queries against a live database and returns a list of
``DbValidationError`` findings — one per cube / field / join that
broke. Use it as a CI gate before promoting a catalogue change.

This package is driver-agnostic: the ``connection`` argument is any
DB-API 2.0 connection. Wire your own ``psycopg.connect`` /
``clickhouse_connect.get_client`` / ``duckdb.connect`` and pass it in.
"""

from __future__ import annotations

from semql_validate_db._validate import (
    DbValidationCode,
    DbValidationError,
    validate_against_db,
)

__all__ = ["DbValidationCode", "DbValidationError", "validate_against_db"]
