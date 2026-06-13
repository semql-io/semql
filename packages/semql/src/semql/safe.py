"""Belt-and-braces guard on emitted SQL.

The compiler emits a single ``SELECT`` by construction. ``is_read_only_statement``
runs a post-hoc check on ``CompiledQuery.sql`` (or any other SQL string)
that a downstream caller wants to gate before it hits the database.
The guard is paranoid by design — multi-statement payloads,
malformed SQL, and any non-SELECT shape (INSERT / UPDATE / DELETE /
CREATE / DROP / ALTER / TRUNCATE / GRANT) all fail.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


def is_read_only_statement(sql: str, *, dialect: str = "postgres") -> bool:
    """Return True iff ``sql`` parses cleanly as exactly one ``SELECT``
    statement under the given ``dialect``.

    Empty input, malformed SQL, multi-statement payloads, and any
    non-SELECT root expression (``Insert`` / ``Update`` / ``Delete`` /
    DDL / privilege statements) return False. ``SELECT`` with CTEs
    (``WITH ... SELECT ...``) and ``UNION`` are accepted — they're
    still a single SELECT root.
    """
    if not sql or not sql.strip():
        return False

    try:
        parsed = sqlglot.parse(sql, dialect=dialect)
    except ParseError:
        return False

    # ``parse`` returns a list of top-level statements (separated by ``;``).
    # Filter out trailing empties (``"SELECT 1;"`` parses as
    # ``[Select, None]``); anything more than one real statement fails.
    statements = [s for s in parsed if s is not None]
    if len(statements) != 1:
        return False

    root = statements[0]
    # ``Select`` covers plain SELECT; ``Union`` (and friends) are
    # composed of SELECTs. Anything else — Insert, Update, Delete,
    # Create, Drop, Alter, Truncate, Grant, ... — is rejected.
    return isinstance(root, exp.Select | exp.Union | exp.Subquery)


__all__ = ["is_read_only_statement"]
