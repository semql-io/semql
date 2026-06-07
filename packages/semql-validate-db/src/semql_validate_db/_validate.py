"""Catalogue ↔ database drift checker.

The compiler is intentionally pure (no I/O). This package picks up the
class of bugs the compiler can't see: an upstream column rename, a
dropped table, a join predicate that suddenly compares incompatible
types. Suitable as a pre-deploy CI gate.

Strategy: for each cube, run cheap ``LIMIT 0`` probes against the
target database — one for the table itself, one per measure /
dimension SQL fragment, one for ``base_predicate``. Any SQL error
gets translated into a ``DbValidationError`` naming the cube and the
field that broke; the function collects all of them rather than
short-circuiting so a single run surfaces the full drift picture.

Transport: accepts any DB-API 2.0 connection. Callers bring their
own driver (psycopg, clickhouse-connect, duckdb) so this package
stays driver-agnostic.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from semql.catalog import Catalog
from semql.model import Cube, Join

DbValidationCode = Literal[
    "missing_table",
    "missing_column",
    "base_predicate_invalid",
    "join_predicate_invalid",
    "required_filter_dimension_missing",
]


@dataclass(frozen=True)
class DbValidationError:
    """One drift finding.

    ``cube`` always names the cube the probe ran against. ``field`` is
    the measure / dimension / join target the probe was checking, or
    ``None`` for cube-level findings (``missing_table``,
    ``base_predicate_invalid``). ``detail`` carries the database's
    own error message so the caller can route the message at the user
    without re-parsing.
    """

    code: DbValidationCode
    cube: str
    field: str | None
    message: str
    detail: str | None = None


class _Cursor(Protocol):
    # DB-API 2.0 cursors return ``self``-like objects from execute, but the
    # exact shape varies by driver. We only care about side-effects + close,
    # so the return type is intentionally untyped via ``object``.
    def execute(self, sql: str, /) -> object: ...
    def close(self) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _resolve_placeholders(sql: str, lookup: dict[str, str]) -> str:
    """Substitute ``{key}`` placeholders the same way the compiler does.

    Mirrors ``semql.compile._resolve_sql``'s behaviour: ``{alias}`` and
    ``{cube_name}`` resolve to the alias; ``{ctx_key}`` resolves from
    the caller-supplied ``context``. Unknown placeholders are left
    in place — the resulting query will fail at the database and that
    failure is the signal we want.
    """

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return lookup.get(key, m.group(0))

    return _PLACEHOLDER_RE.sub(_repl, sql)


def _probe(
    connection: _Connection,
    sql: str,
) -> tuple[bool, str | None]:
    """Run a SQL probe under the connection's cursor.

    Returns ``(True, None)`` on success and ``(False, error_message)``
    on any failure. ``LIMIT 0`` is the caller's responsibility — this
    function just runs whatever it's handed and reports the outcome.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
    except Exception as exc:  # noqa: BLE001 — any DB-API error counts as drift
        return False, str(exc)
    finally:
        # Driver cleanup quirks vary; we don't want a flaky close() to mask
        # a real drift finding so swallow whatever the driver throws here.
        with contextlib.suppress(Exception):
            cursor.close()
    return True, None


def _cube_lookup(cube: Cube, context: dict[str, str]) -> dict[str, str]:
    """The substitution table the cube's SQL fragments resolve against."""
    return {
        cube.alias: cube.alias,
        cube.name: cube.alias,
        **context,
    }


def _validate_required_filters(cube: Cube) -> list[DbValidationError]:
    """Static check: every ``required_filters`` entry must name a real
    dimension on the cube.

    This is technically catalogue-internal (no DB query needed), but
    pre-deploy is the right surface for it — surfacing here means a CI
    gate catches the typo even when the static-validation pass missed."""
    dim_names = {d.name for d in cube.dimensions}
    out: list[DbValidationError] = []
    for req in cube.required_filters:
        if req not in dim_names:
            out.append(
                DbValidationError(
                    code="required_filter_dimension_missing",
                    cube=cube.name,
                    field=req,
                    message=(
                        f"Cube {cube.name!r} declares required_filters=[{req!r}] "
                        f"but has no dimension by that name. Known: {sorted(dim_names)}."
                    ),
                )
            )
    return out


def _validate_cube(
    cube: Cube,
    connection: _Connection,
    context: dict[str, str],
) -> list[DbValidationError]:
    """Probe a single cube. Order: table → fragments → base_predicate.

    Stops on ``missing_table`` (every subsequent probe would just
    repeat the same error). All other findings accumulate so a single
    run reports the whole picture for the cube."""
    errors: list[DbValidationError] = []
    lookup = _cube_lookup(cube, context)
    table = _resolve_placeholders(cube.table, lookup)
    alias = cube.alias

    ok, detail = _probe(connection, f"SELECT * FROM {table} AS {alias} LIMIT 0")
    if not ok:
        errors.append(
            DbValidationError(
                code="missing_table",
                cube=cube.name,
                field=None,
                message=(
                    f"Cube {cube.name!r}: table {table!r} did not respond to a "
                    "trivial SELECT — likely missing, renamed, or "
                    "inaccessible to the connection's role."
                ),
                detail=detail,
            )
        )
        return errors

    for dim in cube.dimensions:
        fragment = _resolve_placeholders(dim.sql, lookup)
        ok, detail = _probe(connection, f"SELECT {fragment} FROM {table} AS {alias} LIMIT 0")
        if not ok:
            errors.append(
                DbValidationError(
                    code="missing_column",
                    cube=cube.name,
                    field=dim.name,
                    message=(
                        f"Cube {cube.name!r}, dimension {dim.name!r}: SQL "
                        f"fragment {dim.sql!r} did not execute against the table."
                    ),
                    detail=detail,
                )
            )

    for td in cube.time_dimensions:
        fragment = _resolve_placeholders(td.sql, lookup)
        ok, detail = _probe(connection, f"SELECT {fragment} FROM {table} AS {alias} LIMIT 0")
        if not ok:
            errors.append(
                DbValidationError(
                    code="missing_column",
                    cube=cube.name,
                    field=td.name,
                    message=(
                        f"Cube {cube.name!r}, time_dimension {td.name!r}: SQL "
                        f"fragment {td.sql!r} did not execute against the table."
                    ),
                    detail=detail,
                )
            )

    for m in cube.measures:
        # ratio measures reference *other* measures by name — there's
        # no raw SQL fragment to probe. The recursive measure references
        # already get validated when we probe their underlying SQL.
        if m.agg == "ratio":
            continue
        # ``count`` over ``*`` is universal — skip; the table probe
        # already covered it.
        if m.agg == "count" and m.sql.strip() == "*":
            continue
        fragment = _resolve_placeholders(m.sql, lookup)
        ok, detail = _probe(connection, f"SELECT {fragment} FROM {table} AS {alias} LIMIT 0")
        if not ok:
            errors.append(
                DbValidationError(
                    code="missing_column",
                    cube=cube.name,
                    field=m.name,
                    message=(
                        f"Cube {cube.name!r}, measure {m.name!r}: SQL "
                        f"fragment {m.sql!r} did not execute against the table."
                    ),
                    detail=detail,
                )
            )

    if cube.base_predicate:
        predicate = _resolve_placeholders(cube.base_predicate, lookup)
        ok, detail = _probe(
            connection,
            f"SELECT 1 FROM {table} AS {alias} WHERE {predicate} LIMIT 0",
        )
        if not ok:
            errors.append(
                DbValidationError(
                    code="base_predicate_invalid",
                    cube=cube.name,
                    field=None,
                    message=(
                        f"Cube {cube.name!r}: base_predicate {cube.base_predicate!r} "
                        "did not execute against the table."
                    ),
                    detail=detail,
                )
            )

    return errors


def _validate_join(
    source: Cube,
    join: Join,
    target: Cube,
    connection: _Connection,
    context: dict[str, str],
) -> list[DbValidationError]:
    """Probe ``source LEFT JOIN target ON <join.on> LIMIT 0``.

    A successful run means the predicate parses against the actual
    column types on both sides. Type-mismatch and unknown-column errors
    surface here under ``join_predicate_invalid``."""
    src_lookup = _cube_lookup(source, context)
    tgt_lookup = _cube_lookup(target, context)
    src_table = _resolve_placeholders(source.table, src_lookup)
    tgt_table = _resolve_placeholders(target.table, tgt_lookup)
    on_clause = _resolve_placeholders(join.on, {**src_lookup, **tgt_lookup})

    sql = (
        f"SELECT 1 FROM {src_table} AS {source.alias} "
        f"LEFT JOIN {tgt_table} AS {target.alias} ON {on_clause} LIMIT 0"
    )
    ok, detail = _probe(connection, sql)
    if ok:
        return []
    return [
        DbValidationError(
            code="join_predicate_invalid",
            cube=source.name,
            field=join.to,
            message=(
                f"Cube {source.name!r} → {join.to!r}: join predicate "
                f"{join.on!r} did not execute against the joined tables."
            ),
            detail=detail,
        )
    ]


def validate_against_db(
    catalog: Catalog,
    *,
    connection: _Connection,
    context: dict[str, str] | None = None,
) -> list[DbValidationError]:
    """Validate every cube and join in ``catalog`` against a live DB.

    Returns the full list of findings (empty on success). Each finding
    names the cube / field that drifted and carries the database's
    own error message in ``detail`` for routing into a CI log.

    ``context`` substitutes ``{key}`` placeholders inside catalog SQL
    (e.g. ``{"schema": "analytics"}`` for cubes whose ``table`` is
    ``"{schema}.orders"``).

    META reflection cubes are skipped — they don't live in the
    physical database.
    """
    ctx = context or {}
    errors: list[DbValidationError] = []
    real_cubes: list[Cube] = []
    for cube in catalog:
        if cube.backend.value == "meta":
            continue
        real_cubes.append(cube)
        errors.extend(_validate_required_filters(cube))
        errors.extend(_validate_cube(cube, connection, ctx))

    by_name = {c.name: c for c in real_cubes}
    for cube in real_cubes:
        for join in cube.joins:
            target = by_name.get(join.to)
            if target is None:
                # Catalog construction already rejects this, but stay
                # defensive — the caller could have edited the catalog.
                continue
            errors.extend(_validate_join(cube, join, target, connection, ctx))

    return errors


__all__ = [
    "DbValidationCode",
    "DbValidationError",
    "validate_against_db",
]
