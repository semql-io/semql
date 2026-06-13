# semql-validate-db

Pre-deploy drift checker for [`semql`](../semql) catalogs. Runs cheap
probe queries against a live database and surfaces the class of bugs
the compiler can't see — missing tables, dropped columns, broken join
predicates, base-predicate drift.

`semql` is intentionally pure (PHILOSOPHY: "the compiler has no I/O").
That keeps the compiler simple, but it also means a catalog can pass
every compile-time check and still blow up at query time because
upstream renamed a column. `semql-validate-db` is the out-of-band
gate that catches it.

Use this for *ongoing* drift detection on a catalog you already
authored. For greenfield scaffolding from a database's existing
schema, see [`semql-introspect`](../semql-introspect) — it generates
`Cube` stubs from the information schema, which is the *opposite*
direction: introspect goes DB → catalog, validate-db goes catalog →
DB.

## Install

```sh
pip install semql-validate-db
```

The package is driver-agnostic. Bring your own DB-API 2.0 connection:

```sh
pip install psycopg              # Postgres
pip install clickhouse-connect   # ClickHouse
pip install duckdb               # DuckDB
```

## Quick start

```python
import duckdb
from semql import Dialect, Catalog, Cube, Dimension, Measure, TimeDimension
from semql_validate_db import validate_against_db

orders = Cube(
    name="orders",
    dialect=Dialect.DUCKDB,
    table="orders",
    alias="o",
    measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
    dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
)
catalog = Catalog([orders])

conn = duckdb.connect(":memory:")
conn.execute(
    "CREATE TABLE orders (amount DOUBLE, region TEXT, created_at TIMESTAMP)"
)

errors = validate_against_db(catalog, connection=conn)
for e in errors:
    print(f"{e.code}: {e.cube}.{e.field or ''} — {e.message}")
```

A clean run returns an empty list. Drift (a missing column, a renamed
table) yields one `DbValidationError` per finding so a single run
gives the full picture instead of bailing on the first failure.

## What it catches

- `missing_table` — `cube.table` doesn't exist or the connection's
  role can't see it.
- `missing_column` — a measure / dimension / time-dimension SQL
  fragment references a column that no longer exists.
- `base_predicate_invalid` — `cube.base_predicate` doesn't execute.
- `join_predicate_invalid` — a `Join.on` predicate references columns
  that aren't there, or compares incompatible types.

(A `required_filters` entry that names no real dimension is now rejected
at catalog construction — it can't reach this pre-deploy stage, so
there's no DB-level check for it.)

## What it doesn't catch

- Semantic drift (a column exists but means something different now).
  Schema is necessary, not sufficient.
- Cross-table referential integrity. The probes are `LIMIT 0`; they
  parse, they don't sample.
- Dialect-specific feature drift (a function got dialect-renamed).
  Use the compiler's snapshot tests for that.

## Why `LIMIT 0`?

Every probe runs `SELECT … LIMIT 0`. The query planner type-checks
identifiers and predicates but does no row work, so the cost is
microseconds per probe — fine for a per-cube fan-out in CI. The
trade-off is that purely runtime drift (e.g. an `enum` value that
got dropped from a check constraint) won't surface here.

## CLI

The package is library-first; a CLI lives in callers' deploy scripts
where the connection / DSN / role are already known.

## Status

Phase A: probe-by-fragment shape. Drift findings are accurate;
performance is "fine for CI, not for runtime gates."
