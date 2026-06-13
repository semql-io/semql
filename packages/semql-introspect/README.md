# semql-introspect

Bootstrap a [semql](../semql) `Catalog` from a live database.

Reads Information Schema, emits Python `Cube` stubs with heuristic
measure / dimension / time-dimension inference and foreign-key derived
joins. Designed for greenfield adoption — a team with 200 tables can
generate the mechanical 80% of a catalog in seconds, then hand-edit
the heuristic guesses.

Use this for *cold-start* scaffolding. For ongoing drift detection
on a catalog you already have, see
[`semql-validate-db`](../semql-validate-db) — it probes a live
database against an authored catalog and surfaces missing tables /
columns / join predicates that the compiler can't see at build time.

## Install

```sh
pip install semql-introspect
```

## Usage

```python
import duckdb
from semql.model import Dialect
from semql_introspect import introspect_to_python

con = duckdb.connect("warehouse.db")
print(introspect_to_python(con, dialect=Dialect.DUCKDB, schema="main"))
```

Or via CLI:

```sh
semql-introspect --backend duckdb --schema main --conn "warehouse.db"
```

## Heuristics

- Numeric columns named `amount` / `price` / `revenue` / `cost` / `total`
  / `value` / `qty` / `quantity` / `count` → `Measure(agg="sum")`.
- Columns ending in `_id` → `Measure(agg="count_distinct")` (the table's
  cardinality is usually interesting).
- `date` / `timestamp` columns → `TimeDimension`.
- Foreign keys → `Join(relationship="many_to_one")` plus the foreign-side
  `Dimension(foreign_key=...)`.
- Everything else → `Dimension` typed by the column's SQL type.

Heuristic guesses get a `# TODO: review` comment so the diff makes the
inference choices reviewable.
