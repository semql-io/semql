# Spec: SQL-like Statement Parser for SemQL

## Overview

Add a new **Parser** role to the prompt pipeline that converts SQL-like statements from an LLM chat agent into a `SemanticQuery`. This enables LLM agents to write queries in familiar SQL syntax while still benefiting from SemQL's semantic layer (catalog resolution, authorization, row-level scope).

## Architecture

```
User's SQL-like statement
           │
           ▼
┌─────────────────────┐
│      PARSER         │  ← NEW: converts SQL → SemanticQuery
│  (SQL → Semantic)   │
└─────────────────────┘
           │
           ▼
    SemanticQuery
           │
           ▼
┌─────────────────────┐
│   EXISTING PIPELINE  │
│ Router → Generator  │
│ Compile → Present   │
└─────────────────────┘
```

## Input Type

```python
# New type in plan.py
class SQLParseInput(BaseModel):
    statement: str           # The SQL-like statement
    catalog: Catalog | None # Optional catalog for validation (can be deferred)
    strict: bool = True     # Fail on unknown references
```

## Output Type

```python
# New type in plan.py
class ParserDecision(BaseModel):
    query: SemanticQuery           # The parsed semantic query
    original_statement: str      # Preserved for reference
    parse_warnings: list[str]    # Non-fatal issues (unknown fields, etc.)
    parse_errors: list[str]      # Fatal errors (unknown cube, syntax)
    resolved_references: dict[str, str]  # "orders.region" → "orders.region"
```

## Supported SQL Syntax

| SQL Construct | SemanticQuery Field |
|---------------|---------------------|
| `SELECT dim1, dim2, SUM(measure)` | `dimensions`, `measures` |
| `FROM cube_name` | Inferred from query refs |
| `WHERE dim = 'value'` | `filters` (implicit AND) |
| `WHERE dim IN ('a', 'b')` | `filters` with `in` operator |
| `WHERE dim > 10` | `filters` with `gt` operator |
| `WHERE (a = 1 OR b = 2)` | `where` (BoolExpr tree) |
| `GROUP BY dim1, dim2` | `dimensions` (deduplicated) |
| `HAVING SUM(measure) > 100` | `having` filters |
| `ORDER BY dim DESC, measure ASC` | `order` |
| `LIMIT 100` | `limit` |
| `OFFSET 50` | `offset` |
| `BETWEEN '2024-01-01' AND '2024-12-31'` | `time_dimension` |
| `COMPARE TO prior_period` | `compare` |

## Operator Mapping

| SQL Operator | Filter Operator |
|--------------|-----------------|
| `=` | `eq` |
| `!=`, `<>` | `neq` |
| `IN (...)` | `in` |
| `NOT IN (...)` | `not_in` |
| `>`, `>=`, `<`, `<=` | `gt`, `gte`, `lt`, `lte` |
| `LIKE '%foo%'` | `contains` |
| `IS NULL` | `is_null` |
| `IS NOT NULL` | `not_null` |

## Parser Implementation

Pure function (no I/O, no globals):

```python
# packages/semql/src/semql/parse.py

def parse_sql_statement(
    statement: str,
    catalog: Catalog | None = None,
    *,
    strict: bool = True,
) -> ParserDecision:
    """
    Parse a SQL-like statement into a SemanticQuery.

    Pure function - no I/O, no external calls.
    Dimension value resolution deferred to lookups.py.
    """
```

**Parsing stages:**

1. **Tokenization**: Split SQL into tokens (keywords, identifiers, literals, operators)
2. **AST Construction**: Build a simple AST (SELECT, FROM, WHERE, GROUP BY, etc.)
3. **Reference Resolution** (if catalog provided): Validate cubes and fields exist
4. **SemanticQuery Construction**: Build the target type
5. **Error Collection**: Collect all errors before failing (better UX)

## Error Handling

- **Unknown cube**: Fatal error → `parse_errors`
- **Unknown field**: Fatal in strict mode; warning in lenient
- **Invalid operator**: Syntax error
- **Unsupported syntax**: Clear error message listing what's supported

## Pipeline Integration

Parser becomes the **first role** in the four-role pipeline:

```python
# prompt.py
def build_parser_prompt_fragment(catalog: Catalog) -> str:
    """Render the parser prompt fragment with SQL syntax reference."""

def to_parser_function(catalog: Catalog) -> dict[str, Any]:
    """Returns {type: "function", function: {name, description, parameters}}"""
```

## File Changes

| File | Change |
|------|--------|
| `plan.py` | Add `SQLParseInput`, `ParserDecision` |
| `parse.py` | New file: parser implementation |
| `prompt.py` | Add `build_parser_prompt_fragment`, `to_parser_function` |
| `tests/test_parse.py` | New test file |

## Multi-cube JOINs — the Malloy-style contract

A `JOIN` in this SQL is a **semantic directive, not a row-level join**.
It declares which cubes a query touches; the catalog is the single
source of truth for how they relate.

| SQL | Meaning |
|-----|---------|
| `FROM orders o JOIN customers c ON ...` | "this query touches `orders` and `customers`" — participation only |
| the `ON o.x = c.y` predicate | **ignored** — the compiler derives the join from `Cube.joins` |
| `o.revenue`, `c.name` | resolved to `orders.revenue`, `customers.name` via the table aliases |
| an unqualified column | resolved to the one cube that declares it; rejected if two do |

**Resolution** is centralised in `ResolveCtx` (table-alias → cube),
which qualifies every column to a `cube.field` ref — the only form the
compiler accepts.

**Refusal**: if the catalog has no join path between the named cubes,
the parser refuses (`_cubes_joinable`) rather than inventing a join
from the SQL `ON`. The `ON` is never used to create a relationship.

**Why ignore `ON` — and the caveat.** A flat `a JOIN b` executed
literally *fans out*: summing a measure across a one-to-many join
multiplies it by the match count. Semantic layers (Malloy, Cube,
LookML) avoid this by treating the join as a modelled relationship and
aggregating safely — not by trusting the query's literal join. SemQL
follows that model: the literal SQL semantics of a flat JOIN are **not
preserved**; the catalog relationship plus the compiler's fan-out
guard (`_check_fan_out`) define the meaning. The literal-fidelity
equivalent is aggregate-then-join on the conformed key (two grouped
subqueries joined on the shared dimension) — which is also the shape a
cross-backend `FederatedPlan` emits, so the same unified SQL drives
single- and multi-backend compilation unchanged.

## Acceptance Criteria

1. Parser correctly converts basic SELECT with dimensions, measures, filters
2. Parser handles WHERE with AND/OR via BoolExpr
3. Parser handles GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET
4. Parser handles time windows (BETWEEN dates)
5. Parser validates cube/field existence against catalog (strict mode)
6. Clear error messages for unsupported SQL constructs
7. Parser is a pure function (no I/O)
8. Parser integrates into the prompt pipeline
9. Multi-cube JOINs resolve via table aliases; the `ON` clause is a
   participation directive (catalog is truth); non-joinable cubes are refused
