"""Tests for the SQL-path planner prompt (``build_sql_planner_prompt_fragment``
and the ``sql_planner_prompt`` catalog convenience).

The SQL-path fragment is the LLM-facing contract for emitting *semantic SQL*
instead of a ``SemanticQuery`` JSON. Two things matter and are pinned here:

1. The grammar contract and section shape are present (so a planner is told
   what SQL it may write).
2. The few-shot examples are real, parseable semantic SQL — they are
   generated from the catalog via ``query_to_sql``, so re-parsing each one
   with ``parse_sql_statement`` must succeed with no errors. That guarantees
   the prompt can never teach SQL the parser would reject.
"""

from __future__ import annotations

import re

import pytest
from semql import Catalog
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension
from semql.parse import parse_sql_statement
from semql_prompt import build_sql_planner_prompt_fragment, sql_planner_prompt

_SQL_BLOCK_RE = re.compile(r"```sql\n(.*?)\n\s*```", re.DOTALL)


def _orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        description="One row per order.",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at", granularities=("day", "month")),
        ],
    )


def _catalog_dict() -> dict[str, Cube]:
    return {"orders": _orders()}


# ---------------------------------------------------------------------------
# Grammar contract + section shape
# ---------------------------------------------------------------------------


def test_fragment_teaches_sql_grammar() -> None:
    rendered = build_sql_planner_prompt_fragment(_catalog_dict())
    assert "## SQL path" in rendered
    assert "semantic SQL" in rendered
    # Key grammar affordances the parser supports are described.
    assert "COUNT(*)" in rendered
    assert "BETWEEN" in rendered
    assert "DATE_TRUNC" in rendered
    assert "COMPARE prior_period" in rendered
    # The catalog block and raw-fallback rule are still present.
    assert "## SEMANTIC CATALOG" in rendered
    assert "`orders.revenue`" in rendered
    assert "raw SQL" in rendered


def test_fragment_omits_json_spec_contract() -> None:
    """The SQL path must not tell the planner to emit a ``SemanticQuery``
    JSON — that's the other path."""
    rendered = build_sql_planner_prompt_fragment(_catalog_dict())
    assert "## Semantic path" not in rendered
    assert "Emit a `SemanticQuery`" not in rendered


# ---------------------------------------------------------------------------
# Few-shot examples are real, parseable semantic SQL
# ---------------------------------------------------------------------------


def test_examples_block_present_and_parseable() -> None:
    catalog = _catalog_dict()
    rendered = build_sql_planner_prompt_fragment(catalog)
    assert "## SQL examples" in rendered
    blocks = [m.strip() for m in _SQL_BLOCK_RE.findall(rendered)]
    assert blocks, "expected at least one ```sql example block"
    for sql in blocks:
        decision = parse_sql_statement(sql, catalog, strict=True)
        assert decision.parse_errors == (), (sql, decision.parse_errors)


def test_examples_demonstrate_grouping_and_time_bucket() -> None:
    """With a time dimension available, the examples show both an aggregation
    and a ``DATE_TRUNC`` time bucket."""
    rendered = build_sql_planner_prompt_fragment(_catalog_dict())
    blocks = _SQL_BLOCK_RE.findall(rendered)
    joined = "\n".join(blocks)
    assert "GROUP BY" in joined
    assert "DATE_TRUNC(" in joined


def test_include_examples_false_omits_block() -> None:
    rendered = build_sql_planner_prompt_fragment(_catalog_dict(), include_examples=False)
    assert "## SQL examples" not in rendered


def test_no_examples_when_no_demonstrable_cube() -> None:
    """A cube with no measure/dimension pair yields no example block (the
    fragment still renders its grammar and catalog)."""
    dims_only = Cube(
        name="lookup",
        dialect=Dialect.POSTGRES,
        table="lookup",
        alias="l",
        dimensions=[Dimension(name="code", sql="{l}.code", type="string")],
    )
    rendered = build_sql_planner_prompt_fragment({"lookup": dims_only})
    assert "## SQL path" in rendered
    assert "## SQL examples" not in rendered


# ---------------------------------------------------------------------------
# Catalog-level convenience
# ---------------------------------------------------------------------------


def test_sql_planner_prompt_convenience() -> None:
    catalog = Catalog([_orders()])
    rendered = sql_planner_prompt(catalog)
    assert "## SQL path" in rendered
    blocks = _SQL_BLOCK_RE.findall(rendered)
    assert blocks
    for sql in blocks:
        decision = parse_sql_statement(sql.strip(), catalog.as_dict(), strict=True)
        assert decision.parse_errors == (), (sql, decision.parse_errors)


@pytest.mark.parametrize("include_examples", [True, False])
def test_sql_planner_prompt_examples_toggle(include_examples: bool) -> None:
    catalog = Catalog([_orders()])
    rendered = sql_planner_prompt(catalog, include_examples=include_examples)
    assert ("## SQL examples" in rendered) is include_examples
