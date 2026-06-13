"""Tests for percentile aggregations (F6).

``median`` / ``p75`` / ``p90`` / ``p95`` extend ``AggLiteral`` and
route through ``DialectStrategy.emit_percentile`` so each dialect can
emit its native shape:

- Postgres / DuckDB / Snowflake: ``PERCENTILE_CONT(q) WITHIN GROUP (ORDER BY ...)``
- ClickHouse: ``quantile(q)(...)``
- BigQuery: ``APPROX_QUANTILES(..., 100)[OFFSET(p)]``

Federation distributive mode refuses percentile measures (the same
``non_distributive_aggregation:<agg>`` reason that gates
``min`` / ``max`` / ``count_distinct``); raw_rows mode emits a
``PERCENTILE_CONT`` at the merge step using DuckDB dialect.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.errors import FederationError
from semql.federate import compile_federated_query


def _cube(dialect: Dialect, name: str = "orders") -> Cube:
    return Cube(
        name=name,
        dialect=dialect,
        table="orders",
        alias="o",
        measures=[
            Measure(name="amount_median", sql="{o}.amount", agg="median"),
            Measure(name="amount_p75", sql="{o}.amount", agg="p75"),
            Measure(name="amount_p90", sql="{o}.amount", agg="p90"),
            Measure(name="amount_p95", sql="{o}.amount", agg="p95"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


# ---------------------------------------------------------------------------
# Postgres / DuckDB / Snowflake — ANSI ``PERCENTILE_CONT`` shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dialect",
    [Dialect.POSTGRES, Dialect.DUCKDB, Dialect.SNOWFLAKE],
)
def test_percentile_emits_ansi_within_group(dialect: Dialect) -> None:
    """The ANSI shape compiles via sqlglot's ``WithinGroup`` node.
    Postgres + Snowflake render the textbook form ``PERCENTILE_CONT(q)
    WITHIN GROUP (ORDER BY ...)``; DuckDB renders its accepted
    shorthand ``PERCENTILE_CONT(q ORDER BY ...)`` — both compile
    against the same node and both are valid in their dialect."""
    cat = Catalog([_cube(dialect)])
    q = SemanticQuery(measures=["orders.amount_median"], dimensions=["orders.region"])
    out = cat.compile(q)
    upper = out.sql.upper()
    assert "PERCENTILE_CONT" in upper
    assert "0.5" in out.sql
    assert "ORDER BY" in upper
    # Either the ANSI ``WITHIN GROUP (ORDER BY ...)`` (PG / SF) or the
    # DuckDB shorthand ``PERCENTILE_CONT(q ORDER BY ...)``.
    assert "WITHIN GROUP" in upper or "PERCENTILE_CONT(0.5 ORDER BY" in upper


def test_percentile_q_values_per_agg() -> None:
    """Each agg literal maps to the right quantile (0.5/0.75/0.9/0.95)."""
    cat = Catalog([_cube(Dialect.DUCKDB)])
    for measure, q_str in [
        ("amount_median", "0.5"),
        ("amount_p75", "0.75"),
        ("amount_p90", "0.9"),
        ("amount_p95", "0.95"),
    ]:
        out = cat.compile(
            SemanticQuery(measures=[f"orders.{measure}"], dimensions=["orders.region"])
        )
        assert q_str in out.sql, f"{measure} should emit q={q_str}, got: {out.sql}"


# ---------------------------------------------------------------------------
# ClickHouse — curried ``quantile(q)(...)`` shape
# ---------------------------------------------------------------------------


def test_percentile_on_clickhouse_uses_quantile_curried_form() -> None:
    cat = Catalog([_cube(Dialect.CLICKHOUSE)])
    q = SemanticQuery(measures=["orders.amount_median"], dimensions=["orders.region"])
    out = cat.compile(q)
    # ClickHouse: quantile(0.5)(expr)
    assert "quantile(0.5)" in out.sql
    # Should not contain the ANSI form.
    assert "WITHIN GROUP" not in out.sql.upper()


# ---------------------------------------------------------------------------
# BigQuery — APPROX_QUANTILES with bracket offset
# ---------------------------------------------------------------------------


def test_percentile_on_bigquery_uses_approx_quantiles() -> None:
    cat = Catalog([_cube(Dialect.BIGQUERY)])
    q = SemanticQuery(measures=["orders.amount_p90"], dimensions=["orders.region"])
    out = cat.compile(q)
    upper = out.sql.upper()
    assert "APPROX_QUANTILES" in upper
    # The offset for p90 is round(0.9 * 100) = 90.
    assert "OFFSET" in upper
    assert "90" in out.sql


# ---------------------------------------------------------------------------
# Federation
# ---------------------------------------------------------------------------


def _federated_catalog_with_percentiles() -> dict[str, Cube]:
    orders = _cube(Dialect.POSTGRES).model_copy(
        update={
            "primary_key": "id",
            "dimensions": [
                Dimension(name="id", sql="{o}.id", type="number"),
                Dimension(
                    name="customer_id",
                    sql="{o}.customer_id",
                    type="number",
                    foreign_key="customers",
                ),
                Dimension(name="region", sql="{o}.region", type="string"),
            ],
        }
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )
    # Catalog auto-derives the cross-backend Join from
    # ``orders.customer_id.foreign_key="customers"``.
    return Catalog([orders, customers]).as_dict()


def test_distributive_federation_refuses_percentiles() -> None:
    """Percentile aggs aren't distributive — distributive mode should
    refuse them with the same ``non_distributive_aggregation:`` reason
    code that gates min / max / count_distinct."""
    catalog = _federated_catalog_with_percentiles()
    q = SemanticQuery(measures=["orders.amount_median"], dimensions=["customers.region"])
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason.startswith("non_distributive_aggregation:")


def test_raw_rows_federation_emits_percentile_at_merge() -> None:
    """Raw-row mode lifts the distributive refusal: percentile measures
    project the raw column at the fragment and apply
    ``PERCENTILE_CONT`` at the merge step."""
    catalog = _federated_catalog_with_percentiles()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.amount_median"], dimensions=["customers.region"]),
        catalog,
        mode="raw_rows",
    )
    # The AST emitter renders DuckDB's QUANTILE_CONT (its canonical spelling
    # of PERCENTILE_CONT ... WITHIN GROUP).
    sql_upper = plan.merge.sql.upper()
    assert "QUANTILE_CONT" in sql_upper or "PERCENTILE_CONT" in sql_upper
    assert "0.5" in plan.merge.sql
