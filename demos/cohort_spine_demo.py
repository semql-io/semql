"""Execute a frozen store cohort, preserving attribution and missing observations."""

from __future__ import annotations

import duckdb
from semql import (
    AuthContext,
    Catalog,
    Cohort,
    CohortComparison,
    CohortMeasurement,
    CohortMetric,
    CohortQuery,
    Cube,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Measure,
    PartitionRank,
    RankOrder,
    SemanticQuery,
    compile_cohort_query,
)
from semql_engine import DuckDBAdapter, Engine, run_cohort


def main() -> None:
    catalog = Catalog(
        [
            Cube(
                name="sales",
                table="sales",
                alias="s",
                dialect=Dialect.DUCKDB,
                security_sql="{s}.owner = {ctx.viewer_id}",
                dimensions=[
                    Dimension(name="store", sql="{s}.store", type="string"),
                    Dimension(name="region", sql="{s}.region", type="string"),
                    Dimension(name="period", sql="{s}.period", type="string"),
                ],
                measures=[
                    Measure(name="conversions", sql="{s}.conversions", agg="sum"),
                    Measure(name="visits", sql="{s}.visits", agg="sum"),
                    Measure(name="observations", sql="*", agg="count"),
                ],
            )
        ]
    )

    def measurement(period: str, *, select: bool = False) -> SemanticQuery:
        return SemanticQuery(
            dimensions=["sales.region", "sales.store"] if select else ["sales.store"],
            measures=["sales.conversions", "sales.visits", "sales.observations"],
            derived_measures=[
                InlineDerived(
                    name="rate", op="ratio", operands=["sales.conversions", "sales.visits"]
                )
            ],
            filters=[Filter(dimension="sales.period", op="eq", values=[period])],
            rank=(
                PartitionRank(
                    partition_by=["sales.region"],
                    order_by=[RankOrder(field="rate", direction="desc")],
                    count=1,
                )
                if select
                else None
            ),
            order=[("sales.store", "asc")],
        )

    query = CohortQuery(
        cohort=Cohort(
            source=measurement("prior", select=True),
            member_keys=["sales.store"],
            attributes=["sales.region"],
            max_members=20,
        ),
        measurements=[
            CohortMeasurement(
                name=period,
                query=measurement(period),
                key_mapping={"sales.store": "sales.store"},
            )
            for period in ("prior", "current")
        ],
        comparisons=[
            CohortComparison(
                name="rate_delta",
                left=CohortMetric(measurement="current", column="rate"),
                right=CohortMetric(measurement="prior", column="rate"),
                op="difference",
            )
        ],
    )
    plan = compile_cohort_query(query, catalog.as_dict(), viewer=AuthContext(viewer_id="reader"))
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE sales(store VARCHAR, region VARCHAR, period VARCHAR, "
            "conversions INTEGER, visits INTEGER, owner VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("A", "West", "prior", 90, 100, "reader"),
                ("B", "West", "prior", 80, 100, "reader"),
                ("C", "East", "prior", 1, 1, "reader"),
                ("D", "East", "prior", 50, 100, "reader"),
                ("E", "North", "prior", 7, 10, "reader"),
                ("J", "South", "prior", 1, 2, "reader"),
                ("K", "Nulls", "prior", 1, 2, "reader"),
                ("A", "East", "current", 20, 100, "reader"),
                ("B", "West", "current", 95, 100, "reader"),
                ("C", "East", "current", 8, 10, "reader"),
                ("D", "East", "current", 99, 100, "reader"),
                ("J", "South", "current", 0, 0, "reader"),
                ("K", "Nulls", "current", None, None, "reader"),
                ("A", "West", "prior", 1000, 1000, "another-reader"),
                ("A", "East", "current", 1000, 1000, "another-reader"),
            ],
        )
        engine = Engine()
        engine.register(Dialect.DUCKDB, DuckDBAdapter(connection))
        result = run_cohort(plan, engine)
        rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
        by_store = {row["selection.store"]: row for row in rows}
        assert set(by_store) == {"A", "C", "E", "J", "K"}
        assert by_store["A"]["selection.region"] == "West"
        assert by_store["A"]["current.rate"] == 0.2
        assert by_store["C"]["current.rate"] == 0.8
        assert by_store["E"]["current.observations"] is None
        assert by_store["E"]["comparison.rate_delta"] is None
        assert by_store["J"]["current.observations"] == 1
        assert by_store["J"]["current.visits"] == 0
        assert by_store["J"]["current.rate"] is None
        assert by_store["K"]["current.observations"] == 1
        assert by_store["K"]["current.conversions"] is None
        assert by_store["K"]["current.rate"] is None
        print("Frozen prior winners:", sorted(by_store))
        print("A retains selection region West despite current observations in East.")
        print("Missing E, zero-denominator J, and observed-NULL K remain distinct:")
        for store in ("E", "J", "K"):
            print(store, by_store[store])
        print("Current rates A=0.2, C=0.8; new winners B/D were not substituted.")
        print("Presence:", result.presence)
        print("Coverage:", result.coverage)
        print("No snapshot or automatic rollup of member metrics is implied.")


if __name__ == "__main__":
    main()
