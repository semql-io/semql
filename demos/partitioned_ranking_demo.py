"""Execute partitioned ranking with authorization and explicit tie semantics."""

from __future__ import annotations

import duckdb
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    PartitionRank,
    RankOrder,
    SemanticQuery,
)


def main() -> None:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE sales(category VARCHAR, product VARCHAR, amount INT, owner VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?)",
            [
                ("A", "a", 6, "reader"),
                ("A", "a", 4, "reader"),
                ("A", "b", 10, "reader"),
                ("A", "c", 9, "reader"),
                ("B", "d", 8, "reader"),
                ("B", "e", 7, "reader"),
                ("A", "private", 1000, "someone-else"),
            ],
        )
        catalog = Catalog(
            [
                Cube(
                    name="sales",
                    table="sales",
                    alias="s",
                    dialect=Dialect.DUCKDB,
                    security_sql="{s}.owner = {ctx.viewer_id}",
                    dimensions=[
                        Dimension(name="category", sql="{s}.category", type="string"),
                        Dimension(name="product", sql="{s}.product", type="string"),
                    ],
                    measures=[Measure(name="revenue", sql="{s}.amount", agg="sum")],
                )
            ]
        )
        viewer = AuthContext(viewer_id="reader")
        query = SemanticQuery(
            dimensions=["sales.category", "sales.product"],
            measures=["sales.revenue"],
            rank=PartitionRank(
                partition_by=["sales.category"],
                order_by=[RankOrder(field="sales.revenue", direction="desc")],
                count=1,
                ties="exact",
            ),
            order=[("sales.category", "asc"), ("sales.product", "asc")],
        )
        exact = catalog.compile(query, viewer=viewer)
        exact_rows = connection.execute(exact.sql, exact.params).fetchall()
        assert exact_rows == [("A", "a", 10), ("B", "d", 8)]
        assert query.rank is not None
        inclusive_query = query.model_copy(
            update={"rank": query.rank.model_copy(update={"ties": "include"})}
        )
        inclusive = catalog.compile(inclusive_query, viewer=viewer)
        inclusive_rows = connection.execute(inclusive.sql, inclusive.params).fetchall()
        assert inclusive_rows == [("A", "a", 10), ("A", "b", 10), ("B", "d", 8)]
        print("Exact one per category:", exact_rows)
        print("Boundary-inclusive ties:", inclusive_rows)
        print("Authorized aggregation precedes ranking; the private high scorer is excluded.")
        print("Exact ties use the remaining product grouping key, not source-row order.")
        print(
            "Selection semantics:",
            exact.analysis.result.model_dump() if exact.analysis.result else None,
        )


if __name__ == "__main__":
    main()
