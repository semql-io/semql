from datetime import datetime

import duckdb
import pytest
from pydantic import ValidationError
from semql.analysis import CatalogContext, compare_analysis
from semql.compile import CompiledQuery, CompileError, compile_plan, compile_query, explain_plan
from semql.model import AuthContext, Cube, Dialect, Dimension, Measure, TimeDimension, View
from semql.parse import ParseError, parse_sql_statement
from semql.spec import (
    CompareWindow,
    InlineDerived,
    PartitionRank,
    RankOrder,
    SemanticQuery,
    TimeWindow,
    rank_shape_errors,
)
from semql.unparse import UnparseError, query_to_sql


def _catalog() -> dict[str, Cube]:
    cube = Cube(
        name="sales",
        dialect=Dialect.DUCKDB,
        table="sales",
        alias="s",
        dimensions=[
            Dimension(
                name="region",
                sql="{s}.region",
                type="string",
                aliases=["territory"],
            ),
            Dimension(name="year", sql="{s}.year", type="number"),
        ],
        measures=[
            Measure(name="revenue", sql="{s}.revenue", agg="sum", unit="currency"),
            Measure(name="cost", sql="{s}.cost", agg="sum", unit="currency"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{s}.created_at", type="time")],
    )
    return {cube.name: cube}


def _execute(compiled: CompiledQuery) -> list[tuple[object, ...]]:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE sales("
            "region VARCHAR, year INTEGER, revenue INTEGER, cost INTEGER, created_at TIMESTAMP)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
            [
                ("east", 2024, 5, 1, "2026-01-02"),
                ("east", 2024, 10, 1, "2026-01-05"),
                ("east", 2025, 20, 10, "2026-01-12"),
                ("east", 2026, 20, 8, "2026-01-20"),
                ("west", 2024, 30, 3, "2026-01-03"),
                ("west", 2025, None, 4, "2026-02-03"),
            ],
        )
        return connection.execute(compiled.sql, compiled.params).fetchall()


def test_rank_models_validate_count_and_deep_freeze() -> None:
    with pytest.raises(ValidationError):
        PartitionRank(
            partition_by=[],
            order_by=[RankOrder(field="revenue", direction="desc")],
            count=0,
        )
    with pytest.raises(ValidationError):
        PartitionRank(
            order_by=[RankOrder(field="revenue", direction="desc")],
            count=True,
        )
    rank = PartitionRank(
        partition_by=["sales.region"],
        order_by=[RankOrder(field="revenue", direction="desc")],
        count=2,
    )
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        rank.partition_by.append("sales.year")  # type: ignore[attr-defined]


def test_rank_semantics_compare_and_ignore_output_aliases() -> None:
    context = CatalogContext(namespace="sales", semantic_revision="r1")
    base = compile_query(
        SemanticQuery(
            measures=["sales.revenue"],
            dimensions=["sales.region"],
            rank=PartitionRank(
                order_by=[RankOrder(field="sales.revenue", direction="desc")],
                count=1,
            ),
        ),
        _catalog(),
        catalog_context=context,
    )
    aliased = compile_query(
        SemanticQuery(
            measures=["sales.revenue"],
            dimensions=["sales.region"],
            aliases={"amount": "sales.revenue"},
            rank=PartitionRank(order_by=[RankOrder(field="amount", direction="desc")], count=1),
        ),
        _catalog(),
        catalog_context=context,
    )
    changed = compile_query(
        SemanticQuery(
            measures=["sales.revenue"],
            dimensions=["sales.region"],
            rank=PartitionRank(
                order_by=[RankOrder(field="sales.revenue", direction="desc")],
                count=2,
            ),
        ),
        _catalog(),
        catalog_context=context,
    )
    assert (
        compare_analysis(base.analysis, aliased.analysis, scope="result").outcome.value
        == "equivalent"
    )
    assert (
        compare_analysis(base.analysis, changed.analysis, scope="result").outcome.value
        == "different"
    )


def test_view_qualified_rank_measure_preserves_semantic_equivalence() -> None:
    context = CatalogContext(namespace="sales", semantic_revision="r1")
    views = {
        "v": View(
            name="v",
            fields={"revenue": "sales.revenue", "region": "sales.region"},
        )
    }
    canonical = compile_query(
        SemanticQuery(
            measures=["sales.revenue"],
            dimensions=["sales.region"],
            rank=PartitionRank(
                order_by=[RankOrder(field="sales.revenue", direction="desc")],
                count=1,
            ),
        ),
        _catalog(),
        catalog_context=context,
    )
    view_query = compile_query(
        SemanticQuery(
            measures=["v.revenue"],
            dimensions=["sales.region"],
            rank=PartitionRank(
                order_by=[RankOrder(field="v.revenue", direction="desc")],
                count=1,
            ),
        ),
        _catalog(),
        views=views,
        catalog_context=context,
    )
    assert compare_analysis(
        canonical.analysis, view_query.analysis, scope="result"
    ).outcome.value == ("equivalent")
    assert _execute(view_query) == [("east", 55)]


def test_view_qualified_time_partition_and_final_order_compile() -> None:
    views = {
        "v": View(
            name="v",
            fields={
                "revenue": "sales.revenue",
                "region": "sales.region",
                "created_at": "sales.created_at",
            },
        )
    }
    query = SemanticQuery(
        measures=["v.revenue"],
        dimensions=["v.region"],
        time_dimension=TimeWindow(
            dimension="v.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
        rank=PartitionRank(
            partition_by=["v.created_at"],
            order_by=[RankOrder(field="v.revenue", direction="desc")],
            count=1,
        ),
        order=[("v.region", "asc"), ("v.created_at", "asc")],
    )
    compiled = compile_query(query, _catalog(), views=views)
    assert compiled.columns == ["region", "created_at_day", "revenue"]
    assert _execute(compiled) == [
        ("east", datetime(2026, 1, 2), 5),
        ("east", datetime(2026, 1, 5), 10),
        ("east", datetime(2026, 1, 12), 20),
        ("east", datetime(2026, 1, 20), 20),
        ("west", datetime(2026, 1, 3), 30),
    ]


@pytest.mark.parametrize("use_plan", [False, True])
@pytest.mark.parametrize("order_ref", ["sales.territory", "area"])
def test_ranked_final_order_uses_selected_output_alias_and_synonym(
    use_plan: bool, order_ref: str
) -> None:
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.territory"],
        aliases={"area": "sales.territory"},
        rank=PartitionRank(
            order_by=[RankOrder(field="sales.revenue", direction="desc")],
            count=2,
        ),
        order=[(order_ref, "asc")],
    )
    catalog = _catalog()
    if use_plan:
        compiled = compile_plan(explain_plan(query, catalog), catalog)
    else:
        compiled = compile_query(query, catalog)
    assert _execute(compiled) == [("east", 55), ("west", 30)]


def test_grouped_rank_is_post_aggregate_and_pre_global_cap() -> None:
    result = compile_query(
        SemanticQuery(
            measures=["sales.revenue"],
            dimensions=["sales.region", "sales.year"],
            rank=PartitionRank(
                partition_by=["sales.region"],
                order_by=[RankOrder(field="revenue", direction="desc")],
                count=1,
            ),
            order=[("sales.revenue", "desc")],
            limit=1,
        ),
        _catalog(),
    )
    assert _execute(result) == [("west", 2024, 30)]
    assert result.columns == ["region", "year", "revenue"]


def test_rank_preserves_unprojected_aggregate_order_and_avoids_alias_collision() -> None:
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region", "sales.year"],
        aliases={"_semql_partition_rank": "sales.revenue"},
        rank=PartitionRank(
            partition_by=["sales.region"],
            order_by=[RankOrder(field="sales.revenue", direction="desc")],
            count=2,
        ),
        order=[("sales.cost", "desc")],
    )
    result = compile_query(query, _catalog())
    assert _execute(result) == [
        ("east", 2025, 20),
        ("east", 2026, 20),
        ("west", 2025, None),
        ("west", 2024, 30),
    ]
    assert result.columns == ["region", "year", "_semql_partition_rank"]


def test_exact_rank_rejects_masked_automatic_tie_breaker() -> None:
    catalog = _catalog()
    cube = catalog["sales"]
    masked_year = cube.dimensions[1].model_copy(
        update={"required_roles": ["analyst"], "mask_roles": ["analyst"]}
    )
    catalog["sales"] = cube.model_copy(update={"dimensions": [cube.dimensions[0], masked_year]})
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region", "sales.year"],
        rank=PartitionRank(
            order_by=[RankOrder(field="sales.revenue", direction="desc")],
            count=2,
        ),
    )
    with pytest.raises(CompileError, match="masked tie-breaker"):
        compile_query(query, catalog, viewer=AuthContext(viewer_id="u", roles=["analyst"]))


def test_include_ties_uses_only_explicit_score_order_and_null_policy() -> None:
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region", "sales.year"],
        rank=PartitionRank(
            partition_by=["sales.region"],
            order_by=[RankOrder(field="sales.revenue", direction="desc", nulls="first")],
            count=1,
            ties="include",
        ),
        order=[("sales.region", "asc"), ("sales.year", "asc")],
    )
    assert _execute(compile_query(query, _catalog())) == [
        ("east", 2025, 20),
        ("east", 2026, 20),
        ("west", 2025, None),
    ]


def test_semantic_sql_refuses_rank_window_round_trip_without_loss() -> None:
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region"],
        rank=PartitionRank(order_by=[RankOrder(field="sales.revenue", direction="desc")], count=1),
    )
    with pytest.raises(UnparseError, match="rank"):
        query_to_sql(query, _catalog())
    with pytest.raises(ParseError, match="Window functions"):
        parse_sql_statement(
            "SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue) FROM sales",
            _catalog(),
        )


def test_rank_shape_rejects_compare_and_dense_fill() -> None:
    rank = PartitionRank(order_by=[RankOrder(field="sales.revenue", direction="desc")], count=2)
    compared = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region"],
        time_dimension=TimeWindow(dimension="sales.created_at", range=("2026-10-01", "2026-11-01")),
        compare=CompareWindow(),
        rank=rank,
    )
    assert "ranking_compare_unsupported" in rank_shape_errors(compared)
    filled = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region"],
        time_dimension=TimeWindow(
            dimension="sales.created_at",
            range=("2026-10-01", "2026-11-01"),
            granularity="day",
            fill_nulls_with=0,
        ),
        rank=rank,
    )
    assert "ranking_fill_unsupported" in rank_shape_errors(filled)


def test_rank_shape_diagnostics_and_masked_value_refusal() -> None:
    grouped = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region"],
        rank=PartitionRank(order_by=[RankOrder(field="sales.revenue", direction="desc")], count=1),
    )
    assert rank_shape_errors(grouped) == ()
    scalar = SemanticQuery(
        measures=["sales.revenue"],
        rank=PartitionRank(order_by=[RankOrder(field="sales.revenue", direction="desc")], count=1),
    )
    assert rank_shape_errors(scalar) == ("ranking_grouping_required",)

    catalog = _catalog()
    cube = catalog["sales"]
    masked_measure = cube.measures[0].model_copy(
        update={"required_roles": ["analyst"], "mask_roles": ["analyst"]}
    )
    catalog["sales"] = cube.model_copy(update={"measures": [masked_measure, *cube.measures[1:]]})
    query = SemanticQuery(
        measures=["sales.revenue"],
        dimensions=["sales.region"],
        rank=PartitionRank(order_by=[RankOrder(field="sales.revenue", direction="desc")], count=1),
    )
    with pytest.raises(CompileError, match="Ranking on masked"):
        compile_query(query, catalog, viewer=AuthContext(viewer_id="u", roles=["analyst"]))


def test_rank_plan_path_and_inline_derived_measure() -> None:
    query = SemanticQuery(
        measures=["sales.revenue", "sales.cost"],
        dimensions=["sales.region"],
        derived_measures=[
            InlineDerived(name="ratio", op="ratio", operands=["sales.revenue", "sales.cost"])
        ],
        rank=PartitionRank(order_by=[RankOrder(field="ratio", direction="desc")], count=4),
    )
    plan = explain_plan(query, _catalog())
    compiled = compile_plan(plan, _catalog())
    assert sorted(_execute(compiled)) == [
        ("east", 55, 20, 2.75),
        ("west", 30, 7, pytest.approx(30 / 7)),
    ]


def test_rank_input_rejects_unknown_nested_fields() -> None:
    with pytest.raises(ValidationError):
        SemanticQuery.model_validate(
            {
                "dimensions": ["sales.region"],
                "rank": {"partition_by": [], "order_by": [], "count": 1, "surprise": True},
            }
        )
    with pytest.raises(ValidationError):
        SemanticQuery.model_validate({"dimensions": ["sales.region"], "unknown": True})
