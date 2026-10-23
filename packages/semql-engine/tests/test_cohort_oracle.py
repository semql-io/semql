"""Independent membership and value oracle for frozen ranked cohorts."""

from __future__ import annotations

import random
from collections import defaultdict
from fractions import Fraction

import duckdb
import pytest
from semql import (
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


@pytest.mark.parametrize("seed", range(24))
def test_ranked_cohort_matches_independent_fraction_oracle(seed: int) -> None:
    rng = random.Random(seed)
    facts: list[tuple[int, str, str, int, int]] = []
    for region in ("East", "West", "North"):
        for offset in range(4):
            store = ("East", "West", "North").index(region) * 4 + offset
            for period in ("prior", "current"):
                if period == "current" and rng.random() < 0.3:
                    continue
                for _ in range(rng.randint(1, 3)):
                    visits = rng.randint(0, 10)
                    facts.append(
                        (
                            store,
                            region if period == "prior" else rng.choice(("East", "West", "North")),
                            period,
                            rng.randint(0, visits),
                            visits,
                        )
                    )
    totals: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    current: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for store, region, period, conversions, visits in facts:
        target = totals[(region, store)] if period == "prior" else current[store]
        target[0] += conversions
        target[1] += visits
    winners: dict[str, int] = {}
    for region in ("East", "West", "North"):
        candidates = [
            (store, values) for (group, store), values in totals.items() if group == region
        ]
        # A defined rate beats NULL; equal rates resolve by member identity.
        winners[region] = min(
            candidates,
            key=lambda item: (
                item[1][1] == 0,
                -Fraction(item[1][0], item[1][1]) if item[1][1] else Fraction(0),
                item[0],
            ),
        )[0]
    cube = Cube(
        name="facts",
        table="facts",
        alias="f",
        dialect=Dialect.DUCKDB,
        dimensions=[
            Dimension(name="store", sql="{f}.store", type="number"),
            Dimension(name="region", sql="{f}.region", type="string"),
            Dimension(name="period", sql="{f}.period", type="string"),
        ],
        measures=[
            Measure(name="conversions", sql="{f}.conversions", agg="sum"),
            Measure(name="visits", sql="{f}.visits", agg="sum"),
        ],
    )

    def stage(period: str, *, select: bool = False) -> SemanticQuery:
        return SemanticQuery(
            dimensions=["facts.region", "facts.store"] if select else ["facts.store"],
            measures=["facts.conversions", "facts.visits"],
            filters=[Filter(dimension="facts.period", op="eq", values=[period])],
            derived_measures=[
                InlineDerived(
                    name="rate", op="ratio", operands=["facts.conversions", "facts.visits"]
                )
            ],
            rank=(
                PartitionRank(
                    partition_by=["facts.region"],
                    order_by=[RankOrder(field="rate", direction="desc")],
                    count=1,
                )
                if select
                else None
            ),
        )

    query = CohortQuery(
        cohort=Cohort(
            source=stage("prior", select=True),
            member_keys=["facts.store"],
            attributes=["facts.region"],
        ),
        measurements=[
            CohortMeasurement(
                name=period,
                query=stage(period),
                key_mapping={"facts.store": "facts.store"},
            )
            for period in ("prior", "current")
        ],
        comparisons=[
            CohortComparison(
                name="delta",
                left=CohortMetric(measurement="current", column="rate"),
                right=CohortMetric(measurement="prior", column="rate"),
            )
        ],
    )
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE facts(store INTEGER, region VARCHAR, period VARCHAR, "
            "conversions INTEGER, visits INTEGER)"
        )
        connection.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?)", facts)
        engine = Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(connection)})
        result = run_cohort(compile_cohort_query(query, {cube.name: cube}), engine)
    rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
    actual = {str(row["selection.region"]): row for row in rows}
    assert {region: row["selection.store"] for region, row in actual.items()} == winners
    for region, store in winners.items():
        row = actual[region]
        prior = totals[(region, store)]
        prior_rate = Fraction(*prior) if prior[1] else None
        observed = current.get(store)
        current_rate = Fraction(*observed) if observed is not None and observed[1] else None
        assert row["prior.rate"] == (
            pytest.approx(float(prior_rate)) if prior_rate is not None else None
        )
        assert row["current.rate"] == (
            pytest.approx(float(current_rate)) if current_rate is not None else None
        )
        if observed is None:
            assert row["current.conversions"] is None
        else:
            assert row["current.conversions"] == observed[0]
            assert row["current.visits"] == observed[1]
        delta = (
            current_rate - prior_rate
            if current_rate is not None and prior_rate is not None
            else None
        )
        assert row["comparison.delta"] == (
            pytest.approx(float(delta)) if delta is not None else None
        )
