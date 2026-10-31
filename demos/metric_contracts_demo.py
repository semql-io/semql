"""Executed semantic counterexamples against compiled SemQL queries."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import duckdb
from semql import (
    Cohort,
    CohortMeasurement,
    CohortQuery,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    compile_cohort_query,
    compile_query,
)
from semql.contracts import MetricContract, SemanticExpectations
from semql.errors import ContractError
from semql.validate import check_semantics
from semql_engine import DuckDBAdapter, DuckDBMergeEngine, Engine, run_cohort

type Number = int | float | Fraction | None


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    observed: tuple[Number, ...]
    expected: tuple[Number, ...]
    detail: str


@dataclass(frozen=True)
class RuntimeCases:
    ratio_of_sums_vs_mean_of_ratios: RuntimeCase
    overlapping_distinct_reaggregation: RuntimeCase
    snapshots_across_time: RuntimeCase
    join_fanout: RuntimeCase
    created_vs_shipped_time_axis: RuntimeCase
    fixed_vs_changing_population: RuntimeCase
    unsafe_fanout_rejected: bool
    duplicate_bridge_observed: tuple[int, int]
    created_axis_check: str
    shipped_axis_check: str
    cohort_comparison_check: str
    cohort_members: tuple[str, ...]
    cohort_measurements: tuple[tuple[str, int | None], ...]


def _execute(
    connection: duckdb.DuckDBPyConnection, query: SemanticQuery, catalog: dict[str, Cube]
) -> tuple[tuple[object, ...], ...]:
    compiled = compile_query(query, catalog)
    return tuple(connection.execute(compiled.sql, compiled.params).fetchall())


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"Expected an exact integer fixture result, got {type(value).__name__}")
    return value


def run_cases() -> RuntimeCases:
    """Run six actual SQL counterexamples; expected values use independent exact oracles."""
    db = duckdb.connect(":memory:")
    try:
        # Ratio-of-sums is not the mean of per-row ratios.
        ratio_cube = Cube(
            name="ratio_data",
            dialect=Dialect.DUCKDB,
            table="ratio_data",
            alias="r",
            measures=[
                Measure(name="numerator", sql="{r}.n", agg="sum"),
                Measure(name="denominator", sql="{r}.d", agg="sum"),
                Measure(
                    name="ratio",
                    sql="",
                    agg="ratio",
                    numerator="numerator",
                    denominator="denominator",
                ),
            ],
        )
        db.execute("CREATE TABLE ratio_data(n INTEGER, d INTEGER)")
        db.execute("INSERT INTO ratio_data VALUES (9, 3), (1, 1)")
        ratio_row = _execute(
            db,
            SemanticQuery(
                measures=[
                    "ratio_data.numerator",
                    "ratio_data.denominator",
                    "ratio_data.ratio",
                ]
            ),
            {"ratio_data": ratio_cube},
        )[0]
        ratio_average = db.execute("SELECT AVG(n::DOUBLE / d) FROM ratio_data").fetchall()[0][0]
        ratio_case = RuntimeCase(
            "ratio_of_sums_vs_mean_of_ratios",
            (cast(Number, ratio_row[2]), cast(Number, ratio_average)),
            (Fraction(10, 4), (Fraction(9, 3) + Fraction(1, 1)) / 2),
            "Ratio is compiled as SUM(n)/SUM(d); raw SQL baseline averages row ratios.",
        )

        # Distinct counts from overlapping groups cannot be added.
        distinct_cube = Cube(
            name="membership",
            dialect=Dialect.DUCKDB,
            table="membership",
            alias="m",
            measures=[Measure(name="members", sql="{m}.person", agg="count_distinct")],
            dimensions=[Dimension(name="group_name", sql="{m}.group_name", type="string")],
        )
        db.execute("CREATE TABLE membership(group_name VARCHAR, person VARCHAR)")
        db.execute(
            "INSERT INTO membership VALUES ('left','a'),('left','b'),('right','b'),('right','c')"
        )
        group_rows = _execute(
            db,
            SemanticQuery(measures=["membership.members"], dimensions=["membership.group_name"]),
            {"membership": distinct_cube},
        )
        total_row = _execute(
            db, SemanticQuery(measures=["membership.members"]), {"membership": distinct_cube}
        )
        distinct_case = RuntimeCase(
            "overlapping_distinct_reaggregation",
            (sum(_integer(row[1]) for row in group_rows), _integer(total_row[0][0])),
            (4, len({"a", "b", "c"})),
            "Compiled group counts versus compiled whole-population count; "
            "independent set union has cardinality 3.",
        )

        # Snapshot values are states, not additive flows across time.
        snapshot_cube = Cube(
            name="balances",
            dialect=Dialect.DUCKDB,
            table="balances",
            alias="b",
            measures=[Measure(name="balance", sql="{b}.amount", agg="sum")],
            time_dimensions=[TimeDimension(name="as_of", sql="{b}.as_of")],
        )
        db.execute("CREATE TABLE balances(as_of DATE, amount INTEGER)")
        db.execute("INSERT INTO balances VALUES ('2026-01-01', 100), ('2026-02-01', 120)")
        snapshots = _execute(
            db,
            SemanticQuery(
                measures=["balances.balance"],
                time_dimension=TimeWindow(
                    dimension="balances.as_of",
                    granularity="month",
                    range=("2026-01-01", "2026-03-01"),
                ),
                order=[("balances.as_of", "asc")],
            ),
            {"balances": snapshot_cube},
        )
        snapshot_case = RuntimeCase(
            "snapshots_across_time",
            (sum(_integer(row[1]) for row in snapshots), _integer(snapshots[-1][1])),
            (220, 120),
            "Compiled monthly snapshot series: summing periods produces 220; latest state is 120.",
        )

        # Unsafe additive measure on the duplicated one-side is refused.
        parent = Cube(
            name="parents",
            dialect=Dialect.DUCKDB,
            table="parents",
            alias="p",
            measures=[Measure(name="value", sql="{p}.value", agg="sum")],
            dimensions=[Dimension(name="id", sql="{p}.id", type="number")],
            joins=[Join(to="children", relationship="one_to_many", on="{p}.id = {c}.parent_id")],
        )
        child = Cube(
            name="children",
            dialect=Dialect.DUCKDB,
            table="children",
            alias="c",
            dimensions=[
                Dimension(name="parent_id", sql="{c}.parent_id", type="number"),
                Dimension(name="label", sql="{c}.label", type="string"),
            ],
        )
        try:
            compile_query(
                SemanticQuery(measures=["parents.value"], dimensions=["children.label"]),
                {"parents": parent, "children": child},
            )
            unsafe_rejected = False
        except ContractError as error:
            unsafe_rejected = error.reason == "unsafe_fanout"
        # Matching executed fixture establishes what fanout would do.
        db.execute("CREATE TABLE parents(id INTEGER, value INTEGER)")
        db.execute("CREATE TABLE children(parent_id INTEGER, label VARCHAR)")
        db.execute("INSERT INTO parents VALUES (1, 100)")
        db.execute("INSERT INTO children VALUES (1, 'x'), (1, 'y')")
        fanout = db.execute(
            "SELECT (SELECT SUM(value) FROM parents), "
            "(SELECT SUM(p.value) FROM parents p JOIN children c ON p.id=c.parent_id)"
        ).fetchall()[0]
        fanout_case = RuntimeCase(
            "join_fanout",
            tuple(_integer(x) for x in fanout),
            (100, 200),
            "Compiler refuses additive fanout; raw execution shows duplicate-row inflation.",
        )

        # Trusted many-to-one declaration does not prove runtime key uniqueness.
        bridge = Cube(
            name="bridge",
            dialect=Dialect.DUCKDB,
            table="bridge",
            alias="x",
            measures=[Measure(name="amount", sql="{x}.amount", agg="sum")],
            dimensions=[Dimension(name="key", sql="{x}.key", type="number")],
            joins=[Join(to="lookup", relationship="many_to_one", on="{x}.key = {l}.key")],
        )
        lookup = Cube(
            name="lookup",
            dialect=Dialect.DUCKDB,
            table="lookup",
            alias="l",
            dimensions=[
                Dimension(name="key", sql="{l}.key", type="number"),
                Dimension(name="label", sql="{l}.label", type="string"),
            ],
        )
        duplicate_compiled = compile_query(
            SemanticQuery(measures=["bridge.amount"], dimensions=["lookup.label"]),
            {"bridge": bridge, "lookup": lookup},
        )
        db.execute("CREATE TABLE bridge(key INTEGER, amount INTEGER)")
        db.execute("CREATE TABLE lookup(key INTEGER, label VARCHAR)")
        db.execute("INSERT INTO bridge VALUES (1, 100)")
        db.execute("INSERT INTO lookup VALUES (1, 'a'), (1, 'b')")
        declared_rows = db.execute(duplicate_compiled.sql, duplicate_compiled.params).fetchall()
        declared_total = sum(_integer(row[1]) for row in declared_rows)
        direct_total = _integer(db.execute("SELECT SUM(amount) FROM bridge").fetchall()[0][0])

        # Compiled created and shipped axis query from same declared metric.
        time_cube = Cube(
            name="orders_rt",
            dialect=Dialect.DUCKDB,
            table="orders_rt",
            alias="o",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(name="created", sql="{o}.created"),
                TimeDimension(name="shipped", sql="{o}.shipped"),
            ],
        )
        db.execute("CREATE TABLE orders_rt(amount INTEGER, created DATE, shipped DATE)")
        db.execute(
            "INSERT INTO orders_rt VALUES "
            "(10,'2026-01-02','2026-02-02'), (20,'2026-02-02','2026-01-02')"
        )
        created_query = SemanticQuery(
            measures=["orders_rt.revenue"],
            time_dimension=TimeWindow(
                dimension="orders_rt.created",
                granularity="month",
                range=("2026-01-01", "2026-03-01"),
            ),
            order=[("orders_rt.created", "asc")],
        )
        shipped_query = SemanticQuery(
            measures=["orders_rt.revenue"],
            time_dimension=TimeWindow(
                dimension="orders_rt.shipped",
                granularity="month",
                range=("2026-01-01", "2026-03-01"),
            ),
            order=[("orders_rt.shipped", "asc")],
        )
        created_compiled = compile_query(created_query, {"orders_rt": time_cube})
        shipped_compiled = compile_query(shipped_query, {"orders_rt": time_cube})
        created = tuple(db.execute(created_compiled.sql, created_compiled.params).fetchall())
        shipped = tuple(db.execute(shipped_compiled.sql, shipped_compiled.params).fetchall())
        contract = MetricContract(
            reference="orders_rt.revenue",
            revision="runtime-v1",
            required_time_axis=True,
            permitted_time_axes=("orders_rt.shipped",),
        )
        expected_axis = SemanticExpectations(
            time_axis="orders_rt.shipped",
            origin="independently_confirmed",
            evidence_ref="runtime fixture intent",
        )
        created_check = check_semantics(
            created_compiled.analysis, {contract.reference: contract}, expected_axis
        )
        shipped_check = check_semantics(
            shipped_compiled.analysis, {contract.reference: contract}, expected_axis
        )
        time_case = RuntimeCase(
            "created_vs_shipped_time_axis",
            (
                _integer(created[0][1]),
                _integer(created[1][1]),
                _integer(shipped[0][1]),
                _integer(shipped[1][1]),
            ),
            (10, 20, 20, 10),
            "Event-time axis changes results; checker rejects created and accepts shipped.",
        )

        # Actual fixed cohort selection via public compile and engine APIs.
        cohort_cube = Cube(
            name="cohort_rt",
            dialect=Dialect.DUCKDB,
            table="cohort_rt",
            alias="q",
            measures=[Measure(name="amount", sql="{q}.amount", agg="sum")],
            dimensions=[
                Dimension(name="person", sql="{q}.person", type="string"),
                Dimension(name="period", sql="{q}.period", type="string"),
            ],
        )
        db.execute("CREATE TABLE cohort_rt(person VARCHAR, period VARCHAR, amount INTEGER)")
        db.execute(
            "INSERT INTO cohort_rt VALUES "
            "('a','prior',10),('b','prior',7),('a','current',20),('c','current',99)"
        )
        cohort_query = CohortQuery(
            cohort=Cohort(
                source=SemanticQuery(
                    dimensions=["cohort_rt.person"],
                    measures=["cohort_rt.amount"],
                    filters=[Filter(dimension="cohort_rt.period", op="eq", values=["prior"])],
                ),
                member_keys=["cohort_rt.person"],
                max_members=10,
            ),
            measurements=[
                CohortMeasurement(
                    name="current",
                    query=SemanticQuery(
                        measures=["cohort_rt.amount"],
                        dimensions=["cohort_rt.person"],
                        filters=[Filter(dimension="cohort_rt.period", op="eq", values=["current"])],
                    ),
                    key_mapping={"cohort_rt.person": "cohort_rt.person"},
                )
            ],
        )
        plan = compile_cohort_query(cohort_query, {"cohort_rt": cohort_cube})
        cohort_result = run_cohort(
            plan,
            Engine(adapters={Dialect.DUCKDB: DuckDBAdapter(db)}, merge_engine=DuckDBMergeEngine()),
        )
        cohort_expectation = SemanticExpectations(
            comparison_basis="fixed_cohort",
            origin="independently_confirmed",
            evidence_ref="runtime fixed-cohort fixture",
        )
        cohort_check = check_semantics(plan.source.plan.analysis, {}, cohort_expectation)
        cohort_rows = {
            str(row[0]): (None if row[1] is None else _integer(row[1]))
            for row in cohort_result.rows
        }
        prior_rows = _execute(db, cohort_query.cohort.source, {"cohort_rt": cohort_cube})
        independently_selected = _execute(
            db,
            cohort_query.measurements[0].query,
            {"cohort_rt": cohort_cube},
        )
        fixed = tuple(sorted(cohort_rows))
        prior_total = sum(_integer(row[1]) for row in prior_rows)
        fixed_total = sum(value for value in cohort_rows.values() if value is not None)
        changing_total = sum(_integer(row[1]) for row in independently_selected)
        # Independent oracle: a=10,b=7 before; a=20,c=99 now. b is missing, not zero.
        population_case = RuntimeCase(
            "fixed_vs_changing_population",
            (fixed_total, changing_total, fixed_total - prior_total, changing_total - prior_total),
            (20, 119, 3, 102),
            "Frozen prior membership yields current 20 and change 3; "
            "independent current membership yields 119 and change 102.",
        )
        return RuntimeCases(
            ratio_case,
            distinct_case,
            snapshot_case,
            fanout_case,
            time_case,
            population_case,
            unsafe_rejected,
            (direct_total, declared_total),
            created_check.semantic_validity.value,
            shipped_check.semantic_validity.value,
            next(
                item.status.value
                for item in cohort_check.alignment
                if item.code == "comparison_basis"
            ),
            fixed,
            tuple(sorted(cohort_rows.items())),
        )
    finally:
        db.close()


def main() -> None:
    cases = run_cases()
    for case in (
        cases.ratio_of_sums_vs_mean_of_ratios,
        cases.overlapping_distinct_reaggregation,
        cases.snapshots_across_time,
        cases.join_fanout,
        cases.created_vs_shipped_time_axis,
        cases.fixed_vs_changing_population,
    ):
        if case.observed != case.expected:
            raise AssertionError(f"{case.name}: {case.observed!r} != {case.expected!r}")
        print(f"{case.name}: {case.observed}")
    print("time-axis checks:", cases.created_axis_check, cases.shipped_axis_check)
    print("declared unsafe fanout rejected:", cases.unsafe_fanout_rejected)
    print("fixed members/measurements:", cases.cohort_members, cases.cohort_measurements)
    print("source-stage cohort alignment:", cases.cohort_comparison_check)


if __name__ == "__main__":
    main()
