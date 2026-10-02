"""Materialized merge obligations and symmetric population rendering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from semql import Cube, Dialect, Dimension, Measure, SemanticQuery, compile_federated_query
from semql.errors import ContractError
from semql.federate import FragmentColumn, MergeKeyRequirement
from semql_engine import AdapterResult, Engine
from semql_engine.merge import render_merge_sql


def _plan() -> Any:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    return compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        {"orders": cube},
    )


class _Rows:
    def __init__(self, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
        self.columns = columns
        self.rows = rows
        self.calls = 0

    def execute(self, sql: str, params: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(columns=self.columns, rows=iter(self.rows))


class _Merge:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.calls = 0

    def merge(self, fragment_results: list[AdapterResult], spec: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(columns=self.columns, rows=[])


def test_known_composite_merge_key_violation_rejects_before_custom_join() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    key_cols = ("status", "revenue")
    assert set(key_cols).issubset(cols)
    adapter = _Rows(cols, [("paid", 1), ("paid", 1)])
    merger = _Merge(list(plan.columns))
    requirement = MergeKeyRequirement(0, key_cols)
    plan = replace(plan, merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)))
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=merger)

    with pytest.raises(ContractError) as raised:
        engine.run(plan)

    assert raised.value.reason == "merge_key_uniqueness_violation"
    assert adapter.calls == 1
    assert merger.calls == 0


def test_null_composite_keys_follow_declared_null_equality() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    key_cols = ("status", "revenue")
    assert set(key_cols).issubset(cols)
    requirement = MergeKeyRequirement(0, key_cols, nulls_equal=True)
    plan = replace(plan, merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)))
    adapter = _Rows(cols, [(None, 1), (None, 1)])
    merger = _Merge(list(plan.columns))
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=merger)

    with pytest.raises(ContractError) as raised:
        engine.run(plan)

    assert raised.value.reason == "merge_key_uniqueness_violation"
    assert merger.calls == 0


def test_composite_null_keys_are_ignored_when_nulls_are_distinct() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    requirement = MergeKeyRequirement(0, ("status", "revenue"))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _Rows(cols, [(None, 1), (None, 1)])
    merger = _Merge(list(plan.columns))

    result = Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=merger).run(plan)

    assert merger.calls == 1
    assert result.validation_evidence[0].status == "validated"


def test_duckdb_nan_values_compare_equal_for_merge_key_uniqueness() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    requirement = MergeKeyRequirement(0, ("status", "revenue"))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _Rows(cols, [("paid", float("nan")), ("paid", float("nan"))])

    with pytest.raises(ContractError) as raised:
        Engine(
            adapters={Dialect.DUCKDB: adapter},
            merge_engine=_Merge(list(plan.columns)),
        ).run(plan)

    assert raised.value.reason == "merge_key_uniqueness_violation"
    assert adapter.calls == 1


def test_duckdb_integer_casts_are_applied_before_key_uniqueness_check() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    requirement = MergeKeyRequirement(0, ("status", "revenue"))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _Rows(cols, [("paid", 1), ("paid", "1")])

    with pytest.raises(ContractError) as raised:
        Engine(
            adapters={Dialect.DUCKDB: adapter},
            merge_engine=_Merge(list(plan.columns)),
        ).run(plan)

    assert raised.value.reason == "merge_key_uniqueness_violation"


def test_uncheckable_explicit_merge_key_fails_before_custom_join() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    requirement = MergeKeyRequirement(0, ("status", "revenue"))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _Rows(cols, [(["unhashable"], 1)])
    merger = _Merge(list(plan.columns))

    with pytest.raises(ContractError) as raised:
        Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=merger).run(plan)

    assert raised.value.reason == "merge_key_validation_uncheckable"
    assert merger.calls == 0


def test_no_materialized_merge_key_obligation_does_not_reject_unhashable_rows() -> None:
    plan = _plan()
    cols = list(plan.fragments[0].columns)
    adapter = _Rows(cols, [(["unhashable"], 1)])
    merger = _Merge(list(plan.columns))

    result = Engine(adapters={Dialect.DUCKDB: adapter}, merge_engine=merger).run(plan)

    assert merger.calls == 1
    assert result.validation_evidence == ()


def test_observed_fact_presence_renders_a_non_null_source_predicate() -> None:
    plan = _plan()
    spec = replace(
        plan.merge_spec,
        observed_fact_sources=(FragmentColumn(0, "status"),),
    )

    sql, _ = render_merge_sql(spec)

    assert '"f0"."status"' in sql
    assert "IS NULL" in sql or "IS NOT NULL" in sql
    assert "COUNT(" not in sql


def test_cache_retains_validation_evidence_for_the_cached_rows() -> None:
    plan = _plan()
    requirement = MergeKeyRequirement(0, ("status",))
    plan = replace(
        plan,
        merge_spec=replace(plan.merge_spec, merge_key_requirements=(requirement,)),
    )
    adapter = _Rows(list(plan.columns), [("paid", 1), ("pending", 2)])
    engine = Engine(adapters={Dialect.DUCKDB: adapter}, cache_size=2)
    first = engine.run(plan)
    cached = engine.run(plan)
    assert first.validation_evidence[0].status == "validated"
    assert cached.validation_evidence == first.validation_evidence
    assert cached.rows == [("paid", 1), ("pending", 2)]
    assert adapter.calls == 1
