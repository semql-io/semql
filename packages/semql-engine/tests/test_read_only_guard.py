"""The engine refuses to execute anything but read-only SELECTs.

Defense-in-depth: the compiler emits SELECT by construction, but RawSQL
escape hatches (DerivedTable.sql, with_ctes, security_sql,
ScopePredicate.sql) splice author-controlled strings into the emitted
fragments. The engine re-checks every fragment + derived source + the
merge SQL at the execution choke point and raises EngineError on
anything non-SELECT — before a single byte reaches a driver.
"""

from __future__ import annotations

# Exercises the engine's module-private guard on purpose.
# pyright: reportPrivateUsage=false
from collections.abc import Mapping

import pytest
from semql.compile import ColumnMeta, CompiledQuery
from semql.federate import FederatedPlan, MergeSpec
from semql.model import Dialect
from semql_engine import AdapterResult, Engine
from semql_engine.engine import (
    EngineError,
    _assert_fragments_read_only,
    _assert_merge_read_only,
)


def _frag(sql: str, *, derived: list[str] | None = None) -> CompiledQuery:
    return CompiledQuery(
        dialect=Dialect.POSTGRES,
        sql=sql,
        params={},
        columns=["x"],
        column_meta=[ColumnMeta(name="x", kind="dimension", display_name="x")],
        derived_sources=derived or [],
    )


def _plan(fragment: CompiledQuery) -> FederatedPlan:
    spec = MergeSpec(
        primary_index=0,
        bridges=[],
        dimensions=[],
        measures=[],
        having=[],
        order_by=[],
        limit=None,
        offset=None,
        mode="distributive",
    )
    return FederatedPlan(
        fragments=[fragment],
        merge_spec=spec,
        columns=["x"],
        column_meta=[ColumnMeta(name="x", kind="dimension", display_name="x")],
    )


def test_clean_select_plan_passes() -> None:
    plan = _plan(_frag("SELECT x FROM t"))
    _assert_fragments_read_only(plan)
    _assert_merge_read_only("SELECT * FROM frag_0")


def test_non_select_fragment_refused() -> None:
    with pytest.raises(EngineError, match="read-only"):
        _assert_fragments_read_only(_plan(_frag("DROP TABLE t")))


def test_non_select_derived_source_refused() -> None:
    with pytest.raises(EngineError, match="read-only"):
        _assert_fragments_read_only(_plan(_frag("SELECT x FROM t", derived=["DELETE FROM audit"])))


def test_non_select_merge_refused() -> None:
    with pytest.raises(EngineError, match="Merge SQL"):
        _assert_merge_read_only("DROP TABLE frag_0")


def test_engine_run_refuses_non_select_before_touching_adapter() -> None:
    """The guard fires in run() before the adapter is ever called."""
    calls: list[str] = []

    class _Adapter:
        def execute(self, sql: str, params: Mapping[str, object]) -> AdapterResult:
            calls.append(sql)
            return AdapterResult(columns=["x"], rows=[])

    engine = Engine()
    engine.register(Dialect.POSTGRES, _Adapter())
    with pytest.raises(EngineError):
        engine.run(_plan(_frag("TRUNCATE t")))
    assert calls == []  # adapter never touched


def test_writable_cte_fragment_refused() -> None:
    """SELECT root with writable CTE body must be rejected (SEMQL-READONLY-WRITABLE-CTE)."""
    with pytest.raises(EngineError, match="read-only"):
        _assert_fragments_read_only(
            _plan(_frag("WITH d AS (DELETE FROM users RETURNING id) SELECT * FROM d"))
        )


# ---------------------------------------------------------------------------
# _quote escapes embedded double-quotes (CAND-semql-engine-alias-ddl-injection)
# ---------------------------------------------------------------------------


def test_quote_escapes_embedded_double_quote() -> None:
    from semql_engine.engine import _quote

    safe = _quote('col"name')
    # Must not produce a bare unescaped " that breaks out of the identifier.
    assert safe == '"col""name"'


def test_quote_ordinary_name_unchanged() -> None:
    from semql_engine.engine import _quote

    assert _quote("revenue") == '"revenue"'
