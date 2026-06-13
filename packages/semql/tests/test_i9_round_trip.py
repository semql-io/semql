# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""I9 — CompiledQuery ↔ Pydantic round-trip.

Asserts that a CompiledQuery can be serialised to a JSON-safe dict (or
JSON string) and reconstructed back, preserving the byte-level fidelity
a tool-call / eval-loop caller needs.

The MCP ``query_semantic`` tool consumes a SemanticQuery JSON payload;
a verifier wrapping the round-trip wants the same guarantee for the
*compiled* form so it can assert "did the planner's emitted query
deserialise back to what we sent it?" without losing type info.
"""

from __future__ import annotations

import json

from semql.compile import CompiledQuery, compile_query
from semql.model import Dialect
from semql.spec import CompareWindow, Filter, SemanticQuery, TimeWindow

from .conftest import CONTEXT


def _compile(catalog: dict, q: SemanticQuery) -> CompiledQuery:
    return compile_query(q, catalog, context=CONTEXT)


def test_compiled_query_round_trip_through_dict(catalog: dict) -> None:
    """CompiledQuery.model_validate(model_dump(...)) preserves every field."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    cq = _compile(catalog, q)
    restored = CompiledQuery.model_validate(cq.model_dump())
    assert restored.sql == cq.sql
    assert restored.params == cq.params
    assert restored.columns == cq.columns
    assert restored.backend == cq.backend
    assert restored.touched_cube_names == cq.touched_cube_names


def test_compiled_query_round_trip_through_json(catalog: dict) -> None:
    """JSON-string round-trip survives str-enum and Literal fields."""
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.status"],
        filters=[Filter(dimension="orders.region", op="eq", values=["emea"])],
        order=[("orders.count", "desc")],
    )
    cq = _compile(catalog, q)
    raw = json.dumps(cq.model_dump())
    restored = CompiledQuery.model_validate(json.loads(raw))
    assert restored.sql == cq.sql
    assert restored.params == cq.params
    assert restored.columns == cq.columns
    assert restored.backend is Dialect.POSTGRES
    assert restored.column_meta == cq.column_meta


def test_compiled_query_round_trip_with_compare(catalog: dict) -> None:
    """Compare-mode derivatives (delta, pct_change) survive round-trip."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-03-31"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    cq = _compile(catalog, q)
    restored = CompiledQuery.model_validate(cq.model_dump())
    # Compare-mode emits delta + pct_change columns; preserve them.
    assert restored.columns == cq.columns
    assert restored.sql == cq.sql


def test_compiled_query_round_trip_preserves_int_param(catalog: dict) -> None:
    """Param values remain their original Python type after round-trip."""
    # A filter with an int value should keep that int after JSON round-trip.
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.amount", op="gt", values=[7])],
    )
    cq = _compile(catalog, q)
    raw = json.dumps(cq.model_dump())
    CompiledQuery.model_validate(json.loads(raw))
    # At least one int bound should be in the params dict, not stringified.
    int_param = next(
        (v for v in cq.params.values() if isinstance(v, int) and v == 7),
        None,
    )
    assert int_param == 7


def test_compiled_query_column_meta_preserved(catalog: dict) -> None:
    """ColumnMeta (kind, format, display_name) round-trips intact."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    cq = _compile(catalog, q)
    raw = json.dumps(cq.model_dump())
    restored = CompiledQuery.model_validate(json.loads(raw))
    assert len(restored.column_meta) == len(cq.column_meta)
    for orig, rest in zip(cq.column_meta, restored.column_meta, strict=True):
        assert orig.name == rest.name
        assert orig.kind == rest.kind
        assert orig.format == rest.format
        assert orig.unit == rest.unit
