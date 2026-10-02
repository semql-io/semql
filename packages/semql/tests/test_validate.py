"""Tests for the collect-all static validator.

`compile()` fails at the first problem; `validate()` collects them all.
Two tools, two contracts (PHILOSOPHY.md).
"""

from __future__ import annotations

from semql import (
    BoolExpr,
    Catalog,
    Cube,
    Dimension,
    Filter,
    InlineDerived,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from semql.model import Dialect, Segment
from semql.validate import ValidationError, validate


def _cat() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="is_paid", sql="{o}.is_paid", type="bool"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
        required_filters=[],
    )
    restricted = Cube(
        name="restricted",
        dialect=Dialect.POSTGRES,
        table="restricted",
        alias="r",
        expose_in_prompt=False,
        required_filters=["flag_type"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="flag_type", sql="{r}.flag_type", type="string")],
    )
    return Catalog([orders, restricted])


def test_valid_query_returns_empty_list() -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    assert validate(q, _cat()) == []


def test_returns_list_never_raises() -> None:
    """Even on catastrophically broken input, validate() must not raise."""
    q = SemanticQuery(
        measures=["nope.boom", "also_nope.thing"],
        dimensions=["nope.x"],
        having=[Filter(dimension="not_a_measure", op="gt", values=[1])],
    )
    # Must not raise.
    errors = validate(q, _cat())
    assert isinstance(errors, list)
    assert all(isinstance(e, ValidationError) for e in errors)


def test_collects_multiple_errors_in_one_pass() -> None:
    q = SemanticQuery(
        measures=["orders.no_such_measure"],
        dimensions=["nope.x"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    errors = validate(q, _cat())
    codes = {e.code for e in errors}
    assert "unknown_field" in codes
    assert "unknown_cube" in codes
    assert "filter_type_mismatch" in codes


def test_required_filter_missing_reported() -> None:
    q = SemanticQuery(measures=["restricted.count"])
    errors = validate(q, _cat())
    codes = [e.code for e in errors]
    assert "missing_required_filter" in codes


def test_granularity_not_allowed_reported() -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="hour",
            range=("2026-01-01", "2026-01-02"),
        ),
    )
    errors = validate(q, _cat())
    codes = [e.code for e in errors]
    assert "bad_granularity" in codes


def test_empty_query_reported() -> None:
    q = SemanticQuery()
    errors = validate(q, _cat())
    codes = [e.code for e in errors]
    assert "empty_query" in codes


def test_ungrouped_without_limit_reported() -> None:
    q = SemanticQuery(dimensions=["orders.region"], ungrouped=True)
    errors = validate(q, _cat())
    codes = [e.code for e in errors]
    assert "ungrouped_no_limit" in codes


def test_having_on_unknown_measure_reported() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        having=[Filter(dimension="orders.profit", op="gt", values=[100])],
    )
    errors = validate(q, _cat())
    codes = [e.code for e in errors]
    assert "having_unknown_measure" in codes


def test_having_on_qualified_measure_in_query_passes() -> None:
    """Mirrors the compile path: qualified HAVING refs resolve to bare."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        having=[Filter(dimension="orders.revenue", op="gt", values=[100])],
    )
    assert validate(q, _cat()) == []


def test_unknown_field_carries_hint() -> None:
    q = SemanticQuery(measures=["orders.reveune"])
    errors = validate(q, _cat())
    err = next(e for e in errors if e.code == "unknown_field")
    assert err.hint == "revenue"


def test_validate_accepts_dict_catalog() -> None:
    """validate() should accept either a Catalog or the dict form."""
    q = SemanticQuery(measures=["orders.revenue"])
    cat = _cat()
    via_obj = validate(q, cat)
    via_dict = validate(q, cat.as_dict())
    assert via_obj == via_dict


# ---------------------------------------------------------------------------
# Shared-walker coverage: validate now follows where-tree leaves and
# segment references through the same resolver compile uses. Pins the
# expanded coverage so a future refactor that drops them surfaces here.
# ---------------------------------------------------------------------------


def test_where_tree_unknown_field_reported() -> None:
    """``where`` leaves are resolved by the shared walker — an unknown
    identifier inside the tree must surface, not pass silently."""
    q = SemanticQuery(
        measures=["orders.count"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.nope", op="eq", values=["x"]),
            ],
        ),
    )
    errors = validate(q, _cat())
    codes = {e.code for e in errors}
    assert "unknown_field" in codes


def test_where_tree_type_mismatch_reported() -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        where=BoolExpr(
            op="and",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.is_paid", op="eq", values=["yes"]),
            ],
        ),
    )
    errors = validate(q, _cat())
    codes = {e.code for e in errors}
    assert "filter_type_mismatch" in codes


def test_segment_unknown_cube_reported() -> None:
    q = SemanticQuery(measures=["orders.count"], segments=["nope.active"])
    errors = validate(q, _cat())
    codes = {e.code for e in errors}
    assert "segment_unknown_cube" in codes


def test_segment_unknown_segment_reported() -> None:
    cat = _cat()
    # Inject a segment on the orders cube via a fresh Catalog.
    orders = cat.as_dict()["orders"].model_copy(
        update={"segments": [Segment(name="paid", sql="o.is_paid = true")]}
    )
    cubes = list(cat.as_dict().values())
    cubes = [orders if c.name == "orders" else c for c in cubes]
    q = SemanticQuery(measures=["orders.count"], segments=["orders.does_not_exist"])
    errors = validate(q, Catalog(cubes))
    codes = {e.code for e in errors}
    assert "segment_unknown_segment" in codes


def test_segment_unqualified_reported() -> None:
    q = SemanticQuery(measures=["orders.count"], segments=["bareword"])
    errors = validate(q, _cat())
    codes = {e.code for e in errors}
    assert "segment_unqualified" in codes


def test_validation_uses_shared_capability_decisions() -> None:
    from semql.capabilities import check_query_capabilities
    from semql.errors import ContractError

    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-01-03"),
            fill_nulls_with=0,
        ),
    )
    expected = check_query_capabilities(query)
    errors = validate(query, _cat())
    capability = next(error for error in errors if error.code == "capability_unsupported")
    assert capability.reason == expected[0].reason == "entity_time_fill_unsupported"
    assert capability.references == expected[0].references
    from semql.capabilities import require_query_capabilities

    try:
        require_query_capabilities(query)
    except ContractError as error:
        assert error.reason == capability.reason
        assert error.references == capability.references
    else:
        raise AssertionError("lowering capability check should reject entity-time fill")


def test_scalar_derived_only_has_own_capability_reason() -> None:
    query = SemanticQuery(
        derived_measures=[
            InlineDerived(name="ratio", op="ratio", operands=["orders.revenue", "orders.count"])
        ]
    )
    errors = validate(query, _cat())
    assert any(
        error.reason == "scalar_derived_only_unsupported" and error.code == "capability_unsupported"
        for error in errors
    )
    assert not any(error.code == "empty_query" for error in errors)


def test_grouped_derived_only_is_not_the_scalar_shape() -> None:
    from semql.capabilities import check_query_capabilities

    query = SemanticQuery(
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="ratio", op="ratio", operands=["orders.revenue", "orders.count"])
        ],
    )
    assert "scalar_derived_only_unsupported" not in {
        diagnostic.reason for diagnostic in check_query_capabilities(query)
    }
