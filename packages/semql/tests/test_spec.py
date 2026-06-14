"""Unit tests for ``semql.spec``.

The spec is the contract the planner emits against. Its validators
(``_check_ungrouped_no_measures``, ``Filter.validate_for_type``) are
load-bearing: they catch shape errors that would otherwise compile to
silently-wrong SQL.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from semql.spec import CompareWindow, Filter, SemanticQuery, SemiJoin, TimeWindow

# ---------------------------------------------------------------------------
# Value-object invariants: every spec type is frozen.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        (SemanticQuery(), "limit"),
        (Filter(dimension="x.y", op="eq", values=["z"]), "dimension"),
        (TimeWindow(dimension="x.t", range=("2026-01-01", "2026-02-01")), "dimension"),
        (CompareWindow(), "mode"),
    ],
)
def test_spec_types_are_frozen(instance: object, field: str) -> None:
    """Planner specs cross trust boundaries (HTTP, MCP, queues) — mutation
    downstream is always a bug. Pin the frozen contract so a future
    refactor doesn't quietly drop it."""
    with pytest.raises(ValidationError):
        setattr(instance, field, "renamed")


# ---------------------------------------------------------------------------
# SemanticQuery._check_ungrouped_no_measures — both branches
# ---------------------------------------------------------------------------


def test_ungrouped_with_measures_rejected() -> None:
    with pytest.raises(ValidationError, match="ungrouped=True is incompatible"):
        SemanticQuery(measures=["orders.revenue"], ungrouped=True, limit=10)


def test_ungrouped_without_measures_allowed() -> None:
    q = SemanticQuery(dimensions=["orders.region"], ungrouped=True, limit=10)
    assert q.ungrouped is True
    assert q.measures == []


def test_aggregated_with_measures_allowed() -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    assert q.ungrouped is False


# ---------------------------------------------------------------------------
# Filter.validate_for_type — is_null / not_null short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["is_null", "not_null"])
def test_null_ops_skip_value_validation(op: str) -> None:
    """is_null / not_null take no values; type check is a no-op."""
    f = Filter(dimension="x", op=op, values=[])  # type: ignore[arg-type]
    # No exception for any dim type, even when values is empty.
    for dim_type in ("string", "number", "time", "bool", "uuid"):
        f.validate_for_type(dim_type)


@pytest.mark.parametrize("op", ["is_null", "not_null"])
def test_null_ops_ignore_values_if_given(op: str) -> None:
    """Even if a caller mistakenly supplies values, null ops short-circuit."""
    f = Filter(dimension="x", op=op, values=["spurious"])  # type: ignore[arg-type]
    f.validate_for_type("string")  # must not raise


# ---------------------------------------------------------------------------
# Filter.validate_for_type — empty values requires at least one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["eq", "neq", "gt", "lt", "gte", "lte", "in", "not_in", "contains"])
def test_non_null_ops_require_at_least_one_value(op: str) -> None:
    f = Filter(dimension="x", op=op, values=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires at least one value"):
        f.validate_for_type("string")


# ---------------------------------------------------------------------------
# Filter.validate_for_type — per dim_type happy + sad
# ---------------------------------------------------------------------------


def test_number_accepts_int_and_float() -> None:
    Filter(dimension="x", op="eq", values=[1]).validate_for_type("number")
    Filter(dimension="x", op="eq", values=[1.5]).validate_for_type("number")


def test_number_rejects_string() -> None:
    f = Filter(dimension="x", op="eq", values=["nope"])
    with pytest.raises(ValueError, match="non-numeric"):
        f.validate_for_type("number")


def test_number_rejects_bool() -> None:
    """``isinstance(True, int)`` is True in Python; we explicitly exclude
    bool from the numeric type to avoid ``count > True`` shenanigans."""
    f = Filter(dimension="x", op="eq", values=[True])
    with pytest.raises(ValueError, match="non-numeric"):
        f.validate_for_type("number")


def test_bool_accepts_bool() -> None:
    Filter(dimension="x", op="eq", values=[True]).validate_for_type("bool")
    Filter(dimension="x", op="eq", values=[False]).validate_for_type("bool")


def test_bool_rejects_string() -> None:
    f = Filter(dimension="x", op="eq", values=["true"])
    with pytest.raises(ValueError, match="non-bool"):
        f.validate_for_type("bool")


def test_bool_rejects_int() -> None:
    f = Filter(dimension="x", op="eq", values=[1])
    with pytest.raises(ValueError, match="non-bool"):
        f.validate_for_type("bool")


def test_time_accepts_iso_8601() -> None:
    Filter(dimension="x", op="gt", values=["2026-01-01T00:00:00"]).validate_for_type("time")
    Filter(dimension="x", op="gt", values=["2026-01-01"]).validate_for_type("time")


def test_time_rejects_non_iso_string() -> None:
    f = Filter(dimension="x", op="gt", values=["last tuesday"])
    with pytest.raises(ValueError, match="non-ISO-8601"):
        f.validate_for_type("time")


def test_time_rejects_non_string_value() -> None:
    f = Filter(dimension="x", op="gt", values=[1700000000])
    with pytest.raises(ValueError, match="non-string"):
        f.validate_for_type("time")


def test_date_accepts_iso_8601() -> None:
    Filter(dimension="x", op="gt", values=["2026-01-01"]).validate_for_type("date")
    Filter(dimension="x", op="gt", values=["2026-01-01T00:00:00"]).validate_for_type("date")


def test_date_rejects_non_iso_string() -> None:
    f = Filter(dimension="x", op="gt", values=["last tuesday"])
    with pytest.raises(ValueError, match="non-ISO-8601"):
        f.validate_for_type("date")


def test_date_rejects_non_string_value() -> None:
    f = Filter(dimension="x", op="gt", values=[20260101])
    with pytest.raises(ValueError, match="non-string"):
        f.validate_for_type("date")


def test_string_accepts_string() -> None:
    Filter(dimension="x", op="eq", values=["us"]).validate_for_type("string")


def test_string_rejects_int() -> None:
    f = Filter(dimension="x", op="eq", values=[42])
    with pytest.raises(ValueError, match="non-string"):
        f.validate_for_type("string")


def test_uuid_accepts_valid_uuid() -> None:
    Filter(
        dimension="x", op="eq", values=["550e8400-e29b-41d4-a716-446655440000"]
    ).validate_for_type("uuid")


def test_uuid_rejects_non_uuid_string() -> None:
    f = Filter(dimension="x", op="eq", values=["not-a-uuid"])
    with pytest.raises(ValueError, match="non-UUID"):
        f.validate_for_type("uuid")


def test_uuid_rejects_non_string() -> None:
    f = Filter(dimension="x", op="eq", values=[42])
    with pytest.raises(ValueError, match="non-string"):
        f.validate_for_type("uuid")


# ---------------------------------------------------------------------------
# TimeWindow — range shape
# ---------------------------------------------------------------------------


def test_timewindow_requires_two_element_range() -> None:
    with pytest.raises(ValidationError):
        TimeWindow(dimension="orders.created_at", range=("2026-01-01",))  # type: ignore[arg-type]


def test_timewindow_accepts_valid_iso_range() -> None:
    tw = TimeWindow(
        dimension="orders.created_at",
        range=("2026-01-01", "2026-02-01"),
    )
    assert tw.range == ("2026-01-01", "2026-02-01")
    assert tw.granularity is None


def test_timewindow_optional_granularity() -> None:
    tw = TimeWindow(
        dimension="orders.created_at",
        granularity="day",
        range=("2026-01-01", "2026-02-01"),
    )
    assert tw.granularity == "day"


def test_timewindow_rejects_unknown_granularity() -> None:
    with pytest.raises(ValidationError):
        TimeWindow(
            dimension="orders.created_at",
            granularity="yearly",  # type: ignore[arg-type]
            range=("2026-01-01", "2026-02-01"),
        )


# ---------------------------------------------------------------------------
# Offset / Limit
# ---------------------------------------------------------------------------


def test_offset_defaults_to_none() -> None:
    assert SemanticQuery(measures=["x.y"]).offset is None


def test_offset_accepts_zero() -> None:
    SemanticQuery(measures=["x.y"], offset=0)


def test_offset_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        SemanticQuery(measures=["x.y"], offset=-1)


# ---------------------------------------------------------------------------
# CompareWindow — default mode, explicit mode
# ---------------------------------------------------------------------------


def test_compare_window_defaults_to_previous_period() -> None:
    cw = CompareWindow()
    assert cw.mode == "previous_period"
    assert cw.range is None


def test_compare_window_explicit_with_range() -> None:
    cw = CompareWindow(mode="explicit", range=("2025-01-01", "2025-02-01"))
    assert cw.mode == "explicit"
    assert cw.range == ("2025-01-01", "2025-02-01")


def test_compare_window_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        CompareWindow(mode="rolling")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SemiJoin — cross-backend value-list semi-join node
# ---------------------------------------------------------------------------


def _sales_subquery() -> SemanticQuery:
    return SemanticQuery(
        dimensions=["employees.id"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )


def test_semijoin_defaults_to_in() -> None:
    sj = SemiJoin(dimension="activity.employee_id", select="employees.id", source=_sales_subquery())
    assert sj.op == "in"


def test_semijoin_is_frozen() -> None:
    sj = SemiJoin(dimension="activity.employee_id", select="employees.id", source=_sales_subquery())
    with pytest.raises(ValidationError):
        sj.op = "not_in"


def test_semijoin_requires_qualified_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be a qualified"):
        SemiJoin(dimension="employee_id", select="employees.id", source=_sales_subquery())


def test_semijoin_requires_qualified_select() -> None:
    with pytest.raises(ValueError, match="select must be a qualified"):
        SemiJoin(dimension="activity.employee_id", select="id", source=_sales_subquery())


def test_semijoin_select_must_be_projected_by_source() -> None:
    with pytest.raises(ValueError, match="must be one of the inner query's dimensions"):
        SemiJoin(
            dimension="activity.employee_id",
            select="employees.id",
            source=SemanticQuery(dimensions=["employees.name"]),
        )


def test_semijoin_rejects_nested_semi_joins() -> None:
    inner = SemanticQuery(
        dimensions=["employees.id"],
        semi_joins=[
            SemiJoin(
                dimension="employees.id",
                select="teams.lead_id",
                source=SemanticQuery(dimensions=["teams.lead_id"]),
            )
        ],
    )
    with pytest.raises(ValueError, match="must not itself contain semi_joins"):
        SemiJoin(dimension="activity.employee_id", select="employees.id", source=inner)


def test_semijoin_only_in_or_not_in() -> None:
    with pytest.raises(ValidationError):
        SemiJoin(
            dimension="activity.employee_id",
            op="eq",  # type: ignore[arg-type]
            select="employees.id",
            source=_sales_subquery(),
        )


def test_semanticquery_semi_joins_default_empty() -> None:
    assert SemanticQuery(measures=["activity.active_secs"]).semi_joins == []


# ---------------------------------------------------------------------------
# JSON Schema descriptions — every property the LLM tool-calling
# layer surfaces must carry a useful description, or the model has to guess
# what each field means.
# ---------------------------------------------------------------------------


def _missing_descriptions(model_cls: type[BaseModel]) -> list[str]:
    """Return property names whose JSON Schema entry has no description.

    Walks ``model_json_schema()`` and complains about any property
    missing a ``description`` key OR carrying an empty string."""
    schema = model_cls.model_json_schema()
    properties = schema.get("properties", {})
    return [
        name for name, prop in properties.items() if not (prop.get("description") or "").strip()
    ]


def test_tool_json_schema_is_object_rooted_despite_recursion() -> None:
    """``SemanticQuery`` is recursive (``semi_joins[].source`` is itself a
    ``SemanticQuery``), so ``model_json_schema()`` emits a root ``$ref``.
    ``tool_json_schema`` must flatten it to an object root for tool-calling
    APIs while keeping ``$defs`` so the self-reference still resolves."""
    raw = SemanticQuery.model_json_schema()
    assert "$ref" in raw  # recursion makes the raw schema root-ref'd

    schema = SemanticQuery.tool_json_schema()
    assert schema.get("type") == "object"
    assert "$ref" not in schema
    assert "semi_joins" in schema["properties"]
    assert "SemanticQuery" in schema["$defs"]
    assert schema["$defs"]["SemiJoin"]["properties"]["source"]["$ref"] == "#/$defs/SemanticQuery"


@pytest.mark.parametrize(
    "model_cls",
    [SemanticQuery, Filter, TimeWindow, CompareWindow],
)
def test_spec_schema_property_descriptions_are_set(model_cls: type[BaseModel]) -> None:
    """Tool-calling frameworks (OpenAI SDK, LangChain, pydantic-ai) hand
    ``model_json_schema()`` to the LLM verbatim. Missing descriptions
    force the model to infer field meaning from names alone, which
    produces malformed filter values."""
    missing = _missing_descriptions(model_cls)
    assert not missing, f"{model_cls.__name__} fields without description: {missing}"


def test_boolexpr_schema_property_descriptions_are_set() -> None:
    """BoolExpr is recursive (children include itself), so pydantic
    emits a $defs entry — walk both top-level and $defs to confirm."""
    from semql.spec import BoolExpr

    missing = _missing_descriptions(BoolExpr)
    assert not missing, f"BoolExpr fields without description: {missing}"


def test_inline_derived_schema_property_descriptions_are_set() -> None:
    from semql.spec import InlineDerived

    missing = _missing_descriptions(InlineDerived)
    assert not missing, f"InlineDerived fields without description: {missing}"


def test_spec_schema_descriptions_are_under_120_chars() -> None:
    """Tool-schema payloads inflate quickly; cap each description so the
    aggregate stays cache-friendly. 120 chars ≈ one wrapped terminal
    line — enough for one self-contained sentence, not enough for an
    essay."""
    for model_cls in (
        SemanticQuery,
        Filter,
        TimeWindow,
        CompareWindow,
    ):
        props = model_cls.model_json_schema().get("properties", {})
        for name, prop in props.items():
            desc = prop.get("description", "")
            assert len(desc) <= 120, (
                f"{model_cls.__name__}.{name} description is {len(desc)} chars; cap is 120."
            )
