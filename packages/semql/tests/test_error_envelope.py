# This module tests internal APIs by design (e.g. _enrich_filter_type_error),
# so cross-class private access is expected.
# pyright: reportPrivateUsage=false
"""Tests for the uniform error envelope.

Errors serve machines and humans (PHILOSOPHY.md). ``to_payload()`` on
every :class:`SemQLError` leaf returns a structured ``dict`` so machine
consumers (MCP, LLM retry loops, API layers) can branch on failure mode
without parsing ``str(err)``.

Contract:

- Every leaf has a stable ``code`` (``class name`` by default; leaves
  may override).
- ``to_payload()`` returns ``{"code", "message", ...}`` where the
  trailing keys are the leaf's structured attrs (kind, name, cube,
  hint, known, backends, ...).
- The base :meth:`SemQLError.to_payload` provides the default
  implementation; leaves extend by adding their attrs.
- ``did_you_mean`` / ``valid_alternatives`` is the LLM-friendly spelling
  of ``hint`` (``UnknownIdentifierError`` already produces one).
- ``next_tool`` affordance: ``FilterTypeError`` for a string dim with a
  registered Lookup carries the lookup tool name and args so the LLM
  can repair without a free-text guess.

A round-trip is the discriminator: ``SemQLError.from_payload(payload)``
rebuilds the same exception class (so the structured attrs survive
across a process boundary — relevant to semql-mcp and HTTP transport).
"""

from __future__ import annotations

import pytest
from semql.catalog import Catalog
from semql.compile import compile_query
from semql.errors import (
    AuthError,
    ContractError,
    CrossDialectError,
    Diagnostic,
    FederationError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    SemQLError,
    UnknownIdentifierError,
    closest_match,
)
from semql.model import Cube, Dialect, Dimension, Lookup, Measure, TimeDimension
from semql.spec import Filter, SemanticQuery

# ---------------------------------------------------------------------------
# base contract
# ---------------------------------------------------------------------------


def test_to_payload_default_on_base() -> None:
    err = SemQLError("boom")
    payload = err.to_payload()
    assert payload == {"code": "SemQLError", "message": "boom"}


def test_to_payload_includes_message_and_code() -> None:
    err = ResolveError("nope")
    payload = err.to_payload()
    assert payload["code"] == "ResolveError"
    assert payload["message"] == "nope"


def test_to_payload_is_json_safe() -> None:
    """Every payload value must round-trip through ``json.dumps`` — MCP
    / API layers serialise the envelope verbatim. Lists, dicts, strings,
    bools, None, ints; no exception instances, no callables."""
    import json

    err = UnknownIdentifierError(
        "Unknown field 'revnue' on cube 'orders'.",
        kind="field",
        name="revnue",
        cube="orders",
        hint="revenue",
    )
    json.dumps(err.to_payload())


# ---------------------------------------------------------------------------
# UnknownIdentifierError — the canonical LLM-repair case
# ---------------------------------------------------------------------------


def test_unknown_identifier_payload_shape(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.reveune"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    payload = exc_info.value.to_payload()
    assert payload["code"] == "UnknownIdentifierError"
    assert payload["kind"] == "field"
    assert payload["name"] == "reveune"
    assert payload["cube"] == "orders"
    assert payload["hint"] == "revenue"


def test_unknown_identifier_payload_includes_valid_alternatives(
    catalog: dict[str, Cube],
) -> None:
    """``valid_alternatives`` is the LLM-friendly spelling of
    ``closest_match`` output. The base envelope omits the key when the
    leaf has nothing to suggest."""
    q = SemanticQuery(measures=["orders.reveune"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    payload = exc_info.value.to_payload()
    # ``revenue`` is the closest match (lowest ratio above cutoff);
    # the list may also include next-best candidates.
    assert "valid_alternatives" in payload
    assert "revenue" in payload["valid_alternatives"]
    assert payload["valid_alternatives"][0] == "revenue"


def test_unknown_identifier_payload_omits_alternatives_when_far(
    catalog: dict[str, Cube],
) -> None:
    q = SemanticQuery(measures=["orders.zzznotathing"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    payload = exc_info.value.to_payload()
    assert "valid_alternatives" not in payload or payload["valid_alternatives"] == []


# ---------------------------------------------------------------------------
# Other leaves — each carries its own structured attrs
# ---------------------------------------------------------------------------


def test_join_path_error_payload_carries_cube_names() -> None:
    a = Cube(
        name="a",
        dialect=Dialect.POSTGRES,
        table="a",
        alias="a",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
    )
    b = Cube(
        name="b",
        dialect=Dialect.POSTGRES,
        table="b",
        alias="b",
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    cat = {"a": a, "b": b}
    q = SemanticQuery(measures=["a.count"], dimensions=["b.x"])
    with pytest.raises(JoinPathError) as exc_info:
        compile_query(q, cat)
    payload = exc_info.value.to_payload()
    assert payload["code"] == "JoinPathError"
    assert payload["root_cube"] == "a"
    assert payload["target_cube"] == "b"


def test_filter_type_error_payload_carries_dimension_op() -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    cat = {
        "orders": Cube(
            name="orders",
            dialect=Dialect.POSTGRES,
            table="orders",
            alias="o",
            measures=[Measure(name="count", sql="*", agg="count", unit="count")],
            dimensions=[Dimension(name="is_paid", sql="{o}.is_paid", type="bool")],
        )
    }
    with pytest.raises(FilterTypeError) as exc_info:
        compile_query(q, cat, context={"schema": "test_schema"})
    payload = exc_info.value.to_payload()
    assert payload["code"] == "FilterTypeError"
    assert payload["dimension"] == "orders.is_paid"
    assert payload["op"] == "eq"
    # ``value`` is the offending literal; the compiler flattens the
    # scalar Filter value (``"yes"``) into the payload as a scalar
    # rather than the list the user passed.
    assert payload["value"] == "yes"


def test_placeholder_error_payload_carries_placeholder_and_known() -> None:
    bad = Cube(
        name="bad",
        dialect=Dialect.POSTGRES,
        table="{nope}.bad",
        alias="b",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    cat = {"bad": bad}
    with pytest.raises(PlaceholderError) as exc_info:
        compile_query(SemanticQuery(measures=["bad.count"]), cat)
    payload = exc_info.value.to_payload()
    assert payload["code"] == "PlaceholderError"
    assert payload["placeholder"] == "nope"
    assert isinstance(payload["known"], list)


def test_cross_backend_error_payload_carries_backends(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue", "sessions.duration"])
    with pytest.raises(CrossDialectError) as exc_info:
        compile_query(q, catalog)
    payload = exc_info.value.to_payload()
    assert payload["code"] == "CrossDialectError"
    assert sorted(payload["backends"]) == ["clickhouse", "postgres"]


def test_phase_deferred_error_payload_carries_feature() -> None:
    err = PhaseDeferredError("compare deferred", feature="compare")
    assert err.to_payload() == {
        "code": "PhaseDeferredError",
        "message": "compare deferred",
        "feature": "compare",
    }


def test_federation_error_payload_carries_reason() -> None:
    err = FederationError("ratio distributive", reason="non_distributive_agg")
    assert err.to_payload() == {
        "code": "FederationError",
        "message": "ratio distributive",
        "reason": "non_distributive_agg",
    }


def test_auth_error_payload_carries_reason() -> None:
    err = AuthError("bad token", reason="expired")
    assert err.to_payload() == {
        "code": "AuthError",
        "message": "bad token",
        "reason": "expired",
    }


def test_auth_error_reason_optional() -> None:
    err = AuthError("rejected")
    payload = err.to_payload()
    assert payload["code"] == "AuthError"
    assert payload["message"] == "rejected"
    # Optional — AuthError.reason defaults to None
    assert payload.get("reason") is None


# ---------------------------------------------------------------------------
# Round-trip: from_payload reconstructs the same class
# ---------------------------------------------------------------------------


def test_from_payload_round_trip_unknown_identifier() -> None:
    original = UnknownIdentifierError(
        "Unknown field 'revnue' on cube 'orders'.",
        kind="field",
        name="revnue",
        cube="orders",
        hint="revenue",
    )
    payload = original.to_payload()
    rebuilt = SemQLError.from_payload(payload)
    assert isinstance(rebuilt, UnknownIdentifierError)
    assert rebuilt.kind == "field"
    assert rebuilt.name == "revnue"
    assert rebuilt.cube == "orders"
    assert rebuilt.hint == "revenue"
    assert str(rebuilt) == str(original)


def test_from_payload_round_trip_join_path() -> None:
    original = JoinPathError("no path", root_cube="a", target_cube="b")
    rebuilt = SemQLError.from_payload(original.to_payload())
    assert isinstance(rebuilt, JoinPathError)
    assert rebuilt.root_cube == "a"
    assert rebuilt.target_cube == "b"


def test_from_payload_round_trip_filter_type() -> None:
    original = FilterTypeError("type mismatch", dimension="orders.is_paid", op="eq", value=["yes"])
    rebuilt = SemQLError.from_payload(original.to_payload())
    assert isinstance(rebuilt, FilterTypeError)
    assert rebuilt.dimension == "orders.is_paid"
    assert rebuilt.op == "eq"
    assert rebuilt.value == ["yes"]


def test_from_payload_unknown_code_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown error code"):
        SemQLError.from_payload({"code": "TotallyMadeUpError", "message": "x"})


def test_from_payload_dispatches_by_code() -> None:
    """The dispatch table is keyed on ``code`` (== class name) and is
    exhaustive over the public leaves. If a new leaf is added without
    registering here the round-trip fails with a clear ValueError —
    the failure mode is loud, not silent."""
    for cls in (
        UnknownIdentifierError,
        JoinPathError,
        FilterTypeError,
        PlaceholderError,
        CrossDialectError,
        PhaseDeferredError,
        FederationError,
        AuthError,
    ):
        # PhaseDeferredError / FederationError / AuthError have no
        # required attrs beyond the message — round-trip is a no-op.
        err: SemQLError
        if cls is PhaseDeferredError:
            err = cls("x", feature="f")
        elif cls is FederationError:
            err = cls("x", reason="r")
        elif cls is AuthError:
            err = cls("x")
        else:
            continue
        rebuilt = SemQLError.from_payload(err.to_payload())
        assert type(rebuilt) is cls


# ---------------------------------------------------------------------------
# Hierarchy: every public leaf exposes to_payload
# ---------------------------------------------------------------------------


def test_all_public_leaves_implement_to_payload() -> None:
    """Defensive: future leaves must extend the contract, not skip it."""
    for cls in (
        UnknownIdentifierError,
        JoinPathError,
        FilterTypeError,
        PlaceholderError,
        CrossDialectError,
        PhaseDeferredError,
        FederationError,
        AuthError,
    ):
        assert hasattr(cls, "to_payload")
        # Round-trip the leaf-specific shape
        instance: SemQLError
        if cls is AuthError:
            instance = cls("x")
        elif cls is PhaseDeferredError:
            instance = cls("x", feature="f")
        elif cls is FederationError:
            instance = cls("x", reason="r")
        elif cls is UnknownIdentifierError:
            instance = cls("x", kind="field", name="n")
        elif cls is JoinPathError:
            instance = cls("x", root_cube="a", target_cube="b")
        elif cls is FilterTypeError:
            instance = cls("x", dimension="d", op="eq")
        elif cls is PlaceholderError:
            instance = cls("x", placeholder="p")
        elif cls is CrossDialectError:
            instance = cls("x", backends=["postgres"])
        else:  # pragma: no cover — defensive; covers future leaves
            continue
        payload = instance.to_payload()
        assert "code" in payload
        assert payload["code"] == cls.__name__


# ---------------------------------------------------------------------------
# next_tool affordance — the lookup-repair case
# ---------------------------------------------------------------------------


def test_filter_type_error_payload_includes_next_tool_for_lookup_dim() -> None:
    """When a ``FilterTypeError`` fires for a dim that has a
    registered ``Lookup``, the envelope enrichment in
    :meth:`Catalog.compile` should attach ``next_tool`` /
    ``next_tool_args`` / ``did_you_mean`` so the LLM can call the
    lookup tool to resolve free-text to canonical.

    We invoke the enrichment directly with a synthetic
    ``FilterTypeError`` — the actual lookup-membership check that
    *raises* such an error is its own commit (separate concern)."""
    from semql.catalog import Catalog
    from semql.errors import FilterTypeError

    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog(
        [orders],
        lookups=[Lookup(dimension="orders.region", values=("EMEA", "APAC", "NA"))],
    )
    err = FilterTypeError(
        "value not in set",
        dimension="orders.region",
        op="eq",
        value="emea",
    )
    enriched = cat._enrich_filter_type_error(err)
    assert enriched is not err
    payload = enriched.to_payload()
    assert payload.get("next_tool") == "resolve_lookup"
    assert payload["next_tool_args"] == {
        "dimension": "orders.region",
        "query": "emea",
    }
    assert "EMEA" in payload.get("did_you_mean", [])


def test_filter_type_error_payload_omits_next_tool_when_no_lookup() -> None:
    """Type mismatches on dims that have no Lookup carry no repair
    affordance — the LLM must reason from the field's type."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="is_paid", sql="{o}.is_paid", type="bool")],
    )
    cat = Catalog([orders])
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    with pytest.raises(FilterTypeError) as exc_info:
        cat.compile(q)
    payload = exc_info.value.to_payload()
    assert "next_tool" not in payload
    assert "did_you_mean" not in payload


# ---------------------------------------------------------------------------
# Collect-all on the query path
# ---------------------------------------------------------------------------


def test_collect_all_on_query_path_returns_validation_records() -> None:
    """When a query has *several* problems (a known cube with
    missing required filters + a non-ISO time-window value) the
    error envelope carries the full list — the LLM repairs in one
    round-trip, not N.

    Implemented by ``Catalog.compile_collect_all`` (returns the
    ``CompiledQuery`` on success or a list of ``ValidationError`` on
    failure; never raises for query-shape problems)."""
    from semql.spec import TimeWindow
    from semql.validate import ValidationError

    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        required_filters=["region"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at"),
        ],
    )
    cat = Catalog([orders])
    q = SemanticQuery(
        measures=["orders.count"],
        time_dimension=TimeWindow(dimension="orders.created_at", range=("not-a-date", "also-bad")),
    )
    errors = cat.compile_collect_all(q)
    assert isinstance(errors, list)
    assert all(isinstance(e, ValidationError) for e in errors)
    codes = {e.code for e in errors}
    assert "missing_required_filter" in codes


def test_collect_all_returns_empty_list_on_success() -> None:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([orders])
    q = SemanticQuery(measures=["orders.count"])
    assert cat.compile_collect_all(q) == []


# ---------------------------------------------------------------------------
# closest_match helper survives
# ---------------------------------------------------------------------------


def test_closest_match_still_callable() -> None:
    assert closest_match("regin", ["region", "status"]) == "region"
    assert closest_match("zzz", ["region"]) is None


def test_contract_error_safe_payload_and_lossless_internal_round_trip() -> None:
    error = ContractError(
        "SQL SECRET_SQL and bound SECRET_BIND repair SECRET_REPAIR",
        reason="binding_missing",
        artifact="fragment-2",
        names=("expected", "actual"),
        references=("orders.revenue",),
        operation="execute",
        stage="binding",
    )
    rebuilt = SemQLError.from_payload(error.to_payload())
    assert isinstance(rebuilt, ContractError)
    assert rebuilt.to_payload() == error.to_payload()
    public = error.to_public_payload()
    rendered = str(public)
    for secret in ("SECRET_SQL", "SECRET_BIND", "SECRET_REPAIR", "fragment-2", "expected"):
        assert secret not in rendered
    assert public["reason"] == "binding_missing"
    assert public["references"] == ["orders.revenue"]


def test_diagnostic_rejects_unbounded_reason_and_keeps_advisory_severity() -> None:
    diagnostic = Diagnostic(
        code="unsafe_fanout",
        severity="advisory",
        reason="private sql fragment!",
    )
    public = diagnostic.to_public_payload()
    assert public["reason"] == "contract_violation"
    assert public["severity"] == "advisory"


def test_filter_repair_payload_remains_lossless_but_public_rendering_is_safe() -> None:
    error = FilterTypeError(
        "bad SECRET_MESSAGE SQL SECRET_SQL",
        dimension="orders.region",
        op="eq",
        value="SECRET_BOUND",
        next_tool="resolve_lookup",
        next_tool_args={"query": "SECRET_REPAIR"},
    )
    rebuilt = SemQLError.from_payload(error.to_payload())
    assert isinstance(rebuilt, FilterTypeError)
    assert rebuilt.to_payload() == error.to_payload()
    public = str(error.to_public_payload())
    for secret in ("SECRET_MESSAGE", "SECRET_SQL", "SECRET_BOUND", "SECRET_REPAIR"):
        assert secret not in public


def test_public_diagnostics_distinguish_compile_resolution_and_federation() -> None:
    from semql.errors import CompileError, FederationError, UnknownIdentifierError

    assert CompileError("private SQL").to_public_payload()["reason"] == "invalid_query"
    unknown = UnknownIdentifierError("private query", kind="field", name="missing")
    assert unknown.to_public_payload()["reason"] == "invalid_reference"
    unsupported = FederationError("private SQL", reason="inline_derived_federation_unsupported")
    public = unsupported.to_public_payload()
    assert public["reason"] == "inline_derived_federation_unsupported"
    assert public["stage"] == "lowering"
    assert "private" not in str(public)
