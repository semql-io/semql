# pyright: reportPrivateUsage=false
from __future__ import annotations

from semql import QueryLocation, SemanticQuery, UnknownIdentifierError, validate
from semql.errors import SemQLError
from semql.spec import BoolExpr, Filter

from tests.test_validate import _cat


def test_validate_preserves_real_filter_and_where_locations_without_filter_values() -> None:
    query = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.missing", op="eq", values=["PRIVATE_FILTER_VALUE"])],
        where=BoolExpr(
            op="and",
            children=[
                Filter(dimension="orders.region", op="eq", values=["private"]),
                Filter(dimension="orders.also_missing", op="eq", values=["another secret"]),
            ],
        ),
    )

    errors = validate(query, _cat())
    unknowns = [error for error in errors if error.code == "unknown_field"]
    public_payloads = [error.to_public_payload() for error in errors]

    assert [error.location for error in unknowns] == [
        QueryLocation(section="filters", index=0),
        QueryLocation(section="where", index=1),
    ]
    rendered = str(public_payloads)
    assert "PRIVATE_FILTER_VALUE" not in rendered
    assert "another secret" not in rendered
    assert public_payloads[0]["category"] == "reference"
    assert public_payloads[0]["repair_hint"]


def test_typed_error_location_round_trips_and_public_payload_is_safe() -> None:
    original = UnknownIdentifierError(
        "private internal text with sentinel",
        kind="field",
        name="secret_field",
        location=QueryLocation(section="derived_measures", index=1, operand_index=0),
    )

    restored = SemQLError.from_payload(original.to_payload())
    public = original.to_public_payload()

    assert isinstance(restored, UnknownIdentifierError)
    assert restored.location == original.location
    assert public["location"] == {
        "section": "derived_measures",
        "index": 1,
        "operand_index": 0,
    }
    assert "private internal text" not in str(public)
    assert "secret_field" not in str(public)
    assert public["category"] == "reference"


def test_location_rejects_negative_query_indexes() -> None:
    try:
        QueryLocation(section="filters", index=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative query location index must be rejected")
