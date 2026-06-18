"""Unit tests for ``semql._resolve``.

The resolver is the single source of truth for parsing ``cube.field``
references against the catalog. Both ``compile.py`` and
``visualize.py`` depend on its contract:

- The regex permits only ``[a-z_][a-z0-9_]*`` segments (case-insensitive).
- Unknown identifiers raise ``UnknownIdentifierError`` with structured
  attrs (``kind``, ``name``, ``cube``, ``hint``).
- The error message lists the known candidates so the reader can pick
  one without re-grepping the catalog.
"""

from __future__ import annotations

import pytest
from semql._resolve import resolve_field, split
from semql.errors import ResolveError, UnknownIdentifierError
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension

# ---------------------------------------------------------------------------
# split() — parsing the qualified reference shape
# ---------------------------------------------------------------------------


def test_split_valid_qualified_reference() -> None:
    assert split("orders.region") == ("orders", "region")


def test_split_allows_underscores_and_digits_after_first_char() -> None:
    assert split("user_events.created_at_2") == ("user_events", "created_at_2")


def test_split_is_case_insensitive() -> None:
    # The regex is IGNORECASE; preserving case lets a downstream layer
    # decide whether to lowercase.
    assert split("Orders.REGION") == ("Orders", "REGION")


def test_split_rejects_missing_dot() -> None:
    with pytest.raises(ResolveError, match="must be 'cube.field'"):
        split("ordersregion")


def test_split_rejects_too_many_dots() -> None:
    with pytest.raises(ResolveError, match="must be 'cube.field'"):
        split("schema.orders.region")


def test_split_rejects_empty_cube_segment() -> None:
    with pytest.raises(ResolveError):
        split(".region")


def test_split_rejects_empty_field_segment() -> None:
    with pytest.raises(ResolveError):
        split("orders.")


def test_split_rejects_leading_digit_in_cube() -> None:
    with pytest.raises(ResolveError):
        split("1orders.region")


def test_split_rejects_leading_digit_in_field() -> None:
    with pytest.raises(ResolveError):
        split("orders.1region")


def test_split_rejects_non_ascii_identifier() -> None:
    # The regex is plain ASCII identifier — non-ASCII letters fail.
    with pytest.raises(ResolveError):
        split("öröders.region")


def test_split_rejects_whitespace() -> None:
    with pytest.raises(ResolveError):
        split("orders .region")


def test_split_rejects_empty_string() -> None:
    with pytest.raises(ResolveError):
        split("")


# ---------------------------------------------------------------------------
# resolve_field() fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="customers",
        alias="c",
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )
    return {"orders": orders, "customers": customers}


# ---------------------------------------------------------------------------
# resolve_field() — happy paths for each field kind
# ---------------------------------------------------------------------------


def test_resolve_measure_returns_cube_and_measure(small_catalog: dict[str, Cube]) -> None:
    cube, fld = resolve_field("orders.revenue", small_catalog)
    assert cube.name == "orders"
    assert isinstance(fld, Measure)
    assert fld.name == "revenue"


def test_resolve_dimension_returns_cube_and_dimension(
    small_catalog: dict[str, Cube],
) -> None:
    cube, fld = resolve_field("orders.region", small_catalog)
    assert cube.name == "orders"
    assert isinstance(fld, Dimension)
    assert fld.name == "region"


def test_resolve_time_dimension_returns_cube_and_time_dimension(
    small_catalog: dict[str, Cube],
) -> None:
    cube, fld = resolve_field("orders.created_at", small_catalog)
    assert cube.name == "orders"
    assert isinstance(fld, TimeDimension)
    assert fld.name == "created_at"


# ---------------------------------------------------------------------------
# resolve_field() — error paths
# ---------------------------------------------------------------------------


def test_resolve_unknown_cube_includes_known_list(small_catalog: dict[str, Cube]) -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("orderz.region", small_catalog)
    err = exc_info.value
    assert err.kind == "cube"
    assert err.name == "orderz"
    assert "Known cubes:" in str(err)
    # The catalog's known cubes are listed in the message.
    assert "orders" in str(err)
    assert "customers" in str(err)


def test_resolve_unknown_cube_suggests_close_match(small_catalog: dict[str, Cube]) -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("order.region", small_catalog)
    err = exc_info.value
    assert err.hint == "orders"
    assert "Did you mean 'orders'" in str(err)


def test_resolve_unknown_field_includes_known_fields(
    small_catalog: dict[str, Cube],
) -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("orders.no_such_field", small_catalog)
    err = exc_info.value
    assert err.kind == "field"
    assert err.name == "no_such_field"
    assert err.cube == "orders"
    assert "Known fields:" in str(err)
    # Knowns include all three kinds — measures, dimensions, time-dims.
    msg = str(err)
    assert "revenue" in msg
    assert "region" in msg
    assert "created_at" in msg


def test_resolve_unknown_field_suggests_close_match(
    small_catalog: dict[str, Cube],
) -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("orders.reveune", small_catalog)
    err = exc_info.value
    assert err.hint == "revenue"
    assert "Did you mean 'revenue'" in str(err)


def test_resolve_no_close_match_omits_hint(small_catalog: dict[str, Cube]) -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("orders.zzzqqq", small_catalog)
    err = exc_info.value
    assert err.hint is None
    assert "Did you mean" not in str(err)


def test_resolve_malformed_reference_raises_resolveerror(
    small_catalog: dict[str, Cube],
) -> None:
    """``split()`` failures surface as plain ``ResolveError`` (not
    ``UnknownIdentifierError``) so callers can branch on the bad-shape
    case separately from the unknown-name case."""
    with pytest.raises(ResolveError) as exc_info:
        resolve_field("orders/region", small_catalog)
    assert not isinstance(exc_info.value, UnknownIdentifierError)


def test_resolve_with_empty_catalog_lists_no_knowns() -> None:
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("anything.x", {})
    err = exc_info.value
    assert err.kind == "cube"
    # No candidates → no hint (closest_match on an empty list returns None).
    assert err.hint is None


def test_resolve_returns_unknown_field_when_cube_has_no_fields() -> None:
    bare = Cube(name="bare", dialect=Dialect.POSTGRES, table="bare", alias="b")
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("bare.something", {"bare": bare})
    err = exc_info.value
    assert err.kind == "field"
    assert err.cube == "bare"
    # ``Known fields:`` is still present even with an empty cube.
    assert "Known fields:" in str(err)


# ---------------------------------------------------------------------------
# Catalog enumeration — filtered catalog hides unauthorized cube names
# (SEMQL-RESOLVER-DIAGNOSTIC-HIDDEN-CATALOG-ENUMERATION)
# ---------------------------------------------------------------------------


def test_resolve_with_filtered_catalog_hides_unauthorized_cube_name(
    small_catalog: dict[str, Cube],
) -> None:
    """Passing only the visible subset of the catalog keeps unauthorized
    cube names out of the 'Known cubes:' list in error messages."""
    visible: dict[str, Cube] = {"orders": small_catalog["orders"]}
    with pytest.raises(UnknownIdentifierError) as exc_info:
        resolve_field("customers.name", visible)
    msg = str(exc_info.value)
    # The "Known cubes:" section must not reveal unauthorized cube names.
    # (The error may still echo the UNKNOWN name itself — that's the query
    # input, not a catalog leak.)
    assert "Known cubes:" in msg
    known_section = msg.split("Known cubes:")[1]
    assert "customers" not in known_section
    assert "orders" in known_section


def test_resolve_with_filtered_catalog_does_not_expose_unknown_cube_in_hint(
    small_catalog: dict[str, Cube],
) -> None:
    """close-match hint must also stay within the visible catalog."""
    visible: dict[str, Cube] = {"orders": small_catalog["orders"]}
    with pytest.raises(UnknownIdentifierError) as exc_info:
        # "custmers" is close to "customers" but customers is not visible.
        resolve_field("custmers.name", visible)
    err = exc_info.value
    # Hint may be None or "orders" — must not be "customers".
    assert err.hint != "customers"
