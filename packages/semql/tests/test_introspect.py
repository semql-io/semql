"""Unit tests for ``semql.introspect``.

The reflection cubes (``CATALOG_CUBES`` / ``CATALOG_MEASURES`` /
``CATALOG_DIMENSIONS``) materialise the catalog snapshot as a
``VALUES`` subquery at compile time so meta queries route through the
same compiler the rest of the catalog does. The contracts here are:

- ``quote_literal`` is the only thing producing SQL string literals;
  apostrophe escaping must be airtight.
- ``build_meta_values`` produces parseable SQL with the cube-/
  measure-/dimension-named rows the META cubes' Dimensions reference.
- The META cube objects themselves expose the dimensions the compiler
  joins against — a rename here without an introspect update would
  silently break ``query_semantic`` against catalog_cubes.
"""

from __future__ import annotations

import pytest
import sqlglot
from semql.introspect import (
    CATALOG_CUBES,
    CATALOG_DIMENSIONS,
    CATALOG_MEASURES,
    META_CUBES,
    build_meta_values,
    quote_literal,
)
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension

# ---------------------------------------------------------------------------
# quote_literal()
# ---------------------------------------------------------------------------


def test_quote_literal_none_returns_NULL() -> None:
    assert quote_literal(None) == "NULL"


def test_quote_literal_plain_string() -> None:
    assert quote_literal("hello") == "'hello'"


def test_quote_literal_escapes_single_quotes() -> None:
    assert quote_literal("o'reilly") == "'o''reilly'"


def test_quote_literal_doubles_every_apostrophe() -> None:
    """Multiple apostrophes — including adjacent — all double."""
    assert quote_literal("a''b'c") == "'a''''b''c'"


def test_quote_literal_empty_string() -> None:
    assert quote_literal("") == "''"


def test_quote_literal_unicode_pass_through() -> None:
    assert quote_literal("héllo") == "'héllo'"


# ---------------------------------------------------------------------------
# build_meta_values() — output shape per META cube
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        description="Order lines",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    return {"orders": orders}


def test_meta_values_for_catalog_cubes_is_parseable_sql(
    sample_catalog: dict[str, Cube],
) -> None:
    sql = build_meta_values("catalog_cubes", sample_catalog)
    # The output is a ``(SELECT * FROM (VALUES ...) AS _v(cols))`` shape;
    # sqlglot should parse it cleanly under the Postgres dialect.
    sqlglot.parse_one(f"SELECT * FROM {sql} AS cc", dialect="postgres")


def test_meta_values_for_catalog_cubes_contains_expected_fields(
    sample_catalog: dict[str, Cube],
) -> None:
    sql = build_meta_values("catalog_cubes", sample_catalog)
    # cube name, backend value, exposed flag, description, alias.
    assert "'orders'" in sql
    assert "'postgres'" in sql
    assert "TRUE" in sql  # expose_in_prompt default
    assert "'Order lines'" in sql
    assert "'o'" in sql
    # Column header tuple in the SELECT.
    for col in ("name", "backend", "exposed", "description", "alias"):
        assert col in sql


def test_meta_values_for_catalog_measures_lists_every_measure(
    sample_catalog: dict[str, Cube],
) -> None:
    sql = build_meta_values("catalog_measures", sample_catalog)
    assert "'revenue'" in sql
    assert "'count'" in sql
    assert "'sum'" in sql
    assert "'currency'" in sql


def test_meta_values_for_catalog_dimensions_marks_time_rows(
    sample_catalog: dict[str, Cube],
) -> None:
    sql = build_meta_values("catalog_dimensions", sample_catalog)
    assert "'region'" in sql
    assert "'created_at'" in sql
    # is_time flag — TRUE for the time dim, FALSE for the plain dimension.
    assert "TRUE" in sql
    assert "FALSE" in sql


# ---------------------------------------------------------------------------
# Empty-catalog edge case — the WHERE FALSE escape hatch
# ---------------------------------------------------------------------------


def test_meta_values_for_empty_catalog_cubes_returns_where_false() -> None:
    """Postgres VALUES with no rows is a syntax error; the helper emits
    a one-row NULL-only VALUES gated by ``WHERE FALSE`` so the query
    still parses and produces zero rows."""
    sql = build_meta_values("catalog_cubes", {})
    assert "WHERE FALSE" in sql.upper() or "WHERE false" in sql


def test_meta_values_for_catalog_measures_with_no_measures() -> None:
    """A catalog with cubes but no measures produces the empty-shape
    VALUES literal."""
    empty = {
        "x": Cube(name="x", backend=Dialect.POSTGRES, table="x", alias="x"),
    }
    sql = build_meta_values("catalog_measures", empty)
    assert "WHERE FALSE" in sql.upper()


def test_meta_values_for_catalog_dimensions_with_no_dimensions() -> None:
    empty = {
        "x": Cube(name="x", backend=Dialect.POSTGRES, table="x", alias="x"),
    }
    sql = build_meta_values("catalog_dimensions", empty)
    assert "WHERE FALSE" in sql.upper()


# ---------------------------------------------------------------------------
# Unknown cube name — must raise so callers don't silently get nothing
# ---------------------------------------------------------------------------


def test_unknown_meta_cube_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        build_meta_values("not_a_meta_cube", {})


# ---------------------------------------------------------------------------
# META cube objects expose the dimensions referenced in build_meta_values
# ---------------------------------------------------------------------------


def test_catalog_cubes_dimensions_match_meta_values_columns() -> None:
    """The dimension names on ``CATALOG_CUBES`` MUST match the SELECT
    column list in its VALUES literal — otherwise a query naming a
    META cube field would compile to a column the subquery doesn't
    expose."""
    expected_cols = {"name", "backend", "exposed", "description", "alias"}
    actual_cols = {d.name for d in CATALOG_CUBES.dimensions}
    assert actual_cols == expected_cols


def test_catalog_measures_dimensions_match_meta_values_columns() -> None:
    expected_cols = {"cube", "name", "agg", "unit", "display_unit", "description"}
    actual_cols = {d.name for d in CATALOG_MEASURES.dimensions}
    assert actual_cols == expected_cols


def test_catalog_dimensions_dimensions_match_meta_values_columns() -> None:
    expected_cols = {
        "cube",
        "name",
        "type",
        "unit",
        "display_unit",
        "description",
        "is_time",
    }
    actual_cols = {d.name for d in CATALOG_DIMENSIONS.dimensions}
    assert actual_cols == expected_cols


def test_meta_cubes_list_contains_all_three() -> None:
    by_name = {c.name for c in META_CUBES}
    assert by_name == {"catalog_cubes", "catalog_measures", "catalog_dimensions"}


def test_meta_cubes_are_meta_backend() -> None:
    for cube in META_CUBES:
        assert cube.backend is Dialect.META


def test_meta_cubes_hidden_from_prompt() -> None:
    """Reflection cubes shouldn't appear in the planner prompt — they
    surface only on explicit introspection."""
    for cube in META_CUBES:
        assert cube.expose_in_prompt is False
