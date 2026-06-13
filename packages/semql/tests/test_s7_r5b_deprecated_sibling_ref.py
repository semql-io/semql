"""Tests for S7-R5b: deprecated-sibling-ref soft warning.

When a cube's ``relations`` narrative references another cube that
has been flagged ``stability="deprecated"``, the validator emits a
``deprecated_sibling_ref`` warning. The whole point of the
``replacement`` field on Cube is to point callers at the new name;
``relations`` text that still uses the old name is a stale reference
the lint pass should surface.

The check is a soft warning (not an error) because the deprecated
cube is still resolvable — it'll compile, the SQL will run, the user
just won't get the new metric. Failing the build would be too loud.
"""

from __future__ import annotations

from semql import (
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    validate,
)
from semql.validate import ValidationWarning


def _orders(relations: str = "") -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="id", sql="{o}.id", type="number")],
        relations=relations,
    )


def _deprecated_orders() -> Cube:
    return Cube(
        name="orders_v1",
        backend=Dialect.POSTGRES,
        table="orders_v1",
        alias="o",
        primary_key="id",
        stability="deprecated",
        replacement="orders",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="id", sql="{o}.id", type="number")],
    )


def test_relations_referencing_deprecated_cube_emits_warning() -> None:
    """A backtick reference to a deprecated cube is a stale pointer."""
    catalog = {
        "orders": _orders(relations="See `orders_v1` for the legacy view."),
        "orders_v1": _deprecated_orders(),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    deprecated_ref_warnings = [
        e for e in errors if isinstance(e, ValidationWarning) and e.code == "deprecated_sibling_ref"
    ]
    assert len(deprecated_ref_warnings) == 1
    w = deprecated_ref_warnings[0]
    assert w.cube == "orders"
    assert "orders_v1" in w.message
    assert "orders" in w.message  # the replacement name should be surfaced


def test_relations_referencing_stable_cube_no_warning() -> None:
    """References to stable cubes are normal, not a deprecation signal."""
    catalog = {
        "orders": _orders(relations="Joins `customers` on id."),
        "customers": Cube(
            name="customers",
            backend=Dialect.POSTGRES,
            table="customers",
            alias="c",
            primary_key="id",
            dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
        ),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    deprecated_ref_warnings = [
        e for e in errors if isinstance(e, ValidationWarning) and e.code == "deprecated_sibling_ref"
    ]
    assert deprecated_ref_warnings == []


def test_relations_referencing_deprecated_cube_with_no_replacement() -> None:
    """A deprecated cube *without* a replacement is going away
    entirely; the warning is even louder — no successor to point
    at."""
    catalog = {
        "orders": _orders(relations="Legacy data in `old_orders`."),
        "old_orders": Cube(
            name="old_orders",
            backend=Dialect.POSTGRES,
            table="old_orders",
            alias="o",
            primary_key="id",
            stability="deprecated",
            replacement=None,  # no successor
            dimensions=[Dimension(name="id", sql="{o}.id", type="number")],
        ),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    deprecated_ref_warnings = [
        e for e in errors if isinstance(e, ValidationWarning) and e.code == "deprecated_sibling_ref"
    ]
    assert len(deprecated_ref_warnings) == 1
    w = deprecated_ref_warnings[0]
    assert "old_orders" in w.message


def test_relations_referencing_beta_cube_no_warning() -> None:
    """Beta is a soft signal (cube_beta is its own warning) — not a
    deprecation. We only flag deprecated siblings, not beta ones."""
    catalog = {
        "orders": _orders(relations="Pilot integration with `experiments`."),
        "experiments": Cube(
            name="experiments",
            backend=Dialect.POSTGRES,
            table="experiments",
            alias="e",
            primary_key="id",
            stability="beta",
            dimensions=[Dimension(name="id", sql="{e}.id", type="number")],
        ),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    deprecated_ref_warnings = [
        e for e in errors if isinstance(e, ValidationWarning) and e.code == "deprecated_sibling_ref"
    ]
    assert deprecated_ref_warnings == []


def test_relations_with_no_backticks_no_warning() -> None:
    catalog = {
        "orders": _orders(relations="Just some narrative text without references."),
        "orders_v1": _deprecated_orders(),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    deprecated_ref_warnings = [
        e for e in errors if isinstance(e, ValidationWarning) and e.code == "deprecated_sibling_ref"
    ]
    assert deprecated_ref_warnings == []


def test_deprecated_sibling_ref_does_not_block_compile() -> None:
    """The warning is non-fatal — the catalog should still be
    usable. (We don't compile here, but we assert the validator
    emits a Warning, not an Error.)"""
    catalog = {
        "orders": _orders(relations="See `orders_v1` for the legacy view."),
        "orders_v1": _deprecated_orders(),
    }
    errors = validate(SemanticQuery(measures=["orders.revenue"]), catalog)
    matches = [e for e in errors if e.code == "deprecated_sibling_ref"]
    assert len(matches) == 1
    assert isinstance(matches[0], ValidationWarning)
