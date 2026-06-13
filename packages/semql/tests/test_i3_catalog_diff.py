"""Tests for CatalogDiff (I3).

Catalog changelog: ``CatalogDiff(old, new) -> MarkdownReport``. The
report enumerates every cube / field / join that was added, removed,
or had its schema-affecting attributes changed. Schema-affecting =
anything that would alter the compiler's emitted SQL or the linter's
findings: ``name``, ``sql``, ``type``, ``agg``, ``unit``, ``primary_key``,
``foreign_key``, ``required_roles``, ``mask_roles``, ``mask_value``,
``aliases``, join ``to``/``on``/``relationship``, table / alias.

Breaking changes (``removed`` / ``type_changed`` / ``agg_changed`` /
``required_roles_narrowed``) are sorted to the top of the report so
they're the first thing a reviewer reads; additive changes
(``added``) and ``aliases_added`` (which are non-breaking) are grouped
beneath.

The test catalogue uses minimal Cube fixtures — the surface is the
diff, not the catalog. We exercise every severity path.
"""

from __future__ import annotations

from semql import (
    CatalogDiff,
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    Segment,
    diff_catalogs,
)


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )


def test_diff_empty_catalogs_is_no_changes() -> None:
    diff = diff_catalogs({}, {})
    assert diff.is_empty
    assert diff.breaking_count == 0
    assert diff.additive_count == 0
    md = diff.to_markdown()
    assert "no changes" in md.lower()


def test_diff_cube_added_is_additive() -> None:
    diff = diff_catalogs({}, {"orders": _orders()})
    assert not diff.is_empty
    assert diff.breaking_count == 0
    assert diff.additive_count == 1
    md = diff.to_markdown()
    assert "added" in md.lower()
    assert "orders" in md


def test_diff_cube_removed_is_breaking() -> None:
    diff = diff_catalogs({"orders": _orders()}, {})
    assert diff.breaking_count >= 1
    md = diff.to_markdown()
    assert "removed" in md.lower()
    assert "orders" in md


def test_diff_measure_added_is_additive() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={
            "measures": old.measures
            + [Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency")],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.additive_count >= 1
    md = diff.to_markdown()
    assert "avg_amount" in md


def test_diff_measure_removed_is_breaking() -> None:
    old = _orders()
    new_cube = old.model_copy(update={"measures": [old.measures[0]]})
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.breaking_count >= 1
    md = diff.to_markdown()
    assert "count" in md  # the removed measure name


def test_diff_measure_agg_changed_is_breaking() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={
            "measures": [
                old.measures[0].model_copy(update={"agg": "avg"}),
                old.measures[1],
            ],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.breaking_count >= 1
    md = diff.to_markdown()
    assert "revenue" in md
    assert "sum" in md and "avg" in md


def test_diff_dimension_type_changed_is_breaking() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={
            "dimensions": [
                old.dimensions[0],
                old.dimensions[1].model_copy(update={"type": "number"}),
            ],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.breaking_count >= 1


def test_dimension_alias_added_is_additive() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={
            "dimensions": [
                old.dimensions[0],
                old.dimensions[1].model_copy(update={"aliases": ["state", "order_state"]}),
            ],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    # Aliases are forward-compatible (new name; old name still works) — additive.
    assert diff.additive_count >= 1
    assert diff.breaking_count == 0


def test_diff_join_added_is_additive() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={
            "joins": old.joins
            + [Join(to="products", relationship="many_to_one", on="{o}.product_id = {p}.id")],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.additive_count >= 1


def test_diff_join_removed_is_breaking() -> None:
    old = _orders()
    new_cube = old.model_copy(update={"joins": []})
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.breaking_count >= 1
    md = diff.to_markdown()
    assert "customers" in md  # the joined-to cube


def test_diff_required_roles_narrowed_is_breaking() -> None:
    """Removing a role requirement widens visibility (additive).
    Adding a role requirement narrows visibility (breaking)."""
    old = _orders()
    new_cube = old.model_copy(
        update={
            "measures": [
                old.measures[0].model_copy(update={"required_roles": ["admin"]}),
                old.measures[1],
            ],
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    md = diff.to_markdown()
    assert "admin" in md
    assert diff.breaking_count >= 1


def test_diff_required_roles_widened_is_additive() -> None:
    old = _orders()
    old_with_role = old.model_copy(
        update={
            "measures": [
                old.measures[0].model_copy(update={"required_roles": ["admin"]}),
                old.measures[1],
            ],
        }
    )
    new_cube = old.model_copy(
        update={
            "measures": [
                old.measures[0].model_copy(update={"required_roles": []}),
                old.measures[1],
            ],
        }
    )
    diff = diff_catalogs({"orders": old_with_role}, {"orders": new_cube})
    assert diff.additive_count >= 1
    assert diff.breaking_count == 0


def test_diff_segment_added_is_additive() -> None:
    old = _orders()
    new_cube = old.model_copy(
        update={"segments": [Segment(name="big_order", sql="{o}.amount > 100")]}
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.additive_count >= 1


def test_diff_segment_removed_is_breaking() -> None:
    old = _orders().model_copy(
        update={"segments": [Segment(name="big_order", sql="{o}.amount > 100")]}
    )
    new_cube = _orders()
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    assert diff.breaking_count >= 1


def test_diff_breaking_changes_appear_before_additive_in_markdown() -> None:
    """The markdown report sorts breaking changes to the top so
    reviewers read them first."""
    old = _orders()
    new_cube = old.model_copy(
        update={
            "measures": [old.measures[0]],  # removed 'count' (breaking)
            "dimensions": old.dimensions
            + [Dimension(name="amount", sql="{o}.amount", type="number")],  # additive
        }
    )
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    md = diff.to_markdown()
    breaking_pos = md.lower().find("breaking")
    additive_pos = md.lower().find("additive")
    assert breaking_pos < additive_pos, (
        f"breaking ({breaking_pos}) should come before additive ({additive_pos})"
    )


def test_diff_no_pii_leakage() -> None:
    """The diff never embeds raw user data — it only references
    catalog names and types. (E.g. the report doesn't echo arbitrary
    ``sql`` content into the markdown; the report mentions which
    attribute changed but doesn't dump the full SQL.)"""
    old = _orders()
    secret = "amount > 1000000 AND user_id = 42"  # noqa: S105 - test fixture
    new_cube = old.model_copy(update={"segments": [Segment(name="vip_order", sql=secret)]})
    diff = diff_catalogs({"orders": old}, {"orders": new_cube})
    md = diff.to_markdown()
    assert secret not in md
    assert "42" not in md


def test_diff_is_pydantic_value_type() -> None:
    """CatalogDiff is a frozen Pydantic model so it round-trips through
    model_dump / model_validate (cf. I9)."""
    diff = diff_catalogs({}, {"orders": _orders()})
    assert isinstance(diff, CatalogDiff)
    restored = CatalogDiff.model_validate(diff.model_dump())
    assert restored.additive_count == diff.additive_count
    assert restored.breaking_count == diff.breaking_count
