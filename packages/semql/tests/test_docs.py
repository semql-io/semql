"""Tests for ``semql.docs.render_catalog_markdown``.

Pins the structural skeleton — TOC, headings, tables — and that
expected fields surface in the output. Doesn't pin exact byte-for-
byte content, since that would brick the test on every cosmetic
change.
"""

from __future__ import annotations

from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    Segment,
    TimeDimension,
)
from semql.docs import render_catalog_markdown


def _full_catalog() -> Catalog:
    customers = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        dimensions=[Dimension(name="id", sql="{c}.id", type="uuid")],
    )
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        description="One row per checkout.",
        display_name="Orders",
        measures=[
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(
                name="paid_count",
                sql="*",
                agg="count",
                filter="{o}.status = 'paid'",
                description="Orders with confirmed payment.",
            ),
            Measure(
                name="conversion_rate",
                sql="",
                agg="ratio",
                numerator="paid_count",
                denominator="count",
                format="percent",
            ),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(
                name="duration_s",
                sql="{o}.duration_s",
                type="number",
                unit="seconds",
                format="duration",
            ),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("hour", "day"),
            ),
        ],
        segments=[
            Segment(name="recent", sql="{o}.created_at > now() - interval '30 days'"),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.cust_id = {c}.id")],
        required_filters=["region"],
        base_predicate="{o}.deleted_at IS NULL",
    )
    return Catalog([orders, customers])


def test_title_appears() -> None:
    out = render_catalog_markdown(_full_catalog(), title="Analytics Catalog")
    assert out.startswith("# Analytics Catalog")


def test_toc_lists_each_cube() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "## Table of contents" in out
    assert "- [`orders`" in out
    assert "- [`customers`" in out


def test_measure_table_rendered() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "### Measures" in out
    assert "`count`" in out
    assert "`paid_count`" in out
    # Filter column surfaces the FILTER fragment.
    assert "`{o}.status = 'paid'`" in out
    # Ratio surfaces its components.
    assert "ratio (paid_count/count)" in out


def test_dimension_table_includes_unit_and_format() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "### Dimensions" in out
    assert "duration_s" in out
    assert "seconds" in out
    assert "duration" in out


def test_time_dimensions_section() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "### Time dimensions" in out
    assert "created_at" in out
    assert "`hour`" in out


def test_segments_section() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "### Segments" in out
    assert "recent" in out


def test_joins_section() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "### Joins" in out
    assert "customers" in out
    assert "many_to_one" in out


def test_base_predicate_surfaces_in_facts() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "**Base predicate:**" in out
    assert "deleted_at IS NULL" in out


def test_required_filters_surface() -> None:
    out = render_catalog_markdown(_full_catalog())
    assert "**Required filters:**" in out
    assert "region" in out


def test_meta_cubes_excluded_by_default() -> None:
    out = render_catalog_markdown(_full_catalog())
    # The Catalog auto-appends META cubes; the docs generator hides
    # them unless include_meta=True.
    assert "catalog_cubes" not in out


def test_meta_cubes_visible_when_include_meta_true() -> None:
    out = render_catalog_markdown(_full_catalog(), include_meta=True)
    assert "catalog_cubes" in out
