# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""I2 — ``semql.lint`` catalog-level static checks.

Walks a ``Catalog`` (or a list of cubes) and surfaces structural
smells. Each rule is a pure function over the catalog; the
``lint_catalog`` entry point returns a ``LintReport`` with one
``LintFinding`` per violation. Severity is ``"warning"`` (catalog
author should look) or ``"error"`` (compile-time misconfig that
will surface at query time).

Initial rule set:

  - ``cube_no_measures`` — a cube with no measures is unreachable
    for any aggregating query.
  - ``count_distinct_non_additive`` — a measure with ``agg='count_distinct'``
    and ``non_additive=False`` is wrong; the answer is wrong when
    rolled up across coarser time grains.
  - ``segment_external_field`` — a Segment's SQL references a field
    that isn't a dimension on the same cube. (The SQL is allowed to
    reference the cube's own table via ``{alias}``; we statically
    check that referenced identifier names match a declared dim.)
  - ``pk_no_fk_joins`` — a cube declares a ``primary_key`` but has
    no ``Join`` to / from any other cube. (Useful signal in
    entities-rich catalogs.)
  - ``empty_cube_name_list`` — a cube with empty ``measures`` AND
    empty ``dimensions`` is a stub.
"""

from __future__ import annotations

from semql.lint import LintFinding, LintReport, lint_catalog
from semql.model import Cube, Dialect, Dimension, Join, Measure, Segment


def _make_cube(**overrides: object) -> Cube:
    """Minimal cube; tests override the fields they care about."""
    defaults: dict = {
        "name": "orders",
        "backend": Dialect.POSTGRES,
        "table": "{schema}.orders",
        "alias": "o",
        "measures": [Measure(name="count", sql="*", agg="count")],
        "dimensions": [Dimension(name="region", sql="{o}.region", type="string")],
    }
    defaults.update(overrides)
    return Cube(**defaults)


# ---------------------------------------------------------------------------
# LintFinding / LintReport shape
# ---------------------------------------------------------------------------


def test_lint_finding_carries_rule_code_and_message() -> None:
    f = LintFinding(
        rule="cube_no_measures",
        severity="warning",
        cube="orders",
        message="Cube 'orders' has no measures.",
    )
    assert f.rule == "cube_no_measures"
    assert f.severity == "warning"
    assert f.cube == "orders"


def test_lint_report_aggregates_findings() -> None:
    r = LintReport(findings=())
    assert r.findings == ()
    assert r.has_errors is False

    f = LintFinding(rule="x", severity="error", cube="c", message="m")
    r2 = LintReport(findings=(f,))
    assert r2.has_errors is True


# ---------------------------------------------------------------------------
# Rule: cube_no_measures
# ---------------------------------------------------------------------------


def test_lint_warns_on_cube_with_no_measures() -> None:
    cube = _make_cube(measures=[])
    report = lint_catalog({"orders": cube})
    assert any(f.rule == "cube_no_measures" and f.cube == "orders" for f in report.findings)


def test_lint_clean_cube_passes() -> None:
    cube = _make_cube()
    report = lint_catalog({"orders": cube})
    assert report.findings == ()


# ---------------------------------------------------------------------------
# Rule: count_distinct_non_additive
# ---------------------------------------------------------------------------


def test_lint_warns_count_distinct_with_non_additive_false() -> None:
    """A count_distinct measure with non_additive=False is wrong on rollup."""
    cube = _make_cube(
        measures=[
            Measure(
                name="unique_users", sql="{o}.user_id", agg="count_distinct", non_additive=False
            ),
        ],
    )
    report = lint_catalog({"orders": cube})
    assert any(
        f.rule == "count_distinct_non_additive" and f.cube == "orders" for f in report.findings
    )


def test_lint_passes_count_distinct_with_non_additive_true() -> None:
    """Marked correctly: non_additive=True, no warning."""
    cube = _make_cube(
        measures=[
            Measure(
                name="unique_users", sql="{o}.user_id", agg="count_distinct", non_additive=True
            ),
        ],
    )
    report = lint_catalog({"orders": cube})
    assert not any(f.rule == "count_distinct_non_additive" for f in report.findings)


# ---------------------------------------------------------------------------
# Rule: segment_external_field
# ---------------------------------------------------------------------------


def test_lint_warns_segment_referencing_unknown_field() -> None:
    """A segment whose SQL references a non-existent field is a smell."""
    cube = _make_cube(
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
    )
    # ``status`` isn't a declared dim on this cube.
    report = lint_catalog({"orders": cube})
    assert any(f.rule == "segment_external_field" for f in report.findings)


def test_lint_passes_segment_referencing_own_dim() -> None:
    """A segment that references a declared dim passes."""
    cube = _make_cube(
        dimensions=[
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
    )
    report = lint_catalog({"orders": cube})
    assert not any(f.rule == "segment_external_field" for f in report.findings)


# ---------------------------------------------------------------------------
# Rule: pk_no_fk_joins
# ---------------------------------------------------------------------------


def test_lint_warns_cube_with_pk_but_no_joins() -> None:
    """A cube declaring primary_key but no joins is an isolated island."""
    cube = _make_cube(
        primary_key="id",
        joins=[],
    )
    report = lint_catalog({"orders": cube})
    assert any(f.rule == "pk_no_fk_joins" and f.cube == "orders" for f in report.findings)


def test_lint_passes_cube_with_pk_and_joins() -> None:
    """A cube with primary_key AND joins is connected."""
    customers = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        primary_key="id",
        measures=[Measure(name="count", sql="*", agg="count")],
        # Bidirectional join — orders -> customers AND customers -> orders.
        joins=[Join(to="orders", relationship="one_to_many", on="{c}.id = {o}.customer_id")],
    )
    cube = _make_cube(
        primary_key="id",
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )
    report = lint_catalog({"orders": cube, "customers": customers})
    # Neither cube is isolated — both have joins.
    assert not any(f.rule == "pk_no_fk_joins" for f in report.findings)


# ---------------------------------------------------------------------------
# Rule: empty_cube (no measures, no dimensions, no time dimensions)
# ---------------------------------------------------------------------------


def test_lint_warns_on_empty_cube() -> None:
    cube = _make_cube(measures=[], dimensions=[])
    report = lint_catalog({"orders": cube})
    assert any(f.rule == "empty_cube" and f.cube == "orders" for f in report.findings)


# ---------------------------------------------------------------------------
# Aggregating across many cubes
# ---------------------------------------------------------------------------


def test_lint_aggregates_across_cubes() -> None:
    good = _make_cube(name="good")
    bad = _make_cube(name="bad", measures=[])
    report = lint_catalog({"good": good, "bad": bad})
    cubes_with_findings = {f.cube for f in report.findings}
    assert "bad" in cubes_with_findings
    assert "good" not in cubes_with_findings
