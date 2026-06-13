"""Tests for the public re-export surface of `semql`.

The README quick-start does ``from semql import Cube, Catalog`` —
this module pins that contract so a future refactor can't silently
move an identifier.
"""

from __future__ import annotations

import semql


def test_public_surface_includes_core_types() -> None:
    expected = {
        # model
        "Backend",
        "Cube",
        "Measure",
        "Dimension",
        "TimeDimension",
        "Join",
        # spec
        "SemanticQuery",
        "Filter",
        "TimeWindow",
        "CompareWindow",
        # compile
        "CompiledQuery",
        "compile_query",
        "MAX_UNGROUPED_ROWS",
        # errors
        "SemQLError",
        "ResolveError",
        "CompileError",
        "UnknownIdentifierError",
        "JoinPathError",
        "FilterTypeError",
        "PlaceholderError",
        "CrossBackendError",
        "PhaseDeferredError",
        # introspect
        "META_CUBES",
        "CATALOG_CUBES",
        "CATALOG_MEASURES",
        "CATALOG_DIMENSIONS",
        # visualize
        "decide_visualization",
        "VizDecision",
        "VizColumn",
        # catalog
        "Catalog",
    }
    missing = expected - set(dir(semql))
    assert not missing, f"semql is missing public exports: {sorted(missing)}"
    # Prompt rendering is not part of the core public surface; it lives in
    # the semql-prompt package.
    assert not hasattr(semql, "build_planner_prompt_fragment")
    assert not hasattr(semql, "render_catalog_block")


def test_no_hello_stub() -> None:
    assert not hasattr(semql, "hello"), "hello() stub should be removed"


def test_all_is_set_explicitly() -> None:
    assert hasattr(semql, "__all__")
    # __all__ entries must all be importable from the package.
    for name in semql.__all__:
        assert hasattr(semql, name), f"__all__ lists {name!r} but it isn't exported"


def test_quick_start_import_shape_works() -> None:
    """Smoke test: the README's quick-start pattern compiles & runs."""
    from semql import Catalog, Cube, Dimension, Measure, SemanticQuery

    orders = Cube(
        name="orders",
        backend=semql.Backend.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([orders])
    out = cat.compile(SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]))
    assert "SUM" in out.sql
