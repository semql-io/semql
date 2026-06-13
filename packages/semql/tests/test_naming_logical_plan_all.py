"""Tests for the naming-review's ``__all__`` discipline.

Naming review 2026-06: ``LogicalPlan`` is the in-flight IR. It is
load-bearing for the compiler today, but the review calls for
removing it from ``semql.__all__`` until it's serialisable (it
isn't a Pydantic model yet — adding ``model_dump`` /
``model_validate`` is a separate piece of work that depends on
the IR's class hierarchy stabilising).

Public callers today import via ``from semql.logical import
LogicalPlan`` (the explicit module path), not via the top-level
``from semql import LogicalPlan`` re-export. Dropping the
re-export is a strict-narrowing change — no production code
references the top-level name.
"""

from __future__ import annotations


def test_logical_plan_not_in_semql_all() -> None:
    """The naming review explicitly says: drop LogicalPlan from
    ``__all__`` until the IR is serialisable. The class itself
    remains importable via ``semql.logical.LogicalPlan`` — only
    the public re-export is removed."""
    import semql

    assert "LogicalPlan" not in semql.__all__


def test_logical_plan_still_importable_via_module_path() -> None:
    """The class is still public via the ``semql.logical`` module
    path; only the top-level re-export was removed. Callers that
    need the IR for debugging or test-time assertions can still
    import it directly."""
    from semql.logical import LogicalPlan

    assert LogicalPlan is not None


def test_logical_plan_not_reexported_at_top_level() -> None:
    """``from semql import LogicalPlan`` no longer binds a name.
    This is a deliberate narrowing of the public surface — the
    IR is implementation detail of the compiler until it carries
    a schema_version + round-trip contract."""
    import semql

    assert not hasattr(semql, "LogicalPlan")
