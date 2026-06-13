"""H3 — ErrorTransformHook Protocol.

ErrorTransformHook: callable Protocol (error: CompileError) -> Exception | None.
Return an exception to replace the CompileError; return None to pass through unchanged.

Wire as ``error_transform`` kwarg on ``Catalog.__init__()``.
Applied in ``Catalog.compile()`` when a CompileError is raised.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.errors import CompileError


def _catalog(error_transform: object = None) -> Catalog:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    if error_transform is not None:
        return Catalog([cube], error_transform=error_transform)
    return Catalog([cube])


# ---------------------------------------------------------------------------
# Protocol importable and structural
# ---------------------------------------------------------------------------


def test_error_transform_hook_importable() -> None:
    from semql.hooks import ErrorTransformHook

    assert ErrorTransformHook is not None


def test_error_transform_hook_is_runtime_checkable() -> None:
    from semql.hooks import ErrorTransformHook

    def my_hook(error: CompileError) -> Exception | None:
        return None

    assert isinstance(my_hook, ErrorTransformHook)


def test_error_transform_hook_non_callable_fails() -> None:
    from semql.hooks import ErrorTransformHook

    assert not isinstance("not callable", ErrorTransformHook)


# ---------------------------------------------------------------------------
# Catalog.__init__ accepts error_transform
# ---------------------------------------------------------------------------


def test_catalog_accepts_error_transform_kwarg() -> None:
    def hook(error: CompileError) -> Exception | None:
        return None

    cat = _catalog(error_transform=hook)
    assert cat is not None


# ---------------------------------------------------------------------------
# Catalog.compile() applies the hook on CompileError
# ---------------------------------------------------------------------------


def test_error_transform_passthrough_none_re_raises_original() -> None:
    """Hook returning None means: keep the original CompileError."""
    called: list[CompileError] = []

    def hook(error: CompileError) -> Exception | None:
        called.append(error)
        return None

    cat = _catalog(error_transform=hook)
    q = SemanticQuery(measures=["orders.nonexistent_measure"])

    with pytest.raises(CompileError):
        cat.compile(q)

    assert len(called) == 1
    assert isinstance(called[0], CompileError)


def test_error_transform_replacement_raises_replacement() -> None:
    """Hook returning a new exception causes that exception to be raised."""

    class CustomError(Exception):
        pass

    def hook(error: CompileError) -> Exception | None:
        return CustomError(f"wrapped: {error}")

    cat = _catalog(error_transform=hook)
    q = SemanticQuery(measures=["orders.nonexistent_measure"])

    with pytest.raises(CustomError, match="wrapped"):
        cat.compile(q)


def test_error_transform_not_called_on_success() -> None:
    """Hook must not fire when compile succeeds."""
    called: list[CompileError] = []

    def hook(error: CompileError) -> Exception | None:
        called.append(error)
        return None

    cat = _catalog(error_transform=hook)
    q = SemanticQuery(measures=["orders.revenue"])
    cat.compile(q)  # should not raise

    assert len(called) == 0


def test_error_transform_receives_compile_error_instance() -> None:
    """Hook receives a real CompileError, not a wrapped one."""
    received: list[Exception] = []

    def hook(error: CompileError) -> Exception | None:
        received.append(error)
        return None

    cat = _catalog(error_transform=hook)
    q = SemanticQuery(dimensions=["orders.nonexistent"])

    with pytest.raises(CompileError):
        cat.compile(q)

    assert len(received) == 1
    assert isinstance(received[0], CompileError)


# ---------------------------------------------------------------------------
# No error_transform — normal behavior unchanged
# ---------------------------------------------------------------------------


def test_no_error_transform_compile_error_propagates() -> None:
    cat = _catalog()  # no hook
    q = SemanticQuery(measures=["orders.nonexistent_measure"])

    with pytest.raises(CompileError):
        cat.compile(q)


# ---------------------------------------------------------------------------
# Exported from semql
# ---------------------------------------------------------------------------


def test_error_transform_hook_exported_from_semql() -> None:
    import semql

    assert hasattr(semql, "ErrorTransformHook")
