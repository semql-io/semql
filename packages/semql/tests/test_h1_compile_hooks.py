"""H1 — CompileHook Protocol: pre_compile / post_compile / on_compile_error.

A ``CompileHook`` registered on ``Catalog`` can:
  - mutate the query before compilation (pre_compile pipe)
  - observe the compiled result (post_compile broadcast)
  - observe compilation errors (on_compile_error broadcast)

``AuditHook(sink)`` is a reference implementation that forwards a
structured ``AuditEvent`` to a caller-supplied sink callable.
"""

from __future__ import annotations

import warnings

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery
from semql.compile import CompiledQuery
from semql.errors import CompileError, SemQLError
from semql.hooks import CompileHook


def _catalog(
    *,
    compile_hooks: list[CompileHook] | None = None,
) -> Catalog:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    return Catalog([cube], compile_hooks=compile_hooks)


# ---------------------------------------------------------------------------
# Protocol importability
# ---------------------------------------------------------------------------


def test_compile_hook_importable() -> None:
    from semql.hooks import CompileHook

    assert CompileHook is not None


def test_base_compile_hook_importable() -> None:
    from semql.hooks import BaseCompileHook

    assert BaseCompileHook is not None


def test_audit_hook_importable() -> None:
    from semql.hooks import AuditEvent, AuditHook

    assert AuditEvent is not None
    assert AuditHook is not None


# ---------------------------------------------------------------------------
# pre_compile — mutation
# ---------------------------------------------------------------------------


def test_pre_compile_mutation_injects_limit() -> None:
    from semql.hooks import BaseCompileHook

    class InjectLimit(BaseCompileHook):
        def pre_compile(self, query: SemanticQuery, **_: object) -> SemanticQuery | None:
            return query.model_copy(update={"limit": 100})

    cat = _catalog(compile_hooks=[InjectLimit()])
    q = SemanticQuery(measures=["orders.revenue"])
    result = cat.compile(q)
    assert "LIMIT 100" in result.sql


def test_pre_compile_none_return_uses_original() -> None:
    from semql.hooks import BaseCompileHook

    class NoOp(BaseCompileHook):
        def pre_compile(self, query: SemanticQuery, **_: object) -> SemanticQuery | None:
            return None

    cat = _catalog(compile_hooks=[NoOp()])
    q = SemanticQuery(measures=["orders.revenue"])
    result = cat.compile(q)
    assert "LIMIT" not in result.sql


def test_pre_compile_pipe_two_hooks() -> None:
    """Second hook receives the output of the first."""
    from semql.hooks import BaseCompileHook

    calls: list[str] = []

    class First(BaseCompileHook):
        def pre_compile(self, query: SemanticQuery, **_: object) -> SemanticQuery | None:
            calls.append("first")
            return query.model_copy(update={"limit": 200})

    class Second(BaseCompileHook):
        def pre_compile(self, query: SemanticQuery, **_: object) -> SemanticQuery | None:
            calls.append(f"second:{query.limit}")
            return query.model_copy(update={"limit": 50})

    cat = _catalog(compile_hooks=[First(), Second()])
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert calls == ["first", "second:200"]
    assert "LIMIT 50" in result.sql


def test_pre_compile_raising_aborts_compilation() -> None:
    from semql.hooks import BaseCompileHook

    class Blocker(BaseCompileHook):
        def pre_compile(self, query: SemanticQuery, **_: object) -> SemanticQuery | None:
            raise CompileError("blocked by hook")

    cat = _catalog(compile_hooks=[Blocker()])
    with pytest.raises(CompileError, match="blocked by hook"):
        cat.compile(SemanticQuery(measures=["orders.revenue"]))


# ---------------------------------------------------------------------------
# post_compile — observation
# ---------------------------------------------------------------------------


def test_post_compile_fires_after_success() -> None:
    from semql.hooks import BaseCompileHook

    received: list[CompiledQuery] = []

    class Observer(BaseCompileHook):
        def post_compile(self, query: SemanticQuery, compiled: CompiledQuery, **_: object) -> None:
            received.append(compiled)

    cat = _catalog(compile_hooks=[Observer()])
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert len(received) == 1
    assert received[0] is result


def test_post_compile_exception_swallowed_and_warned() -> None:
    from semql.hooks import BaseCompileHook

    class Crasher(BaseCompileHook):
        def post_compile(self, query: SemanticQuery, compiled: CompiledQuery, **_: object) -> None:
            raise RuntimeError("post hook crash")

    cat = _catalog(compile_hooks=[Crasher()])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert result is not None
    assert any("post hook crash" in str(warning.message) for warning in w)


def test_post_compile_not_called_on_error() -> None:
    from semql.hooks import BaseCompileHook

    called: list[bool] = []

    class Observer(BaseCompileHook):
        def post_compile(self, query: SemanticQuery, compiled: CompiledQuery, **_: object) -> None:
            called.append(True)

    cat = _catalog(compile_hooks=[Observer()])
    with pytest.raises(CompileError):
        cat.compile(SemanticQuery(measures=["orders.unknown_field"]))
    assert called == []


# ---------------------------------------------------------------------------
# on_compile_error — error observation
# ---------------------------------------------------------------------------


def test_on_compile_error_fires_on_failure() -> None:
    from semql.hooks import BaseCompileHook

    errors: list[Exception] = []

    class ErrorObserver(BaseCompileHook):
        def on_compile_error(self, query: SemanticQuery, error: SemQLError, **_: object) -> None:
            errors.append(error)

    cat = _catalog(compile_hooks=[ErrorObserver()])
    with pytest.raises(CompileError):
        cat.compile(SemanticQuery(measures=["orders.bad_field"]))
    assert len(errors) == 1
    assert isinstance(errors[0], CompileError)


def test_on_compile_error_hook_exception_swallowed() -> None:
    from semql.hooks import BaseCompileHook

    class CrashOnError(BaseCompileHook):
        def on_compile_error(self, query: SemanticQuery, error: SemQLError, **_: object) -> None:
            raise RuntimeError("error hook crash")

    cat = _catalog(compile_hooks=[CrashOnError()])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(CompileError):
            # Original CompileError still propagates; RuntimeError is swallowed.
            cat.compile(SemanticQuery(measures=["orders.bad_field"]))
    assert any("error hook crash" in str(warning.message) for warning in w)


def test_on_compile_error_not_called_on_success() -> None:
    from semql.hooks import BaseCompileHook

    called: list[bool] = []

    class ErrorObserver(BaseCompileHook):
        def on_compile_error(self, query: SemanticQuery, error: SemQLError, **_: object) -> None:
            called.append(True)

    cat = _catalog(compile_hooks=[ErrorObserver()])
    cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert called == []


# ---------------------------------------------------------------------------
# Zero hooks — unchanged behaviour
# ---------------------------------------------------------------------------


def test_zero_hooks_compile_unchanged() -> None:
    cat_no_hooks = _catalog()
    cat_empty_hooks = _catalog(compile_hooks=[])
    q = SemanticQuery(measures=["orders.revenue"])
    assert cat_no_hooks.compile(q).sql == cat_empty_hooks.compile(q).sql


# ---------------------------------------------------------------------------
# AuditHook
# ---------------------------------------------------------------------------


def test_audit_hook_ok_event() -> None:
    from semql.hooks import AuditEvent, AuditHook

    events: list[AuditEvent] = []
    cat = _catalog(compile_hooks=[AuditHook(events.append)])
    cat.compile(SemanticQuery(measures=["orders.revenue"]), context={"tenant_schema": "acme"})
    assert len(events) == 1
    ev = events[0]
    assert ev.outcome == "ok"
    assert "orders" in ev.cubes_accessed
    assert "revenue" in ev.measures_accessed
    assert ev.tenant == "acme"
    assert ev.query_hash
    assert ev.sql_hash
    assert ev.error_code is None


def test_audit_hook_error_event() -> None:
    from semql.hooks import AuditEvent, AuditHook

    events: list[AuditEvent] = []
    cat = _catalog(compile_hooks=[AuditHook(events.append)])
    with pytest.raises(CompileError):
        cat.compile(SemanticQuery(measures=["orders.bad_field"]))
    assert len(events) == 1
    ev = events[0]
    assert ev.outcome == "error"
    assert ev.error_code is not None
    assert ev.sql_hash == ""


def test_audit_hook_filter_dimensions_not_values() -> None:
    from semql.hooks import AuditEvent, AuditHook
    from semql.spec import Filter

    events: list[AuditEvent] = []
    cat = _catalog(compile_hooks=[AuditHook(events.append)])
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
    )
    cat.compile(q)
    ev = events[0]
    assert "status" in ev.filter_dimensions
    assert not hasattr(ev, "query")
    # Values must NOT appear — PII risk
    assert "paid" not in repr(ev)
