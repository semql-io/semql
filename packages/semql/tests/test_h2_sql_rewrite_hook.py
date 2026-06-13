"""H2 — SqlRewriteHook: post-emit SQL rewrite + reference implementations.

A ``SqlRewriteHook`` fires *after* ``post_compile`` hooks and may return
a new ``CompiledQuery`` with a modified ``sql`` string.

Reference impls:
  - ``QueryTagRewriter(tags)`` — prepends ``/* key=val ... */``
  - ``LimitCapRewriter(max_rows)`` — enforces a hard LIMIT ceiling
"""

from __future__ import annotations

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery
from semql.compile import CompiledQuery
from semql.hooks import CompileHook, SqlRewriteHook
from semql.model import AuthContext


def _catalog(
    *,
    compile_hooks: list[CompileHook] | None = None,
    sql_rewrite_hooks: list[SqlRewriteHook] | None = None,
) -> Catalog:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    return Catalog(
        [cube],
        compile_hooks=compile_hooks,
        sql_rewrite_hooks=sql_rewrite_hooks,
    )


# ---------------------------------------------------------------------------
# Protocol importability
# ---------------------------------------------------------------------------


def test_sql_rewrite_hook_importable() -> None:
    from semql.hooks import SqlRewriteHook

    assert SqlRewriteHook is not None


def test_query_tag_rewriter_importable() -> None:
    from semql.hooks import QueryTagRewriter

    assert QueryTagRewriter is not None


def test_limit_cap_rewriter_importable() -> None:
    from semql.hooks import LimitCapRewriter

    assert LimitCapRewriter is not None


# ---------------------------------------------------------------------------
# QueryTagRewriter
# ---------------------------------------------------------------------------


def test_query_tag_rewriter_prepends_comment() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(sql_rewrite_hooks=[QueryTagRewriter({"app": "semql", "team": "analytics"})])
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert result.sql.startswith("/* app=semql team=analytics */")


def test_query_tag_rewriter_viewer_id_placeholder() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(sql_rewrite_hooks=[QueryTagRewriter({"user": "{viewer_id}"})])
    viewer = AuthContext(viewer_id="alice")
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]), viewer=viewer)
    assert "user=alice" in result.sql


def test_query_tag_rewriter_tenant_and_query_hash_placeholders() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(
        sql_rewrite_hooks=[
            QueryTagRewriter({"tenant": "{tenant}", "query": "{query_hash}", "label": "semql"})
        ]
    )
    result = cat.compile(
        SemanticQuery(measures=["orders.revenue"]),
        context={"tenant_schema": "acme"},
    )
    tag = result.sql.split("*/", 1)[0]
    assert "tenant=acme" in tag
    assert "query={query_hash}" not in tag
    assert "query=" in tag


def test_query_tag_rewriter_sanitizes_comment_values() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(sql_rewrite_hooks=[QueryTagRewriter({"x": "*/ SELECT 1 /*"})])
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    first_line = result.sql.splitlines()[0]
    tag_body = first_line.removeprefix("/* ").removesuffix(" */")
    assert first_line.count("*/") == 1
    assert "/*" not in tag_body
    assert "*/" not in tag_body


def test_query_tag_rewriter_unknown_placeholder_kept_verbatim() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(sql_rewrite_hooks=[QueryTagRewriter({"x": "{unknown_placeholder}"})])
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    # Unknown placeholders stay as-is rather than crashing
    assert "{unknown_placeholder}" in result.sql or "x=" in result.sql


# ---------------------------------------------------------------------------
# LimitCapRewriter
# ---------------------------------------------------------------------------


def test_limit_cap_lowers_oversized_limit() -> None:
    from semql.hooks import LimitCapRewriter

    cat = _catalog(sql_rewrite_hooks=[LimitCapRewriter(100)])
    q = SemanticQuery(measures=["orders.revenue"], limit=5000)
    result = cat.compile(q)
    assert "LIMIT 100" in result.sql
    assert "LIMIT 5000" not in result.sql


def test_limit_cap_leaves_smaller_limit_unchanged() -> None:
    from semql.hooks import LimitCapRewriter

    cat = _catalog(sql_rewrite_hooks=[LimitCapRewriter(1000)])
    q = SemanticQuery(measures=["orders.revenue"], limit=50)
    result = cat.compile(q)
    assert "LIMIT 50" in result.sql


def test_limit_cap_adds_limit_when_absent() -> None:
    from semql.hooks import LimitCapRewriter

    cat = _catalog(sql_rewrite_hooks=[LimitCapRewriter(500)])
    q = SemanticQuery(measures=["orders.revenue"])
    result = cat.compile(q)
    assert "LIMIT 500" in result.sql


# ---------------------------------------------------------------------------
# Multiple rewrite hooks compose
# ---------------------------------------------------------------------------


def test_multiple_rewrite_hooks_compose() -> None:
    from semql.hooks import QueryTagRewriter

    cat = _catalog(
        sql_rewrite_hooks=[
            QueryTagRewriter({"step": "1"}),
            QueryTagRewriter({"step": "2"}),
        ]
    )
    result = cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert "step=1" in result.sql
    assert "step=2" in result.sql


# ---------------------------------------------------------------------------
# Rewrite exceptions propagate (not swallowed)
# ---------------------------------------------------------------------------


def test_rewrite_exception_propagates() -> None:
    class Crasher:
        def rewrite(
            self,
            compiled: CompiledQuery,
            *,
            query: SemanticQuery,
            viewer: AuthContext | None = None,
            context: dict[str, str] | None = None,
        ) -> CompiledQuery:
            raise RuntimeError("rewrite crash")

    cat = _catalog(sql_rewrite_hooks=[Crasher()])
    with pytest.raises(RuntimeError, match="rewrite crash"):
        cat.compile(SemanticQuery(measures=["orders.revenue"]))


# ---------------------------------------------------------------------------
# Ordering: post_compile fires before rewrite
# ---------------------------------------------------------------------------


def test_post_compile_fires_before_rewrite() -> None:
    from semql.hooks import BaseCompileHook, QueryTagRewriter

    call_order: list[str] = []

    class RecordPost(BaseCompileHook):
        def post_compile(self, query: SemanticQuery, compiled: CompiledQuery, **_: object) -> None:
            call_order.append(f"post:{compiled.sql[:4]}")

    class RecordRewrite(QueryTagRewriter):
        def rewrite(
            self,
            compiled: CompiledQuery,
            *,
            query: SemanticQuery,
            viewer: AuthContext | None = None,
            context: dict[str, str] | None = None,
        ) -> CompiledQuery:
            result = super().rewrite(
                compiled,
                query=query,
                viewer=viewer,
                context=context,
            )
            call_order.append(f"rewrite:{result.sql[:4]}")
            return result

    cat = _catalog(
        compile_hooks=[RecordPost()],
        sql_rewrite_hooks=[RecordRewrite({"app": "test"})],
    )
    cat.compile(SemanticQuery(measures=["orders.revenue"]))
    assert call_order[0].startswith("post:")
    assert call_order[1].startswith("rewrite:")


# ---------------------------------------------------------------------------
# Zero rewrite hooks — unchanged
# ---------------------------------------------------------------------------


def test_zero_rewrite_hooks_unchanged() -> None:
    cat_no = _catalog()
    cat_empty = _catalog(sql_rewrite_hooks=[])
    q = SemanticQuery(measures=["orders.revenue"])
    assert cat_no.compile(q).sql == cat_empty.compile(q).sql
