"""Extension hook Protocols for the semql compiler and prompt builder."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from semql.dialect import dialect_for as sqlglot_dialect_for

if TYPE_CHECKING:
    from semql.compile import CompiledQuery
    from semql.errors import SemQLError
    from semql.model import AuthContext, Cube
    from semql.spec import SemanticQuery


@runtime_checkable
class CompileHook(Protocol):
    """Protocol for hooks that intercept the compile lifecycle."""

    def pre_compile(
        self,
        query: SemanticQuery,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> SemanticQuery | None: ...

    def post_compile(
        self,
        query: SemanticQuery,
        compiled: CompiledQuery,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None: ...

    def on_compile_error(
        self,
        query: SemanticQuery,
        error: SemQLError,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None: ...


class BaseCompileHook:
    """Convenience base class with no-op implementations for CompileHook."""

    def pre_compile(
        self,
        query: SemanticQuery,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> SemanticQuery | None:
        return None

    def post_compile(
        self,
        query: SemanticQuery,
        compiled: CompiledQuery,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        pass

    def on_compile_error(
        self,
        query: SemanticQuery,
        error: SemQLError,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        pass


def _audit_tenant(
    viewer: AuthContext | None,
    context: dict[str, str] | None,
) -> str | None:
    """Tenant identifier recorded on an audit event.

    Prefers the identity's canonical ``viewer.tenant`` — that covers
    both schema and discriminator tenancy, so "who from tenant X queried
    what" is answerable regardless of mode. Falls back to the legacy
    ``context['tenant_schema']`` / ``context['tenant']`` for callers that
    still thread tenancy through the context dict."""
    if viewer is not None and viewer.tenant is not None:
        return viewer.tenant
    if context:
        return context.get("tenant_schema") or context.get("tenant")
    return None


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    query_hash: str
    viewer_id: str | None
    tenant: str | None
    cubes_accessed: list[str]
    measures_accessed: list[str]
    dimensions_accessed: list[str]
    segments_applied: list[str]
    filter_dimensions: list[str]
    sql_hash: str
    masked_fields: list[str]
    outcome: Literal["ok", "error"]
    error_code: str | None = None


class AuditHook(BaseCompileHook):
    def __init__(self, sink: Callable[[AuditEvent], None]) -> None:
        self.sink = sink

    def _extract_cubes(self, query: SemanticQuery) -> list[str]:
        cubes: set[str] = set()
        for ref in query.measures + query.dimensions:
            if "." in ref:
                cubes.add(ref.split(".")[0])
        for f in query.filters:
            if "." in f.dimension:
                cubes.add(f.dimension.split(".")[0])
        return sorted(list(cubes))

    def _extract_measures(self, query: SemanticQuery) -> list[str]:
        measures: set[str] = set()
        for ref in query.measures:
            if "." in ref:
                measures.add(ref.split(".")[1])
            else:
                measures.add(ref)
        return sorted(list(measures))

    def _extract_filter_dims(self, query: SemanticQuery) -> list[str]:
        dims: set[str] = set()
        for f in query.filters:
            if "." in f.dimension:
                dims.add(f.dimension.split(".")[1])
            else:
                dims.add(f.dimension)
        return sorted(list(dims))

    def _query_hash(self, query: SemanticQuery) -> str:
        return hashlib.sha256(query.model_dump_json().encode("utf-8")).hexdigest()

    def _sql_hash(self, sql: str) -> str:
        return hashlib.sha256(sql.encode("utf-8")).hexdigest()

    def _extract_dimensions(self, query: SemanticQuery) -> list[str]:
        dims = {ref.split(".", 1)[1] if "." in ref else ref for ref in query.dimensions}
        if query.time_dimension is not None:
            ref = query.time_dimension.dimension
            dims.add(ref.split(".", 1)[1] if "." in ref else ref)
        return sorted(dims)

    def _extract_segments(self, query: SemanticQuery) -> list[str]:
        return sorted(ref.split(".", 1)[1] if "." in ref else ref for ref in query.segments)

    def post_compile(
        self,
        query: SemanticQuery,
        compiled: CompiledQuery,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        viewer_id = viewer.viewer_id if viewer else None
        tenant = _audit_tenant(viewer, context)

        event = AuditEvent(
            timestamp=datetime.now(UTC),
            query_hash=self._query_hash(query),
            viewer_id=viewer_id,
            tenant=tenant,
            cubes_accessed=compiled.touched_cube_names,
            measures_accessed=self._extract_measures(query),
            dimensions_accessed=self._extract_dimensions(query),
            segments_applied=self._extract_segments(query),
            filter_dimensions=self._extract_filter_dims(query),
            sql_hash=self._sql_hash(compiled.sql),
            masked_fields=[],
            outcome="ok",
        )
        self.sink(event)

    def on_compile_error(
        self,
        query: SemanticQuery,
        error: SemQLError,
        *,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        viewer_id = viewer.viewer_id if viewer else None
        tenant = _audit_tenant(viewer, context)

        event = AuditEvent(
            timestamp=datetime.now(UTC),
            query_hash=self._query_hash(query),
            viewer_id=viewer_id,
            tenant=tenant,
            cubes_accessed=self._extract_cubes(query),
            measures_accessed=self._extract_measures(query),
            dimensions_accessed=self._extract_dimensions(query),
            segments_applied=self._extract_segments(query),
            filter_dimensions=self._extract_filter_dims(query),
            sql_hash="",
            masked_fields=[],
            outcome="error",
            error_code=type(error).__name__,
        )
        self.sink(event)


@runtime_checkable
class SqlRewriteHook(Protocol):
    """Protocol for hooks that rewrite the compiled SQL string."""

    def rewrite(
        self,
        compiled: CompiledQuery,
        *,
        query: SemanticQuery,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> CompiledQuery: ...


class QueryTagRewriter:
    def __init__(self, tags: dict[str, str]) -> None:
        self.tags = tags

    def rewrite(
        self,
        compiled: CompiledQuery,
        *,
        query: SemanticQuery,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> CompiledQuery:
        rendered_tags: list[str] = []
        for k, v in self.tags.items():
            rendered_tags.append(
                f"{self._sanitize(k)}={self._render_value(v, query, viewer, context)}"
            )

        tag_str = "/* " + " ".join(rendered_tags) + " */\n"
        from dataclasses import replace

        return replace(compiled, sql=tag_str + compiled.sql)

    def _render_value(
        self,
        value: str,
        query: SemanticQuery,
        viewer: AuthContext | None,
        context: dict[str, str] | None,
    ) -> str:
        replacements = {
            "viewer_id": viewer.viewer_id if viewer is not None else "",
            "tenant": context.get("tenant_schema", "") if context else "",
            "query_hash": hashlib.sha256(query.model_dump_json().encode("utf-8")).hexdigest(),
        }
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace("{" + key + "}", replacement)
        return self._sanitize(rendered)

    def _sanitize(self, value: str) -> str:
        value = value.replace("/*", "").replace("*/", "")
        value = re.sub(r"[^A-Za-z0-9_.:=@/-]+", "_", value).strip("_")
        return value


class LimitCapRewriter:
    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows

    def rewrite(
        self,
        compiled: CompiledQuery,
        *,
        query: SemanticQuery,
        viewer: AuthContext | None = None,
        context: dict[str, str] | None = None,
    ) -> CompiledQuery:
        import sqlglot
        from sqlglot import exp
        from sqlglot.errors import ParseError

        # We need to parse the SQL, modify the LIMIT, and emit it back.
        # This is a bit heavy, but it's the correct way.
        dialect = sqlglot_dialect_for(compiled.dialect)
        try:
            ast = sqlglot.parse_one(compiled.sql, dialect=dialect)
        except ParseError:
            # If we can't parse it, leave it alone.
            return compiled

        if not isinstance(ast, exp.Select):
            return compiled

        current_limit = ast.args.get("limit")
        if current_limit is not None:
            # sqlglot limits are expressions, we need to try to parse it
            # For simplicity, if it's a number literal, we cap it.
            if isinstance(current_limit.expression, exp.Literal):
                try:
                    val = int(current_limit.expression.name)
                    if val > self.max_rows:
                        ast = ast.limit(self.max_rows)
                except ValueError:
                    pass
        else:
            ast = ast.limit(self.max_rows)

        new_sql = ast.sql(dialect=dialect, pretty=False, normalize_functions=False)
        from dataclasses import replace

        return replace(compiled, sql=new_sql)


@runtime_checkable
class CubePromptHook(Protocol):
    """Callable appended after a cube's block in the planner prompt.

    Return extra text (e.g. usage notes, example queries, warnings) to
    splice in after that cube's section. Return ``""`` to add nothing.
    """

    def __call__(self, cube: Cube) -> str: ...


@runtime_checkable
class ErrorTransformHook(Protocol):
    """Callable invoked when ``Catalog.compile()`` raises a semantic error.

    Return a replacement exception to raise instead, or ``None`` to
    re-raise the original error unchanged.
    """

    def __call__(self, error: SemQLError) -> Exception | None: ...
