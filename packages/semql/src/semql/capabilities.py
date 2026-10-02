"""Concrete, shared executable-capability checks for semantic queries."""

from __future__ import annotations

from semql.errors import ContractError, Diagnostic
from semql.spec import SemanticQuery


def check_query_capabilities(query: SemanticQuery) -> tuple[Diagnostic, ...]:
    """Return known unsupported shape decisions applicable to ``query``.

    These checks are intentionally independent of catalog resolution. Callers
    must still run authorization, semantic validation, and dialect-specific
    lowering checks before declaring an artifact executable.
    """
    diagnostics: list[Diagnostic] = []
    window = query.time_dimension
    if window is not None and window.fill_nulls_with is not None and query.dimensions:
        diagnostics.append(
            Diagnostic(
                code="capability_unsupported",
                reason="entity_time_fill_unsupported",
                references=(window.dimension, *query.dimensions),
                operation="fill",
                stage="lowering",
            )
        )
    if (
        query.derived_measures
        and not query.measures
        and not query.dimensions
        and (window is None or window.granularity is None)
    ):
        diagnostics.append(
            Diagnostic(
                code="capability_unsupported",
                reason="scalar_derived_only_unsupported",
                references=tuple(
                    operand for derived in query.derived_measures for operand in derived.operands
                ),
                operation="derive",
                stage="lowering",
            )
        )
    return tuple(diagnostics)


def require_query_capabilities(query: SemanticQuery) -> None:
    """Raise the shared typed failure for the first known unsupported shape."""
    diagnostics = check_query_capabilities(query)
    if diagnostics:
        diagnostic = diagnostics[0]
        raise ContractError(
            diagnostic.render_message(),
            reason=diagnostic.reason,
            references=diagnostic.references,
            operation=diagnostic.operation,
            stage=diagnostic.stage,
        )


__all__ = ["check_query_capabilities", "require_query_capabilities"]
