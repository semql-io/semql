"""Collect-all static validation of a SemanticQuery against a catalog.

``compile_query`` fails at the first problem with a CompileError;
``validate`` collects every problem it can find and returns them as a
list of ``ValidationError`` records. Two contracts (PHILOSOPHY.md):

- ``compile()`` is for the hot path — fail-fast, structured exception.
- ``validate()`` is for the planner-feedback path — surface everything
  the user / LLM needs to fix in one round-trip.

``validate`` never raises on input it would otherwise complain about.
It returns an empty list when the query is compile-ready.

The resolution walk itself lives in :mod:`semql._resolve`; this module
maps its :class:`~semql._resolve.ResolutionDiagnostic` records into
:class:`ValidationError` records and layers on the non-resolution
checks (lifecycle, required filters, ungrouped row caps, HAVING
references, cross-backend refusals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import TYPE_CHECKING, Any

from semql._resolve import ResolutionDiagnostic, walk_query_fields
from semql.errors import closest_match
from semql.introspect import PolicyFn, viewer_sees
from semql.model import AuthContext, Cube
from semql.spec import SemanticQuery

if TYPE_CHECKING:
    # ``Catalog`` is the runtime type that callers actually pass, but
    # importing it at top level would create a validate ↔ catalog
    # cycle (catalog's ``compile_collect_all`` method calls back into
    # validate). The TYPE_CHECKING block lets the type signature
    # advertise ``Catalog | dict`` while the runtime check stays
    # duck-typed (``hasattr(catalog, 'relations')``).
    from semql.catalog import Catalog


MAX_UNGROUPED_ROWS = 1000

_BACKTICK_RE = re.compile(r"`([a-z_][a-z0-9_]*)`")

ErrorCode = str  # documented values listed in this module's docstring


@dataclass(frozen=True)
class ValidationError:
    """One problem with a query.

    ``code`` is a stable identifier callers can branch on (see this
    module's leading docstring for the catalog of codes). ``message``
    is a human-readable explanation. The remaining fields carry the
    structure the message refers to — populated when applicable.
    """

    code: ErrorCode
    message: str
    cube: str | None = None
    field: str | None = None
    op: str | None = None
    value: Any = None
    hint: str | None = None
    extra: dict[str, Any] = _dc_field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ValidationWarning(ValidationError):
    """An advisory warning returned by :func:`validate`.

    Subclasses :class:`ValidationError` so existing ``isinstance(e,
    ValidationError)`` checks keep working. Filter by severity with
    ``[e for e in errors if isinstance(e, ValidationWarning)]``.
    """


def _catalog_dict(catalog: Catalog | dict[str, Cube]) -> dict[str, Cube]:
    if isinstance(catalog, dict):
        return catalog
    return catalog.as_dict()


def _to_validation_error(d: ResolutionDiagnostic) -> ValidationError:
    return ValidationError(
        code=d.code,
        message=d.message,
        cube=d.cube,
        field=d.field,
        op=d.op,
        value=d.value,
        hint=d.hint,
        extra=dict(d.extra),
    )


def validate(
    query: SemanticQuery,
    catalog: Catalog | dict[str, Cube],
    *,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
) -> list[ValidationError]:
    """Return every problem the static checker can find in ``query``.

    Returns ``[]`` for a query that ``compile_query`` would also accept.
    Never raises on input.

    ``viewer`` / ``policy`` filter the catalog used for identifier
    resolution so error messages don't enumerate cubes the viewer
    can't access (SEMQL-RESOLVER-DIAGNOSTIC-HIDDEN-CATALOG-ENUMERATION).
    """
    cat = _catalog_dict(catalog)
    if viewer is not None:
        cat = {k: v for k, v in cat.items() if viewer_sees(v, viewer, policy)}
    errors: list[ValidationError] = []

    if not query.measures and not query.dimensions and query.time_dimension is None:
        errors.append(
            ValidationError(
                code="empty_query",
                message=(
                    "SemanticQuery is empty — at least one measure, "
                    "dimension, or time_dimension is required."
                ),
            )
        )

    resolved, diagnostics = walk_query_fields(query, cat)
    for d in diagnostics:
        errors.append(_to_validation_error(d))

    touched = resolved.touched

    filter_dim_refs = {f.dimension for f in query.filters}
    for c in touched:
        for req in c.required_filters:
            ref = f"{c.name}.{req}"
            if ref not in filter_dim_refs:
                errors.append(
                    ValidationError(
                        code="missing_required_filter",
                        message=(
                            f"Cube {c.name!r} requires a filter on {req!r}. "
                            f"Add Filter(dimension='{ref}', op=..., values=[...])."
                        ),
                        cube=c.name,
                        field=req,
                    )
                )

    if query.ungrouped:
        if query.limit is None:
            errors.append(
                ValidationError(
                    code="ungrouped_no_limit",
                    message=(
                        f"ungrouped=True queries must set a limit (maximum {MAX_UNGROUPED_ROWS})."
                    ),
                )
            )
        elif query.limit > MAX_UNGROUPED_ROWS:
            errors.append(
                ValidationError(
                    code="ungrouped_limit_too_high",
                    message=(
                        f"ungrouped=True requires limit <= {MAX_UNGROUPED_ROWS}. "
                        f"Got limit={query.limit}."
                    ),
                    value=query.limit,
                )
            )

    if query.offset is not None and query.offset > 0 and query.limit is None:
        errors.append(
            ValidationError(
                code="offset_without_limit",
                message=(
                    "SemanticQuery has offset set without limit. "
                    "OFFSET is only meaningful in combination with LIMIT."
                ),
                value=query.offset,
            )
        )

    query_measure_short_names: set[str] = set()
    for ref in query.measures:
        if "." in ref:
            query_measure_short_names.add(ref.rsplit(".", 1)[-1])
        else:
            query_measure_short_names.add(ref)
    for hf in query.having:
        short = hf.dimension.rsplit(".", 1)[-1] if "." in hf.dimension else hf.dimension
        if short not in query_measure_short_names:
            hint = (
                closest_match(short, query_measure_short_names)
                if query_measure_short_names
                else None
            )
            errors.append(
                ValidationError(
                    code="having_unknown_measure",
                    message=(
                        f"HAVING references {hf.dimension!r}, which is not a measure in this query."
                    ),
                    field=hf.dimension,
                    hint=hint,
                )
            )

    if query.compare is not None:
        errors.append(
            ValidationError(
                code="compare_unsupported",
                message=("compare windows are not yet supported by the compiler (Phase 2)."),
            )
        )

    # Lifecycle hints. ``deprecated`` is a hard refusal (the
    # compiler raises) so ``validate`` surfaces it as an error too —
    # mirrors ``compile`` for the planner-feedback loop. ``beta`` is a
    # soft advisory: included in the result so a UI can warn the user
    # without blocking the query.
    for c in touched:
        if c.stability == "deprecated":
            if c.replacement is not None:
                msg = f"Cube {c.name!r} is deprecated. Use {c.replacement!r} instead."
            else:
                msg = f"Cube {c.name!r} is deprecated with no replacement; remove the reference."
            errors.append(
                ValidationError(
                    code="cube_deprecated",
                    message=msg,
                    cube=c.name,
                    hint=c.replacement,
                )
            )
        elif c.stability == "beta":
            errors.append(
                ValidationWarning(
                    code="cube_beta",
                    message=(
                        f"Cube {c.name!r} is flagged beta — its surface may "
                        "change. Pin to a stable cube for production workloads."
                    ),
                    cube=c.name,
                )
            )

    backends = {c.dialect for c in touched}
    if len(backends) > 1:
        names = sorted(b.value for b in backends)
        errors.append(
            ValidationError(
                code="cross_backend",
                message=(
                    "Cross-backend queries are not yet supported (Phase 2). "
                    f"Touched backends: {names}."
                ),
                extra={"backends": names},
            )
        )

    # Backtick-name resolution check. Scan relations narratives for
    # `token` patterns and warn when the token doesn't resolve to a
    # cube name, measure, or dimension in the catalog.
    known_tokens: set[str] = set(cat)
    for cube in cat.values():
        for m in cube.measures:
            known_tokens.add(m.name)
        for dim in cube.dimensions:
            known_tokens.add(dim.name)
        for td in cube.time_dimensions:
            known_tokens.add(td.name)

    # Build a quick map: token -> replacement (or True if deprecated
    # with no successor). Used by the deprecated-sibling-ref warning
    # below; only cube-level deprecation is tracked today (Cube is
    # the only model with stability/replacement fields).
    deprecated_replacements: dict[str, str | None] = {
        c.name: c.replacement
        for c in cat.values()
        if c.stability == "deprecated" and c.name in known_tokens
    }

    def _check_relations_text(text: str, source_cube: str | None) -> None:
        for match in _BACKTICK_RE.finditer(text):
            token = match.group(1)
            if token not in known_tokens:
                errors.append(
                    ValidationWarning(
                        code="unresolved_backtick",
                        message=(
                            f"Backtick name `{token}` in "
                            + (f"cube {source_cube!r} " if source_cube else "catalog ")
                            + "relations does not resolve to any known cube, "
                            "measure, or dimension."
                        ),
                        cube=source_cube,
                    )
                )
            elif token in deprecated_replacements:
                replacement = deprecated_replacements[token]
                if replacement is not None:
                    msg = (
                        f"Backtick name `{token}` in "
                        + (f"cube {source_cube!r} " if source_cube else "catalog ")
                        + f"relations points at a deprecated cube. Use "
                        f"`{replacement}` instead."
                    )
                else:
                    msg = (
                        f"Backtick name `{token}` in "
                        + (f"cube {source_cube!r} " if source_cube else "catalog ")
                        + "relations points at a deprecated cube with no replacement; "
                        "remove the reference."
                    )
                errors.append(
                    ValidationWarning(
                        code="deprecated_sibling_ref",
                        message=msg,
                        cube=source_cube,
                    )
                )

    for cube in cat.values():
        if cube.relations:
            _check_relations_text(cube.relations, cube.name)

    # Also check catalog-level relations when a full Catalog is passed.
    # ``hasattr(..., 'relations')`` keeps the check duck-typed so
    # ``validate`` doesn't need to import the Catalog class. The
    # Catalog's ``relations`` attribute is a stable part of the
    # public surface; the alternative ``isinstance`` check would
    # require a top-level catalog import (cycle).
    catalog_relations = getattr(catalog, "relations", None)
    if catalog_relations:
        _check_relations_text(catalog_relations, None)

    return errors


__all__ = ["MAX_UNGROUPED_ROWS", "ValidationError", "ValidationWarning", "validate"]
