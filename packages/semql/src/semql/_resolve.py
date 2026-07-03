"""Shared identifier resolution for the semantic layer.

`compile.py` and `validate.py` both walk a `SemanticQuery` and resolve
every `cube.field` reference against the catalog. The walker lives
here so both modules see exactly the same resolution semantics — the
fail-fast compile path raises on the first batch of diagnostics, while
the collect-all validate path translates them into `ValidationError`s
and runs additional non-resolution checks on top.

`resolve_field` is the single-reference primitive; `walk_query_fields`
is the per-query walker that accumulates `ResolutionDiagnostic` records
without raising.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any

from semql.errors import (
    CompileError,
    FilterTypeError,
    ResolveError,
    UnknownIdentifierError,
    closest_match,
)
from semql.model import Cube, Dimension, Measure, Segment, TimeDimension, View
from semql.refs import is_qualified, parse_qualified_ref
from semql.spec import BoolExpr, Filter, SemanticQuery

if TYPE_CHECKING:
    pass


def split(qualified: str) -> tuple[str, str]:
    """Parse a ``"cube.field"`` qualified reference into its two halves.

    >>> split("orders.revenue")
    ('orders', 'revenue')
    """
    ref = parse_qualified_ref(qualified)
    return ref.cube, ref.field


def resolve_field(
    qualified: str,
    catalog: dict[str, Cube],
) -> tuple[Cube, Measure | Dimension | TimeDimension | Segment]:
    cube_name, field_name = split(qualified)
    if cube_name not in catalog:
        hint = closest_match(cube_name, catalog.keys())
        # Surface a richer alternatives list. The LLM repair loop
        # consumes this directly; we drop the 0.6 difflib cutoff for the
        # list version so any plausible token is offered, and let the
        # caller choose.
        alternatives = difflib.get_close_matches(cube_name, list(catalog), n=3, cutoff=0.4)
        known = ", ".join(sorted(catalog))
        suffix = f" Did you mean {hint!r}?" if hint else ""
        raise UnknownIdentifierError(
            f"Unknown cube: {cube_name!r}. Known cubes: {known}.{suffix}",
            kind="cube",
            name=cube_name,
            hint=hint,
            valid_alternatives=alternatives,
        )
    cube = catalog[cube_name]
    for m in cube.measures:
        if m.name == field_name:
            return cube, m
    for d in cube.dimensions:
        if d.name == field_name or field_name in d.aliases:
            return cube, d
    for td in cube.time_dimensions:
        if td.name == field_name:
            return cube, td
    for seg in cube.segments:
        if seg.name == field_name:
            return cube, seg
    hint = closest_match(field_name, cube.field_names())
    alternatives = difflib.get_close_matches(field_name, list(cube.field_names()), n=3, cutoff=0.4)
    known = ", ".join(sorted(cube.field_names()))
    suffix = f" Did you mean {hint!r}?" if hint else ""
    raise UnknownIdentifierError(
        f"Unknown field {field_name!r} on cube {cube_name!r}. Known fields: {known}.{suffix}",
        kind="field",
        name=field_name,
        cube=cube_name,
        hint=hint,
        valid_alternatives=alternatives,
    )


# ---------------------------------------------------------------------------
# Per-query walker — shared between compile.py and validate.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionDiagnostic:
    """One unresolved or mis-typed reference inside a `SemanticQuery`.

    `code` is the canonical machine-readable identifier used by
    `validate.ValidationError.code`. `source` is the original exception
    when we want the compile path to re-raise it standalone (used to
    preserve `FilterTypeError` typing when it's the only diagnostic)."""

    code: str
    message: str
    cube: str | None = None
    field: str | None = None
    op: str | None = None
    value: Any = None
    hint: str | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict[str, Any])
    source: Exception | None = None


@dataclass
class _ResolvedFields:
    """Field-level resolution output: every `cube.field` reference in the
    query mapped to a concrete `(Cube, Field)` pair, plus the ordered
    list of cubes the query touches.

    `where_leaf_resolutions` is keyed by the leaf's qualified `dimension`
    (the `cube.field` string) so the compiler can look each leaf up during
    AST emission. A leaf's resolved field depends only on `dimension`
    (the resolver calls `resolve_with_views(leaf.dimension)`), so a *copied*
    leaf — which IR transforms such as the federation split-point produce —
    resolves identically to the original. (Keying by `id(leaf)` instead, as
    an earlier version did, broke the moment a transform rebuilt a leaf.)
    The filter list mirrors `q.filters` order; the
    where-leaf dict mirrors a depth-first walk of `q.where`."""

    measure_fields: list[tuple[Cube, Measure]]
    dim_fields: list[tuple[Cube, Dimension]]
    time_cube: Cube | None
    time_dim: TimeDimension | None
    filter_resolutions: list[tuple[Filter, Cube, Dimension | Measure | TimeDimension | Segment]]
    where_leaf_resolutions: dict[str, tuple[Cube, Dimension | Measure | TimeDimension | Segment]]
    segment_resolutions: list[tuple[Cube, Segment]]
    touched: list[Cube]
    # (cube, measure) for every InlineDerived operand that resolves to a
    # measure. Kept apart from ``measure_fields`` so operands are *not*
    # emitted as output columns, yet still feed the fan-out guard — a
    # cross-cube operand that fans out must be refused like any measure.
    derived_operand_fields: list[tuple[Cube, Measure]]


def walk_where_leaves(expr: BoolExpr | Filter) -> list[Filter]:
    """Return all `Filter` leaves from a where tree, depth-first."""
    if isinstance(expr, Filter):
        return [expr]
    leaves: list[Filter] = []
    for child in expr.children:
        leaves.extend(walk_where_leaves(child))
    return leaves


def _filter_field_type(
    fld: Dimension | Measure | TimeDimension | Segment,
) -> str | None:
    """Return the dim_type to pass to `Filter.validate_for_type`, or
    `None` if the field type doesn't participate in filter-value
    typing (Measure on a `having` path is checked separately at the
    emission stage)."""
    if isinstance(fld, Dimension):
        return fld.type
    if isinstance(fld, TimeDimension):
        return "time"
    return None


def _make_view_resolver(
    catalog: dict[str, Cube],
    views_map: dict[str, View],
) -> Callable[[str], tuple[Cube, Measure | Dimension | TimeDimension | Segment]]:
    """Build the per-call view-aware resolver.

    Rewrites `view.local_name` references to the underlying
    `cube.field` BUT re-aliases the returned field so the SELECT
    output column uses the view's local name (not the underlying
    field name)."""

    def _resolve(
        qualified: str,
    ) -> tuple[Cube, Measure | Dimension | TimeDimension | Segment]:
        if is_qualified(qualified):
            ref = parse_qualified_ref(qualified)
            prefix, local = ref.cube, ref.field
            if prefix in views_map:
                view = views_map[prefix]
                if local not in view.fields:
                    raise CompileError(
                        f"View {prefix!r} has no field {local!r}. "
                        f"Known fields on this view: {sorted(view.fields)}."
                    )
                cube, fld = resolve_field(view.fields[local], catalog)
                return cube, fld.model_copy(update={"name": local})
        return resolve_field(qualified, catalog)

    return _resolve


def _diagnostic_from_resolve_exc(ref: str, exc: Exception) -> ResolutionDiagnostic:
    """Turn a resolve-time exception into a typed diagnostic."""
    if isinstance(exc, UnknownIdentifierError):
        code = "unknown_cube" if exc.kind == "cube" else "unknown_field"
        return ResolutionDiagnostic(
            code=code,
            message=str(exc),
            cube=exc.cube,
            field=exc.name if exc.kind == "field" else None,
            hint=exc.hint,
            source=exc,
        )
    if isinstance(exc, CompileError):
        # View-missing-field case from `_make_view_resolver`.
        return ResolutionDiagnostic(
            code="unknown_view_field",
            message=str(exc),
            field=ref,
            source=exc,
        )
    return ResolutionDiagnostic(
        code="bad_reference",
        message=str(exc),
        field=ref,
        source=exc,
    )


def walk_query_fields(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    views_map: dict[str, View] | None = None,
) -> tuple[_ResolvedFields, list[ResolutionDiagnostic]]:
    """Walk every projection / filter / where / segment reference in `q`
    and resolve against `catalog`. Returns the resolved bundle plus a
    list of diagnostics — one per unresolved or mis-typed reference.

    Never raises on bad input. The compile path turns diagnostics into
    `CompileError` (with single-error class preservation); the validate
    path turns them into `ValidationError` records.

    `_ResolvedFields.touched` includes cubes from every reference that
    *did* resolve, so authorisation / lifecycle / cross-backend checks
    downstream still see partially-resolved queries."""
    resolve_with_views = _make_view_resolver(catalog, views_map or {})
    diagnostics: list[ResolutionDiagnostic] = []

    measure_fields: list[tuple[Cube, Measure]] = []
    for ref in q.measures:
        try:
            cube, fld = resolve_with_views(ref)
        except Exception as exc:
            diagnostics.append(_diagnostic_from_resolve_exc(ref, exc))
            continue
        if not isinstance(fld, Measure):
            diagnostics.append(
                ResolutionDiagnostic(
                    code="wrong_field_kind",
                    message=f"{ref!r} is not a measure on cube {cube.name!r}.",
                    cube=cube.name,
                    field=fld.name,
                )
            )
            continue
        measure_fields.append((cube, fld))

    dim_fields: list[tuple[Cube, Dimension]] = []
    for ref in q.dimensions:
        try:
            cube, fld = resolve_with_views(ref)
        except Exception as exc:
            diagnostics.append(_diagnostic_from_resolve_exc(ref, exc))
            continue
        if not isinstance(fld, Dimension):
            diagnostics.append(
                ResolutionDiagnostic(
                    code="wrong_field_kind",
                    message=f"{ref!r} is not a dimension on cube {cube.name!r}.",
                    cube=cube.name,
                    field=fld.name,
                )
            )
            continue
        dim_fields.append((cube, fld))

    time_cube: Cube | None = None
    time_dim: TimeDimension | None = None
    if q.time_dimension is not None:
        tref = q.time_dimension.dimension
        try:
            tcube, tfld = resolve_with_views(tref)
        except Exception as exc:
            diagnostics.append(_diagnostic_from_resolve_exc(tref, exc))
            tcube, tfld = None, None
        if tfld is not None and not isinstance(tfld, TimeDimension):
            diagnostics.append(
                ResolutionDiagnostic(
                    code="wrong_field_kind",
                    message=f"{tref!r} is not a time dimension.",
                    cube=tcube.name if tcube is not None else None,
                    field=tfld.name,
                )
            )
        elif tfld is not None and tcube is not None:
            gran = q.time_dimension.granularity
            if gran is not None and gran not in tfld.granularities:
                diagnostics.append(
                    ResolutionDiagnostic(
                        code="bad_granularity",
                        message=(
                            f"Granularity {gran!r} not supported on {tref!r}. "
                            f"Allowed: {tfld.granularities}."
                        ),
                        cube=tcube.name,
                        field=tfld.name,
                        value=gran,
                        extra={"allowed": list(tfld.granularities)},
                    )
                )
            else:
                time_cube, time_dim = tcube, tfld

    touched: list[Cube] = []
    for c, _ in [*measure_fields, *dim_fields]:
        if c not in touched:
            touched.append(c)
    if time_cube is not None and time_cube not in touched:
        touched.append(time_cube)

    filter_resolutions: list[
        tuple[Filter, Cube, Dimension | Measure | TimeDimension | Segment]
    ] = []
    for f in q.filters:
        try:
            c, fld = resolve_with_views(f.dimension)
        except Exception as exc:
            diagnostics.append(_diagnostic_from_resolve_exc(f.dimension, exc))
            continue
        field_type = _filter_field_type(fld)
        if field_type is not None:
            try:
                f.validate_for_type(field_type)
            except ValueError as exc:
                diagnostics.append(
                    ResolutionDiagnostic(
                        code="filter_type_mismatch",
                        message=str(exc),
                        cube=c.name,
                        field=fld.name,
                        op=f.op,
                        value=f.values[0] if f.values else None,
                        source=FilterTypeError(
                            str(exc),
                            dimension=f.dimension,
                            op=f.op,
                            value=f.values[0] if f.values else None,
                        ),
                    )
                )
                continue
        filter_resolutions.append((f, c, fld))
        if c not in touched:
            touched.append(c)

    where_leaves: list[Filter] = walk_where_leaves(q.where) if q.where is not None else []
    where_leaf_resolutions: dict[
        str, tuple[Cube, Dimension | Measure | TimeDimension | Segment]
    ] = {}
    for leaf in where_leaves:
        try:
            c, fld = resolve_with_views(leaf.dimension)
        except Exception as exc:
            diagnostics.append(_diagnostic_from_resolve_exc(leaf.dimension, exc))
            continue
        field_type = _filter_field_type(fld)
        if field_type is not None:
            try:
                leaf.validate_for_type(field_type)
            except ValueError as exc:
                diagnostics.append(
                    ResolutionDiagnostic(
                        code="filter_type_mismatch",
                        message=str(exc),
                        cube=c.name,
                        field=fld.name,
                        op=leaf.op,
                        value=leaf.values[0] if leaf.values else None,
                        source=FilterTypeError(
                            str(exc),
                            dimension=leaf.dimension,
                            op=leaf.op,
                            value=leaf.values[0] if leaf.values else None,
                        ),
                    )
                )
                continue
        where_leaf_resolutions[leaf.dimension] = (c, fld)
        if c not in touched:
            touched.append(c)

    segment_resolutions: list[tuple[Cube, Segment]] = []
    for seg_ref in q.segments:
        if not is_qualified(seg_ref):
            diagnostics.append(
                ResolutionDiagnostic(
                    code="segment_unqualified",
                    message=f"Segment reference {seg_ref!r} must be qualified as 'cube.segment'.",
                    field=seg_ref,
                )
            )
            continue
        seg = parse_qualified_ref(seg_ref)
        cube_name, seg_name = seg.cube, seg.field
        if cube_name not in catalog:
            diagnostics.append(
                ResolutionDiagnostic(
                    code="segment_unknown_cube",
                    message=f"Segment reference {seg_ref!r}: unknown cube {cube_name!r}.",
                    cube=cube_name,
                    field=seg_name,
                )
            )
            continue
        cube_obj = catalog[cube_name]
        match = next((s for s in cube_obj.segments if s.name == seg_name), None)
        if match is None:
            known = ", ".join(s.name for s in cube_obj.segments) or "(none)"
            diagnostics.append(
                ResolutionDiagnostic(
                    code="segment_unknown_segment",
                    message=(
                        f"Segment reference {seg_ref!r}: cube {cube_name!r} has no segment "
                        f"{seg_name!r}. Known segments: {known}."
                    ),
                    cube=cube_name,
                    field=seg_name,
                )
            )
            continue
        segment_resolutions.append((cube_obj, match))
        if cube_obj not in touched:
            touched.append(cube_obj)

    # Inline derived measures (C17). Resolve each operand and pull its
    # cube into ``touched`` so the join graph joins it — an operand cube
    # referenced *only* through a derivation must still reach the FROM
    # clause. Operands that don't resolve are left for the compile
    # emitter to report precisely; here we just collect what resolves.
    # Declared ``dependencies`` (bridge cubes) are pulled in too.
    derived_operand_fields: list[tuple[Cube, Measure]] = []
    for ir in q.derived_measures:
        for operand_ref in ir.operands:
            try:
                c, fld = resolve_with_views(operand_ref)
            except Exception:
                continue  # emitter raises the precise CompileError.
            if isinstance(fld, Measure):
                derived_operand_fields.append((c, fld))
                if c not in touched:
                    touched.append(c)
        for dep in ir.dependencies:
            dep_cube = catalog.get(dep)
            if dep_cube is None:
                diagnostics.append(
                    ResolutionDiagnostic(
                        code="derived_unknown_dependency",
                        message=(
                            f"InlineDerived({ir.name!r}): dependency {dep!r} "
                            f"is not a cube in the catalog."
                        ),
                        cube=dep,
                    )
                )
                continue
            if dep_cube not in touched:
                touched.append(dep_cube)

    resolved = _ResolvedFields(
        measure_fields=measure_fields,
        dim_fields=dim_fields,
        time_cube=time_cube,
        time_dim=time_dim,
        filter_resolutions=filter_resolutions,
        where_leaf_resolutions=where_leaf_resolutions,
        segment_resolutions=segment_resolutions,
        touched=touched,
        derived_operand_fields=derived_operand_fields,
    )
    return resolved, diagnostics


__all__ = [
    "ResolutionDiagnostic",
    "ResolveError",
    "UnknownIdentifierError",
    "_ResolvedFields",
    "resolve_field",
    "split",
    "walk_query_fields",
    "walk_where_leaves",
]
