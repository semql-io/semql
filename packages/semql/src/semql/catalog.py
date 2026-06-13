"""The Catalog wrapper — one object that owns a list of cubes.

`Catalog` is the high-level API people import. It validates the cube
graph at construction time, auto-appends the reflection META cubes, and
provides convenience methods that wrap the lower-level
``compile_query`` function. Prompt rendering lives in the separate
``semql-prompt`` package (``semql_prompt.planner_prompt(catalog, ...)``).

B5 — CatalogSpec / CatalogRuntime split:
- :class:`CatalogSpec` is a frozen Pydantic value type holding the
  serialisable catalog data (cubes, views, lookups, saved queries,
  glossary, relations, hook names). Round-trips through
  ``model_dump`` / ``from_dict`` so a catalog can cross a process
  boundary (cached specs, migrations, multi-tenant overrides).
- :class:`CatalogRuntime` is a frozen dataclass for the callables
  (policy, scope_fns, unit_registry, error_transform, hooks). Not
  part of the serialised payload.
- :class:`Catalog` pairs a spec with a runtime. The legacy public
  constructor (``Catalog(cubes=..., policy=..., ...)``) builds both
  internally and stays backward-compatible.

Construction-time validation:
- No duplicate cube names.
- Every ``Join.to`` resolves to a cube in the catalog.

Both are reasons a query would fail at compile time later — surfacing
them at catalog construction means the planner and MCP layer can
trust the input. ``CatalogSpec.from_iterables`` is the
collect-all counterpart for callers that want every problem
aggregated into a single list instead of a first-error raise.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from semql._grounding import validate_relations
from semql.compile import CompiledQuery, compile_query, explain_plan
from semql.errors import FilterTypeError, SemQLError
from semql.hooks import CompileHook, SqlRewriteHook
from semql.introspect import META_CUBES, PolicyFn, ScopeFn
from semql.model import (
    AuthContext,
    BaseField,
    Cube,
    DerivedTable,
    Dimension,
    GlossaryEntry,
    Join,
    Lookup,
    View,
)
from semql.retrieve import EmbeddingProvider, Retriever, build_default_retriever
from semql.spec import SavedQuery, SemanticQuery, SemanticQueryDefaults, _apply_query_defaults
from semql.units import DEFAULT_REGISTRY, Registry
from semql.validate import ValidationError, validate

_T = TypeVar("_T", bound=BaseField)


class CatalogSpec(BaseModel):
    """B5 — the serialisable half of a Catalog.

    A :class:`CatalogSpec` carries every field that survives a process
    boundary: cubes, views, lookups, saved queries, glossary, the
    catalog-wide relations narrative, and the names of the compile /
    sql-rewrite hooks (the callables themselves live on the runtime
    — names are the wire-format handle). Round-trips byte-stable
    through ``model_dump`` / ``from_dict``.

    Specs are immutable: mutate via ``model_copy(update=...)`` and
    re-construct the catalog, or use :meth:`from_iterables` for a
    collect-all build that aggregates every construction error into
    a structured list (the PHILOSOPHY.md validate-path promise).

    Hooks as names: ``compile_hook_names`` is a tuple of strings
    (e.g. ``("myapp.audit_hook",)``) — the runtime side is
    responsible for resolving the name back to a callable. This
    keeps the spec serialisable without dragging the callables
    into the payload.

    The ``schema_version`` field is bumped on a breaking change to
    the spec shape. ``from_dict`` surfaces a clear error on a stale
    payload rather than silently corrupting state.
    """

    model_config = ConfigDict(frozen=True)

    #: Bump on a breaking change to the wire format.
    schema_version: int = 1
    cubes: tuple[Cube, ...] = ()
    views: tuple[View, ...] = ()
    lookups: tuple[Lookup, ...] = ()
    saved_queries: tuple[SavedQuery, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()
    relations: str = ""
    compile_hook_names: tuple[str, ...] = ()
    sql_rewrite_hook_names: tuple[str, ...] = ()

    #: Aggregated construction errors from :meth:`from_iterables`.
    #: ``()`` for a clean build; populated when collect-all surfaces
    #: a problem the legacy first-error constructor would have raised.
    construction_errors: tuple[dict[str, Any], ...] = Field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CatalogSpec:
        """Construct a :class:`CatalogSpec` from a dict payload.

        Equivalent to ``CatalogSpec.model_validate(payload)`` —
        named for the wire-format use case (read from JSON / a DB
        row / a serialised cache)."""
        return cls.model_validate(payload)

    @classmethod
    def from_iterables(
        cls,
        *,
        cubes: Sequence[Cube] = (),
        views: Sequence[View] | None = None,
        lookups: Sequence[Lookup] | None = None,
        saved_queries: Sequence[SavedQuery] | None = None,
        glossary: Sequence[GlossaryEntry] | None = None,
        relations: str = "",
        compile_hook_names: Sequence[str] = (),
        sql_rewrite_hook_names: Sequence[str] = (),
    ) -> tuple[CatalogSpec, list[dict[str, Any]]]:
        """Collect-all constructor for a spec.

        Runs every validation the legacy :class:`Catalog` constructor
        would have raised on (duplicate cube names, unknown join
        targets, view ref resolution, lookup dim resolution, saved
        query collisions, glossary token collisions, scope function
        registration, primary-key declarations, replacement
        pointers, ...) and aggregates the failures into a structured
        ``errors`` list. The spec itself is returned regardless of
        whether errors were collected — a caller can inspect the
        partial spec for diagnostics, then decide whether to surface
        the errors or pair with a runtime anyway.

        Each error is a dict shaped ``{"code", "message", ...}``
        (the B8 envelope shape). Codes are stable identifiers:

        - ``duplicate_cube_name``
        - ``unknown_primary_key_dimension``
        - ``unknown_join_target``
        - ``unknown_foreign_key_target``
        - ``foreign_key_target_no_primary_key``
        - ``missing_primary_key_for_foreign_key``
        - ``duplicate_cte_name``
        - ``duplicate_view_name``
        - ``view_collides_with_cube``
        - ``unknown_view_target_cube``
        - ``unknown_view_target_field``
        - ``view_target_field_wrong_type``
        - ``unit_display_without_unit``
        - ``unknown_unit_conversion``
        - ``duplicate_lookup_dimension``
        - ``unknown_lookup_cube``
        - ``unknown_lookup_dimension``
        - ``lookup_dimension_wrong_type``
        - ``invalid_saved_query_name``
        - ``duplicate_saved_query_name``
        - ``saved_query_collides_with_cube_or_view``
        - ``unknown_replacement_cube``
        - ``unknown_replacement_saved_query``
        - ``glossary_token_collision``
        - ``unknown_scope_function``
        """
        errors: list[dict[str, Any]] = []

        def _err(code: str, message: str, **extra: object) -> None:
            payload: dict[str, Any] = {"code": code, "message": message}
            payload.update(extra)
            errors.append(payload)

        cube_list = list(cubes)
        # META reflection cubes are auto-appended by the runtime; the
        # spec carries only user-supplied cubes.
        names = [c.name for c in cube_list]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            _err(
                "duplicate_cube_name",
                f"Catalog has duplicate cube names: {duplicates}. Each cube.name "
                "must be unique within a catalog.",
                duplicates=duplicates,
            )

        # Primary-key declarations
        seen_names: set[str] = set(names)
        for cube in cube_list:
            if cube.primary_key is None:
                continue
            dim_names = {d.name for d in cube.dimensions}
            if cube.primary_key not in dim_names:
                _err(
                    "unknown_primary_key_dimension",
                    f"Cube {cube.name!r} declares primary_key={cube.primary_key!r} "
                    "but the cube has no dimension by that name.",
                    cube=cube.name,
                    primary_key=cube.primary_key,
                )

        # Auto-derived foreign-key joins + join target existence.
        # We duplicate the legacy logic so the error envelope matches
        # what :class:`Catalog` would have raised; refactoring the
        # validation into a shared helper is a follow-up.
        by_name: dict[str, Cube] = {c.name: c for c in cube_list}
        augmented: list[Cube] = []
        for cube in cube_list:
            explicit_targets = {j.to for j in cube.joins}
            inferred: list[Join] = []
            for dim in cube.dimensions:
                fk = dim.foreign_key
                if fk is None:
                    continue
                if fk not in by_name:
                    _err(
                        "unknown_foreign_key_target",
                        f"Cube {cube.name!r}, dimension {dim.name!r}: "
                        f"foreign_key={fk!r} names a cube not in the catalog.",
                        cube=cube.name,
                        dimension=dim.name,
                        foreign_key=fk,
                    )
                    continue
                target = by_name[fk]
                if target.primary_key is None:
                    _err(
                        "foreign_key_target_no_primary_key",
                        f"Cube {cube.name!r}, dimension {dim.name!r}: "
                        f"foreign_key={fk!r} requires the target cube to "
                        f"declare a primary_key.",
                        cube=cube.name,
                        dimension=dim.name,
                        foreign_key=fk,
                    )
                    continue
                if fk in explicit_targets:
                    continue
                inferred.append(
                    Join(
                        to=fk,
                        relationship="many_to_one",
                        on=f"{{{cube.alias}}}.{dim.name} = {{{target.alias}}}.{target.primary_key}",
                    )
                )
            if inferred:
                cube = cube.model_copy(update={"joins": [*cube.joins, *inferred]})
            augmented.append(cube)
        cube_list = augmented
        for cube in cube_list:
            for join in cube.joins:
                if join.to not in seen_names:
                    _err(
                        "unknown_join_target",
                        f"Cube {cube.name!r} declares Join(to={join.to!r}) "
                        f"but {join.to!r} is not in the catalog.",
                        cube=cube.name,
                        join_target=join.to,
                    )

        # CTE name uniqueness across the catalog
        cte_owners: dict[str, str] = {}
        for cube in cube_list:
            src = cube.source
            if not isinstance(src, DerivedTable):
                continue
            for cte in src.with_ctes:
                if cte.name in cte_owners:
                    _err(
                        "duplicate_cte_name",
                        f"Cube {cube.name!r}: CTE name {cte.name!r} in with_ctes "
                        f"collides with cube {cte_owners[cte.name]!r}.",
                        cube=cube.name,
                        cte_name=cte.name,
                    )
                cte_owners[cte.name] = cube.name

        # View validation
        view_list: list[View] = list(views or [])
        view_names: set[str] = set()
        for v in view_list:
            if v.name in view_names:
                _err(
                    "duplicate_view_name",
                    f"Catalog has duplicate view name {v.name!r}.",
                    view_name=v.name,
                )
            view_names.add(v.name)
            if v.name in seen_names:
                _err(
                    "view_collides_with_cube",
                    f"View {v.name!r} collides with cube name {v.name!r}. "
                    "View and cube names share a namespace; rename one.",
                    view_name=v.name,
                )
            for local, target_ref in v.fields.items():
                if "." not in target_ref:
                    _err(
                        "unknown_view_target_field",
                        f"View {v.name!r}, field {local!r}: target "
                        f"{target_ref!r} must be qualified as 'cube.field'.",
                        view_name=v.name,
                        field=local,
                    )
                    continue
                cube_name, field_name = target_ref.split(".", 1)
                if cube_name not in by_name:
                    _err(
                        "unknown_view_target_cube",
                        f"View {v.name!r}, field {local!r}: target cube "
                        f"{cube_name!r} not in the catalog.",
                        view_name=v.name,
                        field=local,
                    )
                    continue
                target_cube = by_name[cube_name]
                cube_field_names = {f.name for f in target_cube.measures}
                cube_field_names |= {f.name for f in target_cube.dimensions}
                cube_field_names |= {f.name for f in target_cube.time_dimensions}
                if field_name not in cube_field_names:
                    _err(
                        "unknown_view_target_field",
                        f"View {v.name!r}, field {local!r}: "
                        f"{cube_name}.{field_name} is not a known measure or "
                        f"dimension on cube {cube_name!r}.",
                        view_name=v.name,
                        field=local,
                    )

        # Lookup validation
        lookup_list: list[Lookup] = list(lookups or [])
        seen_lookup_dims: set[str] = set()
        for lk in lookup_list:
            if lk.dimension in seen_lookup_dims:
                _err(
                    "duplicate_lookup_dimension",
                    f"Catalog has duplicate Lookup for dimension {lk.dimension!r}.",
                    dimension=lk.dimension,
                )
            seen_lookup_dims.add(lk.dimension)
            if "." not in lk.dimension:
                _err(
                    "unknown_lookup_dimension",
                    f"Lookup({lk.dimension!r}): dimension must be qualified as 'cube.dim'.",
                    dimension=lk.dimension,
                )
                continue
            cube_name, dim_name = lk.dimension.split(".", 1)
            if cube_name not in by_name:
                _err(
                    "unknown_lookup_cube",
                    f"Lookup({lk.dimension!r}): cube {cube_name!r} is not in the catalog.",
                    dimension=lk.dimension,
                )
                continue
            target_cube = by_name[cube_name]
            target_dim = next((d for d in target_cube.dimensions if d.name == dim_name), None)
            if target_dim is None:
                _err(
                    "unknown_lookup_dimension",
                    f"Lookup({lk.dimension!r}): cube {cube_name!r} has no "
                    f"dimension named {dim_name!r}.",
                    dimension=lk.dimension,
                )
                continue
            if target_dim.type != "string":
                _err(
                    "lookup_dimension_wrong_type",
                    f"Lookup({lk.dimension!r}): only string-typed dimensions "
                    f"are eligible — {dim_name!r} is type={target_dim.type!r}.",
                    dimension=lk.dimension,
                )

        # Saved queries
        saved_query_list: list[SavedQuery] = list(saved_queries or [])
        seen_saved_names: set[str] = set()
        for sq in saved_query_list:
            if not sq.name or "." in sq.name or " " in sq.name:
                _err(
                    "invalid_saved_query_name",
                    f"SavedQuery has invalid name {sq.name!r}: must be non-empty "
                    "and contain no dots or spaces.",
                    saved_query_name=sq.name,
                )
            if sq.name in seen_saved_names:
                _err(
                    "duplicate_saved_query_name",
                    f"Catalog has duplicate SavedQuery name {sq.name!r}.",
                    saved_query_name=sq.name,
                )
            if sq.name in seen_names or sq.name in view_names:
                _err(
                    "saved_query_collides_with_cube_or_view",
                    f"SavedQuery name {sq.name!r} collides with a cube or view of the same name.",
                    saved_query_name=sq.name,
                )
            seen_saved_names.add(sq.name)

        # Replacement pointers
        for cube in cube_list:
            if cube.replacement is not None and cube.replacement not in by_name:
                _err(
                    "unknown_replacement_cube",
                    f"Cube {cube.name!r}: replacement={cube.replacement!r} "
                    "names a cube not in the catalog.",
                    cube=cube.name,
                    replacement=cube.replacement,
                )
        saved_query_by_name = {sq.name: sq for sq in saved_query_list}
        for sq in saved_query_list:
            if sq.replacement is not None and sq.replacement not in saved_query_by_name:
                _err(
                    "unknown_replacement_saved_query",
                    f"SavedQuery {sq.name!r}: replacement={sq.replacement!r} "
                    "names a saved query not in the catalog.",
                    saved_query_name=sq.name,
                    replacement=sq.replacement,
                )

        # Glossary collisions (case-insensitive)
        glossary_list: list[GlossaryEntry] = list(glossary or [])
        seen_glossary: dict[str, str] = {}

        def _register(token: str, source: str) -> None:
            key = token.lower()
            if key in seen_glossary:
                _err(
                    "glossary_token_collision",
                    f"Catalog glossary: token {token!r} ({source}) collides with "
                    f"{seen_glossary[key]} (case-insensitive).",
                    token=token,
                    conflict_with=seen_glossary[key],
                )
            seen_glossary[key] = source

        for g in glossary_list:
            _register(g.term, f"term {g.term!r}")
            for a in g.aliases:
                _register(a, f"alias on term {g.term!r}")

        # Relations narrative — best-effort; only run if there are no
        # name collisions upstream (otherwise the message is misleading).
        relations_str = relations
        try:
            relations_str = validate_relations("Catalog", "<catalog>", relations)
        except ValueError as exc:
            _err(
                "invalid_relations",
                str(exc),
            )

        spec = cls(
            cubes=tuple(cube_list),
            views=tuple(view_list),
            lookups=tuple(lookup_list),
            saved_queries=tuple(saved_query_list),
            glossary=tuple(glossary_list),
            relations=relations_str,
            compile_hook_names=tuple(compile_hook_names),
            sql_rewrite_hook_names=tuple(sql_rewrite_hook_names),
            construction_errors=tuple(errors),
        )
        return spec, errors


@dataclass(frozen=True)
class CatalogRuntime:
    """B5 — the callable half of a Catalog.

    The runtime holds the callables that can't cross a process
    boundary: the visibility policy, the scope-function registry,
    the unit registry, the error transform, the compile / sql-rewrite
    hooks. The wire-format spec records the *names* of the hooks;
    the runtime resolves them back to callables at construction
    time (or accepts them directly when built in-process).

    Not serialisable by design: ``model_dump`` is intentionally
    absent. Callers serialise the spec and pair it with a fresh
    runtime on the receiving side.
    """

    policy: PolicyFn | None
    scope_fns: dict[str, ScopeFn]
    unit_registry: Registry | None
    error_transform: object | None
    compile_hooks: list[CompileHook]
    sql_rewrite_hooks: list[SqlRewriteHook]


class Catalog:
    """A validated collection of cubes plus the convenience surface
    (``compile``, ``as_dict``, ``with_retrieval``) downstream code wants.

    B5: a :class:`Catalog` pairs a :class:`CatalogSpec` (data) with
    a :class:`CatalogRuntime` (callables). The legacy ``__init__``
    signature still works — internally it builds the spec and runtime
    from the validated state. Use :meth:`from_spec` to construct
    from a pre-built spec, or :meth:`CatalogSpec.from_iterables` for
    a collect-all build.
    """

    #: B5 — public so callers can introspect the serialised shape
    #: without going through model_dump. Populated in ``__init__``.
    spec: CatalogSpec

    #: B5 — the callable half. Not serialisable.
    runtime: CatalogRuntime

    @classmethod
    def from_spec(
        cls,
        spec: CatalogSpec,
        *,
        runtime: CatalogRuntime | None = None,
        scope_fns: dict[str, ScopeFn] | None = None,
        unit_registry: Registry | None = None,
        policy: PolicyFn | None = None,
        error_transform: object | None = None,
        compile_hooks: list[CompileHook] | None = None,
        sql_rewrite_hooks: list[SqlRewriteHook] | None = None,
    ) -> Catalog:
        """Build a Catalog from a pre-built :class:`CatalogSpec`.

        ``runtime`` overrides the per-call callables; any that are
        ``None`` fall back to the keyword arguments (or sensible
        defaults). Use this when you deserialised a spec from a
        cache or migration and want to pair it with a fresh
        runtime in the current process."""
        if spec.construction_errors:
            messages = [e.get("message", "?") for e in spec.construction_errors]
            raise ValueError(
                f"CatalogSpec has {len(spec.construction_errors)} construction "
                f"error(s); build via from_iterables and surface them, or fix "
                f"the spec first. First: {messages[0]!r}"
            )
        # Materialise the spec into a Catalog via the legacy kwargs
        # path: the spec's cubes / views / lookups / saved_queries /
        # glossary / relations are user-supplied; the META auto-append
        # and the rest of the first-error validation runs in __init__.
        return cls(
            list(spec.cubes),
            views=list(spec.views),
            lookups=list(spec.lookups),
            saved_queries=list(spec.saved_queries),
            glossary=list(spec.glossary),
            relations=spec.relations,
            policy=policy,
            scope_fns=scope_fns,
            unit_registry=unit_registry,
            error_transform=error_transform,
            compile_hooks=compile_hooks,
            sql_rewrite_hooks=sql_rewrite_hooks,
        )

    def __init__(
        self,
        cubes: list[Cube],
        *,
        views: list[View] | None = None,
        lookups: list[Lookup] | None = None,
        saved_queries: list[SavedQuery] | None = None,
        policy: PolicyFn | None = None,
        scope_fns: dict[str, ScopeFn] | None = None,
        unit_registry: Registry | None = None,
        glossary: list[GlossaryEntry] | None = None,
        relations: str = "",
        error_transform: object | None = None,
        compile_hooks: list[CompileHook] | None = None,
        sql_rewrite_hooks: list[SqlRewriteHook] | None = None,
    ) -> None:
        self.compile_hooks = compile_hooks or []
        self.sql_rewrite_hooks = sql_rewrite_hooks or []
        names = [c.name for c in cubes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Catalog has duplicate cube names: {duplicates}. "
                "Each cube.name must be unique within a catalog."
            )

        # Auto-append any missing META cubes so reflection always works.
        existing = set(names)
        merged: list[Cube] = list(cubes)
        for meta in META_CUBES:
            if meta.name not in existing:
                merged.append(meta)
                existing.add(meta.name)

        # Resolve ``extends`` chains — flatten inherited measures /
        # dimensions / time_dimensions / segments by name. Detect cycles.
        merged = _resolve_extends(merged)

        # Validate primary_key declarations — must name a real dimension.
        for c in merged:
            if c.primary_key is not None:
                dim_names = {d.name for d in c.dimensions}
                if c.primary_key not in dim_names:
                    raise ValueError(
                        f"Cube {c.name!r} declares primary_key="
                        f"{c.primary_key!r} but the cube has no dimension "
                        f"by that name. Declare it as a Dimension or pick "
                        f"a different primary_key."
                    )

        # Auto-derive Join edges from Dimension.foreign_key declarations.
        # An explicit Join with the same target wins — no duplicates.
        by_name: dict[str, Cube] = {c.name: c for c in merged}
        for cube in merged:
            inferred: list[Join] = []
            explicit_targets = {j.to for j in cube.joins}
            for dim in cube.dimensions:
                fk = dim.foreign_key
                if fk is None:
                    continue
                if fk not in by_name:
                    raise ValueError(
                        f"Cube {cube.name!r}, dimension {dim.name!r}: "
                        f"foreign_key={fk!r} names a cube not in the "
                        f"catalog. Known cubes: {sorted(by_name)}."
                    )
                target = by_name[fk]
                if target.primary_key is None:
                    raise ValueError(
                        f"Cube {cube.name!r}, dimension {dim.name!r}: "
                        f"foreign_key={fk!r} requires the target cube "
                        f"to declare a primary_key. Add primary_key="
                        f"'<dim>' to cube {fk!r}."
                    )
                if fk in explicit_targets:
                    continue  # explicit Join wins
                inferred.append(
                    Join(
                        to=fk,
                        relationship="many_to_one",
                        on=f"{{{cube.alias}}}.{dim.name} = {{{target.alias}}}.{target.primary_key}",
                    )
                )
            if inferred:
                # Replace the cube with a copy carrying the augmented joins.
                # Cube isn't frozen, but stay disciplined and use model_copy.
                merged[merged.index(cube)] = cube.model_copy(
                    update={"joins": [*cube.joins, *inferred]}
                )

        known = {c.name for c in merged}
        for c in merged:
            for j in c.joins:
                if j.to not in known:
                    raise ValueError(
                        f"Cube {c.name!r} declares Join(to={j.to!r}) but "
                        f"{j.to!r} is not in the catalog. "
                        f"Known cubes: {sorted(known)}."
                    )

        # DerivedTable CTE names live in a flat namespace across the
        # whole catalog — the compiler hoists every touched cube's CTEs
        # into a single outer ``WITH`` clause, so two cubes that both
        # declare ``WITH foo AS (...)`` with different bodies would
        # silently lose one. Surface the collision at construction time.
        cte_owners: dict[str, str] = {}
        for cube in merged:
            src = cube.source
            if not isinstance(src, DerivedTable):
                continue
            for cte in src.with_ctes:
                if cte.name in cte_owners:
                    raise ValueError(
                        f"Cube {cube.name!r}: CTE name {cte.name!r} in "
                        f"with_ctes collides with cube "
                        f"{cte_owners[cte.name]!r}. CTE names must be "
                        "unique across the catalog."
                    )
                cte_owners[cte.name] = cube.name

        self._cubes: list[Cube] = merged
        self._by_name: dict[str, Cube] = {c.name: c for c in merged}

        # Validate views: every field target must resolve to a real
        # cube.field, and view names can't collide with cube names.
        view_list: list[View] = list(views or [])
        seen_view_names: set[str] = set()
        for v in view_list:
            if v.name in seen_view_names:
                raise ValueError(f"Catalog has duplicate view name {v.name!r}.")
            seen_view_names.add(v.name)
            if v.name in self._by_name:
                raise ValueError(
                    f"View {v.name!r} collides with cube name {v.name!r}. "
                    "View and cube names share a namespace; rename one."
                )
            for local, target_ref in v.fields.items():
                cube_name, field_name = target_ref.split(".", 1)
                if cube_name not in self._by_name:
                    raise ValueError(
                        f"View {v.name!r}, field {local!r}: target "
                        f"cube {cube_name!r} not in the catalog. "
                        f"Known cubes: {sorted(self._by_name)}."
                    )
                target_cube = self._by_name[cube_name]
                cube_field_names = {f.name for f in target_cube.measures}
                cube_field_names |= {f.name for f in target_cube.dimensions}
                cube_field_names |= {f.name for f in target_cube.time_dimensions}
                if field_name not in cube_field_names:
                    raise ValueError(
                        f"View {v.name!r}, field {local!r}: "
                        f"{cube_name}.{field_name} is not a known measure "
                        f"or dimension on cube {cube_name!r}."
                    )
        self.views: dict[str, View] = {v.name: v for v in view_list}
        self._policy: PolicyFn | None = policy

        # Validate scope_fns: every Cube.scope must resolve to a
        # registered name. Catching it here means the compiler can
        # trust the registry lookup later.
        # Unit conversion registry — used by downstream presenters
        # (visualize, renderers) when applying ``Measure.display_unit``
        # to row values. Defaults to the shared process-wide registry;
        # pass a custom :class:`semql.units.Registry` to isolate a
        # catalog's vocabulary (e.g. per-tenant overrides) without
        # mutating the global one.
        self.unit_registry: Registry = unit_registry or DEFAULT_REGISTRY

        # Unit / display_unit validation — fail fast on configuration
        # errors that would otherwise only surface at render time.
        # Two checks:
        #   1. ``display_unit`` without ``unit`` — the renderer would
        #      have nothing to convert FROM.
        #   2. The (unit, display_unit) pair must be reachable in the
        #      registry. Catches typos like ``display_unit="hour"`` vs
        #      ``"hours"`` that Pydantic accepts as plain strings.
        # Both checks are skipped when ``unit`` matches ``display_unit``
        # (no conversion needed) or when only ``unit`` is set.
        for cube in merged:
            for fld in (*cube.measures, *cube.dimensions):
                self._check_unit_pair(cube, fld)

        # Lookups: dimension-value catalogs addressable by qualified
        # ``cube.dim``. Each Lookup's dimension must resolve to a real
        # ``string``-typed Dimension on a real cube — typo or type
        # mismatches surface here instead of at prompt-render time.
        lookup_list: list[Lookup] = list(lookups or [])
        seen_lookup_dims: set[str] = set()
        for lk in lookup_list:
            if lk.dimension in seen_lookup_dims:
                raise ValueError(
                    f"Catalog has duplicate Lookup for dimension "
                    f"{lk.dimension!r}. Each ``cube.dim`` may have at "
                    "most one Lookup."
                )
            seen_lookup_dims.add(lk.dimension)
            cube_name, dim_name = lk.dimension.split(".", 1)
            if cube_name not in self._by_name:
                raise ValueError(
                    f"Lookup({lk.dimension!r}): cube {cube_name!r} is "
                    f"not in the catalog. Known cubes: "
                    f"{sorted(self._by_name)}."
                )
            target_cube = self._by_name[cube_name]
            target_dim: Dimension | None = next(
                (d for d in target_cube.dimensions if d.name == dim_name), None
            )
            if target_dim is None:
                raise ValueError(
                    f"Lookup({lk.dimension!r}): cube {cube_name!r} has "
                    f"no dimension named {dim_name!r}. Known dimensions: "
                    f"{sorted(d.name for d in target_cube.dimensions)}."
                )
            if target_dim.type != "string":
                raise ValueError(
                    f"Lookup({lk.dimension!r}): only string-typed "
                    f"dimensions are eligible — {dim_name!r} is type="
                    f"{target_dim.type!r}."
                )
        self.lookups: dict[str, Lookup] = {lk.dimension: lk for lk in lookup_list}

        # Saved queries — pre-baked SemanticQueries the MCP layer
        # auto-exposes as zero-arg tools. Validate name uniqueness +
        # name shape; a deeper compile-time validation of each query
        # against the catalog is best-effort and gated behind
        # ``validate_saved_queries`` so a half-built catalog at
        # bootstrap time still loads.
        saved_query_list: list[SavedQuery] = list(saved_queries or [])
        seen_saved_names: set[str] = set()
        for sq in saved_query_list:
            if not sq.name or "." in sq.name or " " in sq.name:
                raise ValueError(
                    f"SavedQuery has invalid name {sq.name!r}: must be "
                    "non-empty and contain no dots or spaces (it becomes "
                    "part of an MCP tool name)."
                )
            if sq.name in seen_saved_names:
                raise ValueError(
                    f"Catalog has duplicate SavedQuery name {sq.name!r}. "
                    "Each saved query's name must be unique."
                )
            if sq.name in self._by_name or sq.name in self.views:
                raise ValueError(
                    f"SavedQuery name {sq.name!r} collides with a cube "
                    "or view of the same name. Saved-query / cube / view "
                    "names share a namespace; rename one."
                )
            seen_saved_names.add(sq.name)
        self.saved_queries: dict[str, SavedQuery] = {sq.name: sq for sq in saved_query_list}

        # S7 — replacement pointers must name a real cube. Cube doesn't
        # know its siblings at construction time; check here.
        for c in merged:
            if c.replacement is not None and c.replacement not in self._by_name:
                raise ValueError(
                    f"Cube {c.name!r}: replacement={c.replacement!r} names "
                    f"a cube not in the catalog. Known cubes: "
                    f"{sorted(self._by_name)}."
                )
        for sq in saved_query_list:
            if sq.replacement is not None and sq.replacement not in self.saved_queries:
                raise ValueError(
                    f"SavedQuery {sq.name!r}: replacement={sq.replacement!r} "
                    f"names a saved query not in the catalog. Known saved "
                    f"queries: {sorted(self.saved_queries)}."
                )

        # S7 — catalog-wide glossary + cross-cube relations narrative.
        # Terms and aliases share one namespace (the retriever indexes
        # aliases as separate documents pointing at the same canonical
        # entry, so a token that's both a term and an alias would be
        # ambiguous). Collisions are raised case-insensitively.
        glossary_list: list[GlossaryEntry] = list(glossary or [])
        seen_glossary: dict[str, str] = {}  # lowercase token → "term:X" / "alias on X"

        def _register(token: str, source: str) -> None:
            key = token.lower()
            if key in seen_glossary:
                raise ValueError(
                    f"Catalog glossary: token {token!r} ({source}) collides "
                    f"with {seen_glossary[key]} (case-insensitive). Terms "
                    "and aliases share one namespace; rename one."
                )
            seen_glossary[key] = source

        for g in glossary_list:
            _register(g.term, f"term {g.term!r}")
            for a in g.aliases:
                _register(a, f"alias on term {g.term!r}")
        self.glossary: list[GlossaryEntry] = glossary_list
        self.relations: str = validate_relations("Catalog", "<catalog>", relations)

        self._scope_fns: dict[str, ScopeFn] = dict(scope_fns or {})
        for c in merged:
            if c.scope is not None and c.scope not in self._scope_fns:
                raise ValueError(
                    f"Cube {c.name!r} declares scope={c.scope!r} but no "
                    f"scope function is registered under that name. "
                    f"Pass scope_fns={{'{c.scope}': fn, ...}} to the Catalog "
                    f"constructor. Registered scopes: {sorted(self._scope_fns)}."
                )
        self._error_transform: object | None = error_transform

        # B5 — build the spec + runtime pair now that validation has
        # passed. The spec carries only the user-supplied cubes
        # (META reflection cubes are appended on read by the
        # ``_by_name`` materialisation, not stored in the spec).
        # The user-supplied cubes are the ones the caller passed in
        # before the auto-append step ran; we recover them by
        # excluding the META names.
        meta_names = {m.name for m in META_CUBES}
        user_cubes = tuple(c for c in merged if c.name not in meta_names)
        self.spec = CatalogSpec(
            cubes=user_cubes,
            views=tuple(view_list),
            lookups=tuple(lookup_list),
            saved_queries=tuple(saved_query_list),
            glossary=tuple(glossary_list),
            relations=self.relations,
        )
        self.runtime = CatalogRuntime(
            policy=policy,
            scope_fns=dict(self._scope_fns),
            unit_registry=self.unit_registry,
            error_transform=error_transform,
            compile_hooks=list(self.compile_hooks),
            sql_rewrite_hooks=list(self.sql_rewrite_hooks),
        )

    def _check_unit_pair(self, cube: Cube, fld: BaseField) -> None:
        """Validate ``unit`` / ``display_unit`` on a single field.

        Raises ``ValueError`` with a pointer to the offending cube and
        field if:
          * ``display_unit`` is set without ``unit``, or
          * the pair can't be converted in ``self.unit_registry``.
        Skips no-op pairs where ``unit == display_unit``.
        """
        unit = getattr(fld, "unit", None)
        display_unit = getattr(fld, "display_unit", None)
        if display_unit is None:
            return
        if unit is None:
            raise ValueError(
                f"Cube {cube.name!r}, field {fld.name!r}: display_unit="
                f"{display_unit!r} requires unit to also be set — there's "
                "nothing to convert from."
            )
        if unit == display_unit:
            return
        try:
            self.unit_registry.factor(unit, display_unit)
        except ValueError as exc:
            raise ValueError(
                f"Cube {cube.name!r}, field {fld.name!r}: cannot convert "
                f"{unit!r} → {display_unit!r}: {exc}. "
                "Register the conversion on the catalog's unit_registry "
                "or correct the spelling."
            ) from exc

    @property
    def policy(self) -> PolicyFn | None:
        """The optional custom-visibility predicate registered at
        construction time. ``None`` means cube visibility is governed
        purely by ``Cube.required_roles``."""
        return self._policy

    @property
    def scope_fns(self) -> dict[str, ScopeFn]:
        """The scope-function registry. Each entry maps a name a
        ``Cube.scope`` field references to the callable that produces
        the row-level predicate for a given (cube, viewer)."""
        return dict(self._scope_fns)

    def as_dict(self) -> dict[str, Cube]:
        """Return ``{cube.name: Cube}`` — the shape ``compile_query`` consumes."""
        return dict(self._by_name)

    def explain(
        self,
        query: SemanticQuery,
        *,
        context: dict[str, str] | None = None,
        viewer: AuthContext | None = None,
    ) -> str:
        """Return a human-readable repr of the LogicalPlan the compiler
        will emit for ``query``.

        Useful for debugging ("what will this turn into?") and for the
        MCP ``explain`` tool.  Output is the ``repr()`` of the
        :class:`semql.logical.LogicalPlan` IR — see the plan-snapshot
        tests for the canonical shapes.

        Same diagnostic surface as :meth:`compile`: resolution errors
        and unauthorised-cube errors raise before any SQL is built.
        Rollup routing is applied so the explained plan matches what
        would actually be executed.
        """
        plan = explain_plan(query, self.as_dict(), context=context, viewer=viewer)
        return repr(plan)

    def compile(
        self,
        query: SemanticQuery,
        *,
        context: dict[str, str] | None = None,
        viewer: AuthContext | None = None,
        query_defaults: object | None = None,
    ) -> CompiledQuery:
        """Compile a ``SemanticQuery`` against this catalog. Thin wrapper
        around ``semql.compile.compile_query``.

        When ``viewer`` is provided, the compiler:
        - Refuses queries that touch a cube the viewer cannot see
          (``Cube.required_roles`` ANY-match + optional ``policy``).
        - Auto-binds ``ctx.viewer_id`` from ``viewer.viewer_id`` so
          ``security_sql`` fragments referencing it get a parameter
          (never a SQL literal).

        On a :class:`FilterTypeError` the catalog enriches the exception
        with the LLM-repair affordance (B8): if the failing dimension
        has a registered ``Lookup`` the error carries
        ``next_tool="resolve_lookup"`` plus the tool args, so a machine
        consumer can call the lookup tool to resolve the free-text
        value instead of guessing.
        """
        if isinstance(query_defaults, SemanticQueryDefaults):
            query = _apply_query_defaults(query, query_defaults)

        try:
            for compile_hook in self.compile_hooks:
                new_q = compile_hook.pre_compile(query, viewer=viewer, context=context)
                if new_q is not None:
                    query = new_q

            compiled = compile_query(
                query,
                self._by_name,
                context=context,
                views=self.views,
                viewer=viewer,
                policy=self._policy,
                scope_fns=self._scope_fns,
            )

            for compile_hook in self.compile_hooks:
                try:
                    compile_hook.post_compile(query, compiled, viewer=viewer, context=context)
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Compile hook {compile_hook} raised exception in post_compile: {e}",
                        stacklevel=2,
                    )

            for rewrite_hook in self.sql_rewrite_hooks:
                compiled = rewrite_hook.rewrite(
                    compiled, query=query, viewer=viewer, context=context
                )

            return compiled
        except SemQLError as exc:
            import warnings

            for compile_hook in self.compile_hooks:
                try:
                    compile_hook.on_compile_error(query, exc, viewer=viewer, context=context)
                except Exception as e:
                    warnings.warn(
                        f"Compile hook {compile_hook} raised exception in on_compile_error: {e}",
                        stacklevel=2,
                    )

            if isinstance(exc, FilterTypeError) and exc.next_tool is None:
                enriched = self._enrich_filter_type_error(exc)
                if enriched is not exc:
                    if self._error_transform is not None:
                        replacement = self._error_transform(enriched)  # type: ignore[operator]
                        if replacement is not None:
                            raise replacement from enriched
                    raise enriched from exc

            if self._error_transform is not None:
                replacement = self._error_transform(exc)  # type: ignore[operator]
                if replacement is not None:
                    raise replacement from exc
            raise

    def _enrich_filter_type_error(self, exc: FilterTypeError) -> FilterTypeError:
        """Add LLM-repair affordance to a FilterTypeError when the failing
        dim has a registered ``Lookup``.

        For string dims with a Lookup, we name the ``resolve_lookup``
        tool (the canonical repair path for free-text values that miss
        the canonical set) and suggest up to 3 close values from the
        lookup's static ``values`` (or its label keys). Returns the
        same exception instance when no Lookup is registered — no
        envelope change in that case.

        Suggestions are case-insensitive — the catalog's canonical
        values are typically uppercase (``"EMEA"``) while the LLM may
        emit mixed case (``"emea"``). We match against a lowercased
        view, then return the *original* case for the suggestion.
        """
        lookup = self.lookups.get(exc.dimension)
        if lookup is None:
            return exc
        # The lookup may be dynamic (loader-backed) — we cannot enumerate
        # its values at error time. The static ``values`` set is the
        # only thing available synchronously, so we suggest from that.
        static_values: list[str] = list(lookup.values) if lookup.values else []
        # Also surface label keys when labels are present (MCP tools
        # commonly render via labels).
        label_keys: list[str] = list(lookup.labels.keys()) if lookup.labels else []
        candidates: list[str] = list(dict.fromkeys([*static_values, *label_keys]))
        did_you_mean: list[str] = []
        if candidates and exc.value is not None:
            query_value = str(exc.value) if not isinstance(exc.value, list) else str(exc.value[0])
            # Build a (lowercase → original) map and match in lowercase
            # space so the difflib ratio is meaningful regardless of
            # the catalog's casing convention.
            lower_to_original: dict[str, str] = {c.lower(): c for c in candidates}
            lower_matches = difflib.get_close_matches(
                query_value.lower(), list(lower_to_original), n=3, cutoff=0.4
            )
            did_you_mean = [lower_to_original[m] for m in lower_matches if m in lower_to_original]
            if not did_you_mean:
                # Fall back to the closest single match so the envelope
                # still offers a concrete suggestion when the difflib
                # cutoff rejected everything.
                single = difflib.get_close_matches(
                    query_value.lower(), list(lower_to_original), n=1, cutoff=0.0
                )
                if single and single[0] in lower_to_original:
                    did_you_mean = [lower_to_original[single[0]]]
        return FilterTypeError(
            str(exc),
            dimension=exc.dimension,
            op=exc.op,
            value=exc.value,
            next_tool="resolve_lookup",
            next_tool_args={"dimension": exc.dimension, "query": str(exc.value)},
            did_you_mean=did_you_mean,
        )

    def compile_collect_all(
        self,
        query: SemanticQuery,
        *,
        context: dict[str, str] | None = None,
        viewer: AuthContext | None = None,
        query_defaults: object | None = None,
    ) -> list[ValidationError]:
        """B8 — collect-all compile path.

        Returns ``[]`` when the query compiles, otherwise returns the
        full list of :class:`~semql.validate.ValidationError` records
        the static checker can find (one round-trip for an LLM, not N).
        Never raises for query-shape problems; only I/O / import errors
        (e.g. a missing transitive dependency) propagate.

        Identifies which cubes the query touches to apply auth /
        policy / scope the same way :meth:`compile` does; on the
        unauthorised-cube path it returns a synthetic
        ``ValidationError`` with code ``"unauthorised_cube"`` and the
        cube name attached. Permission-related errors that don't have
        a structured ``ValidationError`` analogue fall through as
        ``ValidationError(code="compile_error", message=str(exc))``.
        """
        if isinstance(query_defaults, SemanticQueryDefaults):
            query = _apply_query_defaults(query, query_defaults)

        try:
            self.compile(query, context=context, viewer=viewer)
            return []
        except SemQLError as exc:
            errors = validate(query, self)
            if errors:
                return errors
            # The compile error didn't surface through validate (e.g.
            # a backend dialect issue, a non-resolution compile bug).
            # Wrap it as a single ValidationError so the LLM still
            # gets an envelope.
            return [
                ValidationError(
                    code="compile_error",
                    message=str(exc),
                    extra={"error_class": type(exc).__name__},
                )
            ]

    def with_retrieval(
        self,
        *,
        embedder: EmbeddingProvider | None = None,
        mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> Retriever:
        """Build a :class:`semql.retrieve.Retriever` indexed over this
        catalog's cubes + glossary aliases.

        Selection policy mirrors the S7 PRD:
        - No ``embedder`` → :class:`SQLiteBM25Retriever` (lexical only).
        - With ``embedder`` → :class:`HybridRetriever` (BM25 + cosine via RRF).
        - ``mmr=True`` wraps the result in :class:`MMRWrapper` (needs vectors).

        Deprecated cubes are excluded from the index — the compiler
        refuses to materialise them anyway."""
        # Filter deprecated up front so the retriever can't recommend
        # something the compiler will then refuse.
        live_cubes = [c for c in self._cubes if c.stability != "deprecated"]
        return build_default_retriever(
            live_cubes,
            embedder=embedder,
            glossary=self.glossary,
            mmr=mmr,
            mmr_lambda=mmr_lambda,
        )

    def __iter__(self) -> Iterator[Cube]:
        return iter(self._cubes)

    def __len__(self) -> int:
        return len(self._cubes)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name


def _resolve_extends(cubes: list[Cube]) -> list[Cube]:
    """Flatten ``Cube.extends`` chains into self-contained cubes.

    Each child cube inherits the parent's measures / dimensions /
    time_dimensions / segments by name. Child overrides win;
    new items append. Other settings stay on the child.

    Detects cycles and unknown parents."""
    by_name = {c.name: c for c in cubes}

    def _flatten(name: str, stack: tuple[str, ...]) -> Cube:
        cube = by_name[name]
        if cube.extends is None:
            return cube
        if cube.extends == name or cube.extends in stack:
            chain = " -> ".join((*stack, name, cube.extends))
            raise ValueError(f"Cube {name!r}: extends cycle detected ({chain}).")
        if cube.extends not in by_name:
            raise ValueError(
                f"Cube {name!r}: extends={cube.extends!r} names a cube "
                f"not in the catalog. Known cubes: {sorted(by_name)}."
            )
        parent = _flatten(cube.extends, (*stack, name))

        def _merge_by_name(parent_list: list[_T], child_list: list[_T]) -> list[_T]:
            by_field_name: dict[str, _T] = {f.name: f for f in parent_list}
            for f in child_list:
                by_field_name[f.name] = f
            return list(by_field_name.values())

        return cube.model_copy(
            update={
                "measures": _merge_by_name(parent.measures, cube.measures),
                "dimensions": _merge_by_name(parent.dimensions, cube.dimensions),
                "time_dimensions": _merge_by_name(parent.time_dimensions, cube.time_dimensions),
                "segments": _merge_by_name(parent.segments, cube.segments),
            }
        )

    resolved: list[Cube] = []
    for c in cubes:
        if c.extends is None:
            resolved.append(c)
        else:
            resolved.append(_flatten(c.name, ()))
    return resolved


__all__ = ["Catalog"]
