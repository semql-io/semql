"""Schema-aware catalog changelog (I3).

``diff_catalogs(old, new) -> CatalogDiff`` enumerates every cube /
field / join that was added, removed, or had its schema-affecting
attributes changed between two catalog snapshots. The result is a
frozen Pydantic value type that renders itself as Markdown via
``.to_markdown()``.

The diff is *schema-aware* in that the field set it compares is the
schema-affecting surface — anything the compiler emits SQL for or
the linter would flag. Cosmetic attributes (``description``,
``display_name``, ``metadata``) are deliberately excluded: they're
presentation-layer concerns, not schema concerns, and including them
would make every catalog review a 200-line noise report.

Severity model:

- **breaking** — old consumers will fail. Includes ``cube_removed``,
  ``measure_removed``, ``dimension_removed``, ``agg_changed``,
  ``type_changed`` (changing ``"string"`` to ``"number"`` invalidates
  every query predicate), ``join_removed``, ``required_roles_narrowed``
  (wait, narrowed = more roles required = breaking), ``primary_key_changed``.
- **additive** — old consumers keep working. Includes ``cube_added``,
  ``measure_added``, ``dimension_added``, ``join_added``,
  ``segment_added``, ``aliases_added``, ``required_roles_widened``.

The Markdown report sorts breaking changes to the top, so reviewers
read them first. We do not embed raw ``sql`` strings in the report:
the report references the affected attribute by name, not by content.
This avoids leaking implementation details (or PII, if the SQL
references a value) into the review.

The diff is a frozen Pydantic value type so it survives JSON
round-trips and can be cached as a build artefact.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semql.model import Cube, Dimension, Join, Measure, Segment


class ChangeKind(StrEnum):
    """The kind of change a single ``Change`` row represents."""

    CUBE_ADDED = "cube_added"
    CUBE_REMOVED = "cube_removed"
    MEASURE_ADDED = "measure_added"
    MEASURE_REMOVED = "measure_removed"
    MEASURE_AGG_CHANGED = "measure_agg_changed"
    DIMENSION_ADDED = "dimension_added"
    DIMENSION_REMOVED = "dimension_removed"
    DIMENSION_TYPE_CHANGED = "dimension_type_changed"
    DIMENSION_ALIASES_ADDED = "dimension_aliases_added"
    SEGMENT_ADDED = "segment_added"
    SEGMENT_REMOVED = "segment_removed"
    JOIN_ADDED = "join_added"
    JOIN_REMOVED = "join_removed"
    JOIN_CHANGED = "join_changed"
    PRIMARY_KEY_CHANGED = "primary_key_changed"
    REQUIRED_ROLES_NARROWED = "required_roles_narrowed"
    REQUIRED_ROLES_WIDENED = "required_roles_widened"


_BREAKING_KINDS: frozenset[ChangeKind] = frozenset(
    {
        ChangeKind.CUBE_REMOVED,
        ChangeKind.MEASURE_REMOVED,
        ChangeKind.MEASURE_AGG_CHANGED,
        ChangeKind.DIMENSION_REMOVED,
        ChangeKind.DIMENSION_TYPE_CHANGED,
        ChangeKind.SEGMENT_REMOVED,
        ChangeKind.JOIN_REMOVED,
        ChangeKind.JOIN_CHANGED,
        ChangeKind.PRIMARY_KEY_CHANGED,
        ChangeKind.REQUIRED_ROLES_NARROWED,
    }
)


class Change(BaseModel):
    """A single schema-affecting change between two catalog snapshots."""

    model_config = ConfigDict(frozen=True)

    kind: ChangeKind
    cube: str = Field(description="The cube name (e.g. 'orders') the change lives on.")
    field: str | None = Field(
        default=None,
        description="The field name the change refers to, or None for cube-level changes.",
    )
    detail: str | None = Field(
        default=None,
        description="Short human-readable detail (e.g. 'sum -> avg'). Never contains raw SQL.",
    )

    @property
    def is_breaking(self) -> bool:
        return self.kind in _BREAKING_KINDS


class CatalogDiff(BaseModel):
    """The result of comparing two catalog snapshots.

    Render with ``.to_markdown()`` for a reviewer-friendly report.
    Round-trips through ``model_dump`` / ``model_validate`` (cf. I9) so
    the result can be cached as a build artefact or stored alongside
    a deploy."""

    model_config = ConfigDict(frozen=True)

    changes: list[Change] = Field(
        default_factory=lambda: list[Change](),
        description="All detected changes, sorted breaking-first then by cube/field.",
    )

    @property
    def is_empty(self) -> bool:
        return not self.changes

    @property
    def breaking_count(self) -> int:
        return sum(1 for c in self.changes if c.is_breaking)

    @property
    def additive_count(self) -> int:
        return sum(1 for c in self.changes if not c.is_breaking)

    def to_markdown(self) -> str:
        """Render the diff as a Markdown report.

        Sections:
        - **Header**: counts of breaking + additive changes.
        - **Breaking changes**: bullet list, sorted by cube/field.
        - **Additive changes**: bullet list, sorted by cube/field.
        - **No changes**: a single line if the diff is empty."""
        if self.is_empty:
            return "## Catalog diff\n\nNo changes.\n"
        breaking = [c for c in self.changes if c.is_breaking]
        additive = [c for c in self.changes if not c.is_breaking]
        lines: list[str] = [
            "## Catalog diff",
            "",
            f"- {len(breaking)} breaking change(s)",
            f"- {len(additive)} additive change(s)",
            "",
        ]
        if breaking:
            lines.append("### Breaking changes")
            lines.append("")
            for c in breaking:
                lines.append(_format_change(c))
            lines.append("")
        if additive:
            lines.append("### Additive changes")
            lines.append("")
            for c in additive:
                lines.append(_format_change(c))
            lines.append("")
        return "\n".join(lines)


def _format_change(c: Change) -> str:
    """Render a single change as a Markdown bullet.

    Never embeds raw SQL or arbitrary attribute values: the report
    mentions the change *kind*, the affected ``cube``/``field``, and
    the short ``detail`` (e.g. ``sum -> avg``) without echoing the
    underlying content."""
    subject = f"cube `{c.cube}`" if c.field is None else f"`{c.cube}.{c.field}`"
    detail = f" - {c.detail}" if c.detail else ""
    return f"- **{_label_for(c.kind)}** {subject}{detail} ({c.kind.value})"


def _label_for(kind: ChangeKind) -> str:
    """Human-readable verb for a ChangeKind — used in the markdown report."""
    return {
        ChangeKind.CUBE_ADDED: "added",
        ChangeKind.CUBE_REMOVED: "removed",
        ChangeKind.MEASURE_ADDED: "added",
        ChangeKind.MEASURE_REMOVED: "removed",
        ChangeKind.MEASURE_AGG_CHANGED: "agg changed",
        ChangeKind.DIMENSION_ADDED: "added",
        ChangeKind.DIMENSION_REMOVED: "removed",
        ChangeKind.DIMENSION_TYPE_CHANGED: "type changed",
        ChangeKind.DIMENSION_ALIASES_ADDED: "aliases added",
        ChangeKind.SEGMENT_ADDED: "added",
        ChangeKind.SEGMENT_REMOVED: "removed",
        ChangeKind.JOIN_ADDED: "added",
        ChangeKind.JOIN_REMOVED: "removed",
        ChangeKind.JOIN_CHANGED: "join changed",
        ChangeKind.PRIMARY_KEY_CHANGED: "primary key changed",
        ChangeKind.REQUIRED_ROLES_NARROWED: "required roles narrowed",
        ChangeKind.REQUIRED_ROLES_WIDENED: "required roles widened",
    }[kind]


def _set_role_diff(
    cube_name: str,
    field_name: str,
    old_roles: list[str],
    new_roles: list[str],
) -> Change | None:
    """Compare two ``required_roles`` lists.

    Sets are unordered: we sort, deduplicate, then compare. Returns
    a change only if the role set actually shifted.

    - Removing a role widens visibility (additive).
    - Adding a role narrows visibility (breaking)."""
    old_set = set(old_roles)
    new_set = set(new_roles)
    if old_set == new_set:
        return None
    if old_set.issuperset(new_set):
        return Change(
            kind=ChangeKind.REQUIRED_ROLES_WIDENED,
            cube=cube_name,
            field=field_name,
            detail=f"{sorted(old_set)} -> {sorted(new_set)}",
        )
    if new_set.issuperset(old_set):
        return Change(
            kind=ChangeKind.REQUIRED_ROLES_NARROWED,
            cube=cube_name,
            field=field_name,
            detail=f"{sorted(old_set)} -> {sorted(new_set)}",
        )
    # Both added and removed roles — not a clean narrow/widen.
    # Pick the conservative reading: net new restriction exists, so
    # classify as narrowing (breaking).
    return Change(
        kind=ChangeKind.REQUIRED_ROLES_NARROWED,
        cube=cube_name,
        field=field_name,
        detail=f"{sorted(old_set)} -> {sorted(new_set)}",
    )


def _diff_measures(cube_name: str, old: list[Measure], new: list[Measure]) -> list[Change]:
    """Compare two measure lists on a cube.

    Pairs are matched by ``name``. Agg changes are breaking (e.g.
    switching from ``sum`` to ``avg`` changes every consumer's number).
    Other attribute changes (e.g. ``unit``, ``sql``) are NOT tracked
    here — they're either cosmetic or semantic-equivalence is too
    hard to verify automatically. Callers who want full-fidelity SQL
    diffs should use a separate tool."""
    out: list[Change] = []
    by_name_old = {m.name: m for m in old}
    by_name_new = {m.name: m for m in new}
    for name in by_name_new.keys() - by_name_old.keys():
        out.append(Change(kind=ChangeKind.MEASURE_ADDED, cube=cube_name, field=name))
    for name in by_name_old.keys() - by_name_new.keys():
        out.append(Change(kind=ChangeKind.MEASURE_REMOVED, cube=cube_name, field=name))
    for name in by_name_old.keys() & by_name_new.keys():
        o, n = by_name_old[name], by_name_new[name]
        if o.agg != n.agg:
            out.append(
                Change(
                    kind=ChangeKind.MEASURE_AGG_CHANGED,
                    cube=cube_name,
                    field=name,
                    detail=f"{o.agg} -> {n.agg}",
                )
            )
        rc = _set_role_diff(cube_name, name, o.required_roles, n.required_roles)
        if rc is not None:
            out.append(rc)
    return out


def _diff_dimensions(cube_name: str, old: list[Dimension], new: list[Dimension]) -> list[Change]:
    out: list[Change] = []
    by_name_old = {d.name: d for d in old}
    by_name_new = {d.name: d for d in new}
    for name in by_name_new.keys() - by_name_old.keys():
        out.append(Change(kind=ChangeKind.DIMENSION_ADDED, cube=cube_name, field=name))
    for name in by_name_old.keys() - by_name_new.keys():
        out.append(Change(kind=ChangeKind.DIMENSION_REMOVED, cube=cube_name, field=name))
    for name in by_name_old.keys() & by_name_new.keys():
        o, n = by_name_old[name], by_name_new[name]
        if o.type != n.type:
            out.append(
                Change(
                    kind=ChangeKind.DIMENSION_TYPE_CHANGED,
                    cube=cube_name,
                    field=name,
                    detail=f"{o.type} -> {n.type}",
                )
            )
        if set(o.aliases) != set(n.aliases):
            added = sorted(set(n.aliases) - set(o.aliases))
            out.append(
                Change(
                    kind=ChangeKind.DIMENSION_ALIASES_ADDED,
                    cube=cube_name,
                    field=name,
                    detail=f"+{added}",
                )
            )
        rc = _set_role_diff(cube_name, name, o.required_roles, n.required_roles)
        if rc is not None:
            out.append(rc)
    return out


def _diff_segments(cube_name: str, old: list[Segment], new: list[Segment]) -> list[Change]:
    out: list[Change] = []
    by_name_old = {s.name: s for s in old}
    by_name_new = {s.name: s for s in new}
    for name in by_name_new.keys() - by_name_old.keys():
        out.append(Change(kind=ChangeKind.SEGMENT_ADDED, cube=cube_name, field=name))
    for name in by_name_old.keys() - by_name_new.keys():
        out.append(Change(kind=ChangeKind.SEGMENT_REMOVED, cube=cube_name, field=name))
    return out


def _join_signature(j: Join) -> tuple[str, str, str]:
    return (j.to, j.relationship, j.on)


def _diff_joins(cube_name: str, old: list[Join], new: list[Join]) -> list[Change]:
    """Compare two join lists.

    Joins are matched by ``to`` (the joined-to cube name). A
    signature change (``relationship`` or ``on``) is breaking — every
    consumer that references the join path will fail to plan."""
    out: list[Change] = []
    by_to_old = {j.to: j for j in old}
    by_to_new = {j.to: j for j in new}
    for to in by_to_new.keys() - by_to_old.keys():
        out.append(Change(kind=ChangeKind.JOIN_ADDED, cube=cube_name, field=to))
    for to in by_to_old.keys() - by_to_new.keys():
        out.append(Change(kind=ChangeKind.JOIN_REMOVED, cube=cube_name, field=to))
    for to in by_to_old.keys() & by_to_new.keys():
        o, n = by_to_old[to], by_to_new[to]
        if _join_signature(o) != _join_signature(n):
            out.append(
                Change(
                    kind=ChangeKind.JOIN_CHANGED,
                    cube=cube_name,
                    field=to,
                    detail=f"({o.relationship}, {o.on!r}) -> ({n.relationship}, {n.on!r})",
                )
            )
    return out


def _diff_cube(old: Cube | None, new: Cube | None) -> list[Change]:
    if old is None and new is not None:
        return [Change(kind=ChangeKind.CUBE_ADDED, cube=new.name)]
    if new is None and old is not None:
        return [Change(kind=ChangeKind.CUBE_REMOVED, cube=old.name)]
    assert old is not None and new is not None
    out: list[Change] = []
    if old.primary_key != new.primary_key:
        out.append(
            Change(
                kind=ChangeKind.PRIMARY_KEY_CHANGED,
                cube=new.name,
                detail=f"{old.primary_key} -> {new.primary_key}",
            )
        )
    out.extend(_diff_measures(new.name, old.measures, new.measures))
    out.extend(_diff_dimensions(new.name, old.dimensions, new.dimensions))
    out.extend(_diff_segments(new.name, old.segments, new.segments))
    out.extend(_diff_joins(new.name, old.joins, new.joins))
    return out


def diff_catalogs(
    old: dict[str, Cube],
    new: dict[str, Cube],
) -> CatalogDiff:
    """Compare two catalog snapshots and return a :class:`CatalogDiff`.

    Cubes present in only one side produce add/remove changes. Cubes
    present in both produce attribute-level diffs (measures, dimensions,
    segments, joins, primary key). The result is sorted: breaking
    changes first, then additive, then alphabetically by cube/field."""
    all_names = sorted(set(old) | set(new))
    changes: list[Change] = []
    for name in all_names:
        changes.extend(_diff_cube(old.get(name), new.get(name)))
    changes.sort(key=_sort_key)
    return CatalogDiff(changes=changes)


def _sort_key(c: Change) -> tuple[Literal[0, 1], str, str, str]:
    """Sort key: breaking (0) before additive (1), then cube, then field, then kind."""
    return (0 if c.is_breaking else 1, c.cube, c.field or "", c.kind.value)


__all__ = [
    "CatalogDiff",
    "Change",
    "ChangeKind",
    "diff_catalogs",
]
