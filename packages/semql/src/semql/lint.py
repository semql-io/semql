"""I2 — Catalog-level static linting.

Walks a catalog and surfaces structural smells before they hit a
real query. Each rule is a pure function over the catalog. The
``lint_catalog`` entry point returns a :class:`LintReport` with
one :class:`LintFinding` per violation.

Rules:

  - ``cube_no_measures`` — a cube with no measures is unreachable
    for any aggregating query. (warning)
  - ``empty_cube`` — a cube with no measures, no dimensions, and
    no time dimensions is a stub. (warning)
  - ``count_distinct_non_additive`` — a measure with
    ``agg='count_distinct'`` and ``non_additive=False`` is wrong
    when rolled up across coarser time grains. (warning)
  - ``segment_external_field`` — a Segment's SQL references a
    field that isn't a dimension on the same cube. We extract
    identifier-like tokens from the SQL and check that each
    matches a declared dim name. (warning)
  - ``pk_no_fk_joins`` — a cube declares ``primary_key`` but has
    no ``Join`` to / from any other cube. (warning)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from semql.model import Cube

# Identifier-ish tokens: ``{alias}.field_name`` (we strip the alias
# prefix when present) or just ``field_name``. The catch is the
# alias name itself (one word after ``{``); we filter to known dim
# names. The tokeniser is deliberately narrow — we want this rule
# to be a smoke check, not a SQL parser.
_IDENT_RE = re.compile(r"\{[a-z_]+\}\.([a-z_][a-z0-9_]*)")


@dataclass(frozen=True)
class LintFinding:
    """One structural smell. ``severity`` is ``"warning"`` for now
    (catalog author should look); ``"error"`` reserved for compile-
    blocking misconfig that will surface at query time."""

    rule: str
    severity: str
    cube: str
    message: str


@dataclass(frozen=True)
class LintReport:
    """All findings from a single ``lint_catalog`` call."""

    findings: tuple[LintFinding, ...]

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def by_rule(self, rule: str) -> tuple[LintFinding, ...]:
        return tuple(f for f in self.findings if f.rule == rule)


def _cube_no_measures(cube: Cube) -> Iterable[LintFinding]:
    if not cube.measures:
        yield LintFinding(
            rule="cube_no_measures",
            severity="warning",
            cube=cube.name,
            message=(f"Cube {cube.name!r} has no measures — no aggregating query can target it."),
        )


def _empty_cube(cube: Cube) -> Iterable[LintFinding]:
    if not cube.measures and not cube.dimensions and not cube.time_dimensions:
        yield LintFinding(
            rule="empty_cube",
            severity="warning",
            cube=cube.name,
            message=(
                f"Cube {cube.name!r} has no measures, no dimensions, and "
                "no time dimensions — looks like a stub."
            ),
        )


def _count_distinct_non_additive(cube: Cube) -> Iterable[LintFinding]:
    for m in cube.measures:
        if m.agg == "count_distinct" and not m.non_additive:
            yield LintFinding(
                rule="count_distinct_non_additive",
                severity="warning",
                cube=cube.name,
                message=(
                    f"Measure {cube.name}.{m.name!r}: agg='count_distinct' "
                    "with non_additive=False produces a wrong answer when "
                    "rolled up across coarser time grains. Set "
                    "non_additive=True to surface this in the planner prompt."
                ),
            )


def _segment_external_field(cube: Cube) -> Iterable[LintFinding]:
    if not cube.segments:
        return
    declared_dim_names = {d.name for d in cube.dimensions}
    declared_dim_names.update(td.name for td in cube.time_dimensions)
    for seg in cube.segments:
        referenced = set(_IDENT_RE.findall(seg.sql))
        # ``{alias}`` tokens are the cube's own table alias; the
        # right-hand identifier is a column / field name.
        unknown = referenced - declared_dim_names
        if unknown:
            yield LintFinding(
                rule="segment_external_field",
                severity="warning",
                cube=cube.name,
                message=(
                    f"Segment {cube.name}.{seg.name!r}: SQL references "
                    f"unrecognised field(s) {sorted(unknown)} — only "
                    f"dimensions of {cube.name!r} are visible in segments."
                ),
            )


def _pk_no_fk_joins(cube: Cube) -> Iterable[LintFinding]:
    if cube.primary_key is None:
        return
    if not cube.joins:
        yield LintFinding(
            rule="pk_no_fk_joins",
            severity="warning",
            cube=cube.name,
            message=(
                f"Cube {cube.name!r} declares primary_key={cube.primary_key!r} "
                "but has no joins — the cube is isolated from the rest of "
                "the catalog."
            ),
        )


_RULES = (
    _cube_no_measures,
    _empty_cube,
    _count_distinct_non_additive,
    _segment_external_field,
    _pk_no_fk_joins,
)


def lint_catalog(catalog: dict[str, Cube]) -> LintReport:
    """Walk every cube in ``catalog`` and collect rule violations.

    The catalog is the dict shape ``compile_query`` accepts (cube
    name → ``Cube``). The return is a flat ``LintReport`` — no
    cross-cube rules yet (e.g. joins to non-existent cubes), but
    the per-cube rule set catches the most common misconfig.
    """
    findings: list[LintFinding] = []
    for cube in catalog.values():
        for rule in _RULES:
            findings.extend(rule(cube))
    return LintReport(findings=tuple(findings))


__all__ = ["LintFinding", "LintReport", "lint_catalog"]
