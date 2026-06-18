"""DOT-source generation for catalog ER diagrams.

Pure-Python — no third-party imports. ``render_dot(catalog)`` walks
the cubes and joins and produces a string consumable by any Graphviz
renderer.
"""

from __future__ import annotations

from typing import Literal

from semql import Catalog
from semql.model import Cube

RankDir = Literal["LR", "TB", "RL", "BT"]


# ---------------------------------------------------------------------------
# Crow's-foot arrowhead conventions per relationship.
# ``arrowhead`` is the marker at the target end of the edge; ``arrowtail``
# is at the source end. We enable ``dir=both`` so both markers render.
# ---------------------------------------------------------------------------


def _relationship_attrs(relationship: str) -> dict[str, str]:
    if relationship == "many_to_one":
        return {"arrowtail": "crow", "arrowhead": "tee", "dir": "both"}
    if relationship == "one_to_many":
        return {"arrowtail": "tee", "arrowhead": "crow", "dir": "both"}
    if relationship == "one_to_one":
        return {"arrowtail": "tee", "arrowhead": "tee", "dir": "both"}
    return {"arrowhead": "normal"}


# ---------------------------------------------------------------------------
# Escaping for DOT record-shape labels.
# Record labels use ``|`` as a section separator and ``{`` / ``}`` to group;
# ``<`` and ``>`` introduce port names. Escape those plus the obvious
# quote / backslash so a description with apostrophes or pipes doesn't
# break the layout.
# ---------------------------------------------------------------------------


_RECORD_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "|": "\\|",
    "{": "\\{",
    "}": "\\}",
    "<": "\\<",
    ">": "\\>",
    "\n": "\\n",
}


def _escape_record(text: str) -> str:
    return "".join(_RECORD_ESCAPES.get(ch, ch) for ch in text)


def _cube_label(cube: Cube) -> str:
    """Build a DOT record-shape label string for ``cube``.

    Layout: header (cube name + optional display_name + backend) on top,
    measures / dimensions / time-dimensions stacked below, separated by
    horizontal rules (``|`` between sections)."""
    header_parts = [cube.name]
    if cube.display_name:
        header_parts.append(f"({cube.display_name})")
    header_parts.append(f"\n[{cube.dialect.value}]")
    header = " ".join(header_parts)

    sections: list[str] = [header]
    if cube.measures:
        ms = ", ".join(_field_label(m.name, m.unit, m.display_unit) for m in cube.measures)
        sections.append(f"measures: {ms}")
    if cube.dimensions:
        ds = ", ".join(_field_label(d.name, d.unit, d.display_unit) for d in cube.dimensions)
        sections.append(f"dimensions: {ds}")
    if cube.time_dimensions:
        ts = ", ".join(td.name for td in cube.time_dimensions)
        sections.append(f"time: {ts}")

    return "{" + "|".join(_escape_record(s) for s in sections) + "}"


def _field_label(name: str, unit: str | None, display_unit: str | None) -> str:
    """Annotate a field with its storage / display unit when present.

    Renders ``name [unit → display_unit]`` when both are set and
    differ, ``name [unit]`` when only ``unit`` is set, and the bare
    ``name`` otherwise. Mirrors the convention used in the planner
    prompt fragment so the ERD and the prompt stay aligned."""
    if unit and display_unit and display_unit != unit:
        return f"{name} [{unit} → {display_unit}]"
    if unit:
        return f"{name} [{unit}]"
    return name


def _dot_quoted(name: str) -> str:
    """Wrap ``name`` in DOT double-quoted identifier syntax.

    DOT allows arbitrary node IDs inside double quotes; the only
    character that needs escaping is ``"`` itself (SEMQL-DISC-ERD-DOT-INJECTION-004).
    """
    return '"' + name.replace('"', '\\"') + '"'


def _node_id(cube: Cube) -> str:
    """A quoted DOT node identifier for ``cube``."""
    return _dot_quoted(cube.name)


def _cubes_in_scope(catalog: Catalog, *, only_exposed: bool) -> list[Cube]:
    """Filter the catalog for rendering. META reflection cubes are
    always excluded — they're an introspection mechanism, not part of
    the data model. ``only_exposed=True`` (default) also drops cubes
    flagged ``expose_in_prompt=False`` so the diagram matches what the
    planner sees."""
    from semql import iter_cubes

    return list(iter_cubes(catalog, only_exposed=only_exposed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_dot(
    catalog: Catalog,
    *,
    only_exposed: bool = True,
    rankdir: RankDir = "LR",
    title: str | None = None,
) -> str:
    """Render the catalog as a Graphviz DOT source string.

    ``only_exposed`` (default ``True``) mirrors the planner-prompt
    filter — only cubes flagged ``expose_in_prompt=True`` appear.
    ``rankdir`` controls layout direction (LR/TB/RL/BT).
    ``title`` is an optional graph label rendered at the top.
    """
    cubes = _cubes_in_scope(catalog, only_exposed=only_exposed)
    in_scope: set[str] = {c.name for c in cubes}

    lines: list[str] = ["digraph catalog {"]
    lines.append(f'  rankdir="{rankdir}";')
    lines.append('  node [shape=record, fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')
    if title:
        lines.append(f'  label="{_escape_record(title)}";')
        lines.append("  labelloc=t;")
    lines.append("")

    for cube in cubes:
        lines.append(f'  {_node_id(cube)} [label="{_cube_label(cube)}"];')

    lines.append("")
    for cube in cubes:
        for join in cube.joins:
            if join.to not in in_scope:
                # Skip edges that would dangle into filtered-out cubes.
                continue
            attrs = _relationship_attrs(join.relationship)
            attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
            lines.append(f"  {_node_id(cube)} -> {_dot_quoted(join.to)} [{attr_str}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


__all__ = ["RankDir", "render_dot"]
