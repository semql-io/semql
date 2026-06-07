"""Per-cube markdown documentation generator.

``semql.prompt`` already renders the catalogue as an LLM-facing
fragment — terse, with directive language ("the compiler emits
SQL — never write SQL on the semantic path"). This module renders
the same data for *human* documentation: full descriptions, all
the metadata the prompt elides (joins, tenancy, security_sql,
base_predicate, required_filters), per-cube anchors, no LLM
instructions.

Drop the output into ``docs/catalog/<env>.md`` and check it into
the repo — diffs alongside catalog changes are reviewable.
"""

from __future__ import annotations

from semql.catalog import Catalog


def render_catalog_markdown(
    catalog: Catalog,
    *,
    title: str = "Catalogue",
    include_meta: bool = False,
) -> str:
    """Render the full catalogue as a navigable markdown document.

    ``include_meta`` toggles whether the META reflection cubes appear
    in the output. Off by default — most consumers want their own
    cubes documented, not the framework's introspection layer.
    """
    cubes = [c for c in catalog.as_dict().values() if include_meta or c.backend.value != "meta"]

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"_{len(cubes)} cube(s) documented._")
    lines.append("")

    if cubes:
        lines.append("## Table of contents")
        lines.append("")
        for c in cubes:
            anchor = c.name.lower().replace("_", "-")
            label = f"`{c.name}`"
            if c.display_name:
                label = f"`{c.name}` — {c.display_name}"
            lines.append(f"- [{label}](#{anchor})")
        lines.append("")

    for cube in cubes:
        lines.extend(_render_cube(cube))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_cube(cube: object) -> list[str]:  # cube: Cube
    """One ``## <name>`` section per cube."""
    # Imported lazily to keep this module's import surface small.
    from semql.model import Cube

    assert isinstance(cube, Cube)

    out: list[str] = [f"## `{cube.name}`"]
    if cube.display_name:
        out.append(f"_{cube.display_name}_")
    out.append("")
    if cube.description:
        out.append(cube.description)
        out.append("")

    # The "facts" block — every catalogue-level setting on one card.
    facts: list[str] = [
        f"- **Backend:** `{cube.backend.value}`",
        f"- **Table:** `{cube.table}`",
        f"- **Alias:** `{cube.alias}`",
    ]
    if cube.base_predicate:
        facts.append(f"- **Base predicate:** `{cube.base_predicate}`")
    if cube.required_filters:
        reqs = ", ".join(f"`{r}`" for r in cube.required_filters)
        facts.append(f"- **Required filters:** {reqs}")
    if cube.tenancy != "schema":
        facts.append(f"- **Tenancy:** `{cube.tenancy}`")
    if cube.tenancy_column:
        facts.append(f"- **Tenancy column:** `{cube.tenancy_column}`")
    if cube.security_sql:
        facts.append(f"- **Security SQL:** `{cube.security_sql}`")
    if cube.default_chart_type:
        facts.append(f"- **Default chart:** `{cube.default_chart_type}`")
    if not cube.expose_in_prompt:
        facts.append("- **Hidden from planner prompt** (`expose_in_prompt=False`)")
    out.extend(facts)
    out.append("")

    if cube.measures:
        out.append("### Measures")
        out.append("")
        out.append("| Name | Agg | Unit | Format | Filter | Description |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for m in cube.measures:
            agg_label: str = m.agg
            if m.agg == "ratio":
                agg_label = f"ratio ({m.numerator}/{m.denominator})"
            flags: list[str] = []
            if m.non_additive:
                flags.append("non-additive")
            agg_cell = agg_label + (f"  _{' · '.join(flags)}_" if flags else "")
            out.append(
                "| "
                + " | ".join(
                    [
                        f"`{m.name}`",
                        agg_cell,
                        m.unit or "—",
                        m.format or "—",
                        f"`{m.filter}`" if m.filter else "—",
                        m.description or "—",
                    ]
                )
                + " |"
            )
        out.append("")

    if cube.dimensions:
        out.append("### Dimensions")
        out.append("")
        out.append("| Name | Type | Unit | Format | Description |")
        out.append("| --- | --- | --- | --- | --- |")
        for d in cube.dimensions:
            out.append(
                "| "
                + " | ".join(
                    [
                        f"`{d.name}`",
                        f"`{d.type}`",
                        d.unit or "—",
                        d.format or "—",
                        d.description or "—",
                    ]
                )
                + " |"
            )
        out.append("")

    if cube.time_dimensions:
        out.append("### Time dimensions")
        out.append("")
        out.append("| Name | Granularities | Description |")
        out.append("| --- | --- | --- |")
        for td in cube.time_dimensions:
            grans = ", ".join(f"`{g}`" for g in td.granularities)
            out.append(f"| `{td.name}` | {grans} | {td.description or '—'} |")
        out.append("")

    if cube.segments:
        out.append("### Segments")
        out.append("")
        out.append("| Name | SQL | Description |")
        out.append("| --- | --- | --- |")
        for s in cube.segments:
            out.append(f"| `{s.name}` | `{s.sql}` | {s.description or '—'} |")
        out.append("")

    if cube.joins:
        out.append("### Joins")
        out.append("")
        out.append("| → | Relationship | On |")
        out.append("| --- | --- | --- |")
        for j in cube.joins:
            out.append(f"| `{j.to}` | `{j.relationship}` | `{j.on}` |")
        out.append("")

    return out


__all__ = ["render_catalog_markdown"]
