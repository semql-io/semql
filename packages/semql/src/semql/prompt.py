"""Render catalogue + spec contract as a prompt fragment for LLM planners.

`render_catalogue_block` — pure markdown listing of cubes/measures/
dimensions/joins. Shows only cubes flagged `expose_in_prompt=True` by
default; pass `only_exposed=False` for a full catalogue listing.

`build_planner_prompt_fragment` — wraps the catalogue with the spec
contract (what fields `SemanticQuery` takes) and the semantic-vs-raw
guidance. Returns a *fragment*: the caller splices it into the broader
system prompt alongside role description, data-source context, etc.
"""

from __future__ import annotations

from semql.model import Cube


def render_catalogue_block(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
) -> str:
    cubes = [c for c in catalog.values() if c.expose_in_prompt or not only_exposed]
    if not cubes:
        return ""

    lines: list[str] = ["## SEMANTIC CATALOGUE"]
    lines.append(
        "Cubes you can query via the semantic layer. Reference fields as "
        "`cube.field`. The compiler emits SQL — never write SQL on the "
        "semantic path."
    )
    lines.append("")
    for cube in cubes:
        lines.extend(_render_cube(cube))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _human(display_name: str | None) -> str:
    """Render the ``(human: ...)`` suffix when ``display_name`` is set.

    The machine identifier stays as the primary label so the LLM still
    knows what to reference; ``display_name`` rides along as the domain
    label the planner can echo back to users."""
    return f" (human: {display_name})" if display_name else ""


def _render_cube(cube: Cube) -> list[str]:
    header = f"### {cube.name} ({cube.backend.value}){_human(cube.display_name)}"
    out: list[str] = [header]
    if cube.description:
        out.append(cube.description)
    if cube.required_filters:
        reqs = ", ".join(f"`{cube.name}.{r}`" for r in cube.required_filters)
        out.append(f"**Required filters:** {reqs} — compile fails without them.")

    if cube.measures:
        out.append("")
        out.append("**Measures:**")
        for m in cube.measures:
            unit = f" [{m.unit}]" if m.unit else ""
            desc = f" — {m.description}" if m.description else ""
            human = _human(m.display_name)
            flags = " `non-additive`" if m.non_additive else ""
            out.append(f"  - `{cube.name}.{m.name}`{unit} `agg={m.agg}`{flags}{human}{desc}")

    if cube.dimensions:
        out.append("")
        out.append("**Dimensions:**")
        for d in cube.dimensions:
            desc = f" — {d.description}" if d.description else ""
            human = _human(d.display_name)
            out.append(f"  - `{cube.name}.{d.name}` `type={d.type}`{human}{desc}")

    if cube.time_dimensions:
        out.append("")
        out.append("**Time dimensions:**")
        for td in cube.time_dimensions:
            grans = "|".join(td.granularities)
            desc = f" — {td.description}" if td.description else ""
            human = _human(td.display_name)
            out.append(f"  - `{cube.name}.{td.name}` `granularities={grans}`{human}{desc}")

    if cube.joins:
        out.append("")
        out.append("**Joins:**")
        for j in cube.joins:
            out.append(f"  - → `{j.to}` ({j.relationship})")

    return out


_RAW_TRIGGERS: tuple[str, ...] = (
    "Window / rank / lag / lead functions.",
    "Recursive CTEs.",
    "Pivots (rows → columns).",
    "Cross-backend joins — Phase 1 compiler rejects these.",
    "Forecast / predictive shapes.",
    "Columns the catalogue doesn't model.",
)


def _raw_triggers_block(header: str) -> str:
    bullets = "\n".join(f"  - {t}" for t in _RAW_TRIGGERS)
    return f"{header}\n{bullets}"


_SPEC_CONTRACT = """\
## Semantic path

Emit a `SemanticQuery` instead of writing SQL. The compiler turns it
into backend SQL; identifiers, predicates, and parameter binding are
all enforced for you.

Fields:
- `measures: list[str]` — qualified names like `orders.revenue`.
  Aggregated automatically per the catalogue's `agg` field.
- `dimensions: list[str]` — qualified names. Form the GROUP BY when
  measures are present; form the SELECT when ungrouped.
- `time_dimension: {dimension, granularity?, range}` — pre-resolved ISO
  date range, exclusive end. `granularity` truncates to hour/day/week/month.
- `filters: list[{dimension, op, values}]` — pre-aggregation predicates.
  Ops: eq, neq, in, not_in, gt, lt, gte, lte, contains, is_null, not_null.
- `having: list[{dimension, op, values}]` — post-aggregation predicates;
  `dimension` must reference one of the measures you also requested
  (either bare `revenue` or qualified `orders.revenue` — both resolve
  to the same alias).
- `order: list[(field, asc|desc)]` — refer to output column names.
- `limit: int` — required when `ungrouped=True` (capped at 1000).
- `ungrouped: bool` — row-listing mode (no GROUP BY).

Reference fields as `cube.field`. Unknown identifiers fail compile
with a precise message — fix and retry."""


_RAW_FALLBACK = _raw_triggers_block(
    "## When to fall back to raw SQL\n\n"
    "Prefer the semantic path. Fall back to raw SQL only when the "
    "question needs:"
) + (
    "\n\nFalling back is fine — the catalogue earns share by being "
    "preferred where it works, not by being the only option."
)


_INTROSPECTION = """\
## Introspecting the catalogue

The catalogue is itself queryable through three META cubes:
- `catalog_cubes` — one row per cube (name, backend, exposed, alias).
- `catalog_measures` — one row per (cube, measure).
- `catalog_dimensions` — one row per (cube, dim or time_dim); `is_time`
  distinguishes the two.

Use these when the user asks meta questions like "what measures are
available?" or "list available cubes" — same `SemanticQuery` shape."""


def build_planner_prompt_fragment(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
    include_introspection: bool = False,
) -> str:
    """Compose the semantic-layer fragment of a planner's system prompt.

    Returns a fragment that includes the spec contract, the catalogue
    block, and the raw-fallback rule. Splice into your broader system
    prompt alongside role description, data-source context, etc.
    """
    parts: list[str] = [
        _SPEC_CONTRACT,
        render_catalogue_block(catalog, only_exposed=only_exposed).rstrip(),
        _RAW_FALLBACK,
    ]
    if include_introspection:
        parts.append(_INTROSPECTION)
    return "\n\n".join(parts) + "\n"


def build_router_prompt_fragment(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
    include_topic_summary: bool = True,
) -> str:
    """Fragment for the path-routing decision (semantic vs raw SQL).

    Optionally appends a one-liner per exposed cube so the router has a
    sense of what's catalogue-expressible without the full measures tree.
    """
    router_header = _raw_triggers_block(
        "## Path routing — semantic vs raw SQL\n\n"
        "Prefer the semantic path when the question maps cleanly to the "
        "catalogue's measures, dimensions, and filters. Drop to raw SQL "
        "only when the question needs SQL shapes the catalogue can't express:"
    ) + (
        "\n\nIf you're unsure, try the semantic path first — the compiler's "
        "error message will tell you exactly which identifier is missing."
    )

    parts: list[str] = [router_header]
    if include_topic_summary:
        topics: list[str] = ["## Catalogue topics"]
        for cube in catalog.values():
            if only_exposed and not cube.expose_in_prompt:
                continue
            blurb = cube.description.split(".")[0] if cube.description else ""
            blurb = f" — {blurb}." if blurb else "."
            human = _human(cube.display_name)
            topics.append(f"  - `{cube.name}` ({cube.backend.value}){human}{blurb}")
        parts.append("\n".join(topics))
    return "\n\n".join(parts) + "\n"


__all__ = [
    "build_planner_prompt_fragment",
    "build_router_prompt_fragment",
    "render_catalogue_block",
]
