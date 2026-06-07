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

from semql.introspect import PolicyFn, iter_cubes
from semql.model import AuthContext, Cube, Lookup, ResolutionContext, View


def _render_lookup_line(dim_ref: str, lookup: Lookup, ctx: ResolutionContext | None) -> str | None:
    """Render the one-line lookup hint that follows a Dimension entry.

    Inlines values up to ``max_inline``; beyond that emits a tool-hint
    pointing at ``resolve_<dim>`` so the planner narrows via lookup
    instead of stuffing a huge list into the prompt. Returns ``None``
    when the lookup contributes nothing (dynamic with no context)."""
    from semql.lookups import materialize  # local import: avoid module cycle at import time

    materialized = materialize(lookup, ctx)
    cube_name, dim_name = dim_ref.split(".", 1)
    if materialized is None:
        # Dynamic lookup with no context — surface the tool hint anyway
        # so the planner knows resolution is available.
        return (
            f"    Lookup: values resolved at runtime; use "
            f"`resolve_{cube_name}_{dim_name}(query)` to look up canonical ids."
        )
    values, labels = materialized
    if len(values) <= lookup.max_inline:
        if labels:
            rendered = ", ".join(
                f"`{v}` ({labels[v]})" if v in labels else f"`{v}`" for v in values
            )
        else:
            rendered = ", ".join(f"`{v}`" for v in values)
        return f"    Lookup ({len(values)} values): {rendered}"
    # Over the inline cap — surface count + a tool hint.
    sample = ", ".join(f"`{v}`" for v in values[: max(1, lookup.max_inline // 5)])
    return (
        f"    Lookup ({len(values)} values; sample: {sample}, …): use "
        f"`resolve_{cube_name}_{dim_name}(query)` to narrow to a canonical id."
    )


def render_catalogue_block(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    lookups: dict[str, Lookup] | None = None,
    ctx: ResolutionContext | None = None,
) -> str:
    # ``include_meta=True`` here is deliberate: META reflection cubes
    # historically appeared in the planner fragment when callers opted
    # into introspection (downstream ``build_planner_prompt_fragment``
    # gates the META section with ``include_introspection``, but the
    # catalogue block itself stays inclusive — META cubes carry
    # ``expose_in_prompt=False`` so ``only_exposed=True`` hides them
    # by default anyway).
    cubes = list(
        iter_cubes(
            catalog,
            include_meta=True,
            only_exposed=only_exposed,
            viewer=viewer,
            policy=policy,
        )
    )
    if not cubes:
        return ""

    lines: list[str] = ["## SEMANTIC CATALOGUE"]
    lines.append(
        "Cubes you can query via the semantic layer. Reference fields as "
        "`cube.field`. The compiler emits SQL — never write SQL on the "
        "semantic path."
    )
    lines.append("")
    lookups_by_dim = dict(lookups or {})
    for cube in cubes:
        lines.extend(_render_cube(cube, lookups_by_dim, ctx))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _human(display_name: str | None) -> str:
    """Render the ``(human: ...)`` suffix when ``display_name`` is set.

    The machine identifier stays as the primary label so the LLM still
    knows what to reference; ``display_name`` rides along as the domain
    label the planner can echo back to users."""
    return f" (human: {display_name})" if display_name else ""


def _render_cube(
    cube: Cube,
    lookups: dict[str, Lookup],
    ctx: ResolutionContext | None,
) -> list[str]:
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
            # Surface display_unit alongside storage unit so the planner
            # doesn't invent its own conversion (e.g. ``/3600`` to read
            # seconds-stored watch_time in hours).
            if m.unit and m.display_unit and m.display_unit != m.unit:
                unit = f" [{m.unit} → {m.display_unit}]"
            elif m.unit:
                unit = f" [{m.unit}]"
            else:
                unit = ""
            desc = f" — {m.description}" if m.description else ""
            human = _human(m.display_name)
            flags = " `non-additive`" if m.non_additive else ""
            filtered = " `filtered`" if m.filter else ""
            out.append(
                f"  - `{cube.name}.{m.name}`{unit} `agg={m.agg}`{flags}{filtered}{human}{desc}"
            )

    if cube.dimensions:
        out.append("")
        out.append("**Dimensions:**")
        for d in cube.dimensions:
            desc = f" — {d.description}" if d.description else ""
            human = _human(d.display_name)
            out.append(f"  - `{cube.name}.{d.name}` `type={d.type}`{human}{desc}")
            dim_ref = f"{cube.name}.{d.name}"
            lk = lookups.get(dim_ref)
            if lk is not None:
                lookup_line = _render_lookup_line(dim_ref, lk, ctx)
                if lookup_line is not None:
                    out.append(lookup_line)

    if cube.time_dimensions:
        out.append("")
        out.append("**Time dimensions:**")
        for td in cube.time_dimensions:
            grans = "|".join(td.granularities)
            desc = f" — {td.description}" if td.description else ""
            human = _human(td.display_name)
            out.append(f"  - `{cube.name}.{td.name}` `granularities={grans}`{human}{desc}")

    if cube.segments:
        out.append("")
        out.append("**Segments:**")
        for s in cube.segments:
            desc = f" — {s.description}" if s.description else ""
            human = _human(s.display_name)
            out.append(f"  - `{cube.name}.{s.name}`{human}{desc}")

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
- `time_dimension: {dimension, granularity?, range, fill_nulls_with?}` —
  pre-resolved ISO date range, exclusive end. `granularity` truncates
  to hour/day/week/month. `fill_nulls_with: int` emits one row per
  bucket in range and COALESCEs missing measures to the int — use it
  for line charts that need an unbroken time axis.
- `filters: list[{dimension, op, values}]` — pre-aggregation predicates.
  Ops: eq, neq, in, not_in, gt, lt, gte, lte, contains, is_null, not_null.
- `where: BoolExpr | null` — boolean predicate tree for OR / NOT.
  `{op: "and"|"or"|"not", children: [Filter | BoolExpr, ...]}`. Composes
  with `filters` via implicit AND. Use only when `filters` (flat AND)
  isn't expressive enough.
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


def _render_view_block(views: dict[str, View]) -> str:
    """Per-view markdown — each view lists its exposed field names and
    the underlying ``cube.field`` targets. Planners can address fields
    via the view; the compiler rewrites the references at compile time."""
    if not views:
        return ""
    lines: list[str] = ["## VIEWS"]
    lines.append(
        "Curated facades over one or more cubes. Reference view fields "
        "as `view.field`; the compiler maps them to the underlying cube."
    )
    lines.append("")
    for v in views.values():
        header = f"### {v.name}"
        if v.display_name:
            header += f" (human: {v.display_name})"
        lines.append(header)
        if v.description:
            lines.append(v.description)
        lines.append("")
        lines.append("**Fields:**")
        for local, target in v.fields.items():
            lines.append(f"  - `{v.name}.{local}` → `{target}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_planner_prompt_fragment(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
    include_introspection: bool = False,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    lookups: dict[str, Lookup] | None = None,
    ctx: ResolutionContext | None = None,
) -> str:
    """Compose the semantic-layer fragment of a planner's system prompt.

    Returns a fragment that includes the spec contract, the catalogue
    block, and the raw-fallback rule. Splice into your broader system
    prompt alongside role description, data-source context, etc.

    ``viewer`` + ``policy`` (when set) shrink the catalogue block to
    only the cubes the viewer is authorised to see — keeps the planner
    from suggesting a query it can't run.

    ``lookups`` + ``ctx`` inline dimension-value catalogues underneath
    their dimensions. Dynamic ``Lookup`` loaders fire here (the only
    I/O path in the prompt builder).
    """
    parts: list[str] = [
        _SPEC_CONTRACT,
        render_catalogue_block(
            catalog,
            only_exposed=only_exposed,
            viewer=viewer,
            policy=policy,
            lookups=lookups,
            ctx=ctx,
        ).rstrip(),
    ]
    if views:
        parts.append(_render_view_block(views).rstrip())
    parts.append(_RAW_FALLBACK)
    if include_introspection:
        parts.append(_INTROSPECTION)
    return "\n\n".join(parts) + "\n"


_ROUTER_OUTPUT_SCHEMA = """\
## Output

Emit a `RouterDecision`:

```
{
  "path": "semantic" | "raw",
  "cubes": [cube_name, ...],   // empty when path = "raw"
  "views": [view_name, ...],   // empty when path = "raw"
  "reasoning": "<one short sentence>"
}
```

When `path = "semantic"`, list ONLY the cubes / views the next stage
will need (most questions need 1-3). The downstream Query Generator
sees a catalogue trimmed to your picks, so being precise here shrinks
its prompt and sharpens its output.
"""


def build_router_prompt_fragment(
    catalog: dict[str, Cube],
    *,
    only_exposed: bool = True,
    include_topic_summary: bool = True,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
) -> str:
    """Fragment for the path-routing decision (semantic vs raw SQL).

    Optionally appends a one-liner per exposed cube so the router has a
    sense of what's catalogue-expressible without the full measures tree.
    When ``views`` is provided, a parallel one-liner list lets the
    router pick a curated facade instead of (or in addition to) the
    raw cubes.

    Ends with the ``RouterDecision`` output schema so a typed-output
    LLM client (pydantic-ai etc.) can parse the response directly.
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
        for cube in iter_cubes(
            catalog,
            include_meta=True,
            only_exposed=only_exposed,
            viewer=viewer,
            policy=policy,
        ):
            blurb = cube.description.split(".")[0] if cube.description else ""
            blurb = f" — {blurb}." if blurb else "."
            human = _human(cube.display_name)
            topics.append(f"  - `{cube.name}` ({cube.backend.value}){human}{blurb}")
        parts.append("\n".join(topics))

        if views:
            view_lines: list[str] = ["## Views"]
            view_lines.append(
                "Curated facades over one or more cubes. Reference view fields "
                "as `view.field` — the compiler maps each reference back to "
                "the underlying cube."
            )
            for v in views.values():
                blurb = v.description.split(".")[0] if v.description else ""
                blurb = f" — {blurb}." if blurb else ""
                human = _human(v.display_name)
                view_lines.append(f"  - `{v.name}`{human}{blurb}")
            parts.append("\n".join(view_lines))

    parts.append(_ROUTER_OUTPUT_SCHEMA.rstrip())
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Query Generator
# ---------------------------------------------------------------------------


_GENERATOR_OUTPUT_SCHEMA = """\
## Output

Emit a `QueryPlan`:

```
{
  "steps": [
    {
      "query": <SemanticQuery>,
      "intent": "headline" | "breakdown" | "compare" | "context",
      "label": "<one-line description, optional>"
    },
    ...
  ],
  "reasoning": "<one short sentence, optional>"
}
```

Intent vocabulary:
- `headline` — the primary number the user asked for.
- `breakdown` — disaggregation alongside the headline.
- `compare` — sibling number for context (prior period, baseline).
- `context` — supporting data the answer references but doesn't feature.

One headline per plan is the common case; emit 2-4 total steps when
the question naturally decomposes ("revenue this quarter" → headline
+ prior-quarter compare). Empty `steps` means you can't formulate a
query — return that rather than guessing."""


def build_query_generator_prompt_fragment(
    catalog: dict[str, Cube],
    *,
    scope_to: list[str] | None = None,
    only_exposed: bool = True,
    include_introspection: bool = False,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    lookups: dict[str, Lookup] | None = None,
    ctx: ResolutionContext | None = None,
) -> str:
    """Fragment for the second stage of the prompt pipeline.

    Given the Router's pick, this stage turns the question into a
    ``QueryPlan`` (one or more ``QueryStep`` with typed intent).

    ``scope_to`` is the retrieval-pass parameter: when set, the
    rendered catalogue includes only the named cubes (and any views
    in the provided ``views`` dict whose name appears in ``scope_to``).
    Pair with the Router's ``cubes`` + ``views`` output to shrink the
    Generator's prompt to the surface that question actually needs.
    """
    scoped_catalog: dict[str, Cube] = catalog
    scoped_views: dict[str, View] | None = views
    scoped_lookups: dict[str, Lookup] | None = lookups
    if scope_to is not None:
        wanted = set(scope_to)
        scoped_catalog = {n: c for n, c in catalog.items() if n in wanted}
        if views is not None:
            scoped_views = {n: v for n, v in views.items() if n in wanted}
        if lookups is not None:
            scoped_lookups = {d: lk for d, lk in lookups.items() if d.split(".", 1)[0] in wanted}

    parts: list[str] = [
        _SPEC_CONTRACT,
        render_catalogue_block(
            scoped_catalog,
            only_exposed=only_exposed,
            viewer=viewer,
            policy=policy,
            lookups=scoped_lookups,
            ctx=ctx,
        ).rstrip(),
    ]
    if scoped_views:
        parts.append(_render_view_block(scoped_views).rstrip())
    parts.append(_RAW_FALLBACK)
    if include_introspection:
        parts.append(_INTROSPECTION)
    parts.append(_GENERATOR_OUTPUT_SCHEMA.rstrip())
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Presenter
# ---------------------------------------------------------------------------


_PRESENTER_OUTPUT_SCHEMA = """\
## Output

Emit a `Presentation`:

```
{
  "summary": "<one-paragraph user-facing answer>",
  "highlights": ["<bullet>", ...],
  "caveats": ["<bullet>", ...]
}
```

- `summary` is what an executive reads first. Lead with the answer,
  not the methodology. One paragraph; 1-3 sentences.
- `highlights` (optional) call out what's worth noticing — outliers,
  trends, surprising values. Skip when nothing's notable.
- `caveats` (optional) flag small samples, missing data, ambiguous
  comparisons, anything that would make a careful reader hedge.
  Skip when the result is unambiguous."""


def build_presenter_prompt_fragment(
    *,
    query_labels: list[str] | None = None,
    result_summary: str | None = None,
) -> str:
    """Fragment for the third stage of the prompt pipeline.

    ``query_labels`` — optional one-liners describing each query in
    the plan (e.g. ``["Q4 revenue", "Q3 revenue", "Q4 by region"]``);
    splice them in so the Presenter knows what data it received.

    ``result_summary`` — optional caller-supplied prose summary of the
    rows themselves (e.g. ``"3 rows, max=12_400, min=8_200"``).
    The Presenter narrates, but you decide how much data lands inside
    the prompt — pass small samples directly, or summarise large
    results outside.

    The chart-shape decision lives in ``decide_visualization``
    (``semql.visualize``). The Presenter handles prose; the visualiser
    handles chart selection. Keep them decoupled."""
    parts: list[str] = [
        "## Presenter\n\n"
        "Turn query results into a user-facing answer. You receive one or "
        "more `QueryStep`s with their intents (headline / breakdown / "
        "compare / context) and the rows that resulted. Compose a coherent "
        "narrative — headline first, then notable details, then caveats."
    ]
    if query_labels:
        bullet_block = "\n".join(f"  - {label}" for label in query_labels)
        parts.append("## Queries in this plan\n" + bullet_block)
    if result_summary:
        parts.append("## Result snapshot\n" + result_summary)
    parts.append(_PRESENTER_OUTPUT_SCHEMA.rstrip())
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Deep drilldown
# ---------------------------------------------------------------------------


_DRILLDOWN_OUTPUT_SCHEMA = """\
## Output

Emit a `DrilldownSuggestions`:

```
{
  "suggestions": [
    {
      "label": "<short clickable label>",
      "query": <SemanticQuery>,
      "rationale": "<optional one-line why>"
    },
    ...
  ],
  "focus": "<one-line description of the anchor row, optional>"
}
```

3-5 suggestions is the sweet spot. Each must be a runnable
`SemanticQuery` against this cube (or its joined neighbours).
Favour drills the catalogue's `drill_paths` already suggests, but
add cross-cube drills (via declared joins) when they'd be revealing."""


def build_drilldown_prompt_fragment(
    cube: Cube,
    *,
    focused_row: dict[str, str] | None = None,
    drill_paths_hint: bool = True,
) -> str:
    """Fragment for the fourth stage of the prompt pipeline.

    Anchored to one ``cube`` — the drilldown explores within / from
    that cube. ``focused_row`` is the (dimension → value) mapping for
    the row of interest; the suggestions should narrow to or expand
    from it.

    ``drill_paths_hint=True`` (default) renders the cube's declared
    ``drill_paths`` inline as a suggestion baseline."""
    parts: list[str] = [
        f"## Drill down on `{cube.name}`\n\n"
        "Propose follow-up queries an analyst might ask next, given the "
        "focused row. Each suggestion is a clickable next-question — a "
        "complete `SemanticQuery` plus a short label."
    ]
    if cube.description:
        parts.append(f"### Cube: {cube.description}")

    if focused_row:
        row_block = "\n".join(f"  - `{k}`: {v!r}" for k, v in focused_row.items())
        parts.append("## Focused row\n" + row_block)

    if drill_paths_hint and cube.drill_paths:
        path_block = "\n".join(f"  - {' → '.join(path)}" for path in cube.drill_paths)
        parts.append(
            "## Declared drill paths\n"
            "These hierarchies are catalogue-blessed; prefer suggestions "
            "that walk them:\n" + path_block
        )

    if cube.measures:
        ms = ", ".join(f"`{cube.name}.{m.name}`" for m in cube.measures)
        parts.append(f"## Available measures\n{ms}")
    if cube.dimensions:
        ds = ", ".join(f"`{cube.name}.{d.name}`" for d in cube.dimensions)
        parts.append(f"## Available dimensions\n{ds}")
    if cube.time_dimensions:
        ts = ", ".join(f"`{cube.name}.{td.name}`" for td in cube.time_dimensions)
        parts.append(f"## Available time dimensions\n{ts}")

    parts.append(_DRILLDOWN_OUTPUT_SCHEMA.rstrip())
    return "\n\n".join(parts) + "\n"


__all__ = [
    "build_drilldown_prompt_fragment",
    "build_planner_prompt_fragment",
    "build_presenter_prompt_fragment",
    "build_query_generator_prompt_fragment",
    "build_router_prompt_fragment",
    "render_catalogue_block",
]
