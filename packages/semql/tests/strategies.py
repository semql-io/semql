"""Hypothesis strategies for the semql compiler.

Redesign per ``docs/specs/property-testing.md`` §1: feature-flagged swarm
generation, tree-shaped multi-cube join graphs, time dimensions /
base_predicate / numeric dimensions, a hostile value tail, and negative
(``broken_pair``) generation.

Identifiers come from a seeded mimesis pool behind ``st.sampled_from`` so
Hypothesis can shrink them (raw mimesis calls are not shrinkable).

Filter string *values* are wrapped in a unique :data:`SENTINEL` marker by
construction. A value spliced into SQL text would drag its marker along,
so "no SENTINEL ever appears in ``out.sql``" is a sound bind-params oracle
that — unlike a bare substring check — can't false-positive on values that
look like a driver placeholder (``%(p0)s``, ``?``).
"""

from __future__ import annotations

from typing import Literal, cast

from hypothesis import strategies as st
from mimesis import Locale, Text
from semql import (
    Backend,
    BoolExpr,
    Catalog,
    Cube,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from sqlglot.dialects.postgres import Postgres

# ---------------------------------------------------------------------------
# Name pool
# ---------------------------------------------------------------------------

_ALIASES = list("abcdefghijklmnopqrstuvwxyz")

# Postgres reserved words sqlglot won't quote in identifier position — using
# one as a table/field name would yield unparseable SQL.
_SQL_KEYWORDS: frozenset[str] = frozenset(
    k.lower() for k in Postgres.tokenizer_class.KEYWORDS if k.isalpha()
)


def _build_pool(n: int = 600) -> list[str]:
    gen = Text(locale=Locale.EN, seed=0)
    seen: dict[str, None] = {}
    for _ in range(n):
        word = gen.word().lower().replace("-", "_").replace(" ", "_")
        if word.isidentifier() and len(word) >= 3 and word not in _SQL_KEYWORDS:
            seen[word] = None
    return list(seen)


_POOL: list[str] = _build_pool()

identifier: st.SearchStrategy[str] = st.sampled_from(_POOL)

# Backends with a real sqlglot dialect (excludes META). ``Backend.value`` is
# the sqlglot dialect name, so the dialect-validity property can reuse it.
_DIALECT_BACKENDS = [
    Backend.POSTGRES,
    Backend.CLICKHOUSE,
    Backend.DUCKDB,
    Backend.BIGQUERY,
    Backend.SNOWFLAKE,
]


# ---------------------------------------------------------------------------
# Feature-flagged swarm (§1.1)
# ---------------------------------------------------------------------------

FEATURES = frozenset(
    {
        "joins",
        "time_dim",
        "filters",
        "having",
        "order",
        "limit",
        "base_predicate",
        "ungrouped",
        "left_joins",
        "numeric_filter",
        "nested_filter",
    }
)


@st.composite
def feature_set(draw: st.DrawFn) -> frozenset[str]:
    return frozenset(draw(st.sets(st.sampled_from(sorted(FEATURES)))))


# ---------------------------------------------------------------------------
# Value strategies: the hostile tail (§1.4)
# ---------------------------------------------------------------------------

#: Unique marker wrapped around every generated filter string. Its absence
#: from emitted SQL proves the value was bound, not spliced.
SENTINEL = "⟦𝕊⟧"

_sql_meta = st.sampled_from(
    [
        "'",
        "''",
        '"',
        "--",
        "/*",
        "*/",
        ";",
        "; DROP TABLE t;--",
        "\\",
        "%",
        "_",
        "`",
        "$$",
        "${x}",
        "{o}",
        "%(p)s",
        "@p",
        "?",
    ]
)
_weird_unicode = st.sampled_from(
    [
        "\u02bc",  # modifier-letter apostrophe (homoglyph quote)
        "\uff07",  # fullwidth apostrophe
        "\u061b",  # arabic semicolon
        "\x00",  # NUL
        "\u202e",  # RTL override
        "\U0001f98a",  # fox emoji
        "\u03a9",  # capital omega
        "\ufb00",  # ff ligature
        "\u0131",  # dotless i
        "\u00df",  # eszett (case-folding trap)
        "\u200b",  # zero-width space
    ]
)


@st.composite
def filter_value(draw: st.DrawFn) -> str:
    """A sentinel-wrapped string value — benign-weighted with a hostile tail."""
    inner = draw(
        st.one_of(
            st.text(max_size=20),
            st.text(max_size=2000),
            _sql_meta,
            _weird_unicode,
        )
    )
    return f"{SENTINEL}{inner}{SENTINEL}"


# Numeric edges kept finite here (nan/inf live in negative generation, where a
# typed refusal is the expected outcome rather than noise).
_numeric_value = st.one_of(
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.sampled_from([0, -0.0, 1e308, -1e308, 0.1 + 0.2]),
)

temporal_edges = st.sampled_from(
    [
        "1970-01-01",
        "2024-02-29",
        "2026-01-01",
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00Z",
        "9999-12-31",
        "2026-03-08T02:30:00",  # DST gap
    ]
)

# Hostile identifiers feed *negative* properties only (catalog-author SQL is
# trusted input — keep it out of the main path, §3.7).
hostile_identifier = st.sampled_from(
    ["select", "from", "where", "group", "order by", '"x"', "a b", "1col", "x" * 200, "Müller"]
)


# ---------------------------------------------------------------------------
# Catalog strategy: tree-shaped join graph (§1.2)
# ---------------------------------------------------------------------------


@st.composite
def random_cube(
    draw: st.DrawFn,
    features: frozenset[str] = FEATURES,
    *,
    alias: str | None = None,
    name: str | None = None,
    backend: Backend = Backend.POSTGRES,
) -> Cube:
    """A single-backend cube with realistic names. SQL fragments use
    ``{alias}.{field}`` so they always parse; the join key columns
    (``id`` / ``p<i>``) are raw table columns, not declared fields."""
    a = alias if alias is not None else draw(st.sampled_from(_ALIASES))
    cube_name = name if name is not None else draw(identifier)
    table = draw(identifier)

    n_measures = draw(st.integers(min_value=1, max_value=3))
    n_dims = draw(st.integers(min_value=1, max_value=3))
    field_names = draw(
        st.lists(
            identifier, min_size=n_measures + n_dims, max_size=n_measures + n_dims, unique=True
        )
    )
    m_names, d_names = field_names[:n_measures], field_names[n_measures:]

    aggs = draw(
        st.lists(
            st.sampled_from(["sum", "count", "avg", "min", "max"]),
            min_size=n_measures,
            max_size=n_measures,
        )
    )
    measures = [
        Measure(
            name=m,
            sql="*" if agg == "count" else f"{{{a}}}.{m}",
            agg=cast('Literal["sum", "count", "avg", "min", "max"]', agg),
            unit="number",
        )
        for m, agg in zip(m_names, aggs, strict=True)
    ]
    # First dim numeric (enables numeric filters/having); rest string.
    dimensions = [
        Dimension(name=d, sql=f"{{{a}}}.{d}", type=("number" if i == 0 else "string"))
        for i, d in enumerate(d_names)
    ]

    kw: dict[str, object] = {}
    if "base_predicate" in features:
        kw["base_predicate"] = f"{{{a}}}.deleted_at IS NULL"
    if "time_dim" in features:
        kw["time_dimensions"] = [TimeDimension(name="created_at", sql=f"{{{a}}}.created_at")]

    return Cube(
        name=cube_name,
        backend=backend,
        table=table,
        alias=a,
        measures=measures,
        dimensions=dimensions,
        **kw,  # type: ignore[arg-type]
    )


@st.composite
def random_catalog(draw: st.DrawFn, features: frozenset[str] = FEATURES) -> Catalog:
    """A tree-shaped catalog (one backend). Edges are FK-on-child
    (``{child}.p<i> = {parent}.id``); relationships bias to ``many_to_one``
    so most queries don't trip the fan-out guard."""
    backend = draw(st.sampled_from(_DIALECT_BACKENDS))
    n_cubes = draw(st.integers(1, 4)) if "joins" in features else 1
    names = draw(st.lists(identifier, min_size=n_cubes, max_size=n_cubes, unique=True))
    cubes: list[Cube] = []
    for i in range(n_cubes):
        cube = draw(random_cube(features, alias=_ALIASES[i], name=names[i], backend=backend))
        cubes.append(cube)

    if "joins" in features and n_cubes > 1:
        rebuilt: list[Cube] = [cubes[0]]
        for i in range(1, n_cubes):
            parent = draw(st.integers(0, i - 1))
            rel = draw(
                st.sampled_from(
                    ["many_to_one", "many_to_one", "many_to_one", "one_to_one", "one_to_many"]
                )
            )
            ca, pa = _ALIASES[i], _ALIASES[parent]
            join = Join(
                to=cubes[parent].name,
                relationship=cast('Literal["one_to_one", "one_to_many", "many_to_one"]', rel),
                on=f"{{{ca}}}.p{i} = {{{pa}}}.id",
            )
            rebuilt.append(cubes[i].model_copy(update={"joins": [join]}))
        cubes = rebuilt

    return Catalog(cubes)


# ---------------------------------------------------------------------------
# Query strategy (§1.3)
# ---------------------------------------------------------------------------


def _leaf_filter(
    string_dims: list[str], numeric_dims: list[str], features: frozenset[str]
) -> st.SearchStrategy[Filter]:
    options: list[st.SearchStrategy[Filter]] = []
    if string_dims:
        options.append(
            st.builds(
                Filter,
                dimension=st.sampled_from(string_dims),
                op=st.sampled_from(["eq", "neq", "contains"]),
                values=filter_value().map(lambda v: [v]),
            )
        )
        options.append(
            st.builds(
                Filter,
                dimension=st.sampled_from(string_dims),
                op=st.just("in"),
                values=st.lists(filter_value(), min_size=1, max_size=3),
            )
        )
        options.append(
            st.builds(
                Filter,
                dimension=st.sampled_from(string_dims),
                op=st.sampled_from(["is_null", "not_null"]),
                values=st.just([]),
            )
        )
    if numeric_dims and "numeric_filter" in features:
        options.append(
            st.builds(
                Filter,
                dimension=st.sampled_from(numeric_dims),
                op=st.sampled_from(["eq", "gt", "lt", "gte", "lte"]),
                values=_numeric_value.map(lambda v: [v]),
            )
        )
    return st.one_of(options) if options else st.nothing()


def _bool_expr(leaves: st.SearchStrategy[Filter]) -> st.SearchStrategy[object]:
    return st.recursive(
        leaves,
        lambda sub: st.one_of(
            st.builds(
                lambda c: BoolExpr(op="and", children=c), st.lists(sub, min_size=2, max_size=3)
            ),
            st.builds(
                lambda c: BoolExpr(op="or", children=c), st.lists(sub, min_size=2, max_size=3)
            ),
            st.builds(lambda c: BoolExpr(op="not", children=[c]), sub),
        ),
        max_leaves=6,
    )


@st.composite
def random_query(draw: st.DrawFn, catalog: Catalog, features: frozenset[str]) -> SemanticQuery:
    cubes = list(catalog.as_dict().values())
    cubes = [c for c in cubes if c.backend is not Backend.META]
    m_refs = [f"{c.name}.{m.name}" for c in cubes for m in c.measures]
    string_dims = [f"{c.name}.{d.name}" for c in cubes for d in c.dimensions if d.type == "string"]
    numeric_dims = [f"{c.name}.{d.name}" for c in cubes for d in c.dimensions if d.type == "number"]
    all_dims = string_dims + numeric_dims

    ungrouped = "ungrouped" in features and bool(all_dims)
    kw: dict[str, object] = {}

    if ungrouped:
        kw["dimensions"] = draw(
            st.lists(st.sampled_from(all_dims), min_size=1, max_size=3, unique=True)
        )
        kw["ungrouped"] = True
        kw["limit"] = draw(st.integers(min_value=1, max_value=1000))
    else:
        measures = draw(st.lists(st.sampled_from(m_refs), min_size=1, max_size=3, unique=True))
        kw["measures"] = measures
        if all_dims:
            kw["dimensions"] = draw(st.lists(st.sampled_from(all_dims), max_size=2, unique=True))
        if "having" in features:
            bare = [r.split(".", 1)[1] for r in measures]
            kw["having"] = [
                Filter(
                    dimension=draw(st.sampled_from(bare)), op="gt", values=[draw(_numeric_value)]
                )
            ]
        if "limit" in features:
            kw["limit"] = draw(st.integers(min_value=0, max_value=10_000))

    if "filters" in features:
        leaves = _leaf_filter(string_dims, numeric_dims, features)
        if leaves is not st.nothing():
            if "nested_filter" in features:
                expr = draw(_bool_expr(leaves))
                # ``where`` must be a BoolExpr; a shrunk recursion can yield a
                # bare Filter leaf — route that to ``filters`` instead.
                if isinstance(expr, BoolExpr):
                    kw["where"] = expr
                else:
                    kw["filters"] = [expr]
            else:
                kw["filters"] = draw(st.lists(leaves, max_size=3))

    if "time_dim" in features and any(c.time_dimensions for c in cubes):
        tc = next(c for c in cubes if c.time_dimensions)
        lo, hi = draw(temporal_edges), draw(temporal_edges)
        kw["time_dimension"] = TimeWindow(
            dimension=f"{tc.name}.{tc.time_dimensions[0].name}",
            range=(lo, hi),
            granularity=draw(st.sampled_from([None, "day", "month"])),
        )

    return SemanticQuery(**kw)  # type: ignore[arg-type]


@st.composite
def swarm(draw: st.DrawFn) -> tuple[frozenset[str], Catalog, SemanticQuery]:
    """The headline strategy: a feature set, a catalog, and a query drawn
    from it. Properties take ``(features, catalog, query)``."""
    features = draw(feature_set())
    catalog = draw(random_catalog(features))
    query = draw(random_query(catalog, features))
    return features, catalog, query


@st.composite
def catalog_and_query(draw: st.DrawFn) -> tuple[Catalog, SemanticQuery]:
    """Back-compat shim: ``(Catalog, SemanticQuery)`` drawn together."""
    _, catalog, query = draw(swarm())
    return catalog, query


# ---------------------------------------------------------------------------
# Negative generation: one mutation off valid (§1.5)
# ---------------------------------------------------------------------------


@st.composite
def broken_pair(draw: st.DrawFn) -> tuple[Catalog, SemanticQuery, str]:
    """A valid pair with exactly one breakage applied. Returns
    ``(catalog, broken_query, breakage_label)``."""
    catalog, q = draw(catalog_and_query())
    cubes = [c for c in catalog.as_dict().values() if c.backend is not Backend.META]
    a_cube = cubes[0]
    breakage = draw(st.sampled_from(["unknown_measure", "unknown_dimension", "filter_on_measure"]))

    if breakage == "unknown_measure":
        q = q.model_copy(
            update={"measures": [*q.measures, f"{a_cube.name}.{draw(identifier)}_zzz"]}
        )
    elif breakage == "unknown_dimension":
        q = q.model_copy(update={"dimensions": [*q.dimensions, f"{a_cube.name}.nope_zzz"]})
    else:  # filter_on_measure — point a filter at a measure ref
        m = a_cube.measures[0]
        q = q.model_copy(
            update={
                "filters": [
                    *q.filters,
                    Filter(dimension=f"{a_cube.name}.{m.name}", op="eq", values=["x"]),
                ]
            }
        )
    return catalog, q, breakage
