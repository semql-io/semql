"""Hypothesis strategies with mimesis-powered realistic identifiers.

Builds a seeded name pool once at import time so hypothesis can shrink
properly (``st.sampled_from`` is shrinkable; raw mimesis calls are not).
"""

from __future__ import annotations

from hypothesis import strategies as st
from mimesis import Locale, Text
from semql import Backend, Catalog, Cube, Dimension, Measure, SemanticQuery
from sqlglot.dialects.postgres import Postgres

# ---------------------------------------------------------------------------
# Name pool
# ---------------------------------------------------------------------------

_ALIASES = list("abcdefghijklmnopqrstuvwxyz")

# Postgres reserved words that sqlglot won't quote in identifier position —
# using any of these as a table/field name would produce unparseable SQL.
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


# ---------------------------------------------------------------------------
# Catalog strategies
# ---------------------------------------------------------------------------


@st.composite
def random_cube(draw: st.DrawFn) -> Cube:
    """A single-backend Postgres cube with realistic names.

    SQL fragments use ``{alias}.{field_name}`` so they always parse;
    no joins or time-dimensions to keep the strategy self-contained.
    """
    alias = draw(st.sampled_from(_ALIASES))
    cube_name = draw(identifier)
    table = draw(identifier)

    n_measures = draw(st.integers(min_value=1, max_value=3))
    n_dims = draw(st.integers(min_value=0, max_value=4))
    total = n_measures + n_dims

    # Unique field names across both measures and dims within the cube.
    all_names = draw(st.lists(identifier, min_size=total, max_size=total, unique=True))
    m_names, d_names = all_names[:n_measures], all_names[n_measures:]

    measures = [Measure(name=m, sql=f"{{{alias}}}.{m}", agg="sum", unit="number") for m in m_names]
    dimensions = [Dimension(name=d, sql=f"{{{alias}}}.{d}", type="string") for d in d_names]

    return Cube(
        name=cube_name,
        backend=Backend.POSTGRES,
        table=table,
        alias=alias,
        measures=measures,
        dimensions=dimensions,
    )


@st.composite
def catalog_and_query(draw: st.DrawFn) -> tuple[Catalog, SemanticQuery]:
    """A ``(Catalog, SemanticQuery)`` pair guaranteed to be valid together.

    Single-cube catalogs only — no cross-backend or join complexity.
    """
    cube = draw(random_cube())
    catalog = Catalog([cube])

    m_refs = [f"{cube.name}.{m.name}" for m in cube.measures]
    d_refs = [f"{cube.name}.{d.name}" for d in cube.dimensions]

    measures = draw(
        st.lists(st.sampled_from(m_refs), min_size=1, max_size=len(m_refs), unique=True)
    )
    dims: list[str] = []
    if d_refs:
        dims = draw(
            st.lists(
                st.sampled_from(d_refs),
                max_size=min(2, len(d_refs)),
                unique=True,
            )
        )

    return catalog, SemanticQuery(measures=measures, dimensions=dims)
