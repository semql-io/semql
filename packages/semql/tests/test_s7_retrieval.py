"""S7 slice 4 — pluggable retrieval (SQLite BM25, numpy cosine, hybrid RRF, MMR).

The lexical SQLiteBM25Retriever path runs without extras (stdlib FTS5).
The vector-side tests use a deterministic toy embedder so we don't take
a numpy dep just for testing — but the production retrievers need numpy
which we install through ``semql[retrieval]``.

Two layers of coverage:
- Mechanics (each retriever returns scored cubes in plausible order).
- Selection policy (``Catalog.with_retrieval`` picks the right default).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    GlossaryEntry,
    HybridRetriever,
    Measure,
    MMRWrapper,
    NumpyCosineRetriever,
    SQLiteBM25Retriever,
    build_default_retriever,
)


def _cube(name: str, **kwargs: Any) -> Cube:  # noqa: ANN401 — test factory
    defaults: dict[str, Any] = {
        "backend": Dialect.POSTGRES,
        "table": f"public.{name}",
        "alias": name[0],
        "measures": [Measure(name="cnt", sql="*", agg="count")],
        "dimensions": [Dimension(name="status", sql="{o}.status", type="string")],
    }
    defaults.update(kwargs)
    return Cube(name=name, **defaults)


@pytest.fixture
def cubes() -> list[Cube]:
    """A small but distinguishable catalog: each cube has a unique
    keyword surface so retrievers can find the right one."""
    return [
        _cube(
            name="orders",
            description="Customer orders facts.",
            questions=["How many orders shipped last week?", "Top customers by LTV"],
            keywords=["orders", "AOV", "transactions"],
        ),
        _cube(
            name="sessions",
            description="Web session events.",
            questions=["What's the bounce rate this week?"],
            keywords=["pageviews", "bounce", "sessions"],
        ),
        _cube(
            name="payments",
            description="Payment processing facts.",
            questions=["How many failed payments yesterday?"],
            keywords=["MRR", "payments", "billing"],
        ),
    ]


# ---------------------------------------------------------------------------
# SQLiteBM25Retriever
# ---------------------------------------------------------------------------


def test_bm25_finds_keyword_match(cubes: list[Cube]) -> None:
    r = SQLiteBM25Retriever.from_cubes(cubes)
    results = r.top_k("orders shipped", k=3)
    assert results
    assert results[0][0] == "orders"


def test_bm25_finds_acronym(cubes: list[Cube]) -> None:
    """Acronyms like AOV / MRR survive FTS5's tokenizer."""
    r = SQLiteBM25Retriever.from_cubes(cubes)
    results = r.top_k("MRR", k=3)
    assert results
    assert results[0][0] == "payments"


def test_bm25_returns_higher_is_better_scores(cubes: list[Cube]) -> None:
    r = SQLiteBM25Retriever.from_cubes(cubes)
    results = r.top_k("orders", k=3)
    scores = [s for _, s in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_query_returns_empty(cubes: list[Cube]) -> None:
    r = SQLiteBM25Retriever.from_cubes(cubes)
    assert r.top_k("", k=3) == []
    assert r.top_k("   ", k=3) == []


def test_bm25_no_match_returns_empty(cubes: list[Cube]) -> None:
    r = SQLiteBM25Retriever.from_cubes(cubes)
    assert r.top_k("xyzabc_nothing_matches", k=3) == []


def test_bm25_glossary_aliases_route_to_term(cubes: list[Cube]) -> None:
    glossary = [
        GlossaryEntry(
            term="payments",  # using existing cube name as term for routing test
            definition="Payment processing facts.",
            aliases=["charges", "transactions"],
        ),
    ]
    r = SQLiteBM25Retriever.from_cubes(cubes, glossary=glossary)
    results = r.top_k("charges", k=3)
    names = [n for n, _ in results]
    assert "payments" in names


def test_bm25_excludes_deprecated_cubes() -> None:
    """Deprecated cubes never get indexed — saves the planner from a
    follow-up CompileError."""
    cubes = [
        _cube(name="orders_v1", stability="deprecated", replacement="orders_v2"),
        _cube(name="orders_v2", description="Current orders cube.", keywords=["orders"]),
    ]
    r = SQLiteBM25Retriever.from_cubes(cubes)
    names = [n for n, _ in r.top_k("orders", k=5)]
    assert "orders_v1" not in names
    assert "orders_v2" in names


def test_bm25_respects_k_cap(cubes: list[Cube]) -> None:
    r = SQLiteBM25Retriever.from_cubes(cubes)
    assert len(r.top_k("orders", k=1)) <= 1
    assert len(r.top_k("orders sessions payments", k=2)) <= 2


# ---------------------------------------------------------------------------
# NumpyCosineRetriever — uses a deterministic toy embedder
# ---------------------------------------------------------------------------


class _ToyEmbedder:
    """Deterministic 8-dim embedder for tests. Maps each unique token
    to a fixed unit vector via hashing; doc/query vectors are the
    normalised sum of their token vectors.

    Good enough to verify mechanics — real semantic relevance isn't
    expected. ``model_id`` satisfies the EmbeddingProvider Protocol."""

    model_id = "toy/v1"

    def _vec(self, token: str) -> list[float]:
        import hashlib

        # 8-byte hash → 8 dims; map each byte into [-1, 1].
        h = hashlib.sha256(token.lower().encode()).digest()[:8]
        return [(b / 127.5) - 1.0 for b in h]

    def _doc_vec(self, text: str) -> list[float]:
        tokens = [t for t in text.split() if t]
        if not tokens:
            return [0.0] * 8
        acc = [0.0] * 8
        for t in tokens:
            v = self._vec(t)
            for i, x in enumerate(v):
                acc[i] += x
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return [x / norm for x in acc]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._doc_vec(t) for t in texts]

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed_sync(texts)


def test_cosine_retriever_returns_results(cubes: list[Cube]) -> None:
    r = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    results = r.top_k("orders shipped", k=3)
    assert len(results) <= 3
    assert all(isinstance(s, float) for _, s in results)


def test_cosine_scores_in_valid_range(cubes: list[Cube]) -> None:
    """Cosine on normalised vectors lives in [-1, 1]."""
    r = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    for _, score in r.top_k("orders", k=3):
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6


def test_cosine_handles_empty_cubes() -> None:
    r = NumpyCosineRetriever.from_cubes([], _ToyEmbedder())
    assert r.top_k("anything", k=5) == []


def test_cosine_excludes_deprecated_cubes() -> None:
    cubes = [
        _cube(name="orders_v1", stability="deprecated", replacement="orders_v2"),
        _cube(name="orders_v2", description="Current orders cube."),
    ]
    r = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    names = [n for n, _ in r.top_k("orders", k=5)]
    assert "orders_v1" not in names


# ---------------------------------------------------------------------------
# HybridRetriever — RRF over BM25 + cosine
# ---------------------------------------------------------------------------


def test_hybrid_combines_both_retrievers(cubes: list[Cube]) -> None:
    bm25 = SQLiteBM25Retriever.from_cubes(cubes)
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    hybrid = HybridRetriever(bm25=bm25, cosine=cosine)
    results = hybrid.top_k("orders", k=3)
    assert results
    assert results[0][1] > 0


def test_hybrid_includes_cubes_only_in_one_retriever() -> None:
    """If BM25 nails the acronym match but cosine finds something
    different, both should contribute to the merged top-k."""

    cubes = [
        _cube(name="aaa", description="aaa cube", keywords=["AOV"]),
        _cube(name="bbb", description="bbb cube"),
    ]
    bm25 = SQLiteBM25Retriever.from_cubes(cubes)
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    hybrid = HybridRetriever(bm25=bm25, cosine=cosine)
    results = hybrid.top_k("AOV", k=2)
    names = [n for n, _ in results]
    # BM25 should put aaa first (AOV is an indexed keyword); hybrid
    # preserves at least one of the two contributors.
    assert "aaa" in names


def test_hybrid_rrf_scoring_is_monotonic_in_ranks(cubes: list[Cube]) -> None:
    """Sanity: higher rank in either retriever ⇒ higher RRF score."""
    bm25 = SQLiteBM25Retriever.from_cubes(cubes)
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    hybrid = HybridRetriever(bm25=bm25, cosine=cosine, rrf_k=60)
    results = hybrid.top_k("orders", k=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# MMRWrapper — diversity reranker
# ---------------------------------------------------------------------------


def test_mmr_rejects_lambda_out_of_range(cubes: list[Cube]) -> None:
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    with pytest.raises(ValueError, match="lambda_"):
        MMRWrapper(inner=cosine, matrix_source=cosine, lambda_=1.5)
    with pytest.raises(ValueError, match="lambda_"):
        MMRWrapper(inner=cosine, matrix_source=cosine, lambda_=-0.1)


def test_mmr_returns_at_most_k_results(cubes: list[Cube]) -> None:
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    mmr = MMRWrapper(inner=cosine, matrix_source=cosine, lambda_=0.5)
    out = mmr.top_k("orders", k=2)
    assert len(out) <= 2


def test_mmr_passthrough_when_pool_smaller_than_k(cubes: list[Cube]) -> None:
    """If the inner retriever returns fewer than k candidates, MMR has
    nothing to do — passes through the result list intact."""
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    mmr = MMRWrapper(inner=cosine, matrix_source=cosine, lambda_=0.5)
    out = mmr.top_k("orders", k=20)
    # At most 3 cubes; MMR returns them all
    assert len(out) == 3


def test_mmr_with_lambda_zero_prefers_diversity(cubes: list[Cube]) -> None:
    """``lambda_=0`` is pure diversity — first pick is whatever, then
    each subsequent pick maximises distance from already-picked."""
    cosine = NumpyCosineRetriever.from_cubes(cubes, _ToyEmbedder())
    mmr = MMRWrapper(inner=cosine, matrix_source=cosine, lambda_=0.0)
    out = mmr.top_k("orders", k=3)
    names = [n for n, _ in out]
    # All three cubes are distinct, just confirm we got the full set
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# Selection policy — Catalog.with_retrieval + build_default_retriever
# ---------------------------------------------------------------------------


def test_with_retrieval_returns_bm25_without_embedder(cubes: list[Cube]) -> None:
    cat = Catalog(cubes)
    r = cat.with_retrieval()
    assert isinstance(r, SQLiteBM25Retriever)


def test_with_retrieval_returns_hybrid_with_embedder(cubes: list[Cube]) -> None:
    cat = Catalog(cubes)
    r = cat.with_retrieval(embedder=_ToyEmbedder())
    assert isinstance(r, HybridRetriever)


def test_with_retrieval_wraps_in_mmr_when_requested(cubes: list[Cube]) -> None:
    cat = Catalog(cubes)
    r = cat.with_retrieval(embedder=_ToyEmbedder(), mmr=True)
    assert isinstance(r, MMRWrapper)


def test_with_retrieval_mmr_without_embedder_warns_and_falls_back(
    cubes: list[Cube],
) -> None:
    cat = Catalog(cubes)
    with pytest.warns(UserWarning, match="mmr=True"):
        r = cat.with_retrieval(mmr=True)
    assert isinstance(r, SQLiteBM25Retriever)


def test_with_retrieval_skips_deprecated_cubes() -> None:
    cat = Catalog(
        [
            _cube(name="orders_v1", stability="deprecated", replacement="orders_v2"),
            _cube(name="orders_v2", description="Current cube.", keywords=["orders"]),
        ]
    )
    r = cat.with_retrieval()
    names = [n for n, _ in r.top_k("orders", k=5)]
    assert "orders_v1" not in names
    assert "orders_v2" in names


def test_with_retrieval_uses_glossary_for_aliases() -> None:
    cat = Catalog(
        [
            _cube(name="revenue_cube", description="Revenue metrics."),
        ],
        glossary=[
            GlossaryEntry(
                term="revenue_cube",
                definition="Revenue metrics cube.",
                aliases=["ARR", "annual recurring revenue"],
            ),
        ],
    )
    r = cat.with_retrieval()
    names = [n for n, _ in r.top_k("ARR", k=3)]
    assert "revenue_cube" in names


def test_build_default_retriever_picks_bm25_without_embedder(cubes: list[Cube]) -> None:
    r = build_default_retriever(cubes)
    assert isinstance(r, SQLiteBM25Retriever)


def test_build_default_retriever_picks_hybrid_with_embedder(cubes: list[Cube]) -> None:
    r = build_default_retriever(cubes, embedder=_ToyEmbedder())
    assert isinstance(r, HybridRetriever)
