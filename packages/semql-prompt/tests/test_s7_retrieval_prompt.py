"""S7 slice 7 — retrieval-mode prompt rendering.

Verifies that ``semql_prompt.planner_prompt`` / ``render_catalog_*`` narrow the
catalog to the top-k retrieved cubes when a ``user_query`` +
``retriever`` are supplied AND the catalog has more grounding content
than ``retrieval_threshold``. Below the threshold the full catalog
renders — retrieval would just artificially trim a short prompt.
"""

from __future__ import annotations

from typing import Any

from semql import (
    Backend,
    Catalog,
    Cube,
    Dimension,
    Measure,
    Retriever,
    SavedQuery,
    SemanticQuery,
    SQLiteBM25Retriever,
)
from semql_prompt import planner_prompt, planner_prompt_segments


def _cube(name: str, **kwargs: Any) -> Cube:  # noqa: ANN401 — test factory
    defaults: dict[str, Any] = {
        "backend": Backend.POSTGRES,
        "table": f"public.{name}",
        "alias": name[0],
        "measures": [Measure(name="cnt", sql="*", agg="count")],
        "dimensions": [Dimension(name="status", sql="{n}.status", type="string")],
    }
    defaults.update(kwargs)
    return Cube(name=name, **defaults)


class _StaticRetriever:
    """Deterministic retriever for tests — returns a fixed ranking
    regardless of query. Lets us assert on which cubes get spliced."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def top_k(self, user_query: str, k: int) -> list[tuple[str, float]]:
        _ = user_query  # unused, deterministic order
        return [(n, 1.0 - i * 0.1) for i, n in enumerate(self._names[:k])]


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


def test_below_threshold_renders_full_catalog() -> None:
    """Catalog with few questions stays full even when retriever is
    supplied — small prompts don't benefit from retrieval."""
    cubes = [
        _cube(name="orders", questions=["q1", "q2"]),
        _cube(name="sessions"),
        _cube(name="payments"),
    ]
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["orders"])
    p = planner_prompt(cat, user_query="anything", retriever=retriever, retrieval_threshold=50)
    assert "orders" in p
    assert "sessions" in p
    assert "payments" in p


def test_above_threshold_renders_only_top_k() -> None:
    """When question count > threshold, only top-k cubes splice in."""
    # Each cube has 20 questions; 3 cubes × 20 = 60 questions, above the
    # default threshold of 50.
    cubes = [
        _cube(name=name, questions=[f"q{i}" for i in range(20)])
        for name in ["orders", "sessions", "payments"]
    ]
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["payments"])  # only return payments
    p = planner_prompt(
        cat,
        user_query="how about payments?",
        retriever=retriever,
        top_k=1,
        retrieval_threshold=50,
    )
    assert "payments" in p
    # The other two should be filtered out
    assert "### orders " not in p
    assert "### sessions " not in p


def test_retrieval_inactive_without_user_query() -> None:
    cubes = [_cube(name="orders", questions=[f"q{i}" for i in range(60)])]
    cat = Catalog(cubes)
    retriever = _StaticRetriever([])  # would return nothing
    # No user_query → retrieval inactive → full catalog renders.
    p = planner_prompt(cat, retriever=retriever)
    assert "orders" in p


def test_retrieval_inactive_without_retriever() -> None:
    cubes = [_cube(name="orders", questions=[f"q{i}" for i in range(60)])]
    cat = Catalog(cubes)
    p = planner_prompt(cat, user_query="anything")
    assert "orders" in p


def test_retrieval_threshold_zero_means_always_on() -> None:
    """``retrieval_threshold=0`` opts in unconditionally — even tiny
    catalogs get filtered."""
    cubes = [
        _cube(name="orders"),
        _cube(name="sessions"),
    ]
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["sessions"])
    p = planner_prompt(
        cat,
        user_query="x",
        retriever=retriever,
        top_k=1,
        retrieval_threshold=0,
    )
    assert "sessions" in p
    assert "### orders " not in p


def test_saved_queries_count_toward_threshold() -> None:
    """Saved-query questions count toward the threshold per PRD."""
    cubes = [_cube(name="orders", questions=["q1", "q2"])]
    saved = [
        SavedQuery(
            name=f"sq{i}",
            query=SemanticQuery(measures=["orders.cnt"]),
            questions=[f"sq{i}_q{j}" for j in range(20)],
        )
        for i in range(3)  # 60 questions total just from saved queries
    ]
    cat = Catalog(cubes, saved_queries=saved)
    retriever = _StaticRetriever([])  # filter to nothing
    p = planner_prompt(
        cat,
        user_query="x",
        retriever=retriever,
        top_k=1,
        retrieval_threshold=50,
    )
    # Saved-query questions tipped us above threshold so retriever
    # filter applies — cube list narrows to nothing.
    assert "### orders " not in p


# ---------------------------------------------------------------------------
# Header annotation
# ---------------------------------------------------------------------------


def test_retrieval_mode_annotates_header() -> None:
    """When retrieval kicks in the catalog header surfaces a
    "top-k cubes for your question" annotation so the LLM knows it's
    looking at a filtered subset."""
    cubes = [
        _cube(name=name, questions=[f"q{i}" for i in range(20)])
        for name in ["orders", "sessions", "payments"]
    ]
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["orders"])
    p = planner_prompt(cat, user_query="x", retriever=retriever, top_k=2)
    assert "top 2 cubes" in p
    assert "Retrieval-filtered" in p


def test_below_threshold_no_retrieval_annotation() -> None:
    cubes = [_cube(name="orders")]
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["orders"])
    p = planner_prompt(cat, user_query="x", retriever=retriever)
    assert "top " not in p  # no "top N cubes" header annotation
    assert "Retrieval-filtered" not in p


# ---------------------------------------------------------------------------
# Auth interactions
# ---------------------------------------------------------------------------


def test_retrieval_cannot_promote_role_gated_cube() -> None:
    """Auth invariant: a retriever can only *narrow* the visible set
    in the static segment. A role-gated cube must not appear in the
    static segment regardless of how high it ranks."""
    cubes = [_cube(name=f"open_{i}", questions=[f"q{j}" for j in range(20)]) for i in range(3)]
    cubes.append(_cube(name="secret", required_roles=["admin"]))
    cat = Catalog(cubes)
    retriever = _StaticRetriever(["secret"])  # tries to surface it
    segments = planner_prompt_segments(
        cat,
        user_query="x",
        retriever=retriever,
        top_k=5,
        retrieval_threshold=0,
    )
    assert "secret" not in segments.static


# ---------------------------------------------------------------------------
# Integration with Catalog.with_retrieval
# ---------------------------------------------------------------------------


def test_catalog_with_retrieval_feeds_prompt() -> None:
    """End-to-end: build retriever from catalog, feed it back into
    prompt rendering, get a narrowed catalog."""
    cubes = [
        _cube(
            name="orders",
            description="Orders facts.",
            keywords=["AOV"],
            questions=[f"q{i}" for i in range(20)],
        ),
        _cube(
            name="sessions",
            description="Session events.",
            keywords=["bounce"],
            questions=[f"q{i}" for i in range(20)],
        ),
        _cube(
            name="payments",
            description="Payment facts.",
            keywords=["MRR"],
            questions=[f"q{i}" for i in range(20)],
        ),
    ]
    cat = Catalog(cubes)
    retriever: Retriever = cat.with_retrieval()  # SQLite BM25
    assert isinstance(retriever, SQLiteBM25Retriever)
    p = planner_prompt(cat, user_query="MRR", retriever=retriever, top_k=1)
    # BM25 should put payments first
    assert "payments" in p
    assert "### orders " not in p
    assert "### sessions " not in p
