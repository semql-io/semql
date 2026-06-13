# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false, reportPrivateUsage=false
"""LOGb-i — Wire the LogicalPlan into the compile pipeline.

The IR + ``to_logical_plan`` + ``apply_rollup_to_plan`` + ``partition_scans``
+ ``Catalog.explain()`` + plan-snapshot tests all shipped on
``feat/logical-plan-ir``. The plan is *built* in ``_CompileEnv`` but the
emission helpers still reach past it to read the spec tree directly —
``q.time_dimension.granularity`` is read in three places
(``_projection_stage`` / ``_group_by_stage`` / ``_CompileEnv.__init__``)
when ``self.plan.time_window`` carries the same value.

This is the smallest mechanical refactor that makes ``self.plan`` the
single source of truth for time-window / granularity in emission. The
plan is already populated with the right values; we just route reads
through it.

The pass-2 work (routing project / aggregate / order through the plan)
is deferred. This is pass-1: prove the plan is the source of truth for
at least one emission site, and pin that with tests.
"""

from __future__ import annotations

from semql.compile import CompiledQuery, compile_query
from semql.model import Dialect
from semql.spec import SemanticQuery, TimeWindow

from .conftest import CONTEXT


def _compile(catalog: dict, q: SemanticQuery) -> CompiledQuery:
    return compile_query(q, catalog, context=CONTEXT)


def test_plan_is_built_for_time_aggregation(catalog: dict) -> None:
    """``compile_query`` builds a LogicalPlan that mirrors the query."""
    # The IR contract: ``plan.time_window`` equals ``q.time_dimension``
    # (same value, same field). We verify by snapshotting the structure
    # that the plan carries granularity the spec-tree used to own.
    from semql.logical import to_logical_plan

    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-03-31"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    assert plan.time_window is not None
    assert plan.time_window.granularity == "month"
    assert plan.time_window.range == ("2026-01-01", "2026-03-31")


def test_time_aggregation_emits_truncated_bucket(catalog: dict) -> None:
    """Smoke test — granularity actually drives the truncation in SQL.

    Before LOGb-i: emission reads ``q.time_dimension.granularity``
    directly. After LOGb-i: it reads ``self.plan.time_window.granularity``.
    The SQL must be byte-identical — the data structure is the same,
    only the read site moved.
    """
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-03-31"),
        ),
    )
    cq = _compile(catalog, q)
    # The bucketed column is the truncation of orders.created_at; the
    # exact dialect render is snapshotted in test_compile.py.
    assert cq.backend is Dialect.POSTGRES
    assert "revenue" in cq.columns
    assert "created_at_month" in cq.columns


def test_time_window_granularity_via_plan_only(catalog: dict) -> None:
    """The plan and the query carry the same granularity — they're in sync.

    This is the structural property LOGb-i enforces: the plan is the
    single source of truth for emission. If they ever drift, the
    inconsistency is a bug.
    """
    from semql.logical import to_logical_plan

    for granularity in ("day", "week", "month"):
        q = SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity=granularity,
                range=("2026-01-01", "2026-01-31"),
            ),
        )
        plan = to_logical_plan(q, catalog)
        assert plan.time_window is not None
        assert plan.time_window.granularity == granularity

        # And the emitted SQL must mention the bucketed column name.
        cq = _compile(catalog, q)
        expected_col = f"created_at_{granularity}"
        assert expected_col in cq.columns


def test_no_time_dimension_plan_window_is_none(catalog: dict) -> None:
    """A non-time query has ``plan.time_window is None``."""
    from semql.logical import to_logical_plan

    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(q, catalog)
    assert plan.time_window is None


def test_emission_reads_time_window_from_plan(catalog: dict) -> None:
    """The plan is the source of truth for time-window data in emission.

    LOGb-i: ``_CompileEnv`` builds the plan *before* the time-block,
    and the projection / group_by / compare / spine stages all read
    from ``self.plan.time_window`` rather than ``self.q.time_dimension``.
    The latter is now only read in pre-flight invariant checks
    (compare-requires-time, lonely-offset-without-limit, etc.) that
    fire *before* the plan is built.
    """
    import inspect

    from semql.compile import _CompileEnv

    src = inspect.getsource(_CompileEnv)
    # The time-block in __init__ reads from the plan.
    assert "self.plan.time_window" in src
    # The post-init emission helpers (projection, group_by, compare,
    # simple_query) also use the plan, not the spec.
    forbidden = [
        "q.time_dimension.granularity",
        "q.time_dimension.range",
        "q.time_dimension.fill_nulls_with",
    ]
    for needle in forbidden:
        assert needle not in src, f"emission should read from self.plan.time_window, not {needle!r}"
