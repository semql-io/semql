"""Public compiler/checker semantics against independent executed counterexamples."""

from semql.contracts import ProofStatus

from demos.metric_contracts_demo import run_cases


def test_semantic_contracts_with_actual_execution_and_independent_oracles() -> None:
    cases = run_cases()
    for case in (
        cases.ratio_of_sums_vs_mean_of_ratios,
        cases.overlapping_distinct_reaggregation,
        cases.snapshots_across_time,
        cases.join_fanout,
        cases.created_vs_shipped_time_axis,
        cases.fixed_vs_changing_population,
    ):
        assert case.observed == case.expected, case.name
    assert cases.unsafe_fanout_rejected
    assert cases.duplicate_bridge_observed == (100, 200)
    assert cases.created_axis_check == ProofStatus.DISPROVED
    assert cases.shipped_axis_check == ProofStatus.PROVED
    assert cases.cohort_members == ("a", "b")
    assert cases.cohort_measurements == (("a", 20), ("b", None))
    assert cases.cohort_comparison_check == ProofStatus.NOT_ESTABLISHED
