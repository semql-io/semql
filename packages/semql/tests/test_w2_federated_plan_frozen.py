"""W2 stage 7 — ``FederatedPlan`` is frozen and version-stamped.

Review B1: every other node in the federation IR (``MergePlan``,
``MergeSpec``, ``BridgeJoin``, …) is a frozen dataclass, but
``FederatedPlan`` itself was a plain mutable ``@dataclass`` — a caller
could reassign ``plan.merge`` after compilation and silently desync the
fragments from the merge step.  The ``MergeSpec`` already carries a
``# travels across package versions`` caveat in the executor; a format
version on the plan lets a consumer detect a compiler / executor skew
instead of mis-reading a changed shape.

These tests pin: the plan rejects attribute assignment, and it carries
the current format version.
"""

from __future__ import annotations

import dataclasses

import pytest
from semql import (
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.federate import FEDERATED_PLAN_VERSION, FederatedPlan


def _single_backend_plan() -> FederatedPlan:
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="orders",
            backend=Dialect.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    return compile_federated_query(q, catalog)


def test_federated_plan_is_frozen() -> None:
    plan = _single_backend_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.merge = dataclasses.replace(plan.merge, sql="tampered")  # type: ignore[misc]


def test_federated_plan_carries_format_version() -> None:
    plan = _single_backend_plan()
    assert plan.version == FEDERATED_PLAN_VERSION
    assert isinstance(FEDERATED_PLAN_VERSION, int)


def test_federated_plan_replace_round_trips_version() -> None:
    """``dataclasses.replace`` (the supported way to derive a tweaked
    plan) preserves the version stamp."""
    plan = _single_backend_plan()
    tweaked = dataclasses.replace(plan, merge=dataclasses.replace(plan.merge, sql="SELECT 1"))
    assert tweaked.version == FEDERATED_PLAN_VERSION
