"""Cohort envelopes preserve nested semantic-query profile restrictions."""

from __future__ import annotations

from typing import Any, cast

import pytest
from semql import CohortQuery, SchemaProfile, SemanticFeature
from semql_prompt import build_generation_contract


def _payload() -> dict[str, Any]:
    return {
        "cohort": {
            "source": {"dimensions": ["stores.id", "stores.region"]},
            "member_keys": ["stores.id"],
            "attributes": ["stores.region"],
        },
        "measurements": [
            {
                "name": "current",
                "query": {"dimensions": ["sales.store"], "measures": ["sales.revenue"]},
                "key_mapping": {"stores.id": "sales.store"},
            }
        ],
    }


@pytest.mark.parametrize("profile", list(SchemaProfile))
def test_cohort_generation_validates_required_nested_query_payloads(profile: SchemaProfile) -> None:
    contract = build_generation_contract({}, output_model=CohortQuery, profile=profile)
    payload = _payload()
    accepted = contract.validate_payload(payload)
    assert isinstance(accepted, CohortQuery)
    assert accepted.cohort.member_keys == ["stores.id"]
    assert accepted.measurements[0].key_mapping == {"stores.id": "sales.store"}
    assert contract.validate_payload(accepted.model_dump(exclude_unset=True)) == accepted
    with pytest.raises(ValueError):
        contract.validate_payload(
            {"cohort": payload["cohort"], "measurements": [{"name": "current"}]}
        )
    with pytest.raises(ValueError):
        contract.validate_payload({**payload, "steps": []})


@pytest.mark.parametrize("stage", ["source", "measurement"])
def test_cohort_strict_profile_rejects_nested_omitted_features(stage: str) -> None:
    payload = _payload()
    if stage == "source":
        cohort = cast(dict[str, Any], payload["cohort"])
        query = cast(dict[str, Any], cohort["source"])
    else:
        measurements = cast(list[dict[str, Any]], payload["measurements"])
        query = cast(dict[str, Any], measurements[0]["query"])
    query["where"] = {
        "op": "not",
        "children": [{"dimension": "stores.region", "op": "eq", "values": ["West"]}],
    }
    tool = build_generation_contract({}, output_model=CohortQuery, profile=SchemaProfile.TOOL)
    assert isinstance(tool.validate_payload(payload), CohortQuery)
    strict = build_generation_contract({}, output_model=CohortQuery, profile=SchemaProfile.STRICT)
    with pytest.raises(ValueError, match="where"):
        strict.validate_payload(payload)


def test_cohort_rank_disabled_profile_rejects_source_selection_rank() -> None:
    payload = _payload()
    cohort = cast(dict[str, Any], payload["cohort"])
    source = cast(dict[str, Any], cohort["source"])
    source["rank"] = {"order_by": [{"field": "stores.id", "direction": "asc"}], "count": 1}
    contract = build_generation_contract(
        {}, output_model=CohortQuery, disabled_features=[SemanticFeature.RANK]
    )
    with pytest.raises(ValueError, match="rank"):
        contract.validate_payload(payload)
