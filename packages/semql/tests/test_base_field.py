"""Tests for the shared ``BaseField`` supertype.

``Measure``, ``Dimension``, ``TimeDimension``, and ``Segment`` all
carry the same five identity / presentation fields (``name``, ``sql``,
``description``, ``display_name``, ``metadata``). ``BaseField``
declares them once; the subclasses add their type-specific fields
(``agg`` / ``type`` / ``granularities`` / ...).

The refactor is structural — every existing test in the suite remains
the behavioural regression guard. These tests pin only the *base
class relationship* and the contract that the inherited fields work
identically across subclasses.
"""

from __future__ import annotations

import pytest
from semql import BaseField, Dimension, Measure, Segment, TimeDimension


@pytest.mark.parametrize(
    "subclass",
    [Measure, Dimension, TimeDimension, Segment],
)
def test_field_type_is_basefield_subclass(subclass: type) -> None:
    assert issubclass(subclass, BaseField)


def test_basefield_carries_shared_identity_fields() -> None:
    """All five fields declared on the base must be reachable from every
    subclass at the model-field level (Pydantic FieldInfo lookup)."""
    shared = {"name", "sql", "description", "display_name", "metadata"}
    for cls in (Measure, Dimension, TimeDimension, Segment):
        assert shared.issubset(cls.model_fields.keys()), (
            f"{cls.__name__} is missing one of the shared fields: "
            f"{shared - set(cls.model_fields.keys())}"
        )


def test_basefield_subclass_instances_remain_frozen() -> None:
    """Frozen config inherits from the base — no subclass should be
    accidentally mutable after the refactor."""
    m = Measure(name="count", sql="*", agg="count")
    with pytest.raises(Exception):  # noqa: B017, BLE001 — Pydantic raises ValidationError.
        m.name = "renamed"


def test_basefield_metadata_default_factory_is_independent_per_instance() -> None:
    """``default_factory=dict`` must produce a fresh dict per instance —
    not a shared singleton. Regression guard for the easy refactor bug
    where ``metadata: Metadata = {}`` would alias across all instances."""
    a = Dimension(name="x", sql="{c}.x", type="string")
    b = Dimension(name="y", sql="{c}.y", type="string")
    assert a.metadata is not b.metadata


def test_subclasses_keep_their_type_specific_fields() -> None:
    """Subclass-specific fields survive the refactor (smoke test).

    The full surface is covered by the existing test suite; this just
    asserts the obvious specialisation per subclass."""
    assert "agg" in Measure.model_fields
    assert "type" in Dimension.model_fields
    assert "granularities" in TimeDimension.model_fields
    # Segment is structurally the shared fields only — no extras.
    assert set(Segment.model_fields.keys()) == {
        "name",
        "sql",
        "description",
        "display_name",
        "metadata",
    }


def test_isinstance_narrowing_still_works() -> None:
    """The compiler dispatches via ``isinstance(fld, Measure)`` etc. —
    the shared base must NOT make subclasses indistinguishable."""
    m = Measure(name="count", sql="*", agg="count")
    d = Dimension(name="region", sql="{c}.region", type="string")
    assert isinstance(m, Measure) and not isinstance(m, Dimension)
    assert isinstance(d, Dimension) and not isinstance(d, Measure)
    # Both satisfy the base type.
    assert isinstance(m, BaseField)
    assert isinstance(d, BaseField)
