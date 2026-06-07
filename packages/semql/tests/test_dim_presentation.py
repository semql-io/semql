"""Tests for ``Dimension.unit`` and ``Dimension.format`` — presentation
hints that mirror their ``Measure`` siblings.

The compiler ignores both; this just pins the model shape and that
existing constructions (no presentation hints) remain valid.
"""

from __future__ import annotations

from semql import Dimension


def test_dimension_unit_defaults_to_none() -> None:
    d = Dimension(name="region", sql="{c}.region", type="string")
    assert d.unit is None


def test_dimension_format_defaults_to_none() -> None:
    d = Dimension(name="region", sql="{c}.region", type="string")
    assert d.format is None


def test_dimension_accepts_unit() -> None:
    d = Dimension(name="duration", sql="{c}.duration_s", type="number", unit="seconds")
    assert d.unit == "seconds"


def test_dimension_accepts_format() -> None:
    d = Dimension(name="duration", sql="{c}.duration_s", type="number", format="duration")
    assert d.format == "duration"


def test_dimension_format_rejects_unknown_literal() -> None:
    """``FormatLiteral`` is closed — pin so an arbitrary string can't
    sneak in as a format hint."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Dimension(name="x", sql="{c}.x", type="string", format="not-a-format")  # type: ignore[arg-type]
