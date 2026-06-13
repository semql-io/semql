"""C3 (ktx-ports M1) — provenance on resolved output columns.

A projected column tells downstream consumers (MCP tools, presenter /
drilldown prompt roles) how much to trust its value: a pre-defined
measure is VERIFIED, an ad-hoc/derived expression is COMPOSED, a raw
column is a DIMENSION. Derived from the column's ``kind`` so it needs no
extra threading.
"""

from __future__ import annotations

import pytest
from semql.logical import ColumnRef
from semql.model import Cube, Dialect, Dimension, Measure, Provenance


def _cube() -> Cube:
    return Cube(
        name="t",
        backend=Dialect.POSTGRES,
        table="s.t",
        alias="t",
        measures=[Measure(name="m", sql="{t}.x", agg="sum", unit="count")],
        dimensions=[Dimension(name="d", sql="{t}.d", type="string")],
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("measure", Provenance.VERIFIED),
        ("computed", Provenance.COMPOSED),
        ("dimension", Provenance.DIMENSION),
        ("time", Provenance.DIMENSION),
    ],
)
def test_columnref_provenance(kind: str, expected: Provenance) -> None:
    ref = ColumnRef(cube=_cube(), field_name="x", alias="x", kind=kind, field=None)  # type: ignore[arg-type]
    assert ref.provenance is expected
