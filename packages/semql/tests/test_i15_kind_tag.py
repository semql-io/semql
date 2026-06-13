# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false, reportPrivateUsage=false
"""I15 — ``__typename``-style kind tag on resolved fields.

The GraphQL spec at ``docs/specs/graphql-borrowed.md`` Candidate 2
calls for a ``kind: Literal["measure", "dimension", "time",
"segment"]`` accessor on the field returned by
``introspect.resolve_field``. Today, callers (``semql-mcp`` tool
factory, planner logic) ``isinstance``-check on the returned union
to build tool names and to branch on field type — the kind tag
gives the same information structurally, so consumers don't need to
import ``Measure`` / ``Dimension`` / ``TimeDimension`` to read it.

Implemented as a property on ``BaseField`` so the call site reads
``fld.kind`` regardless of which subclass the resolver returned.
"""

from __future__ import annotations

from semql.introspect import resolve_field
from semql.model import Cube, Dimension, Measure, Segment, TimeDimension


def test_resolved_measure_carries_kind_measure(catalog: dict) -> None:
    """Resolving a measure returns a Measure whose ``kind`` is 'measure'."""
    cube, fld = resolve_field("orders.revenue", catalog)
    assert isinstance(fld, Measure)
    assert fld.kind == "measure"


def test_resolved_dimension_carries_kind_dimension(catalog: dict) -> None:
    """Resolving a dimension returns a Dimension whose ``kind`` is 'dimension'."""
    cube, fld = resolve_field("orders.region", catalog)
    assert isinstance(fld, Dimension)
    assert fld.kind == "dimension"


def test_resolved_time_dimension_carries_kind_time(catalog: dict) -> None:
    """Resolving a time dimension returns a TimeDimension whose ``kind`` is 'time'."""
    cube, fld = resolve_field("orders.created_at", catalog)
    assert isinstance(fld, TimeDimension)
    assert fld.kind == "time"


def test_resolved_segment_carries_kind_segment() -> None:
    """Resolving a Segment directly returns ``kind == 'segment'``."""
    # Build a minimal in-memory catalog with a Segment.
    from semql.model import Dialect

    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
    )
    cat = {"orders": cube}
    _, fld = resolve_field("orders.paid", cat)
    assert isinstance(fld, Segment)
    assert fld.kind == "segment"


def test_kind_is_readable_without_subclass_import(catalog: dict) -> None:
    """A caller can branch on ``kind`` without importing ``Measure`` etc."""
    # The point of the spec: callers can read ``kind`` off the returned
    # field without knowing which subclass it is. This is the structural
    # property GraphQL ``__typename`` provides.
    _, fld = resolve_field("orders.count", catalog)
    # No isinstance check — branch purely on the kind tag.
    # The narrow is documented in the kind string itself; mypy can't
    # see that, hence the cast (the test still demonstrates the
    # user-facing API: a consumer dispatches on ``.kind``).
    from typing import cast

    from semql.model import Measure

    agg = cast(Measure, fld).agg if fld.kind == "measure" else None
    assert agg == "count"


def test_kind_preserved_through_model_copy(catalog: dict) -> None:
    """``model_copy`` keeps the kind tag (property reads from class)."""
    _, fld = resolve_field("orders.region", catalog)
    copied = fld.model_copy(update={"display_name": "Region (copied)"})
    assert copied.kind == "dimension"
    assert copied.display_name == "Region (copied)"


def test_mcp_factory_can_branch_on_kind(catalog: dict) -> None:
    """The motivating use case from ``docs/specs/graphql-borrowed.md``.

    The MCP per-cube tool factory (``semql-mcp/server.py``) currently
    does ``isinstance`` checks to build tool names. With ``kind``, it
    can branch structurally — no need to import the model subclasses.
    """
    _, measure = resolve_field("orders.revenue", catalog)
    _, dimension = resolve_field("orders.region", catalog)
    _, td = resolve_field("orders.created_at", catalog)

    kinds = {measure.kind, dimension.kind, td.kind}
    assert kinds == {"measure", "dimension", "time"}
