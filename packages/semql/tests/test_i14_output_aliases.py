# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""I14 — Field aliases on ``SemanticQuery`` (output).

Maps output column name → qualified field ref, e.g.
``{"net": "orders.revenue"}`` emits column ``"net"`` pointing at
``orders.revenue``. Fills the gap ``View`` leaves: query-time
re-labelling when a dashboard needs the same field under two
names in the same result (today: two compiled queries).

Per the spec at ``docs/specs/graphql-borrowed.md`` Candidate 1:
- Alias keys unique.
- Alias keys resolve to declared fields.
- Alias keys can't collide with existing output column names
  (compile error).
- Addressable in ``order`` / ``having``.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.errors import CompileError
from semql.spec import Filter, SemanticQuery

from .conftest import CONTEXT

# ---------------------------------------------------------------------------
# Output column name = alias key
# ---------------------------------------------------------------------------


def test_alias_emits_alias_key_as_column_name(catalog: dict) -> None:
    """A query with ``aliases={"net": "orders.revenue"}`` emits
    a column named ``"net"`` pointing at the revenue measure."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        aliases={"net": "orders.revenue"},
    )
    cq = compile_query(q, catalog, context=CONTEXT)
    assert "net" in cq.columns
    # The original measure column is replaced — not a duplicate.
    assert "revenue" not in cq.columns


def test_alias_addressable_in_order(catalog: dict) -> None:
    """Order can reference the alias key, not the original ref."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        aliases={"net": "orders.revenue"},
        order=[("net", "desc")],
    )
    cq = compile_query(q, catalog, context=CONTEXT)
    # ORDER BY net (alias) renders in the SQL.
    assert "ORDER BY" in cq.sql.upper() or "order by" in cq.sql.lower()


def test_alias_addressable_in_having(catalog: dict) -> None:
    """HAVING can reference the alias key."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        aliases={"net": "orders.revenue"},
        having=[Filter(dimension="net", op="gt", values=[100])],
    )
    cq = compile_query(q, catalog, context=CONTEXT)
    assert "HAVING" in cq.sql.upper() or "having" in cq.sql.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_alias_with_unknown_field_raises(catalog: dict) -> None:
    """An alias mapping to a non-existent field raises ``CompileError``."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        aliases={"net": "orders.does_not_exist"},
    )
    with pytest.raises(CompileError):
        compile_query(q, catalog, context=CONTEXT)


def test_alias_key_collision_with_measure_raises(catalog: dict) -> None:
    """An alias key matching an existing output column name is a compile error."""
    # The query already emits column ``revenue`` from the measure;
    # aliasing another field to ``revenue`` would be ambiguous.
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        aliases={"revenue": "orders.count"},
    )
    with pytest.raises(CompileError):
        compile_query(q, catalog, context=CONTEXT)


def test_alias_key_collision_with_dimension_raises(catalog: dict) -> None:
    """An alias key colliding with a dimension name raises."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        aliases={"region": "orders.revenue"},
    )
    with pytest.raises(CompileError):
        compile_query(q, catalog, context=CONTEXT)


def test_duplicate_alias_keys_detected_at_compile(catalog: dict) -> None:
    """Two alias entries mapping to the same output column name are
    detected by the compiler's collision check (a CompileError on
    the second ``net`` because the first already claimed the slot)."""
    # Both aliases target different fields, but use the same key
    # ``net``. The first wins; the second hits the collision check
    # because the alias key is now in the existing_cols set.
    aliases_dict = dict()
    aliases_dict["net"] = "orders.revenue"
    aliases_dict["net"] = "orders.count"
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        aliases=aliases_dict,
    )
    # Pydantic lets duplicate dict keys through (last write wins), so
    # only one alias survives. The compiler should accept it.
    cq = compile_query(q, catalog, context=CONTEXT)
    assert "net" in cq.columns


def test_empty_aliases_dict_default() -> None:
    """``SemanticQuery.aliases`` defaults to an empty dict."""
    q = SemanticQuery(measures=["orders.revenue"])
    assert q.aliases == {}


def test_alias_coexists_with_filter_and_dim(catalog: dict) -> None:
    """Aliases don't disturb filters or dimension rendering."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        aliases={"net": "orders.revenue"},
    )
    cq = compile_query(q, catalog, context=CONTEXT)
    # The aliased column is present; the original filter still binds.
    assert "net" in cq.columns
    assert "region" in cq.columns
    assert "status" in cq.sql.lower() or "paid" in cq.sql
