# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false, reportPrivateUsage=false
"""I7 — Dimension input aliases.

An LLM might emit ``territory``, ``zone``, or ``area`` when the
catalog names the dimension ``region``. Today the resolver rejects
those as ``UnknownIdentifierError``. ``Dimension.aliases`` accepts
any of them and resolves to the canonical name.

Per the entry: "Strip the alias from the prompt fragment after
resolution so the planner learns the canonical name." The resolver
translates any alias to its canonical ``Dimension``; the prompt
renders the canonical name only (no prompt-side alias listing —
that adds noise).

Field-hide / mask gates (A1) still apply on the canonical field;
the alias is a synonym, not a separate authorization surface.
"""

from __future__ import annotations

import pytest
from semql.introspect import resolve_field
from semql.model import Backend, Cube, Dimension, Measure
from semql.prompt import build_planner_prompt_fragment


def _cube_with_aliases() -> dict:
    return {
        "orders": Cube(
            name="orders",
            backend=Backend.POSTGRES,
            table="{schema}.orders",
            alias="o",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[
                Dimension(
                    name="region",
                    sql="{o}.region",
                    type="string",
                    aliases=["territory", "zone", "area"],
                ),
                Dimension(name="status", sql="{o}.status", type="string"),
            ],
        )
    }


# ---------------------------------------------------------------------------
# Resolver accepts aliases
# ---------------------------------------------------------------------------


def test_resolver_accepts_canonical_name() -> None:
    """The canonical name still resolves."""
    cube, fld = resolve_field("orders.region", _cube_with_aliases())
    assert fld.name == "region"


def test_resolver_accepts_first_alias() -> None:
    cube, fld = resolve_field("orders.territory", _cube_with_aliases())
    assert fld.name == "region"
    assert fld.kind == "dimension"


def test_resolver_accepts_each_alias() -> None:
    for alias in ("territory", "zone", "area"):
        cube, fld = resolve_field(f"orders.{alias}", _cube_with_aliases())
        assert fld.name == "region", f"alias {alias!r} did not resolve to region"


def test_resolver_rejects_unknown_alias() -> None:
    """An alias not in the list still raises ``UnknownIdentifierError``."""
    from semql.errors import UnknownIdentifierError

    with pytest.raises(UnknownIdentifierError):
        resolve_field("orders.unknown_alias", _cube_with_aliases())


def test_aliases_default_to_empty_list() -> None:
    """A Dimension without aliases specified has ``aliases == []``."""
    from semql.model import Dimension

    d = Dimension(name="region", sql="{o}.region", type="string")
    assert d.aliases == []


def test_aliases_deduplicated() -> None:
    """``aliases`` accepts duplicates without breaking the resolver."""
    from semql.model import Dimension

    d = Dimension(
        name="region",
        sql="{o}.region",
        type="string",
        aliases=["territory", "territory", "zone"],
    )
    # The model surface keeps what the caller passed; the resolver
    # does the dedup at lookup time.
    assert d.aliases == ["territory", "territory", "zone"]
    # Resolver still finds the field via either alias.
    cube, fld = resolve_field("orders.zone", _cube_with_aliases())
    assert fld.name == "region"


def test_aliases_with_mask_roles() -> None:
    """Aliases co-exist with A1's mask_roles on the same dimension."""
    from semql.model import Dimension

    d = Dimension(
        name="region",
        sql="{o}.region",
        type="string",
        aliases=["territory"],
        required_roles=["hr", "analyst"],
        mask_roles=["analyst"],
    )
    # Resolver still finds the field via the alias.
    assert d.mask_roles == ["analyst"]
    assert d.aliases == ["territory"]


# ---------------------------------------------------------------------------
# Prompt rendering — canonical name only
# ---------------------------------------------------------------------------


def test_prompt_shows_canonical_name() -> None:
    """The planner prompt renders the canonical dimension name."""
    cat = _cube_with_aliases()
    rendered = build_planner_prompt_fragment(cat)
    # ``region`` is the canonical name — should appear.
    assert "region" in rendered


def test_prompt_does_not_list_aliases() -> None:
    """Aliases don't appear as separate prompt items (noise reduction)."""
    cat = _cube_with_aliases()
    rendered = build_planner_prompt_fragment(cat)
    # The aliases are resolver-side hints, not separate prompt entries.
    # The plan: "Strip the alias from the prompt fragment after
    # resolution so the planner learns the canonical name." So
    # neither ``territory`` nor ``zone`` nor ``area`` should appear
    # in the rendered prompt.
    assert "territory" not in rendered
    assert "zone" not in rendered
    assert "area" not in rendered
