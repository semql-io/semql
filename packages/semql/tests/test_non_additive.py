"""Tests for ``Measure.non_additive`` — declaration only; compiler
refusal lands with the rollup work in §5.

The flag signals "summing this measure across a coarser time grain
gives a wrong answer." The canonical example is
``count_distinct`` — daily distinct user counts can't be re-summed to
get the weekly distinct count.

For this MVP the flag is informational: it surfaces in the prompt
fragment so the planner LLM knows not to ask for naive rollups, and
the model carries it for downstream consumers. The compiler does NOT
yet refuse non-additive + time-grouped queries because the resulting
SQL is still correct *at the chosen grain* — what's wrong is
post-query summing, which the compiler doesn't perform.
"""

from __future__ import annotations

from semql import Catalog, Cube, Dialect, Dimension, Measure
from semql_prompt import planner_prompt


def test_measure_non_additive_defaults_false() -> None:
    m = Measure(name="x", sql="*", agg="count", unit="count")
    assert m.non_additive is False


def test_measure_accepts_non_additive_true() -> None:
    m = Measure(
        name="active_users",
        sql="{u}.id",
        agg="count_distinct",
        unit="count",
        non_additive=True,
    )
    assert m.non_additive is True


# ---------------------------------------------------------------------------
# Prompt fragment surfaces the flag so a planner knows.
# ---------------------------------------------------------------------------


def test_non_additive_flag_appears_in_prompt() -> None:
    cube = Cube(
        name="users",
        backend=Dialect.POSTGRES,
        table="users",
        alias="u",
        measures=[
            Measure(
                name="active",
                sql="{u}.id",
                agg="count_distinct",
                unit="count",
                non_additive=True,
            ),
            Measure(name="signups", sql="*", agg="count", unit="count"),
        ],
        dimensions=[Dimension(name="region", sql="{u}.region", type="string")],
    )
    rendered = planner_prompt(Catalog([cube]))
    # The non-additive measure gets a callout the planner can see.
    line = next(line for line in rendered.splitlines() if "users.active" in line)
    assert "non-additive" in line.lower()
    # The additive measure does NOT get the callout.
    signups_line = next(line for line in rendered.splitlines() if "users.signups" in line)
    assert "non-additive" not in signups_line.lower()


# ---------------------------------------------------------------------------
# Compiler is unchanged: non-additive measures still compile cleanly.
# The flag is declaration only for now; refusal lands with rollups.
# ---------------------------------------------------------------------------


def test_non_additive_measure_still_compiles_at_a_grain() -> None:
    cube = Cube(
        name="users",
        backend=Dialect.POSTGRES,
        table="users",
        alias="u",
        measures=[
            Measure(
                name="active",
                sql="{u}.id",
                agg="count_distinct",
                unit="count",
                non_additive=True,
            ),
        ],
        dimensions=[Dimension(name="region", sql="{u}.region", type="string")],
    )
    out = Catalog([cube]).compile(
        __import__("semql").SemanticQuery(measures=["users.active"], dimensions=["users.region"])
    )
    assert "COUNT(DISTINCT" in out.sql.upper()
