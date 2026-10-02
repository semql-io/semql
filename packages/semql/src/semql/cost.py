"""Referenced-cube size-hint accounting and budget enforcement.

``estimate_cost`` sums declared ``Cube.size_hint`` values for resolvable
query references and keeps the identities of referenced cubes without a hint.
This is a lower-bound guardrail, not a whole-plan scan estimate: it does not
model selectivity, join multiplication, or runtime behavior. A configured row
ceiling always applies to the known subtotal; unknown portions are rejected by
default and can be admitted explicitly with ``unknown_cost_policy="allow"``.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semql._resolve import walk_query_fields
from semql.errors import CompileError
from semql.model import Cube, View
from semql.spec import SemanticQuery

UnknownCostPolicy = Literal["reject", "allow"]


def _referenced_cubes(
    query: SemanticQuery,
    catalog: dict[str, Cube],
    views: dict[str, View] | None,
) -> set[str]:
    """Return resolvable cube dependencies using the compiler's field resolver."""
    resolved, _ = walk_query_fields(query, catalog, views_map=views)
    touched = {cube.name for cube in resolved.touched}
    touched.update(name for name in query.left_joins if name in catalog)
    for semi_join in query.semi_joins:
        touched.update(_referenced_cubes(semi_join.source, catalog, views))
    return touched


class CostEstimate(BaseModel):
    """Referenced-cube size-hint subtotal and any explicitly unknown cubes."""

    model_config = ConfigDict(frozen=True)

    total_rows_scanned: int = Field(
        default=0,
        ge=0,
        description="Sum of declared size_hint values for referenced cubes with a hint.",
    )
    cubes_estimated: dict[str, int] = Field(
        default_factory=lambda: dict[str, int](),
        description="Per-referenced-cube size_hint values.",
    )
    cubes_unknown: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of referenced cubes without a size_hint.",
    )
    rows_scanned_unknown: bool = Field(
        default=False,
        description="True if any referenced cube had size_hint=None.",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_unknown_identities(cls, value: object) -> object:
        """Keep the summary flag and distinct known/unknown identities consistent."""
        if not isinstance(value, dict):
            return value
        # Before-validation values are untyped; Pydantic validates the field types next.
        data = cast(dict[str, Any], value)
        known = data.get("cubes_estimated", {})
        raw_unknown = data.get("cubes_unknown", ())
        if not isinstance(known, dict) or not isinstance(raw_unknown, (list, tuple)):
            return data
        names = cast(list[object] | tuple[object, ...], raw_unknown)
        if not all(isinstance(name, str) for name in names):
            return data
        unknown = cast(tuple[str, ...], tuple(names))
        if len(unknown) != len(set(unknown)):
            raise ValueError("cubes_unknown must contain distinct cube names")
        if set(cast(dict[str, object], known)).intersection(unknown):
            raise ValueError("a cube cannot be both estimated and unknown")
        if data.get("rows_scanned_unknown", False) and not unknown:
            raise ValueError("rows_scanned_unknown requires unknown cube identities")
        return {**data, "cubes_unknown": unknown, "rows_scanned_unknown": bool(unknown)}


class BudgetExceededError(CompileError):
    """Raised by ``QueryBudget.check`` when the estimate exceeds the ceiling.

    Subclass of :class:`CompileError` so existing error-handling
    paths (e.g. agents that catch CompileError) keep working."""


class QueryBudget(BaseModel):
    """A guardrail over referenced-cube size hints.

    Row limits always apply to the known subtotal. By default, any unknown
    referenced cube rejects admission when a row ceiling is configured;
    ``unknown_cost_policy="allow"`` explicitly admits unknown portions only.
    """

    model_config = ConfigDict(frozen=True)

    max_rows_scanned: int | None = Field(
        default=None,
        ge=0,
        description="Maximum allowed known size_hint subtotal. None means no cap.",
    )
    max_cubes: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of referenced cubes, including unknown-size cubes.",
    )
    unknown_cost_policy: UnknownCostPolicy = Field(
        default="reject",
        description="Whether unknown-size referenced cubes are admitted when a row cap is set.",
    )

    def check(self, estimate: CostEstimate) -> None:
        """Raise ``BudgetExceededError`` if the estimate exceeds the budget."""
        if (
            self.max_rows_scanned is not None
            and estimate.total_rows_scanned > self.max_rows_scanned
        ):
            raise BudgetExceededError(
                f"Known referenced-cube size_hint subtotal is {estimate.total_rows_scanned} rows; "
                f"exceeds budget of {self.max_rows_scanned}."
            )
        if (
            self.max_rows_scanned is not None
            and (estimate.rows_scanned_unknown or bool(estimate.cubes_unknown))
            and self.unknown_cost_policy == "reject"
        ):
            raise BudgetExceededError(
                "Query references cubes with unknown size_hint values "
                f"({', '.join(estimate.cubes_unknown)}); unknown-cost policy is 'reject'."
            )
        if self.max_cubes is not None:
            touched = len(estimate.cubes_estimated) + len(estimate.cubes_unknown)
            if touched > self.max_cubes:
                raise BudgetExceededError(
                    f"Query references {touched} cube(s); exceeds budget of {self.max_cubes}."
                )


def estimate_cost(
    query: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    views: dict[str, View] | None = None,
) -> CostEstimate:
    """Compute a referenced-cube size-hint subtotal, not a whole-plan scan estimate.

    Uses the compiler's field resolver for projections, filters, where leaves,
    views, and derived operands/dependencies. The result does not model
    selectivity, join multiplication, or runtime scan behavior.
    """
    touched = _referenced_cubes(query, catalog, views)
    cubes_estimated: dict[str, int] = {}
    cubes_unknown: list[str] = []
    for cube_name in sorted(touched):
        cube = catalog.get(cube_name)
        if cube is None:
            continue
        if cube.size_hint is None:
            cubes_unknown.append(cube_name)
        else:
            cubes_estimated[cube_name] = cube.size_hint

    return CostEstimate(
        total_rows_scanned=sum(cubes_estimated.values()),
        cubes_estimated=cubes_estimated,
        cubes_unknown=tuple(cubes_unknown),
        rows_scanned_unknown=bool(cubes_unknown),
    )


__all__ = [
    "BudgetExceededError",
    "CostEstimate",
    "QueryBudget",
    "UnknownCostPolicy",
    "estimate_cost",
]
