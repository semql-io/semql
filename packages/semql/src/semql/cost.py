"""I1: query cost estimation + budget enforcement.

The compiler emits SQL but the planner has no idea whether a query
costs 100ms or 30 minutes. ``estimate_cost`` derives a rough
``CostEstimate`` from cube ``size_hint`` (declared on ``Cube``) +
the query's touched cubes. ``QueryBudget`` pairs with it: pre-
compile, the caller attaches a ceiling; the budget's ``.check()``
raises if the estimate exceeds the ceiling.

The estimate is intentionally rough — it's a guardrail, not a
planner. We use ``rows_scanned = cube.size_hint`` (a single-table
scan cost) and report the sum across touched cubes. A real planner
would model selectivity, joins, indexes, network, etc.; that's out
of scope. The point is to catch the "you forgot a filter on a
billion-row table" mistake at compile time, not to predict runtime
to the millisecond.

When a cube's ``size_hint`` is ``None`` the estimate is *unknown*
rather than silently zero. The budget's ``.check()`` treats unknown
as a free pass (we can't enforce what we can't estimate), but the
caller can still see the gap by reading ``CostEstimate.rows_scanned_unknown``.

The result is a frozen Pydantic value type (cf. I9) so it survives
JSON round-trips and can be logged alongside a deploy.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from semql.errors import CompileError
from semql.model import Cube
from semql.spec import SemanticQuery


class CostEstimate(BaseModel):
    """The result of a cost estimation.

    ``rows_scanned_unknown`` is True if any touched cube lacked a
    ``size_hint``. ``cubes_estimated`` is the per-cube row counts
    (only includes cubes with a size_hint; the unknown-cube names
    are not in this dict but contribute to ``rows_scanned_unknown``).
    """

    model_config = ConfigDict(frozen=True)

    total_rows_scanned: int = Field(
        default=0,
        ge=0,
        description="Sum of size_hint across all touched cubes with a hint set.",
    )
    cubes_estimated: dict[str, int] = Field(
        default_factory=lambda: dict[str, int](),
        description="Per-cube rows scanned; only includes cubes with size_hint set.",
    )
    rows_scanned_unknown: bool = Field(
        default=False,
        description="True if any touched cube had size_hint=None.",
    )


class BudgetExceededError(CompileError):
    """Raised by ``QueryBudget.check`` when the estimate exceeds the ceiling.

    Subclass of :class:`CompileError` so existing error-handling
    paths (e.g. agents that catch CompileError) keep working."""


class QueryBudget(BaseModel):
    """A guardrail that rejects cost estimates exceeding its ceilings.

    Both ceilings are independent and optional. A budget with only
    ``max_cubes`` set will allow arbitrarily large row counts; a
    budget with only ``max_rows_scanned`` set will allow arbitrarily
    many cubes. Both set means both must pass.

    An unknown estimate (``rows_scanned_unknown=True``) bypasses
    the rows-scanned check: we'd rather let a query run and observe
    the cost than reject a query we can't evaluate. (The caller can
    still see the gap in the estimate object.)
    """

    model_config = ConfigDict(frozen=True)

    max_rows_scanned: int | None = Field(
        default=None,
        ge=0,
        description="Maximum allowed sum of rows scanned. None means no cap.",
    )
    max_cubes: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of cubes touched. None means no cap.",
    )

    def check(self, estimate: CostEstimate) -> None:
        """Raise ``BudgetExceededError`` if the estimate exceeds the budget.

        The check is *non-destructive* — it doesn't change the
        estimate or the budget, just reads. Callers can call
        ``check`` as a guard at the top of a compile pipeline."""
        if (
            self.max_rows_scanned is not None
            and not estimate.rows_scanned_unknown
            and estimate.total_rows_scanned > self.max_rows_scanned
        ):
            raise BudgetExceededError(
                f"Query would scan {estimate.total_rows_scanned} rows; "
                f"exceeds budget of {self.max_rows_scanned}."
            )
        if self.max_cubes is not None:
            touched = len(estimate.cubes_estimated) + (1 if estimate.rows_scanned_unknown else 0)
            # Count unknown as one touched cube; conservative.
            if touched > self.max_cubes:
                raise BudgetExceededError(
                    f"Query touches {touched} cube(s); exceeds budget of {self.max_cubes}."
                )


def estimate_cost(query: SemanticQuery, catalog: dict[str, Cube]) -> CostEstimate:
    """Compute a rough cost estimate for ``query`` against ``catalog``.

    Walks the query's referenced cubes (one per measure, dimension,
    time_dimension, segment, or filter) and sums their ``size_hint``.
    Cubes with ``size_hint=None`` contribute 0 to the total but flag
    the estimate as unknown.

    The estimate is *deliberately rough* — single-table scan cost
    only, no selectivity, no join multiplier. A real planner would
    model those. The point is to surface "you forgot a filter on a
    billion-row table" at compile time.
    """
    if not query.measures and not query.dimensions and query.time_dimension is None:
        return CostEstimate()

    touched: set[str] = set()
    for ref in list(query.measures) + list(query.dimensions):
        if "." in ref:
            touched.add(ref.split(".", 1)[0])
    if query.time_dimension is not None and "." in query.time_dimension.dimension:
        touched.add(query.time_dimension.dimension.split(".", 1)[0])
    for seg in query.segments:
        if "." in seg:
            touched.add(seg.split(".", 1)[0])
    for f in query.filters:
        if "." in f.dimension:
            touched.add(f.dimension.split(".", 1)[0])

    cubes_estimated: dict[str, int] = {}
    rows_unknown = False
    for cube_name in touched:
        cube = catalog.get(cube_name)
        if cube is None:
            # Unknown cube — the validator will reject at compile;
            # cost estimation passes through.
            continue
        if cube.size_hint is None:
            rows_unknown = True
            continue
        cubes_estimated[cube_name] = cube.size_hint

    return CostEstimate(
        total_rows_scanned=sum(cubes_estimated.values()),
        cubes_estimated=cubes_estimated,
        rows_scanned_unknown=rows_unknown,
    )


__all__ = [
    "BudgetExceededError",
    "CostEstimate",
    "QueryBudget",
    "estimate_cost",
]
