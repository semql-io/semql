"""Column → field-kind classification.

Pure functions over :class:`ColumnInfo`. Heuristic choices live here so
the orchestrator can stay glue and tests can exercise each rule in
isolation. Every guess that isn't a hard rule emits a
``heuristic_reason`` the emitter can surface as a ``# TODO: review``
comment — the dev reviewing the diff sees *why* the tool picked
``count_distinct`` over a plain dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from semql.model import DimTypeLiteral

from semql_introspect._probe import ColumnInfo

FieldKind = Literal[
    "measure_sum",
    "measure_count_distinct",
    "time_dimension",
    "dimension",
]


@dataclass(frozen=True)
class Classification:
    """How a single column should be modelled in the catalog.

    ``dim_type`` is set only for ``kind == "dimension"`` — it's the
    semql ``DimTypeLiteral`` the dimension should declare. Other
    kinds don't carry a dim type.

    ``heuristic_reason`` is non-empty when the classification came
    from a guess rather than a hard rule; the emitter renders it as a
    ``# TODO: review`` comment next to the field.
    """

    kind: FieldKind
    dim_type: DimTypeLiteral | None = None
    heuristic_reason: str = ""


# Column-name tokens that mark a numeric column as an additive measure.
# Plural / singular variants both included — the rule fires on a token
# match, not a full-string match.
_MEASURE_NAME_TOKENS = frozenset(
    {
        "amount",
        "amounts",
        "price",
        "prices",
        "revenue",
        "revenues",
        "cost",
        "costs",
        "total",
        "totals",
        "value",
        "values",
        "qty",
        "quantity",
        "quantities",
        "spend",
        "fee",
        "fees",
        "balance",
        "balances",
    }
)


def _normalize_type(data_type: str) -> str:
    """Strip qualifiers/parameters from a SQL type string.

    ``"timestamp without time zone"`` → ``"timestamp"``;
    ``"VARCHAR(255)"`` → ``"varchar"``;
    ``"numeric(18,2)"`` → ``"numeric"``.
    """
    t = data_type.lower().strip()
    if "(" in t:
        t = t.split("(", 1)[0].strip()
    if " " in t:
        t = t.split(" ", 1)[0]
    return t


_NUMERIC_TYPES = frozenset(
    {
        "smallint",
        "integer",
        "int",
        "int2",
        "int4",
        "int8",
        "bigint",
        "decimal",
        "numeric",
        "real",
        "float",
        "float4",
        "float8",
        "double",
        "money",
    }
)


_DATE_TYPES = frozenset(
    {
        "date",
        "timestamp",
        "timestamptz",
        "datetime",
        "time",
        "timetz",
    }
)


_BOOL_TYPES = frozenset({"boolean", "bool"})


def _is_numeric(data_type: str) -> bool:
    return _normalize_type(data_type) in _NUMERIC_TYPES


def _is_date(data_type: str) -> bool:
    return _normalize_type(data_type) in _DATE_TYPES


def _is_bool(data_type: str) -> bool:
    return _normalize_type(data_type) in _BOOL_TYPES


def _dim_type_for(data_type: str) -> DimTypeLiteral:
    """Map a SQL type string onto a semql ``DimTypeLiteral``.

    Falls back to ``"string"`` for unknown types — the catalog author
    can refine it post-emission. ``"string"`` is the safer default
    than ``"number"`` because misclassifying a numeric ID as a number
    invites accidental aggregation."""
    if _is_numeric(data_type):
        return "number"
    if _is_date(data_type):
        return "time"
    if _is_bool(data_type):
        return "bool"
    return "string"


def classify_column(col: ColumnInfo, *, is_fk: bool, is_pk: bool) -> Classification:
    """Pick the field kind + (where applicable) dimension type for a column.

    Rules, in priority order:

    1. **Date / timestamp** → ``time_dimension``. Hard rule.
    2. **Foreign-key columns** → ``dimension`` (with ``foreign_key=``
       wired up by the orchestrator). Numeric-FK heuristic *does not*
       apply — FKs are identifiers, not measurements.
    3. **Primary-key columns** → ``dimension``. Hard rule; the cube's
       ``primary_key`` field tracks identity separately.
    4. **Numeric columns whose name matches a measure-name token**
       (amount / price / revenue / ...) → ``measure_sum``.
    5. **Columns ending in ``_id``** → ``measure_count_distinct``. The
       table's distinct identifier count is a useful default measure
       even when an ID column isn't a measure proper.
    6. Otherwise → ``dimension`` typed by the column's SQL type.
    """
    name = col.name.lower()

    if is_pk:
        return Classification(
            kind="dimension",
            dim_type=_dim_type_for(col.data_type),
        )

    if _is_date(col.data_type):
        return Classification(kind="time_dimension")

    if is_fk:
        return Classification(
            kind="dimension",
            dim_type=_dim_type_for(col.data_type),
        )

    if _is_numeric(col.data_type):
        tokens = set(name.split("_"))
        if tokens & _MEASURE_NAME_TOKENS:
            return Classification(
                kind="measure_sum",
                heuristic_reason=(
                    f"numeric column named {col.name!r} matched a measure-name "
                    "token (amount/price/revenue/...) — confirm this should be a "
                    "summable measure rather than a dimension."
                ),
            )

    if name.endswith("_id"):
        return Classification(
            kind="measure_count_distinct",
            heuristic_reason=(
                f"column ends in ``_id`` ({col.name!r}); inferred a "
                "``count_distinct`` measure. Drop the measure if this column "
                "isn't useful as a count, or move it to a foreign-key "
                "dimension if it should link to another cube."
            ),
        )

    return Classification(
        kind="dimension",
        dim_type=_dim_type_for(col.data_type),
    )


__all__ = ["Classification", "FieldKind", "classify_column"]
