# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""A1 — Field-level visibility and masking on Measure / Dimension.

The four evaluation-order rules (given ``viewer: AuthContext``):

  1. *Cube gate*: ``Cube.required_roles`` refusal (existing).
  2. *Field hide gate*: ``field.required_roles`` non-empty and viewer
     has none → ``CompileError(UnknownIdentifierError)``, message
     indistinguishable from "field doesn't exist".
  3. *Field mask gate*: ``field.mask_roles`` non-empty and viewer has
     any mask role → substitute field SQL with
     ``CAST(NULL AS <inferred_type>)`` (or the literal for
     ``mask_value``) at projection. ``ColumnMeta.masked: bool = True``.
  4. *Real value*: viewer passes both → real SQL.

When ``viewer is None``, both gates are bypassed (unauthed path stays
open for catalog-level tooling). At catalog construction,
``mask_roles ⊆ required_roles`` is enforced — a field that masks a
role that can't even access the field is a configuration error.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.errors import CompileError
from semql.model import (
    AuthContext,
    Backend,
    Cube,
    Dimension,
    Measure,
)
from semql.spec import SemanticQuery

from .conftest import CONTEXT

# ---------------------------------------------------------------------------
# Model surface
# ---------------------------------------------------------------------------


def test_measure_accepts_mask_roles() -> None:
    m = Measure(
        name="salary",
        sql="{o}.salary",
        agg="sum",
        required_roles=["hr", "analyst"],
        mask_roles=["analyst"],
    )
    assert m.mask_roles == ["analyst"]


def test_measure_accepts_mask_value() -> None:
    m = Measure(
        name="salary",
        sql="{o}.salary",
        agg="sum",
        required_roles=["hr", "analyst"],
        mask_roles=["analyst"],
        mask_value="REDACTED",
    )
    assert m.mask_value == "REDACTED"


def test_dimension_accepts_mask_roles() -> None:
    d = Dimension(
        name="ssn",
        sql="{o}.ssn",
        type="string",
        required_roles=["hr", "analyst"],
        mask_roles=["analyst"],
    )
    assert d.mask_roles == ["analyst"]


def test_time_dimension_does_not_have_mask() -> None:
    """Per the A1 spec: time columns are structural, not sensitive."""
    from semql.model import TimeDimension

    # TimeDimension has no ``mask_roles`` field; it's not inherited
    # from Measure / Dimension. Required roles still apply (the
    # field-hide gate) — only masking is excluded.
    assert "mask_roles" not in TimeDimension.model_fields
    assert "mask_value" not in TimeDimension.model_fields


def test_required_roles_default_empty() -> None:
    """`BaseField.required_roles` defaults to empty (open to all viewers)."""
    m = Measure(name="count", sql="*", agg="count")
    assert m.required_roles == []


def test_required_roles_accepts_list() -> None:
    m = Measure(
        name="salary",
        sql="{o}.salary",
        agg="sum",
        required_roles=["hr"],
    )
    assert m.required_roles == ["hr"]


# ---------------------------------------------------------------------------
# Catalog construction: subset validation
# ---------------------------------------------------------------------------


def test_mask_roles_must_be_subset_of_required_roles() -> None:
    """A field that masks a role that can't even access it is a config error."""
    with pytest.raises(ValueError, match="mask_roles"):
        Measure(
            name="salary",
            sql="{o}.salary",
            agg="sum",
            required_roles=["hr"],
            mask_roles=["analyst", "finance"],  # not a subset of ["hr"]
        )


def test_mask_roles_subset_is_valid() -> None:
    """mask_roles ⊆ required_roles passes."""
    m = Measure(
        name="salary",
        sql="{o}.salary",
        agg="sum",
        required_roles=["hr", "analyst"],
        mask_roles=["analyst"],  # subset of ["hr", "analyst"]
    )
    assert m.mask_roles == ["analyst"]


# ---------------------------------------------------------------------------
# Compile-time semantics
# ---------------------------------------------------------------------------


def _hr_only_salary_cube() -> dict:
    """Catalog with a measure that's HR-only and masked-for-analyst."""
    orders = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[
            Measure(name="count", sql="*", agg="count"),
            Measure(
                name="salary",
                sql="{o}.salary",
                agg="sum",
                required_roles=["hr", "analyst"],
                mask_roles=["analyst"],
            ),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
    )
    return {"orders": orders}


def test_viewer_with_required_role_sees_real_value() -> None:
    """A viewer in ``required_roles`` sees the real value."""
    cat = _hr_only_salary_cube()
    q = SemanticQuery(measures=["orders.salary"])
    viewer = AuthContext(viewer_id="u1", roles=["hr"])
    cq = compile_query(q, cat, context=CONTEXT, viewer=viewer)
    assert "salary" in cq.columns
    salary_meta = next(m for m in cq.column_meta if m.name == "salary")
    assert salary_meta.masked is False
    # Real SQL: SUM(o.salary) is in the query.
    assert "salary" in cq.sql


def test_viewer_with_mask_role_gets_null() -> None:
    """A viewer in ``mask_roles`` (but not the full required_roles set) gets NULL."""
    cat = _hr_only_salary_cube()
    q = SemanticQuery(measures=["orders.salary"])
    viewer = AuthContext(viewer_id="u1", roles=["analyst"])
    cq = compile_query(q, cat, context=CONTEXT, viewer=viewer)
    salary_meta = next(m for m in cq.column_meta if m.name == "salary")
    assert salary_meta.masked is True
    # Mask substitutes the field SQL with CAST(NULL AS <type>).
    assert "CAST(NULL" in cq.sql or "NULL AS" in cq.sql


def test_viewer_with_mask_value_substitutes_literal() -> None:
    """``mask_value`` is a SQL literal, not a NULL cast."""
    cube = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[
            Measure(
                name="salary",
                sql="{o}.salary",
                agg="sum",
                required_roles=["hr", "analyst"],
                mask_roles=["analyst"],
                mask_value="'REDACTED'",
            ),
        ],
    )
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.salary"])
    viewer = AuthContext(viewer_id="u1", roles=["analyst"])
    cq = compile_query(q, cat, context=CONTEXT, viewer=viewer)
    salary_meta = next(m for m in cq.column_meta if m.name == "salary")
    assert salary_meta.masked is True
    assert "'REDACTED'" in cq.sql


def test_viewer_without_required_role_gets_compile_error() -> None:
    """A viewer missing the required role gets a CompileError (field-hide gate)."""
    cat = _hr_only_salary_cube()
    q = SemanticQuery(measures=["orders.salary"])
    # The field is restricted to hr+analyst. A viewer with neither
    # role triggers the field-hide gate, indistinguishable from
    # "field doesn't exist" — no information leak.
    viewer = AuthContext(viewer_id="u1", roles=["other"])
    with pytest.raises(CompileError):
        compile_query(q, cat, context=CONTEXT, viewer=viewer)


def test_no_viewer_bypasses_both_gates() -> None:
    """``viewer=None`` keeps the unauthed path open (catalog tooling)."""
    cat = _hr_only_salary_cube()
    q = SemanticQuery(measures=["orders.salary"])
    cq = compile_query(q, cat, context=CONTEXT)
    salary_meta = next(m for m in cq.column_meta if m.name == "salary")
    assert salary_meta.masked is False
    # Real SQL: SUM(o.salary) is in the query.
    assert "salary" in cq.sql


def test_unmasked_field_has_masked_false(catalog: dict) -> None:
    """Regression: non-masked fields get ``ColumnMeta.masked == False``."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    cq = compile_query(q, catalog, context=CONTEXT)
    for col_meta in cq.column_meta:
        assert col_meta.masked is False
