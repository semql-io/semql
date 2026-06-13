"""S7-R5 — ValidationWarning soft warnings.

ValidationWarning: subclass of ValidationError so:
- isinstance(w, ValidationError) still True (backwards compat)
- isinstance(w, ValidationWarning) True only for advisory warnings

New behaviors in validate():
1. Beta cube reference → emit ValidationWarning (code="cube_beta") instead of ValidationError
2. Backtick-name in Cube.relations/Catalog.relations that resolves → no warning
3. Backtick-name that doesn't resolve → ValidationWarning(code="unresolved_backtick")
"""

from __future__ import annotations

from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.validate import ValidationError, validate


def _cube(
    name: str = "orders",
    stability: str = "stable",
    relations: str = "",
) -> Cube:
    return Cube(
        name=name,
        backend=Dialect.POSTGRES,
        table=f"public.{name}",
        alias=name[:2],
        measures=[Measure(name="cnt", sql=f"{{{name[:2]}}}.id", agg="count")],
        dimensions=[Dimension(name="region", sql=f"{{{name[:2]}}}.region", type="string")],
        stability=stability,  # type: ignore[arg-type]
        relations=relations,
    )


# ---------------------------------------------------------------------------
# ValidationWarning importable
# ---------------------------------------------------------------------------


def test_validation_warning_importable() -> None:
    from semql.validate import ValidationWarning

    assert ValidationWarning is not None


def test_validation_warning_is_subclass_of_validation_error() -> None:
    from semql.validate import ValidationWarning

    assert issubclass(ValidationWarning, ValidationError)


def test_validation_warning_exported_from_semql() -> None:
    import semql

    assert hasattr(semql, "ValidationWarning")


# ---------------------------------------------------------------------------
# Beta cube → ValidationWarning (not plain ValidationError)
# ---------------------------------------------------------------------------


def test_beta_cube_emits_validation_warning() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(stability="beta")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    warnings = [e for e in errs if isinstance(e, ValidationWarning)]
    assert any(w.code == "cube_beta" for w in warnings), (
        "Expected a ValidationWarning with code='cube_beta' for a beta cube"
    )


def test_beta_cube_warning_is_also_validation_error() -> None:
    """ValidationWarning must be an instance of ValidationError for backwards compat."""
    cat = Catalog([_cube(stability="beta")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    beta_entries = [e for e in errs if e.code == "cube_beta"]
    assert len(beta_entries) == 1
    assert isinstance(beta_entries[0], ValidationError)


def test_stable_cube_no_beta_warning() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(stability="stable")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    assert not any(isinstance(e, ValidationWarning) and e.code == "cube_beta" for e in errs)


def test_caller_can_filter_warnings() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(stability="beta")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    warnings = [e for e in errs if isinstance(e, ValidationWarning)]
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# Backtick-name resolution check
# ---------------------------------------------------------------------------


def test_resolved_backtick_name_no_warning() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(relations="See `orders` for the primary entity.")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    unresolved = [
        e for e in errs if isinstance(e, ValidationWarning) and e.code == "unresolved_backtick"
    ]
    assert len(unresolved) == 0, f"Expected no unresolved backtick warnings; got {unresolved}"


def test_unresolved_backtick_name_emits_warning() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(relations="See `orders_v1` for historical data.")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    unresolved = [
        e for e in errs if isinstance(e, ValidationWarning) and e.code == "unresolved_backtick"
    ]
    assert len(unresolved) == 1
    assert "orders_v1" in unresolved[0].message


def test_unresolved_backtick_carries_cube_name() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(relations="See `ghost_cube` for details.")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    unresolved = [
        e for e in errs if isinstance(e, ValidationWarning) and e.code == "unresolved_backtick"
    ]
    assert len(unresolved) == 1


def test_backtick_resolving_to_measure_name_no_warning() -> None:
    from semql.validate import ValidationWarning

    # `cnt` is a measure on the orders cube
    cat = Catalog([_cube(relations="The `cnt` metric counts rows.")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    unresolved = [
        e for e in errs if isinstance(e, ValidationWarning) and e.code == "unresolved_backtick"
    ]
    assert len(unresolved) == 0


def test_multiple_unresolved_backticks() -> None:
    from semql.validate import ValidationWarning

    cat = Catalog([_cube(relations="See `ghost_a` and `ghost_b` for context.")])
    q = SemanticQuery(measures=["orders.cnt"])
    errs = validate(q, cat)
    unresolved = [
        e for e in errs if isinstance(e, ValidationWarning) and e.code == "unresolved_backtick"
    ]
    assert len(unresolved) == 2
