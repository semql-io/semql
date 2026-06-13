"""Tests for the CatalogSpec / CatalogRuntime split (B5).

Architecture review B5: ``Catalog`` is a plain class holding callables
(policy, scope_fns, hooks, loaders); no ``model_dump`` / ``from_dict`` /
``version``. No defined ``CatalogSpec`` (data) vs ``CatalogRuntime``
(behaviour) boundary. Also a ~300-line first-error constructor while
PHILOSOPHY promises a collect-all validate path.

The contract:

- ``CatalogSpec`` is a frozen Pydantic value type carrying the
  serialisable data: cubes, views, lookups, saved queries, glossary,
  relations, hook names. ``model_dump`` / ``from_dict`` /
  ``schema_version`` are first-class. Round-trips byte-stable through
  Pydantic's ``model_validate(model_dump(...))``.
- ``CatalogRuntime`` wraps the callables: policy, scope_fns,
  unit_registry, error_transform, compile_hooks, sql_rewrite_hooks.
  Not serialisable.
- ``CatalogSpec.from_iterables`` is the collect-all constructor: it
  runs every construction-time validation (cube-name uniqueness, view
  ref resolution, lookup dim resolution, ...), aggregates the
  failures, and returns ``(spec, errors)`` — the caller can keep the
  partial spec for diagnostics, or surface the errors to the LLM.
- ``Catalog(cubes=..., policy=..., scope_fns=..., ...)`` continues to
  work as today — the public API is unchanged. Internally it builds
  the spec + runtime and pairs them.
"""

from __future__ import annotations

import json

import pytest
from semql.catalog import Catalog
from semql.model import Cube, Dialect, Dimension, Measure

# ---------------------------------------------------------------------------
# CatalogSpec exists, is frozen, and round-trips through model_dump
# ---------------------------------------------------------------------------


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def test_catalog_spec_round_trips_through_model_dump() -> None:
    """``CatalogSpec.model_validate(spec.model_dump())`` is a no-op —
    the I9 round-trip the existing tests pin for Cube / CompiledQuery
    extends to the catalog level."""
    from semql.catalog import CatalogSpec

    spec = CatalogSpec(cubes=(_orders(),))
    payload = spec.model_dump()
    # JSON-safe: every public surface must round-trip through
    # ``json.dumps`` so the spec can cross a process boundary.
    json.dumps(payload)
    restored = CatalogSpec.model_validate(payload)
    assert restored.cubes == spec.cubes
    assert restored.schema_version == spec.schema_version


def test_catalog_spec_schema_version_is_set() -> None:
    """A spec without a version can't be safely deserialised after
    a breaking change. ``schema_version`` is required, and the
    deserialiser surfaces a clear error on mismatch."""
    from semql.catalog import CatalogSpec

    spec = CatalogSpec(cubes=(_orders(),))
    assert isinstance(spec.schema_version, int)
    assert spec.schema_version >= 1


def test_catalog_spec_is_frozen() -> None:
    from pydantic import ValidationError
    from semql.catalog import CatalogSpec

    spec = CatalogSpec(cubes=(_orders(),))
    with pytest.raises(ValidationError):
        spec.cubes = ()


def test_catalog_spec_from_dict_round_trip() -> None:
    """``CatalogSpec.from_dict(dict)`` is the explicit constructor for
    a serialised payload — equivalent to ``model_validate`` but named
    for the wire-format use case."""
    from semql.catalog import CatalogSpec

    spec = CatalogSpec(cubes=(_orders(),))
    rebuilt = CatalogSpec.from_dict(spec.model_dump())
    assert rebuilt.cubes == spec.cubes


# ---------------------------------------------------------------------------
# CatalogRuntime — the behaviour side
# ---------------------------------------------------------------------------


def test_catalog_runtime_holds_callables() -> None:
    """The runtime is the home of callables: policy, scope_fns,
    unit_registry, error_transform, compile_hooks, sql_rewrite_hooks.
    It is not part of the serialised spec."""
    from semql.catalog import CatalogRuntime

    runtime = CatalogRuntime(
        policy=None,
        scope_fns={},
        unit_registry=None,
        error_transform=None,
        compile_hooks=[],
        sql_rewrite_hooks=[],
    )
    assert runtime.policy is None
    assert runtime.scope_fns == {}
    assert runtime.compile_hooks == []


def test_catalog_runtime_does_not_serialise() -> None:
    """The runtime carries callables, which can't cross a process
    boundary. ``model_dump`` is intentionally NOT on the runtime —
    callers serialise the spec, hand the runtime to the in-process
    catalog instance."""
    from semql.catalog import CatalogRuntime

    runtime = CatalogRuntime(
        policy=None,
        scope_fns={},
        unit_registry=None,
        error_transform=None,
        compile_hooks=[],
        sql_rewrite_hooks=[],
    )
    assert not hasattr(runtime, "model_dump")


# ---------------------------------------------------------------------------
# Catalog pairs a spec with a runtime
# ---------------------------------------------------------------------------


def test_catalog_exposes_spec_and_runtime() -> None:
    cat = Catalog([_orders()])
    assert cat.spec is not None
    assert cat.runtime is not None
    assert {c.name for c in cat.spec.cubes} >= {"orders"}


def test_catalog_constructs_from_existing_kwargs() -> None:
    """The legacy public API still works: Catalog(cubes=..., ...).
    Internally it builds a spec + runtime and pairs them."""
    cat = Catalog([_orders()])
    assert "orders" in cat.as_dict()


# ---------------------------------------------------------------------------
# Collect-all constructor — PHILOSOPHY promise
# ---------------------------------------------------------------------------


def test_collect_all_returns_spec_with_no_errors_on_clean_input() -> None:
    from semql.catalog import CatalogSpec

    spec, errors = CatalogSpec.from_iterables(cubes=[_orders()])
    assert errors == []
    # The spec carries only user-supplied cubes; META reflection
    # cubes are auto-appended on read, not stored in the spec.
    assert {c.name for c in spec.cubes} == {"orders"}


def test_collect_all_aggregates_duplicate_cube_name() -> None:
    from semql.catalog import CatalogSpec

    spec, errors = CatalogSpec.from_iterables(cubes=[_orders(), _orders()])
    assert spec is not None
    # The duplicate name surfaces as a structured construction error,
    # not a raise. Each error is the B8 envelope shape (dict with
    # "code" / "message" / ...).
    codes = {e["code"] for e in errors}
    assert "duplicate_cube_name" in codes


def test_collect_all_aggregates_unknown_join_target() -> None:
    from semql.catalog import CatalogSpec
    from semql.model import Join

    bad = Cube(
        name="bad",
        backend=Dialect.POSTGRES,
        table="bad",
        alias="b",
        joins=[Join(to="ghost", relationship="many_to_one", on="{b}.id = 1")],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
    )
    spec, errors = CatalogSpec.from_iterables(cubes=[bad])
    codes = {e["code"] for e in errors}
    assert "unknown_join_target" in codes


def test_collect_all_spec_is_usable_when_no_errors() -> None:
    """A spec with no construction errors can be paired with a
    runtime to build a working Catalog."""
    from semql.catalog import CatalogSpec

    spec, errors = CatalogSpec.from_iterables(cubes=[_orders()])
    assert errors == []
    cat = Catalog.from_spec(spec)
    assert "orders" in cat.as_dict()


# ---------------------------------------------------------------------------
# Hooks (callables) live on the runtime, names serialise into the spec
# ---------------------------------------------------------------------------


def test_spec_records_hook_names() -> None:
    """Compile / rewrite hooks are callables (runtime side) but their
    identifying name is what serialises into the spec — the runtime
    resolves the name back to a callable at construction time."""
    from semql.catalog import CatalogSpec

    spec = CatalogSpec(
        cubes=(_orders(),),
        compile_hook_names=("myapp.audit_hook", "myapp.retry_hook"),
    )
    assert spec.compile_hook_names == ("myapp.audit_hook", "myapp.retry_hook")
    payload = spec.model_dump()
    # Tuples round-trip as lists in the Pydantic model_dump output —
    # the spec stores them as tuples for hashability, the wire format
    # is JSON-array. Either order is fine on the other side.
    assert sorted(payload["compile_hook_names"]) == [
        "myapp.audit_hook",
        "myapp.retry_hook",
    ]


# ---------------------------------------------------------------------------
# Backwards compatibility — the existing public API survives
# ---------------------------------------------------------------------------


def test_existing_catalog_constructor_still_validates() -> None:
    """The legacy first-error constructor still raises ValueError for
    hard misconfiguration. The new collect-all path is opt-in via
    ``CatalogSpec.from_iterables``; the existing API is unchanged."""
    with pytest.raises(ValueError, match="duplicate"):
        Catalog([_orders(), _orders()])
