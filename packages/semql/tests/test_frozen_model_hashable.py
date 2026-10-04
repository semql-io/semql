"""Frozen catalog models must actually be hashable.

``model_config = ConfigDict(frozen=True)`` advertises hashability, but
Pydantic's generated ``__hash__`` raises ``unhashable type: 'list'`` /
``'dict'`` the moment a model carries a ``list`` or ``dict`` field — so
``Measure`` / ``Dimension`` / ``AuthContext`` / ``Join`` / … could not go
in a ``set`` or be a dict key despite claiming to be frozen.

A recursive value-based ``__hash__`` on a shared base restores the
contract while keeping Pydantic's field-wise ``__eq__`` — equal models
hash equal.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel
from semql import Cube, Dialect, MutableEntity, MutableField, Op
from semql.model import (
    AuthContext,
    Dimension,
    Join,
    Measure,
    Rollup,
    ScopePredicate,
    Segment,
    TimeDimension,
)


def _instances() -> list[Any]:
    # Pydantic's stubs mark frozen BaseModel hashes as unavailable, while
    # this suite verifies the concrete _HashableModel runtime contract.
    return [
        Measure(name="rev", sql="{t}.x", agg="sum", unit="count", metadata={"k": "v"}),
        Dimension(name="region", sql="{t}.r", type="string", required_roles=["a"]),
        TimeDimension(name="ts", sql="{t}.ts"),
        Segment(name="active", sql="{t}.active = true"),
        Join(to="other", relationship="many_to_one", on="{t}.id = {o}.t_id"),
        AuthContext(viewer_id="u", roles=["analyst", "hr"], attrs={"team": "x"}),
        ScopePredicate(sql="{t}.a = {ctx.viewer_id}", ctx_keys=["viewer_id"]),
        Rollup(name="daily", physical_table="r.daily", dimensions=["region"], measures=["rev"]),
    ]


def test_frozen_models_with_collection_fields_are_hashable() -> None:
    for obj in _instances():
        # Must not raise — the whole point of the fix.
        hash(obj)
    # And usable in the containers frozen=True implies.
    as_set = set(_instances())
    assert len(as_set) == len(_instances())
    as_key = {obj: i for i, obj in enumerate(_instances())}
    assert len(as_key) == len(_instances())


def test_equal_frozen_models_hash_equal() -> None:
    a = Measure(name="rev", sql="{t}.x", agg="sum", unit="count", metadata={"k": "v"})
    b = Measure(name="rev", sql="{t}.x", agg="sum", unit="count", metadata={"k": "v"})
    assert a == b
    assert hash(a) == hash(b)
    ctx1 = AuthContext(viewer_id="u", roles=["a", "b"])
    ctx2 = AuthContext(viewer_id="u", roles=["a", "b"])
    assert ctx1 == ctx2
    assert hash(ctx1) == hash(ctx2)


def test_distinct_frozen_models_do_not_collapse_in_a_set() -> None:
    m1 = Measure(name="rev", sql="{t}.x", agg="sum", unit="count")
    m2 = Measure(name="rev", sql="{t}.x", agg="avg", unit="count")
    assert m1 != m2
    # _HashableModel defines a real __hash__, but pyright's pydantic plugin
    # treats a non-frozen-config model as __hash__=None; runtime hashability
    # (exactly what this test pins) is correct.
    assert len({m1, m2}) == 2  # pyright: ignore[reportUnhashable]


def test_hashable_models_cannot_mutate_nested_security_or_structural_state() -> None:
    """Hash/equality inputs remain fixed even through collection mutators."""
    roles = ["admin"]
    required_fields = ["region"]
    opaque_tags = {"lineage": "warehouse.region"}
    field = Dimension(
        name="region",
        sql="{o}.region",
        type="string",
        required_roles=roles,
        metadata=opaque_tags,
    )
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        dimensions=[field],
        required_filters=required_fields,
    )
    entity = MutableEntity(
        name="order",
        cubes=["orders"],
        key="orders.region",
        target_cube="orders",
        operations=frozenset({Op.UPDATE}),
        mutable_fields={"region": MutableField(type="string")},
    )
    viewer = AuthContext(viewer_id="u", roles=["reader"], attrs={"groups": ["a"]})
    objects: list[tuple[Any, str]] = [
        (field, "dimension"),
        (entity, "entity"),
        (viewer, "viewer"),
    ]
    buckets = [{value: label} for value, label in objects]
    hashes = [hash(value) for value, _ in objects]
    with pytest.raises(TypeError):
        hash(cube)

    # These casts intentionally bypass the immutable public Sequence/Mapping API
    # to prove that direct mutation cannot alter hash/equality inputs.
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, field.required_roles).append("reader")
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, field.required_roles)[0] = "reader"
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, cube.required_filters).append("id")
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, entity.mutable_fields)["other"] = MutableField(type="string")
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, field.metadata)["lineage"] = "changed"
    with pytest.raises((AttributeError, TypeError)):
        viewer.attrs["groups"].append("b")
    opaque_tags["lineage"] = "changed outside the model"
    assert field.metadata == {"lineage": "warehouse.region"}

    assert [hash(value) for value, _ in objects] == hashes
    assert [bucket[value] for (value, _), bucket in zip(objects, buckets, strict=True)] == [
        label for _, label in objects
    ]
    assert list(field.required_roles) == ["admin"]
    assert list(cube.required_filters) == ["region"]


def test_hashable_models_defensively_copy_inputs_and_model_copy_updates() -> None:
    original_roles = ["analyst"]
    original_attrs = {"teams": ["blue"]}
    context = AuthContext(viewer_id="u", roles=original_roles, attrs=original_attrs)
    cache = {cast(Any, context): "allowed"}
    old_hash = hash(context)
    original_roles.append("admin")
    original_attrs["teams"].append("red")

    assert context.roles == ["analyst"]
    assert context.attrs == {"teams": ["blue"]}
    assert hash(context) == old_hash
    assert cache[cast(Any, context)] == "allowed"

    copied = context.model_copy(update={"roles": ["reader"]})
    copied_cache = {cast(Any, copied): "reader"}
    copied_hash = hash(copied)
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, copied.roles).append("admin")
    assert hash(copied) == copied_hash
    assert copied_cache[cast(Any, copied)] == "reader"


@pytest.mark.parametrize("instance", _instances())
def test_collection_mutation_cannot_change_any_hashable_model_value(
    instance: BaseModel,
) -> None:
    from collections.abc import Mapping, Sequence

    before_hash = hash(instance)
    key = {cast(Any, instance): "stable"}
    before = instance.model_dump()
    model_type: type[BaseModel] = type(instance)
    for name in model_type.model_fields:
        value = getattr(instance, name)
        if isinstance(value, Mapping):
            with pytest.raises(TypeError):
                cast(Any, value)["injected"] = "value"
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            with pytest.raises((AttributeError, TypeError)):
                cast(Any, value).append("injected")

    assert hash(instance) == before_hash
    assert key[cast(Any, instance)] == "stable"
    rebuilt = model_type.model_validate(before)
    assert rebuilt == instance
    assert hash(rebuilt) == before_hash


def test_nested_immutable_model_values_round_trip_through_json() -> None:
    context = AuthContext(
        viewer_id="u",
        roles=["reader"],
        metadata={"owner": "security"},
        attrs={"claims": {"teams": ["blue", "green"]}},
    )
    rebuilt = AuthContext.model_validate_json(context.model_dump_json())

    assert rebuilt == context
    assert rebuilt.roles == ["reader"]
    assert rebuilt.metadata == {"owner": "security"}
    assert rebuilt.attrs == {"claims": {"teams": ["blue", "green"]}}
    assert hash(rebuilt) == hash(context)


def test_hash_rejects_mutable_models_and_unsupported_nested_values() -> None:
    from semql.spec import SavedQuery, SemanticQuery

    query = SemanticQuery(measures=["orders.revenue"])
    saved = SavedQuery(name="revenue", query=query)
    context = AuthContext(viewer_id="u", attrs={"saved_query": saved})

    with pytest.raises(TypeError, match="mutable model"):
        hash(context)
    query.measures.append("orders.cost")
    with pytest.raises(TypeError, match="mutable model"):
        hash(context)

    class MutableTag:
        pass

    unsupported = AuthContext(viewer_id="u", attrs={"tag": MutableTag()})
    with pytest.raises(TypeError, match="unsupported value"):
        hash(unsupported)


def test_model_construct_mutable_state_cannot_be_hashed() -> None:
    raw_roles = ["reader"]
    unsafe_roles = AuthContext.model_construct(viewer_id="u", roles=raw_roles)

    with pytest.raises(TypeError, match="mutable list"):
        hash(unsafe_roles)
    raw_roles.append("admin")
    with pytest.raises(TypeError, match="mutable list"):
        hash(unsafe_roles)

    unsafe_attrs = AuthContext.model_construct(
        viewer_id="u",
        roles=(),
        attrs={"teams": ["blue"]},
    )
    with pytest.raises(TypeError, match="mutable dict"):
        hash(unsafe_attrs)
