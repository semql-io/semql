"""Tests for the row-mode entity MCP tools (M5): per-entity get_/list_/
mutate_ tools, the generic collapse mode, two-step confirm, and the
allow_mutations / role gates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import duckdb
import pytest
from fastmcp import Client
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Entity,
    Measure,
    MutableEntity,
    MutableField,
    Op,
)
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="n", sql="{o}.id", agg="count", unit="count")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        primary_key="id",
    )


def _read_entity() -> Entity:
    return Entity(
        name="order",
        cubes=["orders"],
        key="orders.id",
        list_filters=["orders.region"],
        default_order="orders.id asc",
    )


def _mutable_entity() -> MutableEntity:
    return MutableEntity(
        name="order",
        cubes=["orders"],
        key="orders.id",
        target_cube="orders",
        operations=frozenset({Op.UPDATE, Op.DELETE}),
        mutable_fields={"region": MutableField(type="string")},
    )


class _RecordingExecutor:
    """Returns canned rows for SELECTs; records every (sql, params)."""

    def __init__(self, preview_rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.preview_rows = preview_rows or [{"id": 1, "region": "us"}]

    def __call__(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((sql, params))
        if sql.strip().upper().startswith("SELECT"):
            return list(self.preview_rows)
        return []

    @property
    def dml_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [c for c in self.calls if not c[0].strip().upper().startswith("SELECT")]


def _tool_names(server: MCPServer) -> set[str]:
    async def fetch() -> set[str]:
        async with Client(server.mcp) as c:
            return {t.name for t in await c.list_tools()}

    return _run(fetch())


def _call(server: MCPServer, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        async with Client(server.mcp) as c:
            result = await c.call_tool(tool, args)
            return result.data  # type: ignore[no-any-return]

    return _run(call())


# ---------------------------------------------------------------------------
# Registration + gating
# ---------------------------------------------------------------------------


def test_read_entity_registers_get_and_list() -> None:
    server = MCPServer(Catalog([_cube()], entities=[_read_entity()]), require_viewer=False)
    names = _tool_names(server)
    assert "get_order" in names
    assert "list_order" in names
    assert "mutate_order" not in names  # read-only entity


def test_mutate_tool_absent_without_allow_mutations() -> None:
    server = MCPServer(
        Catalog([_cube()], entities=[_mutable_entity()], allow_mutations=False),
        require_viewer=False,
    )
    assert "mutate_order" not in _tool_names(server)


def test_mutate_tool_present_with_allow_mutations() -> None:
    server = MCPServer(
        Catalog([_cube()], entities=[_mutable_entity()], allow_mutations=True),
        require_viewer=False,
    )
    assert "mutate_order" in _tool_names(server)


def test_generic_entity_tools_mode() -> None:
    server = MCPServer(
        Catalog([_cube()], entities=[_mutable_entity()], allow_mutations=True),
        require_viewer=False,
        generic_entity_tools=True,
    )
    names = _tool_names(server)
    assert {"get_entity", "list_entity", "mutate"} <= names
    assert "get_order" not in names  # collapsed


# ---------------------------------------------------------------------------
# get / list behaviour
# ---------------------------------------------------------------------------


def test_get_entity_compiles_and_executes() -> None:
    ex = _RecordingExecutor()
    server = MCPServer(
        Catalog([_cube()], entities=[_read_entity()]), executor=ex, require_viewer=False
    )
    out = _call(server, "get_order", {"key": 1})
    assert out["sql"] is not None
    assert out["plan"]["source"]["cube"] == "orders"
    assert "rows" in out


def test_list_entity_filter_typed_and_runs() -> None:
    ex = _RecordingExecutor()
    server = MCPServer(
        Catalog([_cube()], entities=[_read_entity()]), executor=ex, require_viewer=False
    )
    out = _call(server, "list_order", {"region": "us"})
    assert out["sql"] is not None
    assert "us" in out["params"].values()


# ---------------------------------------------------------------------------
# Two-step confirm
# ---------------------------------------------------------------------------


def test_mutate_confirm_false_previews_only() -> None:
    ex = _RecordingExecutor(preview_rows=[{"id": 1}, {"id": 2}])
    server = MCPServer(
        Catalog([_cube()], entities=[_mutable_entity()], allow_mutations=True),
        executor=ex,
        require_viewer=False,
    )
    out = _call(
        server, "mutate_order", {"operation": "update", "values": {"region": "x"}, "pk": {"id": 1}}
    )
    assert out["confirmed"] is False
    assert out["executed"] is False
    assert out["affected_rows"] == 2
    assert ex.dml_calls == []  # DML never executed on a preview


def test_mutate_confirm_true_executes_dml() -> None:
    ex = _RecordingExecutor(preview_rows=[{"id": 1}])
    server = MCPServer(
        Catalog([_cube()], entities=[_mutable_entity()], allow_mutations=True),
        executor=ex,
        require_viewer=False,
    )
    out = _call(
        server,
        "mutate_order",
        {"operation": "update", "values": {"region": "x"}, "pk": {"id": 1}, "confirm": True},
    )
    assert out["confirmed"] is True
    assert out["executed"] is True
    assert len(ex.dml_calls) == 1
    assert ex.dml_calls[0][0].strip().upper().startswith("UPDATE")


def test_mutate_cap_exceeded_refuses() -> None:
    ex = _RecordingExecutor(preview_rows=[{"id": i} for i in range(5)])
    server = MCPServer(
        Catalog(
            [_cube()],
            entities=[
                MutableEntity(
                    name="order",
                    cubes=["orders"],
                    key="orders.id",
                    target_cube="orders",
                    operations=frozenset({Op.DELETE}),
                    mutable_fields={},
                    predicate_targeting=True,
                )
            ],
            allow_mutations=True,
            max_mutation_rows=2,
        ),
        executor=ex,
        require_viewer=False,
    )
    out = _call(
        server, "mutate_order", {"operation": "delete", "where": {"region": "us"}, "confirm": True}
    )
    assert out["error"]["code"] == "MutationCapExceeded"
    assert ex.dml_calls == []  # cap blocks the DML


# ---------------------------------------------------------------------------
# Role gate (gate 3) enforced per-call
# ---------------------------------------------------------------------------


def test_mutate_role_gate_denied() -> None:
    cube = _cube().model_copy(update={"required_roles": ["admin"]})
    server = MCPServer(
        Catalog([cube], entities=[_mutable_entity()], allow_mutations=True),
        executor=_RecordingExecutor(),
        viewer_provider=lambda: AuthContext(viewer_id="u1", roles=["viewer"]),
    )
    out = _call(
        server, "mutate_order", {"operation": "update", "values": {"region": "x"}, "pk": {"id": 1}}
    )
    assert "error" in out
    assert out["error"]["code"] == "AuthError"


@pytest.mark.parametrize(("matching", "cap"), [(3, 2), (3, 3), (2, 3)])
@pytest.mark.parametrize("round_trip", [False, True], ids=["original", "json-restored"])
def test_serialized_mutation_cap_controls_actual_confirmed_writes(
    matching: int, cap: int, round_trip: bool
) -> None:
    from semql.catalog import CatalogSpec

    cube = _cube().model_copy(update={"dialect": Dialect.DUCKDB})
    entity = _mutable_entity().model_copy(update={"predicate_targeting": True})
    catalog = Catalog([cube], entities=[entity], allow_mutations=True, max_mutation_rows=cap)
    if round_trip:
        catalog = Catalog.from_spec(CatalogSpec.model_validate_json(catalog.spec.model_dump_json()))
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE orders(id INTEGER, region VARCHAR, amount INTEGER)")
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(i, "us", 1) for i in range(matching)] + [(99, "other", 1)],
        )

        def execute(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            cursor = connection.execute(sql, params)
            names = [column[0] for column in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

        server = MCPServer(catalog, executor=execute, require_viewer=False)
        output = _call(
            server,
            "mutate_order",
            {"operation": "delete", "where": {"region": "us"}, "confirm": True},
        )
        if matching > cap:
            assert output["error"]["code"] == "MutationCapExceeded"
            assert output["executed"] is False
        else:
            assert output["executed"] is True
            assert output["affected_rows"] == matching
        assert connection.execute("SELECT id FROM orders WHERE region = 'other'").fetchall() == [
            (99,)
        ]
        remaining = connection.execute("SELECT count(*) FROM orders WHERE region = 'us'").fetchone()
        assert remaining == (matching if matching > cap else 0,)
    finally:
        connection.close()


@pytest.mark.parametrize("round_trip", [False, True], ids=["original", "json-restored"])
def test_serialized_list_cap_limits_actual_entity_rows(round_trip: bool) -> None:
    from semql.catalog import CatalogSpec

    catalog = Catalog(
        [_cube().model_copy(update={"dialect": Dialect.DUCKDB})],
        entities=[_read_entity()],
        max_list_limit=2,
    )
    if round_trip:
        catalog = Catalog.from_spec(CatalogSpec.model_validate_json(catalog.spec.model_dump_json()))
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE orders(id INTEGER, region VARCHAR, amount INTEGER)")
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)", [(i, "us", 1) for i in range(5)]
        )

        def execute(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            cursor = connection.execute(sql, params)
            names = [column[0] for column in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

        server = MCPServer(catalog, executor=execute, require_viewer=False)
        output = _call(server, "list_order", {"region": "us", "limit": 50})
        assert [row["id"] for row in output["rows"]] == [0, 1]
        assert connection.execute("SELECT count(*) FROM orders").fetchone() == (5,)
    finally:
        connection.close()
