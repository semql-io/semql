"""Real driver/backend cells for the first-release semantic contract.
DuckDB always runs. PostgreSQL uses SEMQL_CONFORMANCE_POSTGRES when set;
ClickHouse uses SEMQL_CONFORMANCE_CLICKHOUSE when set. Otherwise, setting
SEMQL_CONFORMANCE_TESTCONTAINERS=1 opts into pinned local Testcontainers for
external backends. Each external fixture creates and removes its own uniquely
named database/schema. No SQL rewriting or placeholder translation is used.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import os
import urllib.parse
import urllib.request
import uuid
from typing import Any, LiteralString, cast

import duckdb
import psycopg
import pytest
from semql import (
    AuthContext,
    Catalog,
    CatalogContext,
    CompareWindow,
    Cube,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Join,
    Measure,
    Rollup,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    compare_analysis,
    compile_federated_query,
    compile_query,
)
from semql.compile import CompiledQuery
from semql_engine import AdapterResult, DBAPIAdapter, DuckDBAdapter, DuckDBMergeEngine, Engine
from semql_engine.adapter import Adapter
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import ExecWaitStrategy, HttpWaitStrategy

_POSTGRES_IMAGE = "postgres:16.14-alpine"
_CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server:26.8.2.7"
_CONTAINER_STARTUP_TIMEOUT_SECONDS = 90


class _HTTPClickHouse:
    def __init__(
        self, url: str, database: str, *, username: str | None = None, password: str | None = None
    ) -> None:
        self.url = url
        self.database = database
        self.username = username
        self.password = password

    def request(self, sql: str, params: collections.abc.Mapping[str, Any]) -> str:
        options = {"database": self.database, "join_use_nulls": "1"}
        if self.username is not None:
            options["user"] = self.username
        if self.password is not None:
            options["password"] = self.password
        options.update({f"param_{key}": str(value) for key, value in params.items()})
        request = urllib.request.Request(
            self.url + "/?" + urllib.parse.urlencode(options),
            data=sql.encode(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body: bytes = response.read()
            return body.decode()

    def execute(self, sql: str, params: collections.abc.Mapping[str, Any]) -> AdapterResult:
        payload = json.loads(self.request(sql + " FORMAT JSON", params))
        columns = [str(item["name"]) for item in payload["meta"]]
        return AdapterResult(
            columns=columns,
            rows=[tuple(row[column] for column in columns) for row in payload["data"]],
        )


@dataclasses.dataclass
class _Backend:
    dialect: Dialect
    adapter: Adapter
    prefix: str

    def compile(
        self, query: SemanticQuery, *, rollup: bool = False, table: str = "events"
    ) -> CompiledQuery:
        cube = Cube(
            name="events",
            table=f"{self.prefix}{table}",
            alias="e",
            dialect=self.dialect,
            measures=[
                Measure(name="amount", sql="{e}.amount", agg="sum", unit="currency"),
                Measure(name="n", sql="*", agg="count"),
                Measure(name="mean", sql="{e}.amount", agg="avg"),
                Measure(name="minimum", sql="{e}.amount", agg="min"),
                Measure(name="maximum", sql="{e}.amount", agg="max"),
                Measure(name="distinct_ids", sql="{e}.id", agg="count_distinct"),
                Measure(name="median", sql="{e}.amount", agg="median"),
                Measure(name="p75", sql="{e}.amount", agg="p75"),
                Measure(name="p90", sql="{e}.amount", agg="p90"),
                Measure(name="p95", sql="{e}.amount", agg="p95"),
                Measure(name="views", sql="{e}.views", agg="sum"),
                Measure(name="share", sql="", agg="ratio", numerator="amount", denominator="views"),
            ],
            dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
            time_dimensions=[TimeDimension(name="at", sql="{e}.occurred_at")],
            rollups=[
                Rollup(
                    name="by_region",
                    physical_table=f"{self.prefix}event_totals",
                    dimensions=["region"],
                    measures=["amount", "n"],
                )
            ]
            if rollup
            else [],
        )
        return Catalog([cube]).compile(query)

    def rows(self, compiled: CompiledQuery) -> list[dict[str, Any]]:
        result = self.adapter.execute(compiled.sql, compiled.params)
        assert result.columns == compiled.columns
        assert [output.sql_alias for output in compiled.analysis.outputs] == compiled.columns
        return [dict(zip(result.columns, row, strict=True)) for row in result.rows]


@pytest.fixture(scope="module", params=list((Dialect.DUCKDB, Dialect.POSTGRES, Dialect.CLICKHOUSE)))
def backend(request: pytest.FixtureRequest) -> collections.abc.Iterator[_Backend]:
    dialect = request.param
    namespace = "semql_contract_" + uuid.uuid4().hex
    container: DockerContainer | None = None
    cleanup: collections.abc.Callable[[], None] | None = None
    database_created = False

    try:
        if dialect == Dialect.DUCKDB:
            connection = duckdb.connect(":memory:")
            adapter: Adapter = DuckDBAdapter(connection)
            execute: collections.abc.Callable[[str], object] = connection.execute
            prefix = ""
            cleanup = connection.close
        elif dialect == Dialect.POSTGRES:
            dsn = os.environ.get("SEMQL_CONFORMANCE_POSTGRES")
            if not dsn and os.environ.get("SEMQL_CONFORMANCE_TESTCONTAINERS") == "1":
                container = DockerContainer(_POSTGRES_IMAGE)
                container.with_env("POSTGRES_USER", "semql_test")
                container.with_env("POSTGRES_PASSWORD", "semql_test_password")
                container.with_env("POSTGRES_DB", "semql_test")
                container.with_exposed_ports(5432)
                container.start()
                ExecWaitStrategy(
                    ["pg_isready", "-h", "127.0.0.1", "-U", "semql_test", "-d", "semql_test"]
                ).with_startup_timeout(_CONTAINER_STARTUP_TIMEOUT_SECONDS).wait_until_ready(
                    container
                )
                host = container.get_container_host_ip()
                port = container.get_exposed_port(5432)
                dsn = (
                    f"host={host!r} port={port} dbname=semql_test "
                    "user=semql_test password=semql_test_password"
                )
            if not dsn:
                pytest.skip("set SEMQL_CONFORMANCE_POSTGRES or SEMQL_CONFORMANCE_TESTCONTAINERS=1")
            pg = psycopg.connect(dsn, autocommit=True, connect_timeout=10)

            def execute_postgres(sql: str) -> object:
                # Fixture SQL interpolates only its generated namespace, never caller input.
                # Pyright needs LiteralString; mypy treats it as str and flags this cast.
                return pg.execute(cast(LiteralString, sql))  # type: ignore[redundant-cast]

            def cleanup_postgres() -> None:
                try:
                    if database_created:
                        execute_postgres(f"DROP SCHEMA {namespace} CASCADE")
                finally:
                    pg.close()

            cleanup = cleanup_postgres
            execute_postgres(f"CREATE SCHEMA {namespace}")
            database_created = True
            adapter = DBAPIAdapter(pg)
            execute = execute_postgres
            prefix = namespace + "."
        else:
            url = os.environ.get("SEMQL_CONFORMANCE_CLICKHOUSE")
            if not url and os.environ.get("SEMQL_CONFORMANCE_TESTCONTAINERS") == "1":
                container = DockerContainer(_CLICKHOUSE_IMAGE)
                container.with_env("CLICKHOUSE_USER", "semql_test")
                container.with_env("CLICKHOUSE_PASSWORD", "semql_test_password")
                container.with_env("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1")
                container.with_exposed_ports(8123)
                container.start()
                (
                    HttpWaitStrategy(8123, path="/ping")
                    .for_status_code(200)
                    .with_startup_timeout(_CONTAINER_STARTUP_TIMEOUT_SECONDS)
                    .wait_until_ready(container)
                )
                host = container.get_container_host_ip()
                port = container.get_exposed_port(8123)
                url = f"http://{_url_host(host)}:{port}"
            if not url:
                pytest.skip(
                    "set SEMQL_CONFORMANCE_CLICKHOUSE or SEMQL_CONFORMANCE_TESTCONTAINERS=1"
                )
            root = _HTTPClickHouse(
                url,
                "default",
                username="semql_test" if container else None,
                password="semql_test_password" if container else None,
            )

            def cleanup_clickhouse() -> None:
                if database_created:
                    root.request(f"DROP DATABASE {namespace}", {})

            cleanup = cleanup_clickhouse
            root.request(f"CREATE DATABASE {namespace}", {})
            database_created = True
            ch = _HTTPClickHouse(
                url,
                namespace,
                username="semql_test" if container else None,
                password="semql_test_password" if container else None,
            )
            adapter = ch

            def execute_clickhouse(sql: str) -> object:
                return ch.request(sql, {})

            execute = execute_clickhouse
            prefix = namespace + "."

        if dialect == Dialect.CLICKHOUSE:
            execute(
                f"CREATE TABLE {prefix}events (id Int32, region String, amount Nullable(Float64), "
                "views Float64, occurred_at DateTime) ENGINE=Memory"
            )
            execute(
                f"CREATE TABLE {prefix}event_totals (region String, amount Nullable(Float64), "
                "n Int64) ENGINE=Memory"
            )
        elif dialect == Dialect.DUCKDB:
            execute(
                f"CREATE TABLE {prefix}events (id INTEGER, region VARCHAR, "
                "amount DOUBLE PRECISION, views DOUBLE PRECISION, occurred_at TIMESTAMP)"
            )
            execute(
                f"CREATE TABLE {prefix}event_totals "
                "(region VARCHAR, amount DOUBLE PRECISION, n BIGINT)"
            )
        else:
            execute(
                f"CREATE TABLE {prefix}events (id INTEGER, region VARCHAR, "
                "amount DOUBLE PRECISION, views DOUBLE PRECISION, occurred_at TIMESTAMP)"
            )
            execute(
                f"CREATE TABLE {prefix}event_totals "
                "(region VARCHAR, amount DOUBLE PRECISION, n BIGINT)"
            )
        execute(
            f"INSERT INTO {prefix}events VALUES "
            "(1,'A',10,100,'2024-01-01 00:00:00'),"
            "(2,'A',20,100,'2024-01-01 00:00:00'),"
            "(3,'A',30,100,'2024-01-03 00:00:00'),"
            "(4,'B',90,100,'2024-01-03 00:00:00'),"
            "(5,'Z',NULL,100,'2024-01-04 00:00:00')"
        )
        if dialect == Dialect.CLICKHOUSE:
            execute(f"CREATE TABLE {prefix}comparisons AS {prefix}events ENGINE=Memory")
        else:
            execute(f"CREATE TABLE {prefix}comparisons AS SELECT * FROM {prefix}events WHERE FALSE")
        execute(
            f"INSERT INTO {prefix}comparisons VALUES "
            "(1,'negative',-20,100,'2023-12-01 00:00:00'),"
            "(2,'negative',10,100,'2024-01-01 00:00:00'),"
            "(3,'zero',0,0,'2023-12-01 00:00:00'),"
            "(4,'zero',5,0,'2024-01-01 00:00:00'),"
            "(5,'positive',20,100,'2023-12-01 00:00:00'),"
            "(6,'positive',30,100,'2024-01-01 00:00:00')"
        )
        execute(f"INSERT INTO {prefix}event_totals VALUES ('A',60,3),('B',90,1),('Z',NULL,1)")
        if dialect == Dialect.CLICKHOUSE:
            execute(f"CREATE TABLE {prefix}regions (id String, label String) ENGINE=Memory")
            execute(f"CREATE TABLE {prefix}worklog (region String, hours Float64) ENGINE=Memory")
        else:
            execute(f"CREATE TABLE {prefix}regions (id VARCHAR, label VARCHAR)")
            execute(f"CREATE TABLE {prefix}worklog (region VARCHAR, hours DOUBLE PRECISION)")
        execute(
            f"INSERT INTO {prefix}regions VALUES "
            "('A','Shared'),('B','Shared'),('Z','Null'),"
            "('C','Other'),('Absent','Absent'),('Hidden','Hidden')"
        )
        execute(f"INSERT INTO {prefix}worklog VALUES ('A',8),('C',5),('Hidden',99)")
        yield _Backend(dialect=dialect, adapter=adapter, prefix=prefix)
    finally:
        try:
            if cleanup is not None:
                cleanup()
        finally:
            if container is not None:
                container.stop()


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def test_aggregate_numeric_membership_and_hidden_operands(backend: _Backend) -> None:
    query = SemanticQuery(
        dimensions=["events.region"],
        measures=["events.amount", "events.n", "events.mean", "events.minimum", "events.maximum"],
        derived_measures=[
            InlineDerived(name="rate", op="ratio", operands=["events.amount", "events.views"])
        ],
    )
    compiled = backend.compile(query)
    rows = {row["region"]: row for row in backend.rows(compiled)}
    assert set(rows) == {"A", "B", "Z"}
    assert float(rows["A"]["amount"]) == 60
    assert int(rows["A"]["n"]) == 3
    assert float(rows["A"]["mean"]) == 20
    assert float(rows["A"]["minimum"]) == 10
    assert float(rows["A"]["maximum"]) == 30
    assert float(rows["A"]["rate"]) == pytest.approx(0.2)
    assert rows["Z"]["amount"] is None
    assert rows["Z"]["rate"] is None
    assert int(rows["Z"]["n"]) == 1
    assert any(node.catalog_ref == "events.views" for node in compiled.analysis.nodes)


def test_aliases_catalog_ratio_and_output_order(backend: _Backend) -> None:
    query = SemanticQuery(
        dimensions=["events.region"],
        measures=["events.share", "events.mean", "events.amount"],
        aliases={"area": "events.region", "rate": "events.share", "net": "events.amount"},
    )
    compiled = backend.compile(query)
    assert compiled.columns == ["area", "rate", "mean", "net"]
    rows = {row["area"]: row for row in backend.rows(compiled)}
    assert float(rows["A"]["rate"]) == pytest.approx(0.2)
    assert float(rows["B"]["rate"]) == pytest.approx(0.9)
    assert float(rows["A"]["net"]) == 60


def test_distinct_and_median(backend: _Backend) -> None:
    compiled = backend.compile(
        SemanticQuery(
            dimensions=["events.region"],
            measures=[
                "events.distinct_ids",
                "events.median",
                "events.p75",
                "events.p90",
                "events.p95",
            ],
        )
    )
    rows = {row["region"]: row for row in backend.rows(compiled)}
    assert int(rows["A"]["distinct_ids"]) == 3
    assert float(rows["A"]["median"]) == 20
    assert int(rows["B"]["distinct_ids"]) == 1
    assert float(rows["B"]["median"]) == 90
    assert float(rows["A"]["p75"]) == pytest.approx(25)
    assert float(rows["A"]["p90"]) == pytest.approx(28)
    assert float(rows["A"]["p95"]) == pytest.approx(29)


def test_stored_count_rollup_matches_base(backend: _Backend) -> None:
    query = SemanticQuery(dimensions=["events.region"], measures=["events.amount", "events.n"])
    base = backend.compile(query)
    rolled = backend.compile(query, rollup=True)
    assert rolled.applied_rollup == "by_region"
    for artifact in (base, rolled):
        rows = {row["region"]: row for row in backend.rows(artifact)}
        assert {key: int(row["n"]) for key, row in rows.items()} == {"A": 3, "B": 1, "Z": 1}
        assert float(rows["A"]["amount"]) == 60


def test_dense_time_fill_executes_with_untouched_bindings(backend: _Backend) -> None:
    compiled = backend.compile(
        SemanticQuery(
            measures=["events.amount", "events.n"],
            time_dimension=TimeWindow(
                dimension="events.at",
                granularity="day",
                range=("2024-01-01", "2024-01-05"),
                fill_nulls_with=0,
            ),
        )
    )
    rows = backend.rows(compiled)
    time_alias = compiled.columns[0]
    by_day = {str(row[time_alias])[:10]: (float(row["amount"]), int(row["n"])) for row in rows}
    assert by_day == {
        "2024-01-01": (30, 2),
        "2024-01-02": (0, 0),
        "2024-01-03": (120, 2),
        "2024-01-04": (0, 1),
    }


def test_compare_missing_prior_null_and_delta_policy(backend: _Backend) -> None:
    compiled = backend.compile(
        SemanticQuery(
            measures=["events.amount"],
            time_dimension=TimeWindow(dimension="events.at", range=("2024-01-01", "2024-01-05")),
            compare=CompareWindow(mode="previous_period"),
        )
    )
    rows = backend.rows(compiled)
    assert len(rows) == 1
    assert float(rows[0]["amount_current"]) == 150
    assert rows[0]["amount_prior"] is None
    assert float(rows[0]["amount_delta"]) == 150
    assert rows[0]["amount_pct_change"] is None


@pytest.mark.parametrize("mode", ["distributive", "raw_rows"])
def test_real_source_federation_aliases_and_average(backend: _Backend, mode: str) -> None:
    if backend.dialect == Dialect.DUCKDB:
        pytest.skip("cross-backend cells use PostgreSQL/ClickHouse sources with DuckDB merge")
    from typing import cast

    from semql.federate import FederationMode

    local = duckdb.connect(":memory:")
    try:
        local.execute("CREATE TABLE regions (id VARCHAR, label VARCHAR)")
        local.execute("INSERT INTO regions VALUES ('A','Shared'),('B','Shared'),('Z','Null')")
        events = Cube(
            name="events",
            table=f"{backend.prefix}events",
            alias="e",
            dialect=backend.dialect,
            measures=[
                Measure(name="amount", sql="{e}.amount", agg="sum"),
                Measure(name="mean", sql="{e}.amount", agg="avg"),
                Measure(name="views", sql="{e}.views", agg="sum"),
                Measure(name="share", sql="", agg="ratio", numerator="amount", denominator="views"),
            ],
            dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
            joins=[Join(to="regions", on="{e}.region = {r}.id", relationship="many_to_one")],
        )
        regions = Cube(
            name="regions",
            table="regions",
            alias="r",
            dialect=Dialect.DUCKDB,
            primary_key="id",
            dimensions=[
                Dimension(name="id", sql="{r}.id", type="string"),
                Dimension(name="label", sql="{r}.label", type="string"),
            ],
        )
        measures = ["events.mean", "events.amount"]
        if mode == "raw_rows":
            measures.insert(0, "events.share")
        query = SemanticQuery(
            dimensions=["regions.label"],
            measures=measures,
            aliases={
                "area": "regions.label",
                "net": "events.amount",
                **({"rate": "events.share"} if mode == "raw_rows" else {}),
            },
        )
        catalog_context = CatalogContext(namespace="conformance", semantic_revision="1")
        native_regions = regions.model_copy(
            update={"dialect": backend.dialect, "table": f"{backend.prefix}regions"}
        )
        native = compile_query(
            query,
            {"events": events, "regions": native_regions},
            catalog_context=catalog_context,
        )
        native_rows = {row["area"]: row for row in backend.rows(native)}
        plan = compile_federated_query(
            query,
            {"events": events, "regions": regions},
            mode=cast(FederationMode, mode),
            catalog_context=catalog_context,
        )
        engine = Engine(merge_engine=DuckDBMergeEngine())
        engine.register(backend.dialect, backend.adapter)
        engine.register(Dialect.DUCKDB, DuckDBAdapter(local))
        result = engine.run(plan)
        assert result.columns == (
            ["area", "rate", "mean", "net"] if mode == "raw_rows" else ["area", "mean", "net"]
        )
        rows = {row[0]: dict(zip(result.columns, row, strict=True)) for row in result.rows}
        assert set(rows) == {"Shared", "Null"}
        assert float(rows["Shared"]["mean"]) == 37.5
        assert float(rows["Shared"]["net"]) == 150
        if mode == "raw_rows":
            assert float(rows["Shared"]["rate"]) == pytest.approx(0.375)
        assert result.analysis == plan.analysis
        assert rows == native_rows
        assert (
            compare_analysis(
                native.analysis, result.analysis, scope="result", bound_context_equal=True
            ).outcome
            == "equivalent"
        )
    finally:
        local.close()


@pytest.mark.parametrize("exclude_b", [False, True])
def test_real_symmetric_observed_population(backend: _Backend, exclude_b: bool) -> None:
    local = duckdb.connect(":memory:")
    try:
        local.execute("CREATE TABLE regions (id VARCHAR)")
        local.execute("INSERT INTO regions VALUES ('A'),('B'),('C'),('Z'),('Absent'),('Hidden')")
        local.execute("CREATE TABLE worklog (region VARCHAR, hours DOUBLE)")
        local.execute("INSERT INTO worklog VALUES ('A',8),('C',5),('Hidden',99)")
        regions = Cube(
            name="regions",
            table="regions",
            alias="r",
            dialect=Dialect.DUCKDB,
            primary_key="id",
            security_sql="{r}.id <> {ctx.excluded_region}",
            security_ctx_keys=["excluded_region"],
            dimensions=[Dimension(name="id", sql="{r}.id", type="string")],
        )
        events = Cube(
            name="events",
            table=f"{backend.prefix}events",
            alias="e",
            dialect=backend.dialect,
            measures=[Measure(name="amount", sql="{e}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
            joins=[Join(to="regions", on="{e}.region = {r}.id", relationship="many_to_one")],
        )
        worklog = Cube(
            name="worklog",
            table="worklog",
            alias="w",
            dialect=Dialect.DUCKDB,
            measures=[Measure(name="hours", sql="{w}.hours", agg="sum")],
            dimensions=[Dimension(name="region", sql="{w}.region", type="string")],
            joins=[Join(to="regions", on="{w}.region = {r}.id", relationship="many_to_one")],
        )
        query = SemanticQuery(
            dimensions=["regions.id"],
            measures=["events.amount", "worklog.hours"],
            filters=[Filter(dimension="events.region", op="neq", values=["B"])]
            if exclude_b
            else [],
        )
        catalog_context = CatalogContext(namespace="conformance", semantic_revision="1")
        context = {"ctx.excluded_region": "Hidden"}
        viewer = AuthContext(viewer_id="fixture-viewer")
        native_catalog = {
            cube.name: cube.model_copy(
                update={"dialect": backend.dialect, "table": f"{backend.prefix}{cube.name}"}
            )
            for cube in (events, regions, worklog)
        }
        native = compile_query(
            query, native_catalog, context=context, viewer=viewer, catalog_context=catalog_context
        )
        expected: dict[str, tuple[float | None, float | None]] = {
            "A": (60, 8),
            "C": (None, 5),
            "Z": (None, None),
        }
        if not exclude_b:
            expected["B"] = (90, None)
        assert {
            row["id"]: (row["amount"], row["hours"]) for row in backend.rows(native)
        } == expected
        if backend.dialect == Dialect.DUCKDB:
            return
        plan = compile_federated_query(
            query,
            {cube.name: cube for cube in (events, regions, worklog)},
            context=context,
            viewer=viewer,
            catalog_context=catalog_context,
        )
        engine = Engine(merge_engine=DuckDBMergeEngine())
        engine.register(backend.dialect, backend.adapter)
        engine.register(Dialect.DUCKDB, DuckDBAdapter(local))
        result = engine.run(plan)
        assert {row[0]: tuple(row[1:]) for row in result.rows} == expected
        assert (
            compare_analysis(
                native.analysis, result.analysis, scope="result", bound_context_equal=True
            ).outcome
            == "equivalent"
        )
    finally:
        local.close()


def test_real_compare_negative_zero_and_percent_scale(backend: _Backend) -> None:
    compiled = backend.compile(
        SemanticQuery(
            dimensions=["events.region"],
            measures=["events.amount"],
            time_dimension=TimeWindow(dimension="events.at", range=("2024-01-01", "2024-02-01")),
            compare=CompareWindow(mode="explicit", range=("2023-12-01", "2024-01-01")),
        ),
        table="comparisons",
    )
    rows = {row["region"]: row for row in backend.rows(compiled)}
    assert float(rows["negative"]["amount_current"]) == 10
    assert float(rows["negative"]["amount_prior"]) == -20
    assert float(rows["negative"]["amount_delta"]) == 30
    assert rows["negative"]["amount_pct_change"] is None
    assert rows["zero"]["amount_pct_change"] is None
    assert float(rows["positive"]["amount_pct_change"]) == 50


def test_real_ratio_zero_denominator_is_null(backend: _Backend) -> None:
    compiled = backend.compile(
        SemanticQuery(dimensions=["events.region"], measures=["events.share"]),
        table="comparisons",
    )
    rows = {row["region"]: row for row in backend.rows(compiled)}
    assert rows["zero"]["share"] is None
    assert float(rows["negative"]["share"]) == pytest.approx(-0.05)
    assert float(rows["positive"]["share"]) == pytest.approx(0.25)
