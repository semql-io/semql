# Adapter conformance evidence matrix

This is a bounded record of executed test coverage, not a universal backend support promise. It separates actual database drivers from SDK-shaped stand-ins and compiler-only coverage. A collected node is not execution evidence; required external fixtures fail rather than silently skip.

## Reproducible commands and scope

From the repository root, after dependencies are installed:

```sh
uv run pytest -q --no-cov packages/semql-engine/tests/test_adapter_conformance.py
uv run pytest -q --no-cov packages/semql-engine/tests/test_async_dbapi_adapter.py packages/semql-engine/tests/test_async_bq_adapter.py packages/semql-engine/tests/test_async_ch_adapter.py
uv run pytest -q --no-cov packages/semql-engine/tests/test_semantic_backend_conformance.py
SEMQL_CONFORMANCE_REQUIRED=1 SEMQL_CONFORMANCE_TESTCONTAINERS=1 uv run pytest -q --no-cov packages/semql-engine/tests/test_semantic_backend_conformance.py
```

The first command binds the common observable pack to actual in-process DuckDB and actual DuckDB beneath the PEP-249 surface, plus async DuckDB and async DB-API wrappers using that driver. The common cases independently specify expected labels, optional physical schema, result width, duplicate-preserving unordered rows, explicit order, native named binding with two distinct values (including an apostrophe), exact Decimal values, NULL, all-NULL, empty, and a Decimal aggregate computed from input fixture values. Deterministic calls run twice. Fault-provocation cases verify that the shared assertions reject changed, missing, duplicated, incorrectly ordered, mislabeled, and schema-mismatched results; these are harness tests, not claims about a faulty backend.

The second command exercises async DB-API, BigQuery SDK-shaped fake, and ClickHouse SDK-shaped fake test modules. The BigQuery and ClickHouse DuckDB-backed end-to-end cases translate dialect placeholders at their fake client boundary; their execution is explicitly a stand-in, not native backend evidence. Other cases exercise the adapter's handing-off of query/parameter shapes and synthetic schema/row results, not a live SDK service.

The third command runs all fixture-selected cases. DuckDB always binds the actual driver; PostgreSQL uses the configured psycopg connection and ClickHouse uses the module's HTTP/native-parameter adapter. Missing external configurations skip those backend fixture instances. The fourth command makes the external tier required and requests pinned local Testcontainers; absent services, container startup failures, and connection failures fail rather than turn into skips. To use externally managed services instead, set `SEMQL_CONFORMANCE_POSTGRES` and `SEMQL_CONFORMANCE_CLICKHOUSE`, set `SEMQL_CONFORMANCE_REQUIRED=1`, omit `SEMQL_CONFORMANCE_TESTCONTAINERS`, and run the same module command. This requires both external fixtures; each failed fixture is a failure.

The shared pack's actual-backend node is:

```text
packages/semql-engine/tests/test_semantic_backend_conformance.py::test_adapter_shared_pack_against_real_backend_driver
```

Its `backend` fixture expands into DuckDB, PostgreSQL, and ClickHouse instances. The external database cases are executed only if their configured service or Testcontainers instance is available; the required-tier command is the reproducible evidence gate, not mere node collection. The existing suite also includes real single-backend boolean, aggregates/hidden operands, aliases/output order, rollup, dense time, comparison, federation (`distributive` and `raw_rows`), symmetric population, Decimal, predicate boundaries, partition ranking, and cohort cases. The cohort cross-backend selection/measurement case uses one actual external backend plus an in-process DuckDB measurement source. The real-source federation cases similarly use an actual PostgreSQL or ClickHouse source and DuckDB merge source. Their independent expected-value assertions are the numeric/membership oracle; native-versus-federated equality is an additional strategy comparison, not the oracle.

## Adapter inventory

| Built-in implementation / boundary | Test binding | Evidence class | Observed execution |
|---|---|---|---|
| `DuckDBAdapter` | `test_shared_pack_actual_duckdb_driver[duckdb-driver]`; semantic backend shared pack | Actual DuckDB driver | Passed |
| `DBAPIAdapter` | `test_shared_pack_actual_duckdb_driver[dbapi-duckdb-driver]`; semantic backend shared pack | Actual DuckDB beneath PEP-249; actual psycopg/PostgreSQL | Passed |
| `AsyncDuckDBAdapter` | `test_async_shared_pack_actual_duckdb_driver[async-duckdb-driver]` | Actual DuckDB, async dispatch | Passed |
| `AsyncDBAPIAdapter` | `test_async_shared_pack_actual_duckdb_driver[async-dbapi-duckdb-driver]` | Actual DuckDB beneath PEP-249 | Passed |
| `AsyncBigQueryAdapter` | `test_async_shared_pack_sdk_boundary_with_duckdb_stand_in`; `test_async_bq_adapter.py` | Real adapter, SDK-shaped fake over DuckDB; no BigQuery service | Passed at stated stand-in boundary |
| `AsyncClickHouseAdapter` | `test_async_shared_pack_sdk_boundary_with_duckdb_stand_in`; `test_async_ch_adapter.py` | Real adapter, SDK-shaped fake over DuckDB; no native async ClickHouse service | Passed at stated stand-in boundary |
| `to_async_adapter` | `test_async_shared_pack_actual_duckdb_driver[wrapped-sync-driver]` | Sync DuckDB adapter through the public async wrapper | Passed |

The matrix inventory is based on actual adapter instances constructed in tests and public package registration (not a tautological source-text assertion). Existing SQL-only/compiler shapes do not exercise drivers and are not listed as driver coverage. Existing cross-backend engine tests using two DuckDB connections to imitate different dialects remain stand-ins; only the real-source cases above count as PostgreSQL/ClickHouse database evidence.

## Query-shape and execution record

| Query shape / combination | Relevant node or node family | Fixture class | Execution status |
|---|---|---|---|
| Adapter bag/order/binding/Decimal/NULL/empty/aggregate pack | `test_shared_pack_actual_duckdb_driver`; `test_adapter_shared_pack_against_real_backend_driver` | Actual DuckDB, PostgreSQL and ClickHouse | Passed |
| Adapter contract assertion sensitivity | `test_shared_assertions_reject_consumer_visible_faults` | In-memory records; deliberate corruption, including numeric type coercion | Passed |
| Async DuckDB/DB-API/wrapped-sync pack | `test_async_shared_pack_actual_duckdb_driver` | Actual DuckDB under async wrappers | Passed |
| BigQuery shared pack and parameter/schema/async behavior | `test_async_shared_pack_sdk_boundary_with_duckdb_stand_in`; `test_async_bq_adapter.py` | SDK fake over DuckDB | Passed; not native service evidence |
| ClickHouse shared pack and parameter/schema/async behavior | `test_async_shared_pack_sdk_boundary_with_duckdb_stand_in`; `test_async_ch_adapter.py` | SDK fake over DuckDB | Passed; not native async SDK evidence |
| Semantic backend broad contract | `test_semantic_backend_conformance.py` | Actual DuckDB, PostgreSQL and ClickHouse | Passed at named fixture boundaries |
| PostgreSQL/ClickHouse source + DuckDB merge | `test_real_source_federation_aliases_and_average[distributive]`, `[raw_rows]`; symmetric aggregation | Actual PostgreSQL/ClickHouse + DuckDB | Passed; DuckDB-only duplicate tier intentionally skipped |
| Cohort selection vs measurement across sources | `test_cohort_cross_backend_selection_and_measurement` | Actual PostgreSQL/ClickHouse selection + DuckDB measurement | Passed; DuckDB-only duplicate tier intentionally skipped |
| Compiler-only / SQL parse shapes elsewhere in module | SQL construction and parsing tests in `test_semantic_backend_conformance.py` | No adapter execution where explicitly SQL-only | Collection does not imply driver execution |

The shared actual-backend node includes eight named shapes (`bag`, `ordered`, `bound`, `rebound`, `decimal-null`, `all-null`, `empty`, `aggregate`). ClickHouse HTTP JSON Decimal strings are decoded from authoritative column types with `Decimal(text)`, never binary float; incidental trailing-zero formatting is not a contract. The required local external tier used `postgres:16.14-alpine` and `clickhouse/clickhouse-server:26.8.2.7`; DuckDB was 1.5.4.

Residual gaps include a live BigQuery service, the native asynchronous ClickHouse SDK/service, other DB-API drivers, driver-level cancellation guarantees, and heterogeneous combinations beyond the PostgreSQL/ClickHouse plus DuckDB combinations named above. The pack does not assert a global backend capability list.
