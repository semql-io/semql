# Threat Model

## Overview

SemQL is a library boundary, not an end-to-end authentication or session system. Its security promise is to preserve invariants when untrusted or semi-trusted callers submit semantic queries, natural-language-derived plans, MCP tool arguments, saved query names, lookup terms, viewer identities, or database schema metadata.

Protected assets:

- Tenant-isolated data
- Authorization policy decisions
- Parameterized SQL boundaries
- Database credentials held by host applications
- Generated prompt/tool surfaces shown to LLMs
- Operator trust in validation and introspection output

## Trust Boundaries and Assumptions

### Trust Boundaries

- **Query caller → SemQL compiler.** `SemanticQuery`, filters, dimensions, measures, time windows, pagination, saved query identifiers, lookup terms, and MCP arguments can be attacker-controlled when SemQL is embedded in chat, API, or MCP workflows.
- **LLM output → typed planning models.** Generated `QueryPlan`, `SemanticQuery`, drilldown suggestions, and presentation choices are untrusted even when Pydantic validation succeeds.
- **Viewer/auth context → catalog controls.** `AuthContext(viewer_id, roles, metadata)` is request-scoped and must be supplied by a trusted application boundary. If a caller can forge it, SemQL authorization cannot protect data.
- **Catalog author → query caller.** Cube definitions, field SQL templates, join definitions, `security_sql`, scope function names, saved queries, lookup metadata, descriptions, and prompt text are operator/developer-controlled. In SaaS or multi-tenant metadata products, catalog content may be partly tenant-controlled and must then be treated as attacker-influenced.
- **SemQL compiler → SQL backend.** Compiled SQL and bound params cross into Postgres, ClickHouse, DuckDB, BigQuery, Snowflake, Redshift, Trino, Databricks, and experimental dialects. SQL text must preserve parameterization and authorization predicates across dialect rendering.
- **Execution engine → database adapters.** `semql-engine` runs compiled/federated plans through adapter objects and may process rows from multiple backends. Adapter objects and connections are trusted host-application dependencies; row data is not trusted for rendering or downstream display.
- **MCP client → MCP server.** MCP clients can invoke tools directly. If the server is run with an executor, it crosses from compile-only semantics into live data access.
- **Introspection/validation tools → live databases.** Operator credentials, database schemas, table/column names, types, and probe errors cross into generated catalog code or validation reports.
- **Local developer tooling and docs** are lower-priority except where scripts interact with package publishing, live credentials, or generated code consumed at runtime.

### Assumptions

- Host applications authenticate users and construct `AuthContext` correctly. SemQL enforces authorization against that context but does not prove its origin.
- Catalog definitions and scope functions are trusted code by default. If a product allows user-authored cubes or SQL snippets, those snippets become privileged inputs and severity increases sharply.
- Database credentials and network reachability are held by the embedding app, MCP server, validation CLI, introspection CLI, or adapter — not by the core compiler.
- The compiler should be pure: no network, no database I/O, no time-of-day, no mutable global authorization state.
- Authorization visibility checks and row-level predicates must fail closed: unauthorized cubes are not silently exposed in prompts, discovery tools, or compiled SQL.
- Values from queries, viewer metadata, lookup terms, and scope contexts must bind as params rather than be interpolated as SQL literals.

## Attack Surface

- **SQL generation and rendering** — `packages/semql/src/semql/compile.py`, `backend.py`, `dialect.py`, `federate.py`, `semijoin.py`, `partition.py`, `parse.py`, `rewrite.py`, and related resolver/model code. Relevant attacks: SQL injection through field references, filter values, time ranges, raw SQL parser/rewrite paths, unsafe identifier handling, parameter collision, dialect-specific literal escaping mistakes, and generated SQL that changes query semantics across backends.
- **Authorization enforcement** — `packages/semql/src/semql/catalog.py`, `introspect.py`, `_resolve.py`, `compile.py`, `model.py`, and MCP/prompt projections. Relevant attacks: unauthorized cube discovery, inconsistent visibility filtering between prompt/MCP/compiler, row-level `ScopePredicate` placement errors, bypass by outer ORs/joins, missing scope on derived tables or joins, and policy fail-open behavior.
- **MCP tooling** — `packages/semql-mcp/src/semql_mcp/server.py` and visualization helpers. Relevant attacks: tool calls bypassing compile-only restrictions, executor mode performing writes or multi-statements, saved-query lookup confusion, unauthorized entity/lookup tools, prompt or HTML injection in visualization responses, and excessive data exposure through tool schemas.
- **Prompt/tool projection** — `packages/semql-prompt/src/semql_prompt/*`. Relevant attacks: prompt injection through catalog descriptions/metadata, leaking hidden cubes or fields into planner prompts, and unsafe tool descriptions that induce callers to execute beyond the viewer's allowed surface.
- **Execution and federation** — `packages/semql-engine/src/semql_engine/*`. Relevant attacks: executing non-read-only SQL, merge-spec injection, unsafe temporary table names, cross-backend row mixing that loses authorization context, adapter parameter binding mistakes, cache key confusion, and uncontrolled resource consumption.
- **Introspection and validation** — `packages/semql-introspect/src/semql_introspect/*` and `packages/semql-validate-db/src/semql_validate_db/*`. Relevant attacks: generated Python/catalog injection from malicious database identifiers, unsafe probe SQL, credential leakage in errors, and misleading validation that lets drifted or unscoped catalogs deploy.
- **Auth helpers** — `packages/semql-auth/src/semql_auth/auth.py`. Relevant attacks: accepting weak token claims, role confusion, missing issuer/audience checks, and metadata trust confusion.

## Mitigations

- The core compiler produces `CompiledQuery(sql, params, columns)` with values bound through a `bind(value, dim_type)` closure rather than literal interpolation.
- Catalog and spec value types are frozen Pydantic models, reducing mutation-after-validation and shared-state hazards.
- `AuthContext`, `Cube.required_roles`, `Catalog.policy`, and scope functions are first-class concepts. Unauthorized cubes must be refused loudly; row-level scope predicates must be injected inside alias subqueries.
- Prompt and MCP surfaces are expected to shrink to the viewer's authorized catalog surface, reducing LLM-side discovery of forbidden cubes.
- Explicit tests cover SQL injection, read-only statements, auth, field masking, compile plan trust, raw SQL, MCP auth, MCP read-only guard, prompt data fences, grounding, cache hazards, and merge parameter binding.

## Attacker Stories

### Realistic

- A tenant or chat user submits crafted filters, lookup terms, saved query names, or LLM-generated plans that try to alter SQL semantics, select unauthorized dimensions, bypass row-level predicates, force expensive queries, or reveal hidden catalog structure.
- A malicious or compromised MCP client calls lower-level tools directly, tries executor mode with write statements, or probes differences between validate/explain/query errors to infer hidden cubes.
- A catalog author accidentally uses unsafe raw SQL or a scope function with unbound context values; in a self-service catalog product, a tenant author can intentionally weaponize SQL templates, descriptions, or database identifiers.
- A database schema controlled by an adversary is introspected into generated catalog code containing hostile identifiers or descriptions.
- A prompt-injection string in catalog descriptions or saved-query text tries to override LLM instructions or expose hidden fields through generated prompt fragments.

### Out of Scope

- Remote code execution in the core compiler from query values alone is out of scope unless code emission, eval-like behavior, Graphviz/browser rendering, or database adapter execution is involved.
- Web-session vulnerabilities (CSRF, cookies, browser XSS, password reset) are not central unless a host application adds HTTP endpoints around the MCP/server APIs.
- Network SSRF is lower priority in the core packages, which primarily transform local objects, but becomes relevant for live database adapters, introspection, validation, cloud clients, and LLM/provider integration code.

## Severity Calibration

### Critical

- A query caller bypasses cube authorization or row-level scope and reads another tenant's data through the compiler, MCP query tools, prompt-generated plans, or execution engine.
- Attacker-controlled query/filter/context values are rendered as executable SQL and can perform arbitrary SELECT/DDL/DML or exfiltrate unrelated data across a supported backend.
- MCP executor/read-only controls are bypassed to run writes or multi-statements against a live production database.
- Introspection or ERD/rendering paths allow code or command execution from attacker-controlled database/catalog identifiers in a realistic operator workflow.

### High

- Hidden cubes, fields, saved queries, lookup values, or metadata leak through prompts, tool schemas, introspection APIs, MCP errors, or validation paths despite viewer restrictions.
- Scope predicates are applied outside a subquery or omitted for derived/federated/semijoin/time-spine paths, enabling semantic bypass for some query shapes.
- Parameter binding is inconsistent in a dialect or merge adapter, allowing injection or parameter confusion for a material subset of supported backends.
- Auth helper token validation accepts wrong issuer/audience/algorithm or confuses roles in a way likely to grant privileged catalog access when used as documented.

### Medium

- Catalog or database identifiers break generated SQL, generated Python catalogs, Graphviz DOT, HTML visualization, or prompt/tool text in ways that cause denial of service, misleading output, or limited injection without data exfiltration.
- Read-only guards reject common writes but miss a less-common backend-specific write form in a path normally used by trusted operators rather than public users.
- Prompt injection through catalog text can influence an LLM planner but does not itself bypass compiler/MCP authorization or SQL parameterization.
- Resource-exhaustion issues allow untrusted callers to generate very large prompts, query plans, joins, time spines, or merge workloads without configured limits.

### Low

- Developer-only scripts, benchmarks, generated docs, or examples mishandle local input without touching credentials, live databases, published packages, or runtime library paths.
- Error messages reveal internal file/module names or non-sensitive schema details to an already-authorized operator.
- Validation or linting produces confusing diagnostics that could hide configuration mistakes but does not change runtime enforcement.
