# Changelog

All notable changes to the `semql` packages are recorded here. The eight
packages version in lockstep. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 0.6.0
(0.5.0 and earlier predate it — see the git history).

## [0.6.0]

### Added

- **Cross-source Auto-Planner.** Routes filter-only cross-backend filters,
  choosing between a `semi_join` and a `bridge_merge` by operator and size
  hint, with cost-based selection between the two strategies.
- **Cross-backend primitives.** A semi-join value-list primitive (filter a
  backend by a value list from another, no join) and cross-backend
  symmetric aggregation in the federated compile path.
- **Resolution stage.** Maps free-text filter phrases to canonical keys.
- `QualifiedRef` type for `cube.field` references, with `_resolve` routed
  through it; `JoinPath` type carrying ambiguity + fan-out flags.
- Cross-cube `InlineDerived` over reachable joins.
- A compile-path warnings channel.
- `SemanticQuery.tool_json_schema` and a `Bedrock Converse` root-`$ref`
  schema flattener (moved to core), for tool-schema-constrained LLM APIs.
- Field-ref codegen in the CLI.
- MCP Apps visualisation surface (beta) and a `ShapeStats` decision hook.

### Changed

- Repository moved to the `semql-io` GitHub org; project URLs now point at
  `github.com/semql-io/semql`.
- `DerivedTable.sql` now refuses a top-level `WITH`.

### Fixed

- Filter-only cross-source merge now INNER-joins at the bridge (was LEFT,
  which inflated results) — Gap C.
- `semql-mcp` now pins `semql-prompt` in lockstep (`>=0.6.0,<0.7`); the
  prior `<0.3` pin made the published `semql-mcp` unresolvable.

### Security

- Resolved 13 findings from a security audit: THREAT_MODEL restructure plus
  code fixes across the viewer-filter MCP/prompt surfaces and plan auth.
- Added an auth-containment property and bijective bind-params to the test
  suite; documented the accepted MCP tool-listing metadata disclosure.

### Testing / infra

- Tiered CI (fast PR lane / merge gate / nightly) with a per-package
  coverage ratchet; branch coverage and test-tier markers.
- DuckDB-backed metamorphic test suite; pathology guards for CNF blowup,
  deep nesting, and prompt scale; dialect-validity pinned to every backend.
- Module-boundary enforcement (`tach`) and a frozen-model discipline lint.
- `actionlint` + `zizmor` workflow linting and weekly Dependabot updates.
- OIDC Trusted Publishing release workflow (see `docs/RELEASING.md`).
