# Boolean semantics repair — 2026-10-23

## Goal and baseline

Repair GitHub issues #13 and #14 and the related class of Boolean predicate loss and precedence errors. Start at origin/main `9932c22fd3e74586b81053dfe7a2e3c4c9b50875` in the isolated `fix/boolean-semantics` worktree; leave the original dirty checkout unchanged. User authorized a lockstep patch bump, branch push, PR, and merge, with both issues closed by the PR. No package publication or release tag is authorized.

## Contracts

- SQL parsing must preserve mixed AND/OR grouping and supported negation, independently of SQL emission. Every accepted predicate must be represented; unsupported leaves or Boolean placements must produce diagnostics (strict mode raises), never disappear silently.
- Preserve SQL three-valued semantics. Do not reinterpret arbitrary SQL expressions as scalar literals. LIKE is not interchangeable with SemQL's escaped, case-insensitive `contains`; reject non-representable LIKE predicates rather than invert or reinterpret them.
- Keep the existing semantic-SQL time-window convention for top-level BETWEEN. Do not move a window out of OR/NOT, overwrite multiple windows, or use the time-window slot for a non-time field without an explicit error.
- HAVING remains an implicit-AND measure-filter list. Flatten only equivalent conjunctions; reject predicates that cannot be represented in that contract.
- Boolean SQL AST composition must preserve precedence at every existing construction site, including compiler, mutation, and backend helpers. Use SQLGlot's grouping-aware combinators instead of ad hoc wrapping or raw binary nodes where appropriate.
- Mandatory tenancy/security/scope constraints remain inside isolation subqueries. No public schema changes, new filter operators, unrelated arithmetic refactors, or policy changes.

## Implementation slices

1. Parser: replace cross-operator flattening with structural recursive conversion; preserve compound NOT, validate predicate leaves and all tree references; fail closed on unsupported WHERE/HAVING forms. Extend parser tests with independent tree/SQL result oracles.
2. Emitter: repair grouping-aware Boolean composition in all matching core sites. Extend execution-based compiler, security/mutation/backend tests as warranted by the matched sites.
3. Integration: extend real-backend predicate conformance if existing fixtures support it, review duplicate Boolean construction sites and affected callers, update documentation and bump all eight workspace packages from 0.11.0 to 0.11.1 via `just bump patch`, regenerate the lockfile.

## Verification

- Independent parser and direct semantic-query reproductions for both issues; reject unsupported syntax explicitly.
- Exhaustive small truth fixtures, including NULL, nested mixed AND/OR, NOT/double NOT, IN/NOT IN and IS NULL/IS NOT NULL. Compare exact row membership to independent SQL, not emitted SQL spelling or parser/compiler agreement.
- Exercise flat-filter/segment/time constraints and policy isolation with excluded rows.
- Run the full affected test modules and native `just check` (format, lint, mypy, pyright, structural boundaries, full suite), with real PostgreSQL/ClickHouse conformance where local Docker supports the existing suite.
- Run public parse/compile/execute smoke paths against actual DuckDB; verify all eight package versions and internal ranges after the bump.
- Commit explicit scoped paths, push branch normally, create PR with `Closes #13` and `Closes #14`, inspect checks at the exact head, merge without bypassing protections, and verify merged main and both issue closures.

## Evidence

- Implemented structural predicate parsing with compound NOT, scalar-only operands,
  complete Boolean reference validation, constrained time-window extraction, and
  diagnostic rejection of unsupported WHERE/HAVING conditions. Removed the
  obsolete cross-operator flattening/classification helpers.
- Compiler, backend, and mutation Boolean construction uses SQLGlot's
  grouping-aware combinators. Reviewed the remaining engine merge constructors:
  they negate atomic leaves or combine homogeneous OR clauses; surrounding
  SELECT builders already group clauses, so no engine-source cutover was needed.
- SQL serialization and the parser prompt no longer pretend LIKE is equivalent
  to `contains`. Direct JSON `contains` compilation remains supported and was
  executed successfully with case-insensitive matching.
- Original issue probes now select IDs `[1, 2]` and compile to count `2`.
  Independent DuckDB SQL matched direct and parsed query execution across all
  27 three-valued inputs for five mixed/negated Boolean shapes.
- Additional security reproduction at the base revision: tenancy combined with
  `owner_id = 'alice' OR public = TRUE` selected `[1, 2, 5]`, including an
  out-of-tenant row. The repaired compiler selects exactly `[1, 2]`. Permanent
  execution regressions cover both `security_sql` and ScopeFn policy sources.
- Final `env -u SEMQL_CONFORMANCE_POSTGRES -u SEMQL_CONFORMANCE_CLICKHOUSE
  SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check` passed: **3420 passed, 4 skipped,
  1 xfailed; 80 snapshots**. Ruff, mypy (271 files), Pyright, Tach, core boundary,
  and frozen-model checks passed. This includes the full affected test modules
  and actual DuckDB/PostgreSQL/ClickHouse conformance; 30 new backend cells
  exercise direct semantic queries and semantic-SQL parse/compile execution.
- Public API comparison against `origin/main` reported no breaking signature
  changes in the modules covered by the repository checker. Read-only code
  review returned no findings.
- All eight package manifests, installed versions, internal dependency ranges,
  and the regenerated lockfile are at **0.11.1** (`>=0.11.1,<0.12` internally).
  No release tag or package publication was performed.
