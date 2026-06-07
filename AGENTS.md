# SemQL — agent context

For agents (Claude Code, Cursor, etc.) working in this repository.
The README is for end users; this file is for the people *changing*
SemQL itself.

## Repository layout

```
packages/
  semql/             # core: model, Cube, Catalog, compile_query, prompt rendering
  semql-mcp/         # FastMCP-backed MCPServer exposing a Catalog
  semql-erd/         # graphviz ER-diagram generator for catalogues
  semql-validate-db/ # pre-deploy drift check (LIMIT 0 probes per cube/join)
scripts/
  gen_api_docs.py     # griffe → docs/api/*.md
  check_api_break.py  # griffe-driven public-surface diff
.github/workflows/    # CI (lint, types, tests, breakage, build)
skills/               # installable Claude Code skills
docs/notes/           # source material (gitignored)
docs/api/             # generated API reference (gitignored)
PHILOSOPHY.md         # design invariants — read this before changing scope
TODOS.org             # open work, DAG, ribbon (gitignored)
```

Each package is a uv workspace member declared in the top-level
`pyproject.toml`. There is no monorepo tooling beyond `uv` itself.

## Commands

`just check` is the canonical "everything is green" target — runs
`fmt`, `lint`, `typecheck`, and `test` in that order.

| Action                     | Command                                                |
|----------------------------|--------------------------------------------------------|
| Add a workspace dependency | `cd packages/<pkg> && uv add <name>` (never edit `pyproject.toml`) |
| Add a dev dependency       | `uv add --dev <name>` from the repo root              |
| Run one test               | `uv run pytest packages/semql/tests/test_X.py::test_Y` |
| Update a snapshot          | `uv run pytest --snapshot-update`                     |
| Regenerate API docs        | `uv run scripts/gen_api_docs.py`                      |
| Check API breakage         | `uv run scripts/check_api_break.py --base <ref>`      |
| Install pre-commit hooks   | `just hooks`                                          |
| Build all packages         | `uv build --package <name>` (per package)             |

The strict-typing setup runs both `mypy` and `pyright` — they overlap
heavily but disagree just often enough that keeping both has caught
real bugs. (Item #59 in `TODOS.org` is to revisit this trade-off.)

## Conventions

### Testing

Red / Green TDD is the default cadence — write a failing test first,
then make it pass. The existing 568-test suite acts as the regression
guard for refactors.

Per-module unit-tests live next to the package they target
(`packages/semql/tests/test_<module>.py`). Hypothesis property tests
go in `tests/test_property.py`; syrupy SQL snapshots in
`tests/test_snapshots.py` + `tests/__snapshots__/`.

### Imports

Fully qualified, package-relative: `from semql.compile import ...`
not `from .compile import ...`. The exception is intra-test helpers,
which can use relative imports.

### Linting / Types

- `ruff` enforces `E,F,I,UP,B,SIM,ANN` rule sets.
- `mypy` runs in `--strict` mode; so does `pyright`.
- Per-file pragmas (`# pyright: reportXxx=false`) are preferred over
  global ones — narrow the suppression to the file that needs it,
  with a comment explaining *why*.
- If you must disable a lint rule inline, leave a comment naming the
  reason. "Pydantic raises ValidationError on frozen mutation; the
  exact class isn't load-bearing here" is the right shape.

### Model design

- Catalogue value types (`Measure`, `Dimension`, `TimeDimension`,
  `Segment`, `Join`) are frozen Pydantic models. Construct, never
  mutate.
- Spec value types (`SemanticQuery`, `Filter`, `TimeWindow`,
  `CompareWindow`, `BoolExpr`) are also frozen.
- Shared identity / presentation fields (`name`, `sql`,
  `description`, `display_name`, `metadata`) live on `BaseField`.
  Subclasses add only their type-specific fields.
- The user-owned `metadata: dict[str, str]` field on every
  catalogue type is opaque to SemQL — k8s-annotation flavoured.
  Never read or validate its contents.

### Compiler

- Pure: no I/O, no globals, no time-of-day. The compiler turns
  `SemanticQuery + Catalog` into `Compiled(sql, params, columns)`.
- Identifier resolution → join graph BFS → sqlglot AST composition →
  dialect render. Each phase is independently testable.
- Dialect-specific shapes (placeholder syntax, `date_trunc`,
  contains) come from `BackendStrategy` — not branches in the
  compiler body.

### Commits

`~/.config/git/message` is the template. Subject line uses
imperative-uppercase verbs (`Add`, `Drop`, `Fix`, `Bump`, `Make`,
`Start`, `Stop`). Body should explain *why*, not what — the diff
shows what.

Stage explicitly by path (`git add packages/...`), never `git add .`
or `-A`. Commit messages omit empty template fields.

### Don't

- Don't mutate frozen value types.
- Don't bypass the compiler with hand-built SQL strings — emit
  sqlglot AST and let the dialect renderer handle it.
- Don't add a new field's value as a SQL literal — bind it via the
  `bind(value, dim_type)` closure so it appears in `Compiled.params`.
- Don't change PHILOSOPHY.md invariants without an explicit
  discussion in the PR — they're load-bearing.

## When working on a feature

1. Read PHILOSOPHY.md first if the feature changes scope.
2. Check `TODOS.org` for prior context — most items have a paragraph
   of rationale plus links to related items.
3. Write the failing test first.
4. Make it pass.
5. Run `just check` before committing.
6. If the feature changes the public surface, re-run
   `uv run scripts/check_api_break.py --base main` and document any
   intentional breakages in the commit body.
