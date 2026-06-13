# SemQL Philosophy

The rules a contributor gets told "no" on.

## Compiler

The compiler has no I/O. `Catalog` is Python data; `compile()` returns
SQL + bound params. Running the SQL is the caller's job.

Compile errors beat runtime errors. Runtime errors beat wrong results.

## Authorisation

Identity is the caller's. Authorisation is the compiler's.

`AuthContext` is request-scoped. A catalog without a viewer compiles
unscoped. A catalog with a viewer refuses queries that touch a cube
the viewer can't see, and injects row-level predicates inside the
cube's alias subquery — not the outer query — so outer `OR`s can't
reach rows the scope excludes.

Identity values bind as parameters. Never as SQL literals.

`expose_in_prompt` is for LLMs. `required_roles` and `ScopeFn` are
for access control. Don't conflate them.

## SQL

The emitted SQL must be readable by the engineer debugging a
production incident at 2am. If it isn't, the abstraction failed.

Raw SQL is an escape hatch. When SemQL emits it, it says so.

## Errors

`compile()` fails at the first problem. `validate()` collects them.

When a field is unknown, name the closest match.

## Catalog

Catalogs must be serialisable and versioned. A catalog that can't
cross a process boundary can't serve a real system.

## Growth

One package per concern. Core stays minimal — `CompiledQuery` in,
`CompiledQuery` out. Executors, enrichers, MCP, prompt fragments,
and call-site opinions live in sibling packages and stay swappable.

Federation belongs out of core SemQL, declared by the caller.

Break freely before v1. Lock strictly after.
