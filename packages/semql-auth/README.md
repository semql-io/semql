# semql-auth

Credential→identity adapters for [semql](https://github.com/npalladium/semql).

`semql` threads an `AuthContext` (identity + roles) through
`Catalog.compile(viewer=...)` to enforce `required_roles` cube/field
visibility and `security_sql` row-level scoping. This package turns a
transport credential into that `AuthContext`:

- **`TokenVerifier`** — verify a bearer token and return its claims.
  - `HMACVerifier` — symmetric HS256/384/512.
  - `JWKSVerifier` — asymmetric RS/ES, fetching keys from a JWKS URL
    (needs the `jwks` extra: `pip install semql-auth[jwks]`).
- **`TokenMapper`** — map a verified credential to an `AuthContext`.
  - `DictMapper` — static, in-memory `token → AuthContext` table.
  - `IntrospectMapper` — OAuth2 token introspection (`introspect` extra).
  - `X509Mapper` — derive identity from an mTLS client cert subject / SAN
    (the reference cryptography decoder needs the `x509` extra).

`AuthContext` itself lives in `semql.model` — the compiler depends on it,
so it stays in the pure core. This package holds only the adapters, which
carry optional third-party dependencies (PyJWT, httpx, cryptography) that
the core shouldn't.

## Install

```sh
pip install semql-auth
pip install semql-auth[jwks]        # JWKS verifier (httpx)
pip install semql-auth[introspect]  # OAuth2 introspection
pip install semql-auth[x509]        # mTLS client cert decoder
```

## Quick start

```python
from semql import Catalog
from semql_auth import HMACVerifier, DictMapper

verifier = HMACVerifier(secret="...")
mapper = DictMapper({"tok-abc": ...})
# In your transport: verify the token, map to AuthContext, then
#   catalog.compile(query, viewer=auth_context)
```

See [API reference](../../docs/api/semql_auth.md) for the full adapter
surface.

## License

BSD-3-Clause.
