"""Public surface of the semql-auth package.

Reference ``TokenVerifier`` / ``TokenMapper`` implementations for turning
a transport credential (bearer token, mTLS client cert) into a
:class:`semql.model.AuthContext` — the identity ``semql`` threads through
``Catalog.compile(viewer=...)`` for ``required_roles`` visibility and
``security_sql`` row scoping.

``AuthContext`` itself lives in ``semql.model`` (the compiler depends on
it); this package is only the credential→identity adapters, which carry
optional third-party deps (PyJWT, httpx, cryptography) the pure core
shouldn't.
"""

from __future__ import annotations

from semql_auth.auth import (
    DictMapper,
    HMACVerifier,
    IntrospectMapper,
    JWKSVerifier,
    TokenMapper,
    TokenVerifier,
    X509Mapper,
)

__all__ = [
    "DictMapper",
    "HMACVerifier",
    "IntrospectMapper",
    "JWKSVerifier",
    "TokenMapper",
    "TokenVerifier",
    "X509Mapper",
]
