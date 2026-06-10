# mypy: disable-error-code=unused-ignore
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnusedImport=false
"""``TokenVerifier`` Protocol + reference HMAC / JWKS implementations.

Decodes incoming bearer tokens into a fully-populated
:class:`~semql.model.AuthContext`. Callers wire one of these into
their request middleware; downstream code (compile, prompt, MCP)
just receives the ``AuthContext`` and never sees the token.

Reference claim mapping (used by both ``HMACVerifier`` and
``JWKSVerifier``):

  - ``sub`` → ``AuthContext.viewer_id``
  - ``roles`` claim (list[str]) → ``AuthContext.roles``
  - everything else → ``AuthContext.attrs``, with the original JSON
    type preserved (list, bool, int, str, dict).

Convention (documented, not enforced): namespace claim names
k8s-style (e.g. ``acme/allowed_regions``) to avoid collisions
with standard JWT claims like ``sub`` / ``iss`` / ``exp``.

The Protocol is the integration point: callers with their own
token stack (OAuth introspect, opaque session tokens, mTLS
certificates) implement ``TokenVerifier`` without depending on
this module.

Both reference impls require ``PyJWT``; the ``JWKSVerifier``
additionally requires ``httpx``. Both are import-guarded with an
actionable message — install ``semql[jwt]`` to enable them.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from semql.errors import AuthError
from semql.model import AuthContext

# Reserved claims — never copied into ``attrs`` so a token can't
# shadow the structural fields of ``AuthContext``.
_RESERVED_CLAIMS = frozenset(
    {
        "sub",
        "iss",
        "aud",
        "exp",
        "iat",
        "nbf",
        "jti",
        "roles",  # mapped to AuthContext.roles, not attrs
    }
)


def _payload_to_auth_context(payload: dict[str, Any]) -> AuthContext:
    """Map a verified JWT payload to an ``AuthContext``.

    ``sub`` is required; missing ``sub`` raises ``AuthError``. Other
    reserved claims (``exp`` / ``iat`` / etc.) are validated by
    PyJWT before this is called.
    """
    if "sub" not in payload or not payload["sub"]:
        raise AuthError("Token is missing required 'sub' claim.", reason="missing_sub")
    viewer_id = str(payload["sub"])
    roles_raw_obj: object = payload.get("roles", [])
    if not isinstance(roles_raw_obj, list) or not all(
        isinstance(r, str)
        for r in roles_raw_obj  # type: ignore[union-attr]
    ):
        raise AuthError(
            "Token 'roles' claim must be a list[str].",
            reason="bad_roles_claim",
        )
    roles_list: list[str] = [r for r in roles_raw_obj]  # type: ignore[union-attr]
    attrs: dict[str, Any] = {k: v for k, v in payload.items() if k not in _RESERVED_CLAIMS}
    return AuthContext(viewer_id=viewer_id, roles=roles_list, attrs=attrs)


@runtime_checkable
class TokenVerifier(Protocol):
    """Decode a bearer token into an ``AuthContext``.

    Implementations raise :class:`~semql.errors.AuthError` on
    invalid, expired, or otherwise unverifiable tokens. The contract
    is intentionally narrow: a token string in, an
    ``AuthContext`` out (or an exception).
    """

    def verify(self, token: str) -> AuthContext: ...


# ---------------------------------------------------------------------------
# HMACVerifier
# ---------------------------------------------------------------------------


class HMACVerifier:
    """HMAC shared-secret JWT verification (HS256 / HS384 / HS512).

    Reference implementation of :class:`TokenVerifier`. Suits
    single-tenant deployments and any context where the platform
    issues its own tokens. For multi-tenant with rotating keys,
    use :class:`JWKSVerifier`.

    Args:
        secret: HMAC shared secret. Must be at least the algorithm's
            required key length (32 bytes for HS256/HS384, 64 for HS512);
            PyJWT raises on shorter secrets.
        algorithm: One of ``"HS256"`` / ``"HS384"`` / ``"HS512"``.
            Defaults to ``"HS256"``.
        audience: Optional ``aud`` claim to enforce.
        issuer: Optional ``iss`` claim to enforce.
    """

    def __init__(
        self,
        secret: bytes | str,
        *,
        algorithm: str = "HS256",
        audience: str | None = None,
        issuer: str | None = None,
    ) -> None:
        try:
            import jwt  # noqa: F401  — import-time guard
        except ImportError as exc:
            raise ImportError(
                "HMACVerifier requires PyJWT. Install with `pip install semql[jwt]`."
            ) from exc
        if isinstance(secret, str):
            secret = secret.encode("utf-8")
        self._secret = secret
        self._algorithm = algorithm
        self._audience = audience
        self._issuer = issuer

    def verify(self, token: str) -> AuthContext:
        import jwt

        options: dict[str, Any] = {}
        decode_kwargs: dict[str, Any] = {
            "key": self._secret,
            "algorithms": [self._algorithm],
            "options": options,
        }
        if self._audience is not None:
            decode_kwargs["audience"] = self._audience
        if self._issuer is not None:
            decode_kwargs["issuer"] = self._issuer
        try:
            payload = jwt.decode(token, **decode_kwargs)
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Token has expired.", reason="expired") from exc
        except jwt.InvalidSignatureError as exc:
            raise AuthError("Token signature is invalid.", reason="bad_signature") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError("Token audience is invalid.", reason="bad_audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError("Token issuer is invalid.", reason="bad_issuer") from exc
        except jwt.DecodeError as exc:
            raise AuthError("Token is malformed.", reason="malformed") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"Token failed verification: {exc}", reason="invalid") from exc
        return _payload_to_auth_context(payload)


# ---------------------------------------------------------------------------
# JWKSVerifier — RS256 / ES256 against a JWKS endpoint
# ---------------------------------------------------------------------------


class JWKSVerifier:
    """RS256 / ES256 JWT verification against a JWKS endpoint.

    Reference implementation of :class:`TokenVerifier` for the case
    where an external identity provider signs tokens with an
    asymmetric key. Fetches the JWKS document once and caches
    keys; rotate keys by re-fetching (see ``ttl``).

    Args:
        jwks_url: URL of the JWKS endpoint (e.g. an OIDC provider's
            ``/jwks.json`` route).
        algorithms: Tuple of allowed algorithms. Defaults to
            ``("RS256",)`` — extend cautiously, never accept ``none``.
        audience: Optional ``aud`` claim to enforce.
        issuer: Optional ``iss`` claim to enforce.
        ttl: Seconds the cached JWKS lives before refetching. Default
            300 (5 minutes). Set ``0`` to disable caching.
    """

    def __init__(
        self,
        jwks_url: str,
        *,
        algorithms: tuple[str, ...] = ("RS256",),
        audience: str | None = None,
        issuer: str | None = None,
        ttl: int = 300,
    ) -> None:
        try:
            import httpx  # noqa: F401
            import jwt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "JWKSVerifier requires PyJWT + httpx. Install with `pip install semql[jwt]`."
            ) from exc
        if "none" in (a.lower() for a in algorithms):
            raise ValueError(
                "JWKSVerifier must never accept the 'none' algorithm — "
                "this is a security invariant, not a config knob."
            )
        self._jwks_url = jwks_url
        self._algorithms = algorithms
        self._audience = audience
        self._issuer = issuer
        self._ttl = ttl
        self._cached_jwks: dict[str, Any] | None = None
        self._cached_at: float = 0.0

    def _fetch_jwks(self) -> dict[str, Any]:
        import httpx

        now = __import__("time").monotonic()
        cached = self._cached_jwks
        if cached is not None and self._ttl > 0 and (now - self._cached_at) < self._ttl:
            return cached
        response = httpx.get(self._jwks_url, timeout=10.0)
        response.raise_for_status()
        new_jwks: dict[str, Any] = response.json()
        self._cached_jwks = new_jwks
        self._cached_at = now
        return new_jwks

    def verify(self, token: str) -> AuthContext:
        import jwt

        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.DecodeError as exc:
            raise AuthError("Token header is malformed.", reason="malformed") from exc
        kid = unverified_header.get("kid")
        if not kid:
            raise AuthError(
                "Token header is missing 'kid' (key id).",
                reason="missing_kid",
            )
        jwks = self._fetch_jwks()
        key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
        if key is None:
            # Cache miss — refetch once before giving up. Key rotations
            # land between TTL windows; this handles the common case.
            self._cached_jwks = None
            jwks = self._fetch_jwks()
            key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
            if key is None:
                raise AuthError(
                    f"No JWKS key matches token kid={kid!r}.",
                    reason="unknown_kid",
                )

        # PyJWT accepts a JWK dict directly via from_jwk, but we have
        # the key in raw form; serialise to a PEM-like public key.
        public_key: Any = jwt.algorithms.RSAAlgorithm.from_jwk(  # type: ignore[attr-defined]
            json.dumps(key)
        )

        decode_kwargs: dict[str, Any] = {
            "key": public_key,
            "algorithms": list(self._algorithms),
        }
        if self._audience is not None:
            decode_kwargs["audience"] = self._audience
        if self._issuer is not None:
            decode_kwargs["issuer"] = self._issuer
        try:
            payload = jwt.decode(token, **decode_kwargs)
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Token has expired.", reason="expired") from exc
        except jwt.InvalidSignatureError as exc:
            raise AuthError("Token signature is invalid.", reason="bad_signature") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError("Token audience is invalid.", reason="bad_audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError("Token issuer is invalid.", reason="bad_issuer") from exc
        except jwt.DecodeError as exc:
            raise AuthError("Token is malformed.", reason="malformed") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"Token failed verification: {exc}", reason="invalid") from exc
        return _payload_to_auth_context(payload)


__all__ = [
    "HMACVerifier",
    "JWKSVerifier",
    "TokenVerifier",
    "_payload_to_auth_context",
]
