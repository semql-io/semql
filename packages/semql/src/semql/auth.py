# mypy: disable-error-code=unused-ignore
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnusedImport=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""``TokenVerifier`` / ``TokenMapper`` Protocols + reference implementations.

Two sibling integration points that turn an inbound auth identity
into a fully-populated :class:`~semql.model.AuthContext`:

- :class:`TokenVerifier` is the JWT-shaped path
  (``verify(token: str) -> AuthContext``). Reference impls:
  :class:`HMACVerifier` (HS256/384/512) and :class:`JWKSVerifier`
  (RS256 / ES256 against a JWKS endpoint).
- :class:`TokenMapper` is the structured-identity path
  (``verify(identity: object) -> AuthContext``) for stacks that
  don't return a string token. Reference impls:
  :class:`DictMapper` (always available, no extras — for tests and
  pre-shaped introspect responses),
  :class:`IntrospectMapper` (RFC 7662 OAuth 2.0 Token Introspection
  — ``httpx`` required, ``semql[jwt]`` extras),
  :class:`X509Mapper` (mTLS client cert subject / SAN mapping —
  ``cryptography`` required, ``semql[mTLS]`` extras).

Callers wire either protocol into their request middleware;
downstream code (compile, prompt, MCP) just receives the
``AuthContext`` and never sees the token.

Reference claim mapping (used by ``HMACVerifier``,
``JWKSVerifier``, ``DictMapper``, ``IntrospectMapper``):

  - ``sub`` (or ``username``, or ``client_id`` for client-credentials
    grants) → ``AuthContext.viewer_id``
  - ``roles`` claim (list[str]) or RFC 7662 ``scope`` (space-delimited)
    → ``AuthContext.roles``
  - everything else → ``AuthContext.attrs``, with the original JSON
    type preserved (list, bool, int, str, dict).

Convention (documented, not enforced): namespace claim names
k8s-style (e.g. ``acme/allowed_regions``) to avoid collisions
with standard JWT claims like ``sub`` / ``iss`` / ``exp``.

The Protocols are the integration point: callers with their own
auth stack implement either one without depending on this module.

``HMACVerifier`` and ``JWKSVerifier`` require ``PyJWT``;
``JWKSVerifier`` additionally requires ``httpx``; ``X509Mapper``
requires ``cryptography``. Each is import-guarded with an
actionable message — install the matching extras group to enable it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from semql.errors import AuthError
from semql.model import AuthContext

# Reserved claims — never copied into ``attrs`` so a token can't
# shadow the structural fields of ``AuthContext``. Note: ``iss``,
# ``aud``, ``exp``, ``iat``, ``nbf``, ``jti`` are NOT reserved
# here — they're preserved in ``attrs`` so callers can reference
# them (e.g. introspect-endpoint ``exp`` for cache TTL). The
# JWT path's PyJWT-decoded payload and the introspect path's
# flat JSON response share the same claim-mapping contract.
_RESERVED_CLAIMS = frozenset(
    {
        "sub",  # mapped to AuthContext.viewer_id
        "roles",  # mapped to AuthContext.roles
        "metadata",  # mapped to AuthContext.metadata
    }
)


def _payload_to_auth_context(payload: dict[str, Any]) -> AuthContext:
    """Map a verified JWT payload (or pre-shaped claims dict) to
    an ``AuthContext``.

    ``sub`` is required; missing ``sub`` raises ``AuthError``. Other
    reserved claims (``exp`` / ``iat`` / etc.) are validated by
    PyJWT before this is called for the JWT path; for the
    non-JWT ``DictMapper`` path, those claims are treated as
    regular attrs (preserved as-is).

    The ``metadata`` claim is treated as the structural
    ``AuthContext.metadata`` field rather than being merged into
    ``attrs`` — legacy callers that already pass ``metadata``
    keep their existing field name.
    """
    if "sub" not in payload or not payload["sub"]:
        raise AuthError("Identity is missing required 'sub' claim.", reason="missing_sub")
    viewer_id = str(payload["sub"])
    roles_raw_obj: object = payload.get("roles", [])
    if not isinstance(roles_raw_obj, list) or not all(
        isinstance(r, str)
        for r in roles_raw_obj  # type: ignore[union-attr]
    ):
        raise AuthError(
            "Identity 'roles' claim must be a list[str].",
            reason="bad_roles_claim",
        )
    roles_list: list[str] = [r for r in roles_raw_obj]  # type: ignore[union-attr]
    metadata_obj: object = payload.get("metadata", {})
    if not isinstance(metadata_obj, dict):
        metadata_obj = {}
    metadata_dict: dict[str, str] = {str(k): str(v) for k, v in metadata_obj.items()}  # type: ignore[union-attr]
    attrs: dict[str, Any] = {k: v for k, v in payload.items() if k not in _RESERVED_CLAIMS}
    return AuthContext(
        viewer_id=viewer_id,
        roles=roles_list,
        attrs=attrs,
        metadata=metadata_dict,
    )


@runtime_checkable
class TokenVerifier(Protocol):
    """Decode a bearer token into an ``AuthContext``.

    Implementations raise :class:`~semql.errors.AuthError` on
    invalid, expired, or otherwise unverifiable tokens. The contract
    is intentionally narrow: a token string in, an
    ``AuthContext`` out (or an exception).
    """

    def verify(self, token: str) -> AuthContext: ...


@runtime_checkable
class TokenMapper(Protocol):
    """Map a structured auth identity into an ``AuthContext``.

    Sibling of :class:`TokenVerifier` for stacks that don't
    return a string token. ``TokenMapper.verify`` takes whatever
    the caller's auth stack natively provides — an OAuth
    introspect response dict, an mTLS x509 cert, a SAML
    assertion, an opaque session token — and produces the
    canonical ``AuthContext``.

    The Protocol is parameterised on ``object`` (any structured
    identity). Concrete mappers narrow the parameter to their
    own type via a runtime check. The contract is:

      - Return an :class:`~semql.model.AuthContext` on a valid
        identity.
      - Raise :class:`~semql.errors.AuthError` on malformed /
        unidentifiable input.
      - Never return a "default" AuthContext — a missing
        viewer id is an auth error, not a fallback.

    Application code wires one of ``TokenVerifier`` /
    ``TokenMapper`` into its request middleware; downstream
    code (compile, prompt, MCP) just receives the
    ``AuthContext`` and never sees the underlying identity.
    """

    def verify(self, identity: object) -> AuthContext: ...


class DictMapper:
    """Map a ``Mapping[str, Any]``-shaped identity to an ``AuthContext``.

    Always-available reference implementation of
    :class:`TokenMapper`. No external dependencies — useful
    for tests, for callers that pre-shape their auth response
    (e.g. a middleware that already decoded the JWT and
    passes the claims dict), and for any custom auth stack
    that can return a ``Mapping``-compatible view of its
    identity.

    The claim mapping is the same as the JWT path:
    ``sub`` → ``viewer_id``, ``roles`` (list[str]) →
    ``roles``, ``metadata`` → ``metadata``, everything else
    → ``attrs``.
    """

    def verify(self, identity: object) -> AuthContext:
        if not isinstance(identity, Mapping):
            raise AuthError(
                "DictMapper.verify requires a Mapping-shaped identity "
                f"(got {type(identity).__name__}).",
                reason="bad_identity_type",
            )
        # ``Mapping`` doesn't enforce string keys; the contract
        # is ``Mapping[str, Any]`` but the runtime check is
        # cheap. We normalise to a dict for the existing
        # ``_payload_to_auth_context`` helper.
        items_any: Any = identity
        payload: dict[str, Any] = {str(k): v for k, v in items_any.items()}
        return _payload_to_auth_context(payload)


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


# ---------------------------------------------------------------------------
# IntrospectMapper — RFC 7662 OAuth 2.0 Token Introspection
# ---------------------------------------------------------------------------


class _IntrospectHttpClient(Protocol):
    """Structural protocol for the HTTP client used by
    :class:`IntrospectMapper`. The real implementation is
    ``httpx``; tests inject a fake with the same shape."""

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        auth: tuple[str, str],
        timeout: float,
    ) -> Mapping[str, object]: ...


def _default_introspect_client() -> _IntrospectHttpClient:
    """Lazy default — ``httpx`` is required for
    :class:`IntrospectMapper` but not for any other part of
    ``semql``. The import-time guard below gives a clean
    error message at construction; this function exists so the
    failure is at mapper-instantiation time, not at module
    import.
    """
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "IntrospectMapper requires httpx. Install with `pip install semql[jwt]`."
        ) from exc

    class _HttpxAdapter:
        def post(
            self,
            url: str,
            *,
            data: dict[str, str],
            auth: tuple[str, str],
            timeout: float,
        ) -> Mapping[str, object]:
            return httpx.post(url, data=data, auth=auth, timeout=timeout).json()  # type: ignore[no-any-return]

    return _HttpxAdapter()


class IntrospectMapper:
    """OAuth 2.0 Token Introspection (RFC 7662) → ``AuthContext``.

    Reference implementation of :class:`TokenMapper` for the
    case where the auth stack has an OAuth 2.0 introspection
    endpoint. ``verify(access_token)`` POSTs the token to the
    configured endpoint with HTTP Basic client credentials,
    parses the JSON response, and maps it to an
    ``AuthContext``.

    Requires ``httpx`` — guarded at construction time with
    an actionable ``ImportError`` (``pip install semql[jwt]``).

    Args:
        introspect_url: The introspection endpoint URL (e.g.
            ``https://idp.example.com/oauth2/introspect``).
        client_id: The OAuth client's id.
        client_secret: The OAuth client's secret.
        http_client: An optional HTTP client. Defaults to
            ``httpx``. Tests inject a fake with the shape
            ``post(url, *, data, auth, timeout) -> Any``.
        timeout: HTTP request timeout in seconds. Default 5.0.
    """

    def __init__(
        self,
        introspect_url: str,
        *,
        client_id: str,
        client_secret: str,
        http_client: _IntrospectHttpClient | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._introspect_url = introspect_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._http_client: _IntrospectHttpClient = (
            http_client if http_client is not None else _default_introspect_client()
        )

    def verify(self, identity: object) -> AuthContext:
        # The IntrospectMapper only handles string access tokens.
        if not isinstance(identity, str):
            raise AuthError(
                "IntrospectMapper.verify requires a string access token "
                f"(got {type(identity).__name__}).",
                reason="bad_identity_type",
            )
        try:
            # The ``object`` annotation (not ``Mapping[str, object]``)
            # is intentional: it keeps the runtime ``isinstance``
            # check live rather than being narrowed away. Test
            # fakes and custom http-client implementations can
            # misbehave at runtime; the type-level ``Mapping``
            # contract is the static guarantee, the
            # ``isinstance`` is the dynamic one.
            response: object = self._http_client.post(  # type: ignore[assignment]
                self._introspect_url,
                data={"token": identity, "token_type_hint": "access_token"},
                auth=(self._client_id, self._client_secret),
                timeout=self._timeout,
            )
        except Exception as exc:
            raise AuthError(
                f"Introspection endpoint returned an error: {exc}",
                reason="introspect_failed",
            ) from exc
        # The ``isinstance`` is intentional despite the static
        # ``Mapping[str, object]`` annotation — test fakes and
        # custom http-client implementations can misbehave at
        # runtime. The type-level contract is the static
        # guarantee; the ``isinstance`` is the dynamic one.
        if not isinstance(response, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise AuthError(
                "Introspection endpoint did not return a JSON object.",
                reason="introspect_bad_response",
            )
        # RFC 7662 §2.2: ``active`` MUST be a boolean. The
        # only true indicator of validity is ``active=true``.
        if not response.get("active", False):
            raise AuthError(
                "Introspection reported token inactive (expired, revoked, or unknown).",
                reason="inactive",
            )
        # Build a payload shaped like the JWT claims dict, then
        # re-use ``_payload_to_auth_context`` so claim mapping
        # is identical to the JWT path.
        # ``sub`` is the canonical viewer id; fall back to
        # ``username`` (older Auth0 / Keycloak) then to
        # ``client_id`` (client-credentials grants) so the
        # contract is "we always have a viewer id when the
        # token is active".
        if "sub" not in response or not response["sub"]:
            if "username" in response and response["username"]:
                response = dict(response)
                response["sub"] = response["username"]
            elif "client_id" in response and response["client_id"]:
                # Response carries a ``client_id`` claim
                # (the token was issued for that client —
                # client-credentials grant or
                # service-account delegation).
                response = dict(response)
                response["sub"] = response["client_id"]
        # RFC 7662 §2.2: ``scope`` is OPTIONAL but when present
        # it's a space-delimited string. Treat it as the
        # canonical roles surface (overriding any ``roles``
        # claim — OAuth scopes are the standard way to express
        # authorisation grants).
        scope_value = response.get("scope")
        if isinstance(scope_value, str):
            response = dict(response)
            response["roles"] = scope_value.split()
        return _payload_to_auth_context(dict(response))


# ---------------------------------------------------------------------------
# X509Mapper — mTLS client cert subject / SAN mapping
# ---------------------------------------------------------------------------


class _X509Cert(Protocol):
    """Structural protocol for the client cert shape used by
    :class:`X509Mapper`. Real ``cryptography.x509.Certificate``
    objects expose these via attribute access; tests inject a
    plain object with the right shape (the mapper doesn't
    require ``cryptography`` for the test path)."""

    @property
    def subject_cn(self) -> str: ...

    @property
    def subject_ou(self) -> tuple[str, ...]: ...

    @property
    def subject_o(self) -> str | None: ...

    @property
    def subject_c(self) -> str | None: ...

    @property
    def sans(self) -> tuple[str, ...]: ...

    @property
    def fingerprint(self) -> str | None: ...


class X509Mapper:
    """Map an mTLS client x509 cert to an ``AuthContext``.

    Reference implementation of :class:`TokenMapper` for
    mTLS deployments where the caller's middleware extracts
    the client cert from the TLS handshake. ``verify(cert)``
    reads the cert's subject Common Name (preferred) or its
    Subject Alternative Names (URI > DNS > email) and maps it
    to ``viewer_id``.

    The mapper is structurally typed — the call site is
    responsible for decoding the cert into a shape with the
    right attributes (``subject_cn``, ``subject_ou``,
    ``subject_o``, ``subject_c``, ``sans``, ``fingerprint``).
    The ``cryptography`` adapter is shipped as a small
    reference function in ``semql.auth._cryptography_adapter``
    (separate file, import-guarded) for the common case;
    call sites with their own cert decoder don't need it.

    Args:
        ou_to_role: If ``True``, each ``OU`` (Organizational
            Unit) in the cert subject becomes a role in the
            resulting ``AuthContext``. Default ``False`` —
            ``OU``s land in ``attrs`` instead. Opt in to the
            structural mapping when your PKI uses ``OU`` as
            the team / project grouping.
    """

    def __init__(self, *, ou_to_role: bool = False) -> None:
        self._ou_to_role = ou_to_role

    def verify(self, identity: object) -> AuthContext:
        cert = self._coerce(identity)
        viewer_id = self._viewer_id(cert)
        if not viewer_id:
            raise AuthError(
                "X509 cert has no usable identity (empty subject CN and no "
                "URI / DNS / email SAN). Cannot derive viewer_id.",
                reason="no_identity",
            )
        attrs: dict[str, Any] = {}
        if cert.subject_o is not None:
            attrs["subject_o"] = cert.subject_o
        if cert.subject_c is not None:
            attrs["subject_c"] = cert.subject_c
        if cert.subject_ou:
            attrs["subject_ou"] = list(cert.subject_ou)
        if cert.fingerprint is not None:
            attrs["fingerprint"] = cert.fingerprint
        if cert.sans:
            attrs["sans"] = list(cert.sans)
        roles: list[str] = list(cert.subject_ou) if self._ou_to_role and cert.subject_ou else []
        return AuthContext(viewer_id=viewer_id, roles=roles, attrs=attrs)

    @staticmethod
    def _coerce(identity: object) -> _X509Cert:
        # The mapper is structurally typed. The ``_X509Cert``
        # Protocol declares the attribute shape; we just
        # access them and let any missing attribute raise
        # AttributeError. A clearer error helps callers fix
        # the adapter.
        try:
            cn = getattr(identity, "subject_cn", "")
            if not isinstance(cn, str):
                cn = ""
            return _CertAdapter(  # type: ignore[return-value]
                subject_cn=cn,
                subject_ou=tuple(getattr(identity, "subject_ou", ()) or ()),
                subject_o=getattr(identity, "subject_o", None),
                subject_c=getattr(identity, "subject_c", None),
                sans=tuple(getattr(identity, "sans", ()) or ()),
                fingerprint=getattr(identity, "fingerprint", None),
            )
        except AttributeError as exc:
            raise AuthError(
                "X509Mapper.verify requires a cert-like object with "
                "attributes subject_cn / subject_ou / subject_o / "
                f"subject_c / sans / fingerprint. ({exc})",
                reason="bad_cert_shape",
            ) from exc

    @staticmethod
    def _viewer_id(cert: _X509Cert) -> str:
        # CN first (RFC 6125 §6.4.4 — CN is the canonical
        # mTLS identity). Then SANs in priority order:
        # URI > DNS > email. The first hit wins.
        if cert.subject_cn:
            return cert.subject_cn
        for san in cert.sans:
            if san.startswith(("spiffe://", "https://", "urn:")):
                return san
        for san in cert.sans:
            if "." in san and "@" not in san and ":" not in san:
                # DNS-style: contains a dot, no @, no port.
                return san
        for san in cert.sans:
            if "@" in san:
                # Email-style: use the *full* address as the viewer
                # id. The local part alone is not unique —
                # alice@a.com and alice@b.com are different
                # principals, and collapsing them to "alice" would
                # cross-wire row-level security between tenants.
                return san
        return ""


class _CertAdapter:
    """Wrap a duck-typed cert so the rest of the mapper can
    use ``_X509Cert``-typed attribute access uniformly. Internal
    helper — not exported."""

    __slots__ = (
        "subject_cn",
        "subject_ou",
        "subject_o",
        "subject_c",
        "sans",
        "fingerprint",
    )

    def __init__(
        self,
        *,
        subject_cn: str,
        subject_ou: tuple[str, ...],
        subject_o: str | None,
        subject_c: str | None,
        sans: tuple[str, ...],
        fingerprint: str | None,
    ) -> None:
        self.subject_cn = subject_cn
        self.subject_ou = subject_ou
        self.subject_o = subject_o
        self.subject_c = subject_c
        self.sans = sans
        self.fingerprint = fingerprint


__all__ = [
    "DictMapper",
    "HMACVerifier",
    "IntrospectMapper",
    "JWKSVerifier",
    "TokenMapper",
    "TokenVerifier",
    "X509Mapper",
    "_payload_to_auth_context",
]
