"""A2-extension — ``TokenMapper`` Protocol + reference impls for
non-JWT token stacks.

``TokenVerifier`` is the JWT-shaped integration point
(``verify(token: str) -> AuthContext``). ``TokenMapper`` is the
sibling for stacks that don't return a string token: the caller
already has the auth identity in some structured form (an OAuth
introspect response dict, an mTLS x509 cert, a SAML assertion,
an opaque session token) and wants a uniform ``AuthContext``
output.

The test surface pins:

- ``TokenMapper`` is a runtime-checkable ``Protocol`` with a
  ``verify(identity) -> AuthContext`` method.
- ``DictMapper`` (always available, no extra deps) maps a
  ``Mapping[str, Any]`` to ``AuthContext`` — useful for tests
  and for callers that pre-shape their auth response.
- ``IntrospectMapper`` (RFC 7662) calls an OAuth introspection
  endpoint and maps the response. ``httpx`` import-guarded.
- ``X509Mapper`` maps a client x509 cert's subject + SAN to
  ``AuthContext``. ``cryptography`` import-guarded.
- All mappers raise ``AuthError`` on malformed input.
- ``AuthContext`` is the canonical output — the protocol is
  intentionally narrow (identity in, ``AuthContext`` out).
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from typing import Any

import pytest
from semql import AuthContext
from semql.auth import DictMapper, TokenMapper
from semql.errors import AuthError

# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


def test_token_mapper_protocol_is_runtime_checkable() -> None:
    """The protocol is ``@runtime_checkable`` so callers can
    ``isinstance(obj, TokenMapper)`` without subclassing.
    """
    m = DictMapper()
    assert isinstance(m, TokenMapper)


def test_token_mapper_protocol_declares_verify_method() -> None:
    """``TokenMapper.verify`` is the integration point — same
    shape as ``TokenVerifier.verify`` but parameter type is
    intentionally ``object`` (concrete classes narrow)."""
    assert hasattr(TokenMapper, "verify")


def test_dict_mapper_returns_auth_context_for_valid_dict() -> None:
    """A dict with the canonical claims maps to an
    ``AuthContext`` — the most useful shape for tests and for
    pre-shaped introspect responses."""
    m = DictMapper()
    ctx = m.verify({"sub": "alice", "roles": ["analyst"]})
    assert isinstance(ctx, AuthContext)
    assert ctx.viewer_id == "alice"
    assert ctx.roles == ["analyst"]


def test_dict_mapper_missing_sub_raises_auth_error() -> None:
    """Missing ``sub`` is the canonical "this isn't a valid
    identity" error — same gate as the JWT path."""
    m = DictMapper()
    with pytest.raises(AuthError, match="sub"):
        m.verify({"roles": ["analyst"]})


def test_dict_mapper_empty_sub_raises_auth_error() -> None:
    """An empty ``sub`` is not a valid viewer id."""
    m = DictMapper()
    with pytest.raises(AuthError, match="sub"):
        m.verify({"sub": "", "roles": []})


def test_dict_mapper_preserves_extra_claims_in_attrs() -> None:
    """Claims beyond ``sub`` / ``roles`` land in ``attrs`` so
    downstream code can reference them. Same shape as the JWT
    path's ``_payload_to_auth_context``."""
    m = DictMapper()
    ctx = m.verify(
        {
            "sub": "alice",
            "roles": ["analyst"],
            "tenant": "acme",
            "tier": "premium",
        }
    )
    assert ctx.attrs["tenant"] == "acme"
    assert ctx.attrs["tier"] == "premium"


def test_dict_mapper_omits_reserved_claims_from_attrs() -> None:
    """Reserved claims (those that map to structural ``AuthContext``
    fields) don't leak into ``attrs`` — same contract as the
    JWT path. The dict path has no JWT-specific reserved
    claims; ``sub`` and ``roles`` are mapped to structural
    fields and never appear in ``attrs``."""
    m = DictMapper()
    ctx = m.verify({"sub": "alice", "roles": ["x"]})
    assert "sub" not in ctx.attrs
    assert "roles" not in ctx.attrs


def test_dict_mapper_defaults_roles_to_empty() -> None:
    """A dict without ``roles`` gets an empty roles list, not
    a missing field — ``AuthContext.roles`` is required."""
    m = DictMapper()
    ctx = m.verify({"sub": "alice"})
    assert ctx.roles == []


def test_dict_mapper_rejects_non_list_roles() -> None:
    """``roles`` must be a list[str] — same contract as the
    JWT path. A non-list value is an auth error, not a
    coercion."""
    m = DictMapper()
    with pytest.raises(AuthError, match="roles"):
        m.verify({"sub": "alice", "roles": "analyst"})


def test_dict_mapper_rejects_list_with_non_string_items() -> None:
    """``roles`` items must be strings — same gate as JWT path."""
    m = DictMapper()
    with pytest.raises(AuthError, match="roles"):
        m.verify({"sub": "alice", "roles": ["analyst", 42]})


def test_dict_mapper_preserves_list_and_dict_attrs() -> None:
    """``attrs`` keeps the original JSON types — list / dict
    / bool / int / str all survive. Same contract as the JWT
    path."""
    m = DictMapper()
    ctx = m.verify(
        {
            "sub": "alice",
            "roles": [],
            "regions": ["us", "ca"],
            "tier": {"name": "premium", "level": 2},
            "active": True,
        }
    )
    assert ctx.attrs["regions"] == ["us", "ca"]
    assert ctx.attrs["tier"] == {"name": "premium", "level": 2}
    assert ctx.attrs["active"] is True


def test_dict_mapper_accepts_typed_mapping() -> None:
    """The Protocol allows any ``Mapping``-shaped input — a
    real OAuth introspect response is a dict, but a custom
    auth stack might pass a ``TypedDict`` or ``MappingProxyType``.
    """
    from types import MappingProxyType

    m = DictMapper()
    frozen = MappingProxyType({"sub": "alice", "roles": ["a"]})
    ctx = m.verify(frozen)
    assert ctx.viewer_id == "alice"


def test_dict_mapper_with_metadata_field() -> None:
    """``metadata`` is a structural field on ``AuthContext`` for
    legacy callers; the mapper respects the source claim name."""
    m = DictMapper()
    ctx = m.verify(
        {
            "sub": "alice",
            "roles": [],
            "metadata": {"region": "us"},
        }
    )
    # ``metadata`` is a structural field; the mapper hands it
    # through as-is rather than re-routing to ``attrs``.
    assert ctx.metadata == {"region": "us"}


# ---------------------------------------------------------------------------
# IntrospectMapper — OAuth 2.0 Token Introspection (RFC 7662)
# ---------------------------------------------------------------------------


def test_introspect_mapper_uses_post_with_basic_auth() -> None:
    """RFC 7662 mandates a POST to the introspection endpoint
    with the token in the body (``application/x-www-form-urlencoded``)
    and client credentials in HTTP Basic auth. The mapper must
    do exactly that — no GET, no query string.
    """
    from semql.auth import IntrospectMapper

    captured: dict[str, Any] = {}

    class FakeClient:
        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            captured["url"] = url
            captured["data"] = data
            captured["auth"] = auth
            return {
                "active": True,
                "sub": "alice",
                "username": "alice",
                "roles": ["analyst"],
            }

    m = IntrospectMapper(
        "https://idp.example.com/oauth2/introspect",
        client_id="client",
        client_secret="secret",
        http_client=FakeClient(),
    )
    ctx = m.verify("opaque-access-token-123")
    assert captured["url"] == "https://idp.example.com/oauth2/introspect"
    assert captured["data"] == {
        "token": "opaque-access-token-123",
        "token_type_hint": "access_token",
    }
    assert captured["auth"] == ("client", "secret")
    assert ctx.viewer_id == "alice"
    assert ctx.roles == ["analyst"]


def test_introspect_mapper_inactive_token_raises_auth_error() -> None:
    """RFC 7662: ``{"active": false}`` is the canonical "this
    token doesn't exist / has been revoked" response. The
    mapper surfaces this as ``AuthError``."""
    from semql.auth import IntrospectMapper

    class FakeClient:
        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            return {"active": False}

    m = IntrospectMapper(
        "https://idp.example.com/oauth2/introspect",
        client_id="c",
        client_secret="s",
        http_client=FakeClient(),
    )
    with pytest.raises(AuthError, match="inactive"):
        m.verify("token")


def test_introspect_mapper_uses_sub_then_username_then_client_id() -> None:
    """Some IdPs return ``sub``, some return ``username`` (older
    Auth0 / Keycloak), some return neither and you have to fall
    back to ``client_id``. The mapper tries them in order so
    the contract is 'we'll get a viewer_id if there's any
    identifier at all'."""
    from semql.auth import IntrospectMapper

    class FakeClient:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            return self._payload

    # username only
    m = IntrospectMapper(
        "https://idp/",
        client_id="c",
        client_secret="s",
        http_client=FakeClient({"active": True, "username": "bob"}),
    )
    assert m.verify("t").viewer_id == "bob"

    # client_id in the response (client-credentials grant).
    # The mapper treats the response's ``client_id`` claim as
    # the token's identity (the token was issued for that
    # client).
    m = IntrospectMapper(
        "https://idp/",
        client_id="oauth-client",
        client_secret="s",
        http_client=FakeClient({"active": True, "client_id": "machine-123"}),
    )
    assert m.verify("t").viewer_id == "machine-123"

    # no identifier — AuthError. The mapper refuses to invent
    # a viewer id from the OAuth client's own credentials
    # (those are the introspect caller, not the token subject).
    m = IntrospectMapper(
        "https://idp/", client_id="c", client_secret="s", http_client=FakeClient({"active": True})
    )
    with pytest.raises(AuthError):
        m.verify("t")


def test_introspect_mapper_handles_scope_string() -> None:
    """RFC 7662 says ``scope`` is a space-delimited string. The
    mapper splits it into a list of roles (overriding any
    ``roles`` claim — ``scope`` is the canonical OAuth
    surface)."""
    from semql.auth import IntrospectMapper

    class FakeClient:
        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            return {
                "active": True,
                "sub": "alice",
                "scope": "read write admin",
                "roles": ["ignored-because-scope-present"],
            }

    m = IntrospectMapper("https://idp/", client_id="c", client_secret="s", http_client=FakeClient())
    ctx = m.verify("t")
    assert ctx.roles == ["read", "write", "admin"]


def test_introspect_mapper_handles_expiry() -> None:
    """RFC 7662: ``exp`` is an integer epoch. The mapper passes
    it through as an ``attrs`` claim so the application can
    enforce its own expiry check (the introspect endpoint
    already validated the token, but the result's ``exp`` is
    useful for cache TTL)."""
    from semql.auth import IntrospectMapper

    class FakeClient:
        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            return {
                "active": True,
                "sub": "alice",
                "exp": 1735689600,
            }

    m = IntrospectMapper("https://idp/", client_id="c", client_secret="s", http_client=FakeClient())
    ctx = m.verify("t")
    assert ctx.attrs["exp"] == 1735689600


def test_introspect_mapper_http_error_raises_auth_error() -> None:
    """A non-2xx response from the introspection endpoint is an
    auth error (the token can't be validated)."""
    from semql.auth import IntrospectMapper

    class FakeClient:
        def post(
            self, url: str, *, data: dict[str, str], auth: tuple[str, str], timeout: float = 5.0
        ) -> dict[str, Any]:
            raise RuntimeError("401 Unauthorized")

    m = IntrospectMapper("https://idp/", client_id="c", client_secret="s", http_client=FakeClient())
    with pytest.raises(AuthError):
        m.verify("t")


def test_introspect_mapper_uses_default_httpx_client() -> None:
    """When no ``http_client`` is supplied, the mapper uses
    ``httpx`` — the canonical async/sync client. Verifies the
    import-time guard works (httpx is required, not optional)."""
    from semql.auth import IntrospectMapper

    m = IntrospectMapper(
        "https://idp.example.com/oauth2/introspect",
        client_id="c",
        client_secret="s",
    )
    # The default client is lazy — created on first ``verify``,
    # so the constructor doesn't fail. Just check the attribute
    # name to avoid a real HTTP call.
    assert m._http_client is not None


def test_introspect_mapper_allows_url_trailing_slash_or_not() -> None:
    """The endpoint URL is whatever the IdP exposes — the
    mapper doesn't normalise it (callers configure the
    canonical form)."""
    from semql.auth import IntrospectMapper

    m = IntrospectMapper("https://idp.example.com/introspect", client_id="c", client_secret="s")
    assert m._introspect_url == "https://idp.example.com/introspect"


# ---------------------------------------------------------------------------
# X509Mapper — mTLS client cert subject / SAN mapping
# ---------------------------------------------------------------------------


def test_x509_mapper_uses_cn_for_viewer_id() -> None:
    """The cert's Common Name is the canonical viewer id for
    mTLS deployments (per RFC 6125 §6.4.4). The mapper
    extracts it from the cert's subject."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="alice", sans=())
    ctx = m.verify(cert)
    assert ctx.viewer_id == "alice"


def test_x509_mapper_uses_ou_for_roles() -> None:
    """Organizational Unit maps to a role — convention for
    grouping certs by team / project. Multiple OUs become
    multiple roles."""
    from semql.auth import X509Mapper

    m = X509Mapper(ou_to_role=True)
    cert = _fake_cert(subject_cn="alice", subject_ou=("Data Platform", "Analytics"))
    ctx = m.verify(cert)
    assert ctx.roles == ["Data Platform", "Analytics"]


def test_x509_mapper_ou_to_role_disabled_keeps_attrs() -> None:
    """When ``ou_to_role=False`` (the default), the OU is kept
    in ``attrs`` for callers that want to inspect it. Opt-in
    to the structural mapping."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="alice", subject_ou=("Data Platform",))
    ctx = m.verify(cert)
    assert ctx.roles == []
    assert "Data Platform" in ctx.attrs.get("subject_ou", [])


def test_x509_mapper_uses_san_for_viewer_when_no_cn() -> None:
    """Some certs (especially service identities) have no CN
    but do have a URI / DNS / email SAN. The mapper falls back
    to the first SAN — URI > DNS > email > IP — in that order."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(
        subject_cn="",
        sans=("spiffe://example.com/service/payments", "payments.example.com"),
    )
    ctx = m.verify(cert)
    assert ctx.viewer_id == "spiffe://example.com/service/payments"


def test_x509_mapper_uses_dns_san_when_no_uri_san() -> None:
    """DNS-only SANs are the common case for service mesh /
    ingress-controller certs. The mapper extracts the DNS name
    as the viewer id."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="", sans=("payments.example.com",))
    ctx = m.verify(cert)
    assert ctx.viewer_id == "payments.example.com"


def test_x509_mapper_uses_email_san_when_no_uri_or_dns() -> None:
    """Email SANs are common in client certs (e.g. per-user
    mTLS). The mapper uses the *full* email as the viewer id —
    the local part alone is not a unique identity (the domain is
    also kept in ``attrs`` via ``sans``)."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="", sans=("alice@example.com",))
    ctx = m.verify(cert)
    assert ctx.viewer_id == "alice@example.com"


def test_x509_mapper_email_san_keeps_distinct_domains_distinct() -> None:
    """Two users with the same local part but different domains
    (alice@a.com vs alice@b.com) are different principals — the
    mapper must not collapse them to a shared viewer id, or one
    tenant's row-level security would leak to the other."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    a = m.verify(_fake_cert(subject_cn="", sans=("alice@a.com",)))
    b = m.verify(_fake_cert(subject_cn="", sans=("alice@b.com",)))
    assert a.viewer_id != b.viewer_id
    assert (a.viewer_id, b.viewer_id) == ("alice@a.com", "alice@b.com")


def test_x509_mapper_no_cn_no_san_raises_auth_error() -> None:
    """A cert with neither CN nor any usable SAN has no
    identity — the mapper raises ``AuthError`` rather than
    emitting a viewer-less context."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="", sans=())
    with pytest.raises(AuthError, match="identity"):
        m.verify(cert)


def test_x509_mapper_preserves_subject_dn_attrs() -> None:
    """The full subject DN lands in ``attrs`` for callers that
    want to inspect the organisation, country, etc. The
    mapper is non-destructive."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(
        subject_cn="alice",
        subject_o="Acme Corp",
        subject_c="US",
    )
    ctx = m.verify(cert)
    assert ctx.attrs.get("subject_o") == "Acme Corp"
    assert ctx.attrs.get("subject_c") == "US"


def test_x509_mapper_preserves_fingerprint() -> None:
    """The cert's SHA-256 fingerprint is a stable identity
    (key rotation changes the cert but a session-scoped
    fingerprint is useful for audit). It lands in ``attrs``."""
    from semql.auth import X509Mapper

    m = X509Mapper()
    cert = _fake_cert(subject_cn="alice", fingerprint="aa:bb:cc")
    ctx = m.verify(cert)
    assert ctx.attrs.get("fingerprint") == "aa:bb:cc"


# ---------------------------------------------------------------------------
# Test helpers — fake cert, fake introspect client
# ---------------------------------------------------------------------------


def _fake_cert(
    *,
    subject_cn: str = "",
    subject_ou: tuple[str, ...] = (),
    subject_o: str | None = None,
    subject_c: str | None = None,
    sans: tuple[str, ...] = (),
    fingerprint: str | None = None,
) -> Any:
    """A duck-typed cert object — the mapper uses
    structural-typing (attribute access), not a hard
    ``cryptography.x509.Certificate`` import, so a plain
    object with the right attribute shape is sufficient for
    unit tests.

    The mapper contract is: ``cert.subject_cn``,
    ``cert.subject_ou``, ``cert.subject_o``,
    ``cert.subject_c``, ``cert.sans``, ``cert.fingerprint``.
    Real certs are decoded into this shape at the call
    site (the mapper doesn't require ``cryptography``
    for the test).
    """

    class _Cert:
        subject_cn: str
        subject_ou: tuple[str, ...]
        subject_o: str | None
        subject_c: str | None
        sans: tuple[str, ...]
        fingerprint: str | None

    c = _Cert()
    c.subject_cn = subject_cn
    c.subject_ou = subject_ou
    c.subject_o = subject_o
    c.subject_c = subject_c
    c.sans = sans
    c.fingerprint = fingerprint
    return c


def test_dict_mapper_protocol_uses_object_typed_identity() -> None:
    """The ``TokenMapper`` Protocol declares ``verify(identity)``
    as accepting ``object`` — the runtime-checked shape. Real
    mappers narrow this to their own type. A bare object
    passes the ``isinstance`` check, but invoking ``verify``
    on it is the mapper's problem."""
    from semql.auth import DictMapper

    # The protocol is duck-typed; the impl narrows.
    m: TokenMapper = DictMapper()
    # Implementation accepts Mapping.
    assert isinstance(m.verify({"sub": "alice"}), AuthContext)


def test_token_mapper_and_token_verifier_are_siblings() -> None:
    """``TokenMapper`` and ``TokenVerifier`` are alternative
    integration points — a caller wires one or the other
    into their request middleware. They are NOT chained
    (a ``TokenMapper`` doesn't depend on ``TokenVerifier``).
    """
    from semql.auth import TokenVerifier

    assert TokenMapper is not TokenVerifier
