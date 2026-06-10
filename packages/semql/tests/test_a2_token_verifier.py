# pyright: reportUnusedImport=false, reportPrivateUsage=false
"""A2 — ``TokenVerifier`` Protocol + reference HMAC / JWKS impls.

The model-side ``AuthContext.attrs`` bag already landed; this
delivers the integration point — a Protocol the platform wires up
to decode incoming bearer tokens into a fully-populated
``AuthContext``. Reference implementations:

  - ``HMACVerifier(secret, algorithm="HS256")`` — HMAC shared-secret
    verification. Extracts ``sub`` → ``viewer_id``, ``roles`` claim
    → ``roles``, remaining claims → ``attrs``.
  - ``JWKSVerifier(jwks_url, algorithms=("RS256",))`` — RS256
    public-key verification against a JWKS endpoint. Same claim
    mapping; caches keys with a configurable TTL.

Both behind ``semql[jwt]`` extras (``PyJWT``); import-time guard
with an actionable ``ImportError``. ``TokenVerifier`` is the
extension point — callers with their own token stack implement it
without depending on the reference impls.
"""

from __future__ import annotations

import time

import pytest
from semql.auth import HMACVerifier, JWKSVerifier, TokenVerifier
from semql.errors import AuthError

# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_hmac_verifier_satisfies_token_verifier_protocol() -> None:
    """``HMACVerifier`` is a runtime-checkable ``TokenVerifier``."""
    v = HMACVerifier(secret=b"k" * 32)
    assert isinstance(v, TokenVerifier)


def test_token_verifier_protocol_has_verify_method() -> None:
    """The Protocol declares ``verify(token) -> AuthContext``."""
    assert hasattr(TokenVerifier, "verify")


# ---------------------------------------------------------------------------
# HMACVerifier — happy path
# ---------------------------------------------------------------------------


def test_hmac_verifier_decodes_sub_to_viewer_id() -> None:
    """A token's ``sub`` claim becomes the viewer's id."""
    import jwt

    secret = b"k" * 32
    token = jwt.encode({"sub": "alice"}, secret, algorithm="HS256")
    v = HMACVerifier(secret=secret)
    ctx = v.verify(token)
    assert ctx.viewer_id == "alice"


def test_hmac_verifier_decodes_roles_claim() -> None:
    """A ``roles`` claim (list) becomes ``AuthContext.roles``."""
    import jwt

    secret = b"k" * 32
    token = jwt.encode({"sub": "alice", "roles": ["hr", "analyst"]}, secret, algorithm="HS256")
    v = HMACVerifier(secret=secret)
    ctx = v.verify(token)
    assert ctx.roles == ["hr", "analyst"]


def test_hmac_verifier_remaining_claims_become_attrs() -> None:
    """Everything except ``sub`` / ``roles`` lands in ``attrs`` with
    the original JSON type preserved (list, bool, int, str)."""
    import jwt

    secret = b"k" * 32
    payload = {
        "sub": "alice",
        "roles": ["hr"],
        "allowed_regions": ["west", "central"],
        "hr_clearance": True,
        "level": 3,
        "org_id": "acme",
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    v = HMACVerifier(secret=secret)
    ctx = v.verify(token)
    assert ctx.attrs["allowed_regions"] == ["west", "central"]
    assert ctx.attrs["hr_clearance"] is True
    assert ctx.attrs["level"] == 3
    assert ctx.attrs["org_id"] == "acme"


# ---------------------------------------------------------------------------
# HMACVerifier — error path
# ---------------------------------------------------------------------------


def test_hmac_verifier_rejects_tampered_token() -> None:
    """A token with a wrong signature raises ``AuthError``."""
    import jwt

    token = jwt.encode({"sub": "alice"}, b"k" * 32, algorithm="HS256")
    v = HMACVerifier(secret=b"other-secret")
    with pytest.raises(AuthError):
        v.verify(token)


def test_hmac_verifier_rejects_expired_token() -> None:
    """An expired token raises ``AuthError``."""
    import jwt

    secret = b"k" * 32
    token = jwt.encode(
        {"sub": "alice", "exp": int(time.time()) - 60},
        secret,
        algorithm="HS256",
    )
    v = HMACVerifier(secret=secret)
    with pytest.raises(AuthError):
        v.verify(token)


def test_hmac_verifier_rejects_malformed_token() -> None:
    """A garbage string raises ``AuthError``, not some other exception."""
    v = HMACVerifier(secret=b"k" * 32)
    with pytest.raises(AuthError):
        v.verify("not.a.token")


def test_hmac_verifier_uses_configured_algorithm() -> None:
    """Custom ``algorithm=`` is honoured."""
    import jwt

    secret = b"k" * 32
    token = jwt.encode({"sub": "alice"}, secret, algorithm="HS512")
    v = HMACVerifier(secret=secret, algorithm="HS512")
    ctx = v.verify(token)
    assert ctx.viewer_id == "alice"


# ---------------------------------------------------------------------------
# AuthError — new leaf in the error hierarchy
# ---------------------------------------------------------------------------


def test_auth_error_is_a_semql_error() -> None:
    """``AuthError`` is a ``SemQLError`` so callers catch the base."""
    from semql.errors import SemQLError

    err = AuthError("bad token")
    assert isinstance(err, SemQLError)


def test_auth_error_carries_message() -> None:
    err = AuthError("token expired")
    assert "expired" in str(err)


# ---------------------------------------------------------------------------
# JWKSVerifier — test double, no live network
# ---------------------------------------------------------------------------


def test_jwks_verifier_rejects_none_algorithm() -> None:
    """The ``none`` algorithm is a security invariant, not a config knob."""
    with pytest.raises(ValueError, match="none"):
        JWKSVerifier(
            jwks_url="https://example.invalid/jwks.json",
            algorithms=("none",),
        )


def test_jwks_verifier_handles_unknown_kid() -> None:
    """A token whose kid isn't in the JWKS raises AuthError."""
    import jwt
    from cryptography.hazmat.primitives import serialization

    # Build a real RSA keypair so we can sign with it.
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_numbers = private_key.public_key().public_numbers()

    def _b64uint(n: int) -> str:
        import base64

        # JWT JWK format: big-endian, no leading zero padding.
        byte_length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(byte_length, "big")).rstrip(b"=").decode("ascii")

    real_kid = "real-kid-1"
    other_kid = "other-kid-2"
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": real_kid,
                "use": "sig",
                "alg": "RS256",
                "n": _b64uint(public_numbers.n),
                "e": _b64uint(public_numbers.e),
            }
        ]
    }

    # Sign a token with a kid that isn't in the JWKS.
    token = jwt.encode(
        {"sub": "alice", "roles": ["hr"]},
        private_pem,
        algorithm="RS256",
        headers={"kid": other_kid},
    )

    # Inject the JWKS by patching ``_fetch_jwks`` to skip the network
    # call (this is the test-double contract).
    verifier = JWKSVerifier(jwks_url="https://example.invalid/jwks.json")
    verifier._fetch_jwks = lambda: jwks  # type: ignore[method-assign]

    with pytest.raises(AuthError) as exc_info:
        verifier.verify(token)
    assert exc_info.value.reason == "unknown_kid"


def test_jwks_verifier_decodes_real_token() -> None:
    """A token signed with a JWKS-published key verifies end-to-end."""
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_numbers = private_key.public_key().public_numbers()

    def _b64uint(n: int) -> str:
        import base64

        byte_length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(byte_length, "big")).rstrip(b"=").decode("ascii")

    kid = "real-kid-1"
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "use": "sig",
                "alg": "RS256",
                "n": _b64uint(public_numbers.n),
                "e": _b64uint(public_numbers.e),
            }
        ]
    }
    token = jwt.encode(
        {
            "sub": "alice",
            "roles": ["hr", "analyst"],
            "allowed_regions": ["west"],
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )

    verifier = JWKSVerifier(jwks_url="https://example.invalid/jwks.json")
    verifier._fetch_jwks = lambda: jwks  # type: ignore[method-assign]

    ctx = verifier.verify(token)
    assert ctx.viewer_id == "alice"
    assert ctx.roles == ["hr", "analyst"]
    assert ctx.attrs["allowed_regions"] == ["west"]


def test_jwks_verifier_refetches_on_cache_miss() -> None:
    """When the cached JWKS has no matching kid, refetch once."""
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_numbers = private_key.public_key().public_numbers()

    def _b64uint(n: int) -> str:
        import base64

        byte_length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(byte_length, "big")).rstrip(b"=").decode("ascii")

    kid = "real-kid-1"
    rotated_jwks: dict[str, object] = {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "use": "sig",
                "alg": "RS256",
                "n": _b64uint(public_numbers.n),
                "e": _b64uint(public_numbers.e),
            }
        ]
    }
    token = jwt.encode(
        {"sub": "alice"},
        private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )

    call_count = {"n": 0}

    def _fetch() -> dict[str, object]:
        call_count["n"] += 1
        return rotated_jwks

    verifier = JWKSVerifier(jwks_url="https://example.invalid/jwks.json")
    verifier._fetch_jwks = _fetch  # type: ignore[method-assign]
    # Pre-populate the cache with an empty / stale JWKS so the
    # first lookup misses.
    verifier._cached_jwks = {"keys": []}
    verifier._cached_at = 0.0

    ctx = verifier.verify(token)
    assert ctx.viewer_id == "alice"
    # The verifier pre-populated the cache with an empty JWKS, so the
    # first _fetch_jwks() returns the cached value (no lambda call).
    # On cache miss (no matching kid) it clears the cache and refetches —
    # the lambda fires once, returning the rotated JWKS.
    assert call_count["n"] == 1
