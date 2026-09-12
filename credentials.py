"""
Server-side credential storage.

Why this exists:
    Connection details were previously round-tripped through a Dash `dcc.Store`,
    which lives in the browser. That meant a connection string like

        postgresql://admin:SuperSecret@prod-db.company.com:5432/analytics

    was serialised into the customer's DOM, readable by any browser extension,
    anyone with devtools open, or an XSS injection. It would also fail any
    serious security review.

    Credentials now stay on the server. The browser only ever sees an opaque
    connection id, which is useless on its own.

Two kinds of credential:
    • Static secrets — API keys, database URLs, service-account JSON
    • OAuth tokens — access + refresh pairs that expire and must be renewed

Encryption:
    Secrets are encrypted at rest with Fernet. The key comes from
    INSIGHTERA_SECRET_KEY. If that variable is absent the store still works but
    refuses to persist to disk, so a misconfigured deployment fails loudly at
    startup rather than silently writing plaintext secrets to a file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
    _HAS_CRYPTO = True
except ImportError:                                    # pragma: no cover
    _HAS_CRYPTO = False
    InvalidToken = Exception                           # type: ignore

_ENV_KEY = "INSIGHTERA_SECRET_KEY"


# ══════════════════════════════════════════════════════════════════════════════
#  ENCRYPTION
# ══════════════════════════════════════════════════════════════════════════════

def _derive_key(raw: str) -> bytes:
    """Fernet needs a 32-byte urlsafe-base64 key; accept any passphrase."""
    return base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())


class SecretBox:
    """Encrypt/decrypt secret values.

    Falls back to an in-memory ephemeral key when none is configured. That keeps
    local development working, but `can_persist` is False so nothing is ever
    written to disk in plaintext or under a key that vanishes on restart.
    """

    def __init__(self, key: Optional[str] = None):
        self._configured = bool(key or os.environ.get(_ENV_KEY))
        raw = key or os.environ.get(_ENV_KEY) or secrets.token_urlsafe(32)
        self._fernet = Fernet(_derive_key(raw)) if _HAS_CRYPTO else None

    @property
    def can_persist(self) -> bool:
        """True only when a real key is configured and crypto is available."""
        return self._configured and _HAS_CRYPTO

    @property
    def available(self) -> bool:
        return _HAS_CRYPTO

    def encrypt(self, value: str) -> str:
        if not self._fernet:
            raise RuntimeError(
                "cryptography is not installed — cannot encrypt credentials. "
                "Install it with: pip install cryptography")
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, token: str) -> str:
        if not self._fernet:
            raise RuntimeError("cryptography is not installed.")
        return self._fernet.decrypt(token.encode()).decode()


# ══════════════════════════════════════════════════════════════════════════════
#  RECORDS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Connection:
    """One configured data source for one workspace."""
    connection_id: str
    workspace_id: str
    provider: str                       # "stripe" | "google_ads" | "postgres" ...
    kind: str                           # "api_key" | "oauth" | "database"
    label: str = ""
    secret_enc: str = ""                # encrypted payload (JSON)
    created_at: str = ""
    # OAuth-only
    expires_at: Optional[str] = None
    scopes: list[str] = field(default_factory=list)
    # Operational
    last_sync_at: Optional[str] = None
    last_error: Optional[str] = None

    def public_view(self) -> dict:
        """Safe to send to the browser — no secret material."""
        return {
            "connection_id": self.connection_id,
            "provider": self.provider,
            "kind": self.kind,
            "label": self.label,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
            "healthy": self.last_error is None,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STORE
# ══════════════════════════════════════════════════════════════════════════════

class CredentialStore:
    """Server-side connection registry, keyed by opaque connection id.

    Deliberately simple — a JSON file behind a lock. The interface is what
    matters; swapping the backing store for Postgres later changes only the
    `_load`/`_persist` methods.
    """

    def __init__(self, path: Optional[str] = None,
                 secret_box: Optional[SecretBox] = None):
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".credentials", "connections.json")
        self.box = secret_box or SecretBox()
        self._lock = threading.Lock()
        self._conns: dict[str, Connection] = {}
        self._load()

    # ── persistence ─────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            self._conns = {k: Connection(**v) for k, v in raw.items()}
        except Exception:
            self._conns = {}

    def _persist(self) -> None:
        if not self.box.can_persist:
            # Refuse to write rather than persist under an ephemeral key.
            return
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({k: asdict(v) for k, v in self._conns.items()}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    # ── writes ──────────────────────────────────────────────────────────
    def add(self, workspace_id: str, provider: str, kind: str,
            secret: dict, label: str = "",
            expires_at: Optional[datetime] = None,
            scopes: Optional[list[str]] = None) -> str:
        """Store a credential. Returns the opaque id safe to hand to the UI."""
        cid = f"conn_{secrets.token_urlsafe(16)}"
        with self._lock:
            self._conns[cid] = Connection(
                connection_id=cid,
                workspace_id=workspace_id,
                provider=provider,
                kind=kind,
                label=label or provider,
                secret_enc=self.box.encrypt(json.dumps(secret)),
                created_at=datetime.now(timezone.utc).isoformat(),
                expires_at=expires_at.isoformat() if expires_at else None,
                scopes=scopes or [],
            )
            self._persist()
        return cid

    def update_secret(self, connection_id: str, secret: dict,
                      expires_at: Optional[datetime] = None) -> None:
        with self._lock:
            c = self._conns[connection_id]
            c.secret_enc = self.box.encrypt(json.dumps(secret))
            if expires_at:
                c.expires_at = expires_at.isoformat()
            self._persist()

    def mark_sync(self, connection_id: str, error: Optional[str] = None) -> None:
        with self._lock:
            c = self._conns.get(connection_id)
            if not c:
                return
            c.last_sync_at = datetime.now(timezone.utc).isoformat()
            c.last_error = error
            self._persist()

    def delete(self, connection_id: str) -> bool:
        """Remove a connection and its secret. Used by the disconnect flow —
        required for the platform data-deletion commitments."""
        with self._lock:
            existed = self._conns.pop(connection_id, None) is not None
            if existed:
                self._persist()
        return existed

    # ── reads ───────────────────────────────────────────────────────────
    def get_secret(self, connection_id: str) -> dict:
        """Decrypt and return secret material. Server-side callers only."""
        c = self._conns.get(connection_id)
        if c is None:
            raise KeyError(f"unknown connection: {connection_id}")
        try:
            return json.loads(self.box.decrypt(c.secret_enc))
        except InvalidToken as e:
            raise RuntimeError(
                f"Could not decrypt credentials for {connection_id}. The "
                f"{_ENV_KEY} in use does not match the one they were stored "
                f"with.") from e

    def get(self, connection_id: str) -> Optional[Connection]:
        return self._conns.get(connection_id)

    def list_for_workspace(self, workspace_id: str) -> list[Connection]:
        return [c for c in self._conns.values()
                if c.workspace_id == workspace_id]

    def public_list(self, workspace_id: str) -> list[dict]:
        """What the UI is allowed to see."""
        return [c.public_view() for c in self.list_for_workspace(workspace_id)]

    def delete_workspace(self, workspace_id: str) -> int:
        """Delete every credential for a workspace — account termination path."""
        with self._lock:
            ids = [k for k, v in self._conns.items()
                   if v.workspace_id == workspace_id]
            for k in ids:
                del self._conns[k]
            if ids:
                self._persist()
        return len(ids)


# ══════════════════════════════════════════════════════════════════════════════
#  OAUTH
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OAuthToken:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scopes: list[str] = field(default_factory=list)

    @property
    def expired(self) -> bool:
        if self.expires_at is None:
            return False
        # Refresh slightly early so a token can't expire mid-request.
        return datetime.now(timezone.utc) >= self.expires_at - timedelta(minutes=5)


class OAuthTokenStore:
    """OAuth tokens with automatic refresh.

    Six of the ten planned connectors need this. Each registers a refresh
    callable once; individual connectors then just ask for a valid token and
    never deal with expiry themselves.
    """

    def __init__(self, store: Optional[CredentialStore] = None):
        self.store = store or CredentialStore()
        self._refreshers: dict[str, Callable[[str], OAuthToken]] = {}
        self._lock = threading.Lock()

    def register_refresher(self, provider: str,
                           fn: Callable[[str], OAuthToken]) -> None:
        """`fn(refresh_token) -> OAuthToken` for one provider."""
        self._refreshers[provider] = fn

    def save(self, workspace_id: str, provider: str, token: OAuthToken,
             label: str = "") -> str:
        return self.store.add(
            workspace_id, provider, "oauth",
            secret={"access_token": token.access_token,
                    "refresh_token": token.refresh_token},
            label=label, expires_at=token.expires_at, scopes=token.scopes)

    def _read(self, connection_id: str) -> OAuthToken:
        c = self.store.get(connection_id)
        if c is None:
            raise KeyError(f"unknown connection: {connection_id}")
        s = self.store.get_secret(connection_id)
        return OAuthToken(
            access_token=s.get("access_token", ""),
            refresh_token=s.get("refresh_token"),
            expires_at=(datetime.fromisoformat(c.expires_at)
                        if c.expires_at else None),
            scopes=c.scopes,
        )

    def get_access_token(self, connection_id: str) -> str:
        """Return a currently-valid access token, refreshing if needed."""
        with self._lock:
            token = self._read(connection_id)
            if not token.expired:
                return token.access_token

            conn = self.store.get(connection_id)
            refresher = self._refreshers.get(conn.provider) if conn else None
            if refresher is None or not token.refresh_token:
                raise RuntimeError(
                    f"Token for {connection_id} has expired and cannot be "
                    f"refreshed automatically. The customer needs to reconnect "
                    f"{conn.provider if conn else 'this source'}.")

            fresh = refresher(token.refresh_token)
            self.store.update_secret(
                connection_id,
                {"access_token": fresh.access_token,
                 "refresh_token": fresh.refresh_token or token.refresh_token},
                expires_at=fresh.expires_at)
            return fresh.access_token


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL DEFAULT
# ══════════════════════════════════════════════════════════════════════════════

_default_store: Optional[CredentialStore] = None
_default_oauth: Optional[OAuthTokenStore] = None


def default_store() -> CredentialStore:
    global _default_store
    if _default_store is None:
        _default_store = CredentialStore()
    return _default_store


def default_oauth_store() -> OAuthTokenStore:
    global _default_oauth
    if _default_oauth is None:
        _default_oauth = OAuthTokenStore(default_store())
    return _default_oauth


def startup_check() -> list[str]:
    """Configuration warnings to surface at boot."""
    warnings: list[str] = []
    if not _HAS_CRYPTO:
        warnings.append(
            "cryptography is not installed — credentials cannot be encrypted. "
            "Install with: pip install cryptography")
    if not os.environ.get(_ENV_KEY):
        warnings.append(
            f"{_ENV_KEY} is not set — credentials will be kept in memory only "
            f"and lost on restart. Set it before storing real credentials.")
    return warnings
