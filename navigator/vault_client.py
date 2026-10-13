"""Vault permission client for Navigator (doc 11 §4 / doc 21 §6).

Navigator asks Vault which data categories a user may read, then searches only the matching
collections. Vault's REST API requires a Bearer JWT, so this client authenticates with a
service token. It never guesses: if Vault cannot give a trustworthy answer it raises
``VaultUnavailable`` and the caller must deny access, not widen it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import quote

import httpx

from .models import SearchResult

log = logging.getLogger(__name__)

# Environment fallback for the service token (used when no token or token file is configured).
ENV_TOKEN = "NAVIGATOR_VAULT_AUTH_TOKEN"

# Vault reports data categories as IDs (vault/internal/model: DC-01..DC-03);
# Navigator names them by content (orchestrator._CATEGORY_TO_COLLECTION).
_VAULT_CATEGORY_TO_NAVIGATOR: dict[str, str] = {
    "DC-01": "customer_data",
    "DC-02": "manufacturing_data",
    "DC-03": "hr_data",
}


class VaultUnavailable(Exception):
    """Vault could not answer (network error, rejected credentials, bad reply)."""


class VaultClient:
    """Fetches per-user allowed categories from the Vault REST API, authenticating as a service."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 5.0,
        auth_token: str = "",
        auth_token_file: str = "",
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._auth_token = auth_token
        self._auth_token_file = auth_token_file
        self._http = client or httpx.Client(timeout=timeout)
        if not (auth_token or auth_token_file or os.environ.get(ENV_TOKEN)):
            log.warning(
                "[vault_client] no Vault credentials configured (vault.auth_token, vault.auth_token_file "
                "or %s): requests only succeed if Vault runs with auth.skip_auth", ENV_TOKEN,
            )

    # ── credentials ──────────────────────────────────────────────────────────

    def _token(self) -> str:
        """Current service token. A token file is re-read on every call so rotation needs no restart."""
        if self._auth_token_file:
            try:
                tok = Path(self._auth_token_file).read_text().strip()
            except OSError as exc:
                raise VaultUnavailable(f"cannot read Vault token file: {exc}") from exc
            if not tok:
                raise VaultUnavailable("Vault token file is empty")
            return tok
        return self._auth_token or os.environ.get(ENV_TOKEN, "")

    def _headers(self) -> dict[str, str]:
        tok = self._token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    # ── permissions ──────────────────────────────────────────────────────────

    def allowed_categories(
        self,
        user_id: str,
        *,
        tenant_id: str = "",
        department: str = "",
        roles: Sequence[str] = (),
    ) -> list[str]:
        """Navigator category names the user may read; ``[]`` means no access.

        Vault decides from department and role. Without them it would answer for the *service's*
        own role, so when they are missing no request is made and access is denied.

        Raises VaultUnavailable when Vault cannot be consulted.
        """
        roles = [r for r in (roles or ()) if r]
        if not department or not roles:
            log.warning("[vault_client] no department/role for %s: cannot determine permissions, denying", user_id)
            return []

        headers = self._headers()
        allowed: list[str] = []
        for role in roles:
            params = {"department": department, "role": role}
            if tenant_id:
                params["tenant_id"] = tenant_id
            for cat in self._fetch(user_id, params, headers):
                if cat not in allowed:
                    allowed.append(cat)
        return allowed

    def _fetch(self, user_id: str, params: dict[str, str], headers: dict[str, str]) -> list[str]:
        url = f"{self._endpoint}/v1/vault/permissions/{quote(user_id, safe='')}"
        try:
            resp = self._http.get(url, params=params, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"Vault request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VaultUnavailable(f"Vault returned HTTP {resp.status_code}")
        try:
            cats = resp.json()["allowed_categories"]
        except (ValueError, KeyError, TypeError) as exc:
            raise VaultUnavailable(f"unexpected Vault response: {exc!r}") from exc
        if not isinstance(cats, list):
            raise VaultUnavailable("unexpected Vault response: allowed_categories is not a list")
        return [_VAULT_CATEGORY_TO_NAVIGATOR.get(c, c) for c in cats if isinstance(c, str)]

    def filter_results(self, results: list[SearchResult], allowed: list[str]) -> list[SearchResult]:
        """Remove results whose data_category is not in allowed."""
        allowed_set = set(allowed)
        return [r for r in results if r.metadata.get("data_category", "") in allowed_set]


class NoopVaultClient:
    """No-op Vault client: Vault is disabled, so nothing is restricted (``None`` = no restriction)."""

    def allowed_categories(self, user_id: str, **_: object) -> Optional[list[str]]:  # noqa: ARG002
        return None

    def filter_results(self, results: list[SearchResult], allowed: list[str]) -> list[SearchResult]:  # noqa: ARG002
        return results
