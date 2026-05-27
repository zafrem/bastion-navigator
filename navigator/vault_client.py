"""Vault permission client for Navigator (doc 11 §4 / doc 21 §6)."""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from .models import SearchResult

log = logging.getLogger(__name__)

# Maps data category strings to collection names.
_CATEGORY_MAP: dict[str, str] = {
    "customer_data": "customer_docs",
    "manufacturing_data": "manufacturing_docs",
    "hr_data": "hr_docs",
}


class VaultClient:
    """Fetches per-user allowed_categories from the Vault REST API."""

    def __init__(self, endpoint: str, timeout: float = 5.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def allowed_categories(self, user_id: str) -> Optional[list[str]]:
        """Return the list of allowed data categories for user_id, or None on error."""
        try:
            resp = httpx.get(
                f"{self._endpoint}/v1/vault/permissions/{user_id}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("allowed_categories")
        except Exception as exc:
            log.warning("[vault_client] failed to fetch permissions for %s: %s", user_id, exc)
            return None

    def filter_results(self, results: list[SearchResult], allowed: list[str]) -> list[SearchResult]:
        """Remove results whose data_category is not in allowed."""
        allowed_set = set(allowed)
        return [r for r in results if r.metadata.get("data_category", "") in allowed_set]


class NoopVaultClient:
    """No-op Vault client — allows everything (used when Vault is disabled)."""

    def allowed_categories(self, user_id: str) -> Optional[list[str]]:  # noqa: ARG002
        return None

    def filter_results(self, results: list[SearchResult], allowed: list[str]) -> list[SearchResult]:  # noqa: ARG002
        return results
