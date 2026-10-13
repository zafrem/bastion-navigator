"""Navigator -> Vault: authentication, request shape, error handling, and fail-closed permissions."""
from __future__ import annotations

import httpx
import pytest

from navigator.config import Config
from navigator.models import SearchOptions, SearchRequest, UserContext
from navigator.orchestrator import Orchestrator
from navigator.vault_client import NoopVaultClient, VaultClient, VaultUnavailable


# ─── helpers ──────────────────────────────────────────────────────────────────

def _client(handler, **kw) -> VaultClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return VaultClient("http://vault:8081", client=http, **kw)


def _ok(allowed):
    return lambda req: httpx.Response(200, json={"allowed_categories": allowed})


class Recorder:
    """Records every request the client sends and answers with a fixed payload."""

    def __init__(self, allowed=("DC-01",)):
        self.requests: list[httpx.Request] = []
        self.allowed = list(allowed)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"allowed_categories": self.allowed})


# ─── credentials ──────────────────────────────────────────────────────────────

def test_sends_bearer_token_from_config():
    rec = Recorder()
    _client(rec, auth_token="svc-token").allowed_categories("alice", department="marketing", roles=["manager"])
    assert rec.requests[0].headers["authorization"] == "Bearer svc-token"


def test_no_credentials_sends_no_authorization_header(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_VAULT_AUTH_TOKEN", raising=False)
    rec = Recorder()
    _client(rec).allowed_categories("alice", department="marketing", roles=["manager"])
    assert "authorization" not in rec.requests[0].headers


def test_token_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("NAVIGATOR_VAULT_AUTH_TOKEN", "env-token")
    rec = Recorder()
    _client(rec).allowed_categories("alice", department="marketing", roles=["manager"])
    assert rec.requests[0].headers["authorization"] == "Bearer env-token"


def test_token_file_is_reread_so_rotation_needs_no_restart(tmp_path):
    f = tmp_path / "vault.token"
    f.write_text("first\n")
    rec = Recorder()
    c = _client(rec, auth_token_file=str(f))
    c.allowed_categories("alice", department="marketing", roles=["manager"])
    f.write_text("second\n")
    c.allowed_categories("alice", department="marketing", roles=["manager"])
    assert [r.headers["authorization"] for r in rec.requests] == ["Bearer first", "Bearer second"]


def test_unreadable_token_file_fails_closed(tmp_path):
    rec = Recorder()
    c = _client(rec, auth_token_file=str(tmp_path / "missing.token"))
    with pytest.raises(VaultUnavailable):
        c.allowed_categories("alice", department="marketing", roles=["manager"])
    assert rec.requests == []  # never called Vault without the credentials it was told to use


# ─── request shape ────────────────────────────────────────────────────────────

def test_user_id_is_url_quoted_not_a_path_injection():
    rec = Recorder()
    _client(rec).allowed_categories("a/../admin", department="marketing", roles=["manager"])
    assert rec.requests[0].url.raw_path.startswith(b"/v1/vault/permissions/a%2F..%2Fadmin")


def test_sends_department_role_and_tenant_so_vault_decides_for_the_user_not_the_service():
    rec = Recorder()
    _client(rec).allowed_categories("alice", tenant_id="acme", department="marketing", roles=["manager"])
    q = dict(rec.requests[0].url.params)
    assert q == {"tenant_id": "acme", "department": "marketing", "role": "manager"}


def test_multiple_roles_are_unioned():
    def handler(req: httpx.Request) -> httpx.Response:
        role = req.url.params["role"]
        return httpx.Response(200, json={"allowed_categories": ["DC-01"] if role == "analyst" else ["DC-02"]})

    got = _client(handler).allowed_categories("alice", department="marketing", roles=["analyst", "manager"])
    assert sorted(got) == ["customer_data", "manufacturing_data"]


def test_without_department_or_role_access_is_denied_and_vault_is_not_asked():
    # Vault would otherwise answer for the *service* identity's role.
    rec = Recorder()
    assert _client(rec).allowed_categories("alice") == []
    assert rec.requests == []


# ─── response handling ────────────────────────────────────────────────────────

def test_maps_vault_category_ids_to_navigator_categories():
    got = _client(_ok(["DC-01", "DC-02", "DC-03"])).allowed_categories(
        "alice", department="marketing", roles=["manager"])
    assert got == ["customer_data", "manufacturing_data", "hr_data"]


def test_empty_allowed_list_is_returned_as_empty_not_none():
    got = _client(_ok([])).allowed_categories("alice", department="marketing", roles=["manager"])
    assert got == []


@pytest.mark.parametrize("response", [
    httpx.Response(401, json={"error": "unauthorized"}),
    httpx.Response(403, json={"error": "forbidden"}),
    httpx.Response(500, text="boom"),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"unexpected": True}),
])
def test_vault_errors_raise_instead_of_returning_none(response):
    c = _client(lambda req: response)
    with pytest.raises(VaultUnavailable):
        c.allowed_categories("alice", department="marketing", roles=["manager"])


def test_network_error_raises():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(VaultUnavailable):
        _client(handler).allowed_categories("alice", department="marketing", roles=["manager"])


def test_noop_client_means_unrestricted_and_accepts_the_new_arguments():
    assert NoopVaultClient().allowed_categories("alice", tenant_id="t", department="d", roles=["r"]) is None


# ─── orchestrator: fail closed ────────────────────────────────────────────────

class _Searcher:
    def __init__(self):
        self.searched: list[str] = []

    def vector_search(self, collection, vec, filters, over_fetch):
        self.searched.append(collection)
        return []

    def sparse_search(self, collection, query, filters, over_fetch):
        return []

    def collections(self):
        return []


class _Embedder:
    def embed(self, text):
        return [0.1] * 8

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class _Reranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class _FakeVault:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def allowed_categories(self, user_id, **kw):
        self.calls.append((user_id, kw))
        if self.exc:
            raise self.exc
        return self.result


def _orch(vault):
    searcher = _Searcher()
    orch = Orchestrator(cfg=Config(), embedder=_Embedder(), searcher=searcher, reranker=_Reranker(), vault=vault)
    return orch, searcher


def _req(**user_kw):
    return SearchRequest(
        request_id="r1", tenant_id="acme", query="quarterly report",
        user=UserContext(user_id="alice", department="marketing", roles=["manager"], **user_kw),
        options=SearchOptions(top_k=3),
    )


def test_vault_outage_denies_instead_of_searching_every_collection():
    orch, searcher = _orch(_FakeVault(exc=VaultUnavailable("401")))
    assert orch._collections_for_categories(orch._resolve_permissions(_req())) == []
    orch.search(_req())
    assert searcher.searched == []


def test_user_with_no_permitted_categories_gets_no_collections():
    orch, searcher = _orch(_FakeVault(result=[]))
    assert orch._collections_for_categories(orch._resolve_permissions(_req())) == []
    orch.search(_req())
    assert searcher.searched == []


def test_only_permitted_collections_are_searched():
    orch, searcher = _orch(_FakeVault(result=["customer_data"]))
    orch.search(_req())
    assert set(searcher.searched) == {"customer_docs"}


def test_unrecognised_categories_do_not_widen_access():
    orch, _ = _orch(_FakeVault(result=["something-new"]))
    assert orch._collections_for_categories(orch._resolve_permissions(_req())) == []


def test_vault_disabled_is_still_unrestricted():
    orch, _ = _orch(NoopVaultClient())
    cols = orch._collections_for_categories(orch._resolve_permissions(_req()))
    assert set(cols) == {"customer_docs", "manufacturing_docs", "hr_docs"}


def test_orchestrator_passes_tenant_department_and_roles_to_vault():
    vault = _FakeVault(result=["customer_data"])
    orch, _ = _orch(vault)
    orch._resolve_permissions(_req())
    assert vault.calls == [("alice", {"tenant_id": "acme", "department": "marketing", "roles": ["manager"]})]


def test_caller_supplied_categories_can_only_narrow_never_widen():
    # Vault says customer only; the caller claims hr as well. Vault stays authoritative.
    vault = _FakeVault(result=["customer_data"])
    orch, _ = _orch(vault)
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data", "customer_data"])) == ["customer_data"]
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data"])) == []
    assert len(vault.calls) == 2  # Vault was consulted; the claim did not bypass it


def test_caller_may_narrow_what_vault_allows():
    orch, _ = _orch(_FakeVault(result=["customer_data", "hr_data"]))
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data"])) == ["hr_data"]


def test_vault_outage_denies_even_when_the_caller_supplies_categories():
    orch, _ = _orch(_FakeVault(exc=VaultUnavailable("down")))
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data"])) == []


def test_trusted_caller_categories_bypass_vault_only_when_explicitly_enabled():
    vault = _FakeVault(result=[])
    orch, _ = _orch(vault)
    orch._cfg.server.trust_caller_permissions = True
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data"])) == ["hr_data"]
    assert vault.calls == []


def test_vault_disabled_caller_categories_still_restrict():
    orch, _ = _orch(NoopVaultClient())
    assert orch._resolve_permissions(_req(allowed_categories=["hr_data"])) == ["hr_data"]


def test_request_without_a_user_is_denied_when_vault_enforces_permissions():
    def anonymous():
        return SearchRequest(request_id="r2", tenant_id="acme", query="q", options=SearchOptions(top_k=3))

    http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"allowed_categories": ["DC-01"]})))
    orch, searcher = _orch(VaultClient("http://vault:8081", client=http, auth_token="t"))
    assert orch._collections_for_categories(orch._resolve_permissions(anonymous())) == []
    orch.search(anonymous())
    assert searcher.searched == []


def test_request_without_a_user_is_unrestricted_when_vault_is_disabled():
    orch, _ = _orch(NoopVaultClient())
    req = SearchRequest(request_id="r3", tenant_id="acme", query="q", options=SearchOptions(top_k=3))
    assert orch._resolve_permissions(req) is None


def test_anonymous_lookup_does_not_ask_vault():
    rec = Recorder()
    assert _client(rec, auth_token="t").allowed_categories("") == []
    assert rec.requests == []
