"""Navigator caller authentication (REST + gRPC), federation peer tokens, and startup validation."""
from __future__ import annotations

from concurrent import futures

import grpc
import pytest
from fastapi.testclient import TestClient

from navigator.config import Config
from navigator.events import Publisher
from navigator.federation import PeerClient, PeerConfig
from navigator.grpc_server import GRPCServer
from navigator.hooks import HookManager
from navigator.rest import build_app

TOKEN = "navigator-shared-token-0123456789"


# ─── helpers ──────────────────────────────────────────────────────────────────

class _Orch:
    """Just enough orchestrator for the routes under test."""

    def search(self, req, **_):
        from navigator.models import SearchMetadata, SearchResponse
        return SearchResponse(request_id=req.request_id, results=[], metadata=SearchMetadata(), processing_time_ms=0.0)

    def collections(self):
        return []


def _cfg(token: str = "") -> Config:
    cfg = Config()
    cfg.server.auth_token = token
    return cfg


def _client(token: str = "") -> TestClient:
    return TestClient(build_app(_cfg(token), _Orch(), Publisher(""), HookManager()))


SEARCH = {"request_id": "r1", "tenant_id": "acme", "query": "q"}
INDEX = {"document_id": "d1", "tenant_id": "acme", "category": "customer_docs", "title": "t", "content": "hello"}


# ─── REST ─────────────────────────────────────────────────────────────────────

def test_rest_requires_the_token_when_configured():
    c = _client(TOKEN)
    assert c.post("/v1/navigator/search", json=SEARCH).status_code == 401
    assert c.post("/v1/navigator/search", json=SEARCH, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/navigator/search", json=SEARCH, headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert c.post("/v1/navigator/search", json=SEARCH, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_rest_write_endpoints_are_protected_too():
    c = _client(TOKEN)
    assert c.post("/v1/navigator/index", json=INDEX).status_code == 401
    assert c.get("/v1/navigator/collections").status_code == 401


def test_rest_probes_stay_public():
    c = _client(TOKEN)
    for path in ("/v1/health", "/v1/health/live", "/v1/health/ready", "/v1/metrics"):
        assert c.get(path).status_code == 200, path


def test_rest_is_open_when_no_token_is_configured(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    assert _client("").post("/v1/navigator/search", json=SEARCH).status_code == 200


def test_rest_token_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("NAVIGATOR_AUTH_TOKEN", TOKEN)
    c = _client("")
    assert c.post("/v1/navigator/search", json=SEARCH).status_code == 401
    assert c.post("/v1/navigator/search", json=SEARCH, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


# ─── gRPC ─────────────────────────────────────────────────────────────────────

SEARCH_RPC = "/bastion.navigator.v1.NavigatorService/Search"
HEALTH_RPC = "/bastion.navigator.v1.NavigatorService/Health"


@pytest.fixture()
def grpc_channel():
    servers = []

    def make(token: str):
        srv = GRPCServer(_Orch(), Publisher(""), HookManager(), auth_token=token)._make_server()
        port = srv.add_insecure_port("localhost:0")
        srv.start()
        servers.append(srv)
        return grpc.insecure_channel(f"localhost:{port}")

    yield make
    for s in servers:
        s.stop(0)


def _call(channel, method, metadata=None):
    return channel.unary_unary(method)(b"{}", metadata=metadata or [], timeout=5)


def test_grpc_requires_the_token_when_configured(grpc_channel):
    ch = grpc_channel(TOKEN)
    for md in (None, [("authorization", "Bearer nope")], [("authorization", f"Basic {TOKEN}")]):
        with pytest.raises(grpc.RpcError) as e:
            _call(ch, SEARCH_RPC, md)
        assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    # With the right token the call reaches the handler (the empty payload is rejected there, not by auth).
    with pytest.raises(grpc.RpcError) as e:
        _call(ch, SEARCH_RPC, [("authorization", f"Bearer {TOKEN}")])
    assert e.value.code() != grpc.StatusCode.UNAUTHENTICATED


def test_grpc_health_stays_public(grpc_channel):
    assert _call(grpc_channel(TOKEN), HEALTH_RPC)


@pytest.mark.parametrize("service", ["bastion.navigator.v1.NavigatorService", "navigator.NavigatorService"])
def test_grpc_serves_both_service_names_and_both_are_authenticated(grpc_channel, service):
    ch = grpc_channel(TOKEN)
    with pytest.raises(grpc.RpcError) as e:
        _call(ch, f"/{service}/Search")
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED       # not UNIMPLEMENTED: the route exists
    assert _call(ch, f"/{service}/Health")                         # public probe


def test_grpc_is_open_when_no_token_is_configured(grpc_channel, monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    with pytest.raises(grpc.RpcError) as e:
        _call(grpc_channel(""), SEARCH_RPC)
    assert e.value.code() != grpc.StatusCode.UNAUTHENTICATED


# ─── federation: peers present the token ──────────────────────────────────────

def test_peer_calls_carry_the_peers_token():
    peer = PeerConfig(id="b", endpoint="b:9092", auth_token="peer-token")
    md = dict(PeerClient(peer, "a", 1000)._call_metadata(1))
    assert md["authorization"] == "Bearer peer-token"
    assert md["x-hop-depth"] == "1" or "hop" in "".join(md)  # loop-prevention headers are still sent


def test_peer_without_a_token_sends_no_authorization():
    md = dict(PeerClient(PeerConfig(id="b", endpoint="b:9092"), "a", 1000)._call_metadata(0))
    assert "authorization" not in md


# ─── startup validation ───────────────────────────────────────────────────────

def test_startup_refuses_without_an_auth_token(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="auth_token"):
        Config().validate_security()


def test_startup_accepts_a_configured_or_environment_token(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    cfg = Config()
    cfg.server.auth_token = TOKEN
    cfg.validate_security()

    monkeypatch.setenv("NAVIGATOR_AUTH_TOKEN", TOKEN)
    Config().validate_security()


def test_unauthenticated_mode_needs_an_explicit_dev_opt_in(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    cfg = Config()
    assert cfg.server.allow_unauthenticated is False  # secure by default
    cfg.server.allow_unauthenticated = True
    cfg.validate_security()


def test_trusting_caller_permissions_requires_authenticated_callers(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    cfg = Config()
    cfg.server.trust_caller_permissions = True
    with pytest.raises(ValueError, match="trust_caller_permissions"):
        cfg.validate_security()
    # The dev opt-in does not make unauthenticated callers trustworthy.
    cfg.server.allow_unauthenticated = True
    with pytest.raises(ValueError, match="trust_caller_permissions"):
        cfg.validate_security()
    cfg.server.auth_token = TOKEN
    cfg.validate_security()


def test_default_config_does_not_trust_callers():
    assert Config().server.trust_caller_permissions is False


def test_the_shipped_sample_config_requires_the_operator_to_provide_a_token(monkeypatch):
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    cfg = Config.load("config/config.yaml")
    assert cfg.server.auth_token == "" and cfg.server.allow_unauthenticated is False
    with pytest.raises(ValueError):
        cfg.validate_security()
