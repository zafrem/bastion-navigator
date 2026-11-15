"""Navigator service entry point: starts FastAPI (REST) + gRPC in parallel."""
from __future__ import annotations

import argparse
import logging
import os
import socket
import threading

import uvicorn

from .config import Config
from .embedder import build as build_embedder, MockEmbedder
from .reranker import build as build_reranker, MockReranker
from .searcher import QdrantSearcher, MockSearcher
from .vault_client import VaultClient, NoopVaultClient
from .orchestrator import Orchestrator
from .token_rewriter import TokenRewriter
from .router import Router
from .evaluator import Evaluator
from .federation import (
    FederationConfig as _FedCfg,
    PeerConfig as _PeerCfg,
    build_federated_orchestrator,
)
from .auth import resolve_token
from .events import Publisher
from .hooks import HookManager
from .rest import build_app as _build_rest_app
from .grpc_server import GRPCServer
from .tls import TLSConfigError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_instance_id(cfg: Config) -> str:
    if cfg.instance_id:
        return cfg.instance_id
    try:
        return socket.gethostname()
    except Exception:
        return "navigator-unknown"


def _or_mock(cfg: Config, what: str, build, mock):
    """Build a real component; fall back to its mock only in explicit development mode.

    A silent fallback in production would serve meaningless embeddings, ranking or search results behind a
    healthy status, so outside ``server.allow_unauthenticated`` a failure stops startup instead.
    """
    try:
        return build()
    except Exception as exc:
        if not cfg.server.allow_unauthenticated:
            raise RuntimeError(
                f"{what} unavailable ({exc}); refusing to fall back to a mock outside development "
                "(server.allow_unauthenticated)"
            ) from exc
        log.warning("[navigator] %s unavailable (%s), using mock (development mode)", what, exc)
        return mock()


def _build_runtime(cfg: Config) -> tuple[Orchestrator, Publisher, HookManager]:
    embedder = _or_mock(cfg, "embedder", lambda: build_embedder(cfg.embedder), MockEmbedder)
    reranker = _or_mock(cfg, "reranker", lambda: build_reranker(cfg.reranker), MockReranker)
    searcher = _or_mock(cfg, "vector search", lambda: QdrantSearcher(cfg.vector_db.hosts), MockSearcher)

    vault = (
        VaultClient(
            cfg.vault.endpoint,
            auth_token=cfg.vault.auth_token,
            auth_token_file=cfg.vault.auth_token_file,
        )
        if cfg.vault.enabled
        else NoopVaultClient()
    )

    pub = Publisher(cfg.events.nats_url)
    rewriter = TokenRewriter(token_pattern=cfg.token_rewriter.token_pattern)
    base_orch = Orchestrator(cfg, embedder, searcher, reranker, vault, rewriter=rewriter, publisher=pub)

    if cfg.modular_rag.enabled:
        base_orch.configure_modular_rag(Router(cfg.modular_rag.router), Evaluator())
        log.info("[navigator] modular RAG enabled (max_iterations=%d)", cfg.modular_rag.loop.max_iterations)

    if cfg.mode in ("federation", "agent") and cfg.federation.peers:
        own_id = _resolve_instance_id(cfg)
        fed_cfg = _FedCfg(
            confidence_threshold=cfg.federation.confidence_threshold,
            routing_threshold=cfg.federation.routing_threshold,
            max_peers_per_query=cfg.federation.max_peers_per_query,
            max_depth=cfg.federation.max_depth,
            peer_timeout_ms=cfg.federation.peer_timeout_ms,
            rrf_k=cfg.federation.rrf_k,
            peers=[
                _PeerCfg(
                    id=p.id,
                    endpoint=p.endpoint,
                    topic_affinity=p.topic_affinity,
                    capability=p.capability,
                    # Peers are authenticated like any other caller: use the peer's own token,
                    # else this node's shared token.
                    auth_token=p.auth_token or resolve_token(cfg.server.auth_token),
                    tls=p.tls,
                    tls_ca_file=p.tls_ca_file,
                    tls_server_name=p.tls_server_name,
                    tls_client_cert=p.tls_client_cert,
                    tls_client_key=p.tls_client_key,
                )
                for p in cfg.federation.peers
            ],
        )
        orch = build_federated_orchestrator(base_orch, fed_cfg, own_id)
        log.info("[navigator] federation enabled with %d peers", len(cfg.federation.peers))
    else:
        orch = base_orch
        log.info("[navigator] mode=search (standalone)")

    hm = HookManager()
    return orch, pub, hm


def build_app(
    cfg: Config | None = None,
    orch: Orchestrator | None = None,
    pub: Publisher | None = None,
    hm: HookManager | None = None,
):
    """Build the FastAPI application.

    When called with no arguments (e.g. ``uvicorn navigator.main:build_app --factory``),
    loads configuration from ``CONFIG_PATH`` (default ``config.yaml``) and constructs
    default runtime components.
    """
    if cfg is None:
        cfg_path = os.environ.get("CONFIG_PATH", "config.yaml")
        cfg = Config.load(cfg_path)
        if cfg_path == os.devnull:
            cfg.embedder.type = "mock"
            cfg.reranker.type = "mock"
        # The factory is an entry point of its own (uvicorn --factory): apply the same startup checks as main().
        try:
            cfg.validate_security()
        except ValueError as exc:
            raise SystemExit(f"refusing to start: {exc}")
    if orch is None or pub is None or hm is None:
        default_orch, default_pub, default_hm = _build_runtime(cfg)
        orch = orch or default_orch
        pub = pub or default_pub
        hm = hm or default_hm
    return _build_rest_app(cfg, orch, pub, hm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bastion Navigator service")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    try:
        cfg.validate_security()
    except ValueError as exc:
        raise SystemExit(f"refusing to start: {exc}")
    if cfg.server.allow_unauthenticated:
        log.warning("[navigator] allow_unauthenticated is set: callers are NOT authenticated (development only)")
    log.info("[navigator] starting v%s mode=%s (REST :%d, gRPC :%d)",
             cfg.version, cfg.mode, cfg.server.rest_port, cfg.server.grpc_port)

    try:
        orch, pub, hm = _build_runtime(cfg)
    except RuntimeError as exc:
        raise SystemExit(f"refusing to start: {exc}")

    # gRPC in background thread
    try:
        grpc_srv = GRPCServer(
            orch, pub, hm,
            auth_token=cfg.server.auth_token,
            tls_cert=cfg.server.grpc_tls_cert,
            tls_key=cfg.server.grpc_tls_key,
            tls_client_ca=cfg.server.grpc_tls_client_ca,
        )
    except TLSConfigError as exc:
        raise SystemExit(f"refusing to start: gRPC TLS: {exc}")
    grpc_thread = threading.Thread(
        target=grpc_srv.serve,
        args=(cfg.server.grpc_port,),
        daemon=True,
        name="navigator-grpc",
    )
    grpc_thread.start()

    # REST in main thread
    app = _build_rest_app(cfg, orch, pub, hm)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=cfg.server.rest_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
