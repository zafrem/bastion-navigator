"""Navigator service entry point: starts FastAPI (REST) + gRPC in parallel."""
from __future__ import annotations

import argparse
import logging
import socket
import threading

import uvicorn

from .config import Config
from .embedder import build as build_embedder
from .reranker import build as build_reranker
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
from .events import Publisher
from .hooks import HookManager
from .rest import build_app
from .grpc_server import GRPCServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_instance_id(cfg: Config) -> str:
    if cfg.instance_id:
        return cfg.instance_id
    try:
        return socket.gethostname()
    except Exception:
        return "navigator-unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bastion Navigator service")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    log.info("[navigator] starting v%s mode=%s (REST :%d, gRPC :%d)",
             cfg.version, cfg.mode, cfg.server.rest_port, cfg.server.grpc_port)

    # Build base components
    embedder = build_embedder(cfg.embedder)
    reranker = build_reranker(cfg.reranker)

    try:
        searcher = QdrantSearcher(cfg.vector_db.hosts)
    except Exception as exc:
        log.warning("[navigator] qdrant unavailable (%s), using mock searcher", exc)
        searcher = MockSearcher()

    vault = VaultClient(cfg.vault.endpoint) if cfg.vault.enabled else NoopVaultClient()

    pub = Publisher(cfg.events.nats_url)
    rewriter = TokenRewriter(token_pattern=cfg.token_rewriter.token_pattern)
    base_orch = Orchestrator(cfg, embedder, searcher, reranker, vault, rewriter=rewriter, publisher=pub)

    if cfg.modular_rag.enabled:
        base_orch.configure_modular_rag(Router(cfg.modular_rag.router), Evaluator())
        log.info("[navigator] modular RAG enabled (max_iterations=%d)", cfg.modular_rag.loop.max_iterations)

    # Build orchestrator based on mode
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

    # gRPC in background thread
    grpc_srv = GRPCServer(orch, pub, hm)
    grpc_thread = threading.Thread(
        target=grpc_srv.serve,
        args=(cfg.server.grpc_port,),
        daemon=True,
        name="navigator-grpc",
    )
    grpc_thread.start()

    # REST in main thread
    app = build_app(cfg, orch, pub, hm)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=cfg.server.rest_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
