"""Navigator Federation extension — distributed vector search across peer Navigators.

Implements FederationRouter (topic-affinity + confidence-based peer selection)
and PeerClient (gRPC calls to peer Search endpoints with loop-prevention headers).

See docs/22_extension_navigator_federation_v1.md.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import grpc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import SearchRequest, SearchResponse, SearchResult, SearchMetadata

log = logging.getLogger(__name__)


# ─── config models ────────────────────────────────────────────────────────────

@dataclass
class PeerConfig:
    id: str
    endpoint: str
    topic_affinity: list[str] = field(default_factory=list)
    capability: str = "search"          # "search" | "agent"
    affinity_embedding: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


@dataclass
class FederationConfig:
    confidence_threshold: float = 0.70
    routing_threshold: float = 0.40
    max_peers_per_query: int = 3
    max_depth: int = 2
    peer_timeout_ms: int = 2000
    rrf_k: float = 60.0
    peers: list[PeerConfig] = field(default_factory=list)


# ─── loop-prevention constants ────────────────────────────────────────────────

_HEADER_ORIGIN_ID = "x-origin-id"
_HEADER_HOP_DEPTH = "x-hop-depth"


# ─── FederationRouter ─────────────────────────────────────────────────────────

class FederationRouter:
    """Selects which peers to query based on topic affinity and local confidence.

    Affinity embeddings for peers are pre-computed at startup so no per-query
    embedding of peer tags is needed (NFR-FED-007).
    """

    def __init__(self, peers: list[PeerConfig], embedder) -> None:
        self._peers = peers
        self._embedder = embedder
        self._precompute_affinity_embeddings()

    def _precompute_affinity_embeddings(self) -> None:
        for peer in self._peers:
            if peer.topic_affinity:
                tag_text = " ".join(peer.topic_affinity)
                try:
                    peer.affinity_embedding = np.array(self._embedder.embed(tag_text))
                except Exception as exc:
                    log.warning("federation: could not embed affinity tags for peer %s: %s", peer.id, exc)
                    peer.affinity_embedding = None

    def route(
        self,
        query_embedding: list[float],
        local_confidence: float,
        cfg: FederationConfig,
        origin_id: str = "",
        hop_depth: int = 0,
    ) -> list[PeerConfig]:
        """Return peers to query. Returns [] when local results are sufficient."""
        if local_confidence >= cfg.confidence_threshold:
            return []
        if hop_depth >= cfg.max_depth:
            return []

        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored: list[tuple[float, PeerConfig]] = []
        for peer in self._peers:
            if peer.id == origin_id:
                continue  # loop prevention
            if peer.affinity_embedding is None:
                continue
            a_norm = np.linalg.norm(peer.affinity_embedding)
            if a_norm == 0:
                continue
            score = float(np.dot(q_vec, peer.affinity_embedding) / (q_norm * a_norm))
            if score >= cfg.routing_threshold:
                scored.append((score, peer))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[: cfg.max_peers_per_query]]


# ─── PeerClient ───────────────────────────────────────────────────────────────

class PeerClient:
    """gRPC client for a single peer Navigator.

    Uses the peer's existing Search endpoint — no new interface required on the
    peer side. Loop-prevention headers are injected on every call.
    """

    def __init__(self, peer: PeerConfig, own_id: str, timeout_ms: int) -> None:
        self._peer = peer
        self._own_id = own_id
        self._timeout_s = timeout_ms / 1000.0

    async def search(
        self,
        req: SearchRequest,
        hop_depth: int,
    ) -> Optional[SearchResponse]:
        """Send a Search call to the peer. Returns None on timeout or error."""
        try:
            return await asyncio.wait_for(
                self._do_search(req, hop_depth),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("federation: peer %s timed out after %.1fs", self._peer.id, self._timeout_s)
            return None
        except Exception as exc:
            log.warning("federation: peer %s error: %s", self._peer.id, exc)
            return None

    async def _do_search(self, req: SearchRequest, hop_depth: int) -> SearchResponse:
        metadata = [
            (_HEADER_ORIGIN_ID, self._own_id),
            (_HEADER_HOP_DEPTH, str(hop_depth)),
        ]
        channel = grpc.aio.insecure_channel(self._peer.endpoint)
        try:
            payload = json.dumps(req.model_dump()).encode()
            stub_response = await channel.unary_unary(
                "/bastion.navigator.v1.NavigatorService/Search",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )(payload, metadata=metadata)
            data = json.loads(stub_response)
            return SearchResponse.model_validate(data)
        finally:
            await channel.close()

    async def agent_generate(
        self,
        query: str,
        context_results: list[SearchResult],
        tenant_id: str,
        max_tokens: int = 500,
    ) -> Optional[dict]:
        """Call the peer's agent/generate endpoint (agent mode only)."""
        import httpx
        host = self._peer.endpoint.split(":")[0]
        port_str = self._peer.endpoint.split(":")[-1] if ":" in self._peer.endpoint else "8082"
        url = f"http://{host}:{port_str}/v1/navigator/agent/generate"
        payload = {
            "query": query,
            "context": [r.model_dump() for r in context_results],
            "max_tokens": max_tokens,
            "tenant_id": tenant_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            log.warning("federation: agent generate from peer %s failed: %s", self._peer.id, exc)
            return None


# ─── RRF merge ────────────────────────────────────────────────────────────────

def rrf_merge_multi(
    result_lists: list[list[SearchResult]],
    k: float = 60.0,
) -> list[SearchResult]:
    """Merge N result lists using Reciprocal Rank Fusion.

    Extends the existing 2-list RRF to support federation across N sources.
    Formula: score(d) = Σ 1 / (k + rank_i(d))
    """
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, r in enumerate(result_list):
            scores[r.document_id] = scores.get(r.document_id, 0.0) + 1.0 / (k + rank + 1)
            if r.document_id not in by_id:
                by_id[r.document_id] = r

    out = []
    for doc_id, result in by_id.items():
        r = result.model_copy()
        r.score = scores[doc_id]
        out.append(r)

    out.sort(key=lambda r: r.score, reverse=True)
    return out


# ─── FederatedOrchestrator ────────────────────────────────────────────────────

class FederatedOrchestrator:
    """Extends the base Orchestrator with federation fan-out.

    Wraps the existing Orchestrator and adds peer queries when local results
    fall below the confidence threshold.
    """

    def __init__(
        self,
        base_orchestrator,
        cfg: FederationConfig,
        own_id: str,
        router: FederationRouter,
        peer_clients: dict[str, PeerClient],
    ) -> None:
        self._base = base_orchestrator
        self._cfg = cfg
        self._own_id = own_id
        self._router = router
        self._peer_clients = peer_clients

    def search(self, req: SearchRequest, hop_depth: int = 0, origin_id: str = "") -> SearchResponse:
        """Run local search, fan out to peers if local confidence is low."""
        start = time.perf_counter()

        # Step 1: local search (full pipeline)
        local_resp = self._base.search(req)
        local_confidence = max((r.score for r in local_resp.results), default=0.0)

        # Step 2: route decision
        query_vec = self._base.embed(req.query)
        peers = self._router.route(
            query_vec, local_confidence, self._cfg,
            origin_id=origin_id or self._own_id,
            hop_depth=hop_depth,
        )

        if not peers:
            return local_resp  # local results are sufficient or loop/depth limit hit

        # Step 3: fan out to peers in parallel.
        # Run in a fresh thread so asyncio.run() never collides with FastAPI's
        # already-running event loop when this method is called from a sync handler.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            peer_results = ex.submit(
                asyncio.run, self._fan_out(req, peers, hop_depth + 1)
            ).result()

        # Step 4: RRF merge: local + all peer result lists
        all_lists = [local_resp.results] + [r.results for r in peer_results if r]

        if len(all_lists) == 1:
            return local_resp  # all peers failed or timed out

        merged = rrf_merge_multi(all_lists, k=self._cfg.rrf_k)

        top_k = (req.options.top_k if req.options and req.options.top_k else 10)
        final = merged[:top_k]

        return SearchResponse(
            request_id=local_resp.request_id,
            results=final,
            metadata=SearchMetadata(
                total_candidates=sum(
                    (r.metadata.total_candidates if r else 0) for r in [local_resp] + [r for r in peer_results if r]
                ),
                filtered_out=local_resp.metadata.filtered_out,
                final_count=len(final),
                strategy="federated+" + local_resp.metadata.strategy,
            ),
            processing_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    async def _fan_out(
        self,
        req: SearchRequest,
        peers: list[PeerConfig],
        hop_depth: int,
    ) -> list[Optional[SearchResponse]]:
        tasks = [
            self._peer_clients[peer.id].search(req, hop_depth)
            for peer in peers
            if peer.id in self._peer_clients
        ]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # Delegate non-search methods to base orchestrator
    def embed(self, text: str) -> list[float]:
        return self._base.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_batch(texts)

    def rerank(self, query: str, candidates, top_k: int):
        return self._base.rerank(query, candidates, top_k)

    def collections(self):
        return self._base.collections()


# ─── factory ──────────────────────────────────────────────────────────────────

def build_federated_orchestrator(
    base_orchestrator,
    fed_cfg: FederationConfig,
    own_id: str,
) -> FederatedOrchestrator:
    """Build a FederatedOrchestrator from config and a base Orchestrator."""
    router = FederationRouter(fed_cfg.peers, base_orchestrator)
    peer_clients = {
        peer.id: PeerClient(peer, own_id, fed_cfg.peer_timeout_ms)
        for peer in fed_cfg.peers
    }
    return FederatedOrchestrator(base_orchestrator, fed_cfg, own_id, router, peer_clients)
