"""Foundation-standard NATS event publisher for Navigator (doc 02)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

log = logging.getLogger(__name__)

_MODULE = "navigator"
_MODULE_VERSION = "3.0.0"
_SCHEMA_VER = "1.0"
_SUBJECT_PFX = "bastion.events.navigator"


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    request_id: str = ""


@dataclass
class NavigatorEvent:
    event_id: str
    event_type: str
    schema_version: str
    trace_id: str
    span_id: str
    parent_span_id: str
    module: str
    module_version: str
    timestamp: int
    tenant_id: str
    severity: str
    category: str
    user_id: str = ""
    request_id: str = ""
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    action_taken: str = ""


def _new_event(tc: TraceContext, event_type: str, severity: str, category: str, data: dict) -> NavigatorEvent:
    return NavigatorEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        schema_version=_SCHEMA_VER,
        trace_id=tc.trace_id,
        span_id=tc.span_id,
        parent_span_id=tc.parent_span_id,
        module=_MODULE,
        module_version=_MODULE_VERSION,
        timestamp=time.time_ns(),
        tenant_id=tc.tenant_id,
        user_id=tc.user_id,
        request_id=tc.request_id,
        severity=severity,
        category=category,
        data=data,
    )


def event_search_started(tc: TraceContext, query: str, pipeline_type: str) -> NavigatorEvent:
    ev = _new_event(tc, "search_started", "info", "operational", {
        "query_length": len(query),
        "pipeline_type": pipeline_type,
    })
    ev.status = "started"
    ev.action_taken = "search"
    return ev


def event_search_completed(tc: TraceContext, result_count: int, filtered_out: int, duration_ms: float) -> NavigatorEvent:
    ev = _new_event(tc, "search_completed", "info", "operational", {
        "result_count": result_count,
        "filtered_out": filtered_out,
        "duration_ms": duration_ms,
    })
    ev.status = "completed"
    ev.action_taken = "search"
    return ev


def event_permission_filtered(tc: TraceContext, total: int, filtered: int, categories: list[str]) -> NavigatorEvent:
    ev = _new_event(tc, "permission_filtered", "info", "security", {
        "total_candidates": total,
        "filtered_out": filtered,
        "allowed_categories": categories,
    })
    ev.status = "filtered"
    ev.action_taken = "filter"
    return ev


def event_federation_started(tc: TraceContext, peers_queried: int, local_confidence: float) -> NavigatorEvent:
    ev = _new_event(tc, "federation_started", "info", "operational", {
        "peers_queried": peers_queried,
        "local_confidence": local_confidence,
    })
    ev.status = "started"
    ev.action_taken = "federation_search"
    return ev


def event_federation_completed(tc: TraceContext, total: int, local_count: int, remote_count: int, duration_ms: float) -> NavigatorEvent:
    ev = _new_event(tc, "federation_completed", "info", "operational", {
        "total_results": total,
        "local_results": local_count,
        "remote_results": remote_count,
        "duration_ms": duration_ms,
    })
    ev.status = "completed"
    ev.action_taken = "federation_search"
    return ev


def event_peer_timeout(tc: TraceContext, peer_id: str, timeout_ms: int) -> NavigatorEvent:
    ev = _new_event(tc, "peer_timeout", "warning", "operational", {
        "peer_id": peer_id,
        "timeout_ms": timeout_ms,
    })
    ev.status = "timeout"
    return ev


def event_agent_generated(tc: TraceContext, peer_id: str, model: str, confidence: float, latency_ms: float) -> NavigatorEvent:
    ev = _new_event(tc, "agent_generated", "info", "operational", {
        "peer_id": peer_id,
        "model": model,
        "confidence": confidence,
        "latency_ms": latency_ms,
    })
    ev.status = "completed"
    ev.action_taken = "agent_generate"
    return ev


def event_embed_completed(tc: TraceContext, text_length: int, dim_count: int, duration_ms: float) -> NavigatorEvent:
    """Emitted after a single-text embedding operation (SRS doc 22 lineage)."""
    ev = _new_event(tc, "embed_completed", "info", "operational", {
        "text_length": text_length,
        "dim_count": dim_count,
        "duration_ms": duration_ms,
    })
    ev.status = "completed"
    ev.action_taken = "embed"
    return ev


def event_batch_embed_completed(tc: TraceContext, text_count: int, dim_count: int, duration_ms: float) -> NavigatorEvent:
    """Emitted after a batch embedding operation (SRS doc 22 lineage)."""
    ev = _new_event(tc, "batch_embed_completed", "info", "operational", {
        "text_count": text_count,
        "dim_count": dim_count,
        "duration_ms": duration_ms,
    })
    ev.status = "completed"
    ev.action_taken = "batch_embed"
    return ev


def event_rerank_completed(tc: TraceContext, candidate_count: int, top_k: int, duration_ms: float) -> NavigatorEvent:
    """Emitted after a rerank operation (SRS doc 22 lineage)."""
    ev = _new_event(tc, "rerank_completed", "info", "operational", {
        "candidate_count": candidate_count,
        "top_k": top_k,
        "duration_ms": duration_ms,
    })
    ev.status = "completed"
    ev.action_taken = "rerank"
    return ev


def event_batch_search_completed(tc: TraceContext, query_count: int, total_results: int) -> NavigatorEvent:
    """Emitted after a batch search operation (SRS doc 22 lineage)."""
    ev = _new_event(tc, "batch_search_completed", "info", "operational", {
        "query_count": query_count,
        "total_results": total_results,
    })
    ev.status = "completed"
    ev.action_taken = "batch_search"
    return ev


def event_query_routed(
    tc: TraceContext,
    intent: str,
    strategy: str,
    collections: list[str],
    excluded: list[str],
    confidence: float,
    routing_ms: float,
) -> NavigatorEvent:
    """Emitted after each routing decision (MR-01, FR-MR-01-004)."""
    ev = _new_event(tc, "query_routed", "info", "operational", {
        "intent": intent,
        "strategy": strategy,
        "collections": collections,
        "excluded": excluded,
        "confidence": round(confidence, 3),
        "routing_ms": round(routing_ms, 2),
    })
    ev.status = "routed"
    ev.action_taken = "route"
    return ev


def event_search_iteration(
    tc: TraceContext,
    iteration: int,
    verdict: str,
    top_score: float,
    coverage: float,
    refinement: Optional[str],
    iteration_ms: float,
) -> NavigatorEvent:
    """Emitted after each re-search loop iteration (MR-03, FR-MR-03-004)."""
    ev = _new_event(tc, "search_iteration", "info", "operational", {
        "iteration": iteration,
        "verdict": verdict,
        "top_score": round(top_score, 4),
        "coverage": round(coverage, 4),
        "refinement": refinement,
        "iteration_ms": round(iteration_ms, 2),
    })
    ev.status = verdict
    ev.action_taken = "search_iteration"
    return ev


def event_loop_completed(
    tc: TraceContext,
    total_iterations: int,
    termination: str,
    final_result_count: int,
    total_ms: float,
) -> NavigatorEvent:
    """Emitted when the re-search loop finishes (MR-03, FR-MR-03-004)."""
    ev = _new_event(tc, "loop_completed", "info", "operational", {
        "total_iterations": total_iterations,
        "termination": termination,
        "final_result_count": final_result_count,
        "total_ms": round(total_ms, 2),
    })
    ev.status = "completed"
    ev.action_taken = "search_loop"
    return ev


def event_chunk_retrieved(
    tc: TraceContext,
    chunk_id: str,
    document_id: str,
    score: float,
    rank: int,
    collection: str = "",
) -> NavigatorEvent:
    """Emitted once per returned chunk for data lineage tracking (MR-05-001)."""
    ev = _new_event(tc, "chunk_retrieved", "info", "lineage", {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "score": round(score, 4),
        "rank": rank,
        "collection": collection,
    })
    ev.status = "retrieved"
    ev.action_taken = "retrieve"
    return ev


def event_honey_token_retrieved(tc: TraceContext, token_id: str, document_id: str) -> NavigatorEvent:
    ev = _new_event(tc, "honey_token_retrieved", "critical", "security", {
        "honey_token_id": token_id,
        "document_id": document_id,
        "detecting_module": _MODULE,
    })
    ev.status = "detected"
    ev.action_taken = "alert_raised"
    return ev


def extract_trace_context(
    trace_id: str = "",
    span_id: str = "",
    parent_span_id: str = "",
    tenant_id: str = "",
    user_id: str = "",
    request_id: str = "",
) -> TraceContext:
    return TraceContext(
        trace_id=trace_id or str(uuid.uuid4()),
        span_id=span_id or str(uuid.uuid4())[:16],
        parent_span_id=parent_span_id,
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=request_id,
    )


class Publisher:
    """Thread-safe async NATS publisher. Fire-and-forget; never blocks callers."""

    def __init__(self, nats_url: str) -> None:
        self._url = nats_url
        self._nc = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        if nats_url:
            self._start()

    def _start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="navigator-nats")
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    async def _connect(self) -> None:
        try:
            import nats
            self._nc = await nats.connect(
                self._url,
                max_reconnect_attempts=5,
                reconnect_time_wait=2,
                error_cb=lambda e: log.warning("[navigator-events] nats error: %s", e),
            )
        except Exception as exc:
            log.warning("[navigator-events] nats unavailable (%s), events disabled", exc)

    def publish(self, ev: NavigatorEvent) -> None:
        if not self._loop or not self._nc:
            return
        asyncio.run_coroutine_threadsafe(self._publish(ev), self._loop)

    async def _publish(self, ev: NavigatorEvent) -> None:
        if not self._nc:
            return
        try:
            subject = f"{_SUBJECT_PFX}.{ev.event_type}"
            payload = json.dumps(asdict(ev)).encode()
            await self._nc.publish(subject, payload)
        except Exception as exc:
            log.debug("[navigator-events] publish failed: %s", exc)

    def close(self) -> None:
        if self._nc and self._loop:
            asyncio.run_coroutine_threadsafe(self._nc.drain(), self._loop)
