"""gRPC server for Navigator (doc 12 §5.3)."""
from __future__ import annotations

import json
import logging
import uuid
from concurrent import futures
from typing import Any

import grpc

from .config import Config
from .orchestrator import Orchestrator
from .events import (
    Publisher, TraceContext,
    event_search_started, event_search_completed, event_permission_filtered,
    event_honey_token_retrieved, event_embed_completed, event_batch_embed_completed,
    event_rerank_completed, event_batch_search_completed,
)
from .hooks import (
    HookManager, HookEvent,
    EVENT_HONEY_TOKEN_RETRIEVED, EVENT_EMBED_COMPLETED, EVENT_BATCH_EMBED_COMPLETED,
    EVENT_RERANK_COMPLETED, EVENT_BATCH_SEARCH_COMPLETED,
)
from .models import BatchEmbedRequest, BatchSearchRequest, CollectionsResponse, EmbedRequest, RerankRequest, SearchOptions, SearchRequest

log = logging.getLogger(__name__)


def _deserialize(data: bytes) -> dict:
    return json.loads(data)


def _serialize(obj: Any) -> bytes:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump()).encode()
    return json.dumps(obj).encode()


def _tc_from_context(context) -> TraceContext:
    meta = dict(context.invocation_metadata() or [])
    return TraceContext(
        trace_id=meta.get("x-trace-id", str(uuid.uuid4())),
        span_id=meta.get("x-span-id", ""),
        parent_span_id=meta.get("x-parent-span-id", ""),
        tenant_id=meta.get("x-tenant-id", ""),
        user_id=meta.get("x-user-id", ""),
        request_id=meta.get("x-request-id", ""),
    )


class _Empty:
    @classmethod
    def model_validate(cls, data: dict) -> "_Empty":
        return cls()

    def model_dump(self) -> dict:
        return {}


class _NavigatorServiceHandler(grpc.GenericRpcHandler):
    def __init__(self, orch: Orchestrator, pub: Publisher, hm: HookManager) -> None:
        self._orch = orch
        self._pub = pub
        self._hm = hm
        self._routes = {
            "/navigator.NavigatorService/Search": self._search,
            "/navigator.NavigatorService/SearchWithPermissions": self._search_with_permissions,
            "/navigator.NavigatorService/HybridSearch": self._hybrid_search,
            "/navigator.NavigatorService/BatchSearch": self._batch_search,
            "/navigator.NavigatorService/Embed": self._embed,
            "/navigator.NavigatorService/BatchEmbed": self._batch_embed,
            "/navigator.NavigatorService/Rerank": self._rerank,
            "/navigator.NavigatorService/GetCollections": self._get_collections,
            "/navigator.NavigatorService/Health": self._health,
        }

    def service_name(self) -> str:
        return "navigator.NavigatorService"

    def service(self, handler_call_details):
        method = handler_call_details.method
        fn = self._routes.get(method)
        if fn is None:
            return None

        def _wrapped(request_bytes: bytes, context):
            try:
                return fn(request_bytes, context)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return grpc.unary_unary_rpc_method_handler(lambda b, ctx: _wrapped(b, ctx))

    def _fire_honey_tokens(self, resp, tc: TraceContext) -> None:
        for result in getattr(resp, "results", []):
            if getattr(result, "metadata", {}).get("is_honey_token") == "true":
                token_id = result.metadata.get("honey_token_id", "")
                self._pub.publish(event_honey_token_retrieved(tc, token_id, result.document_id))
                self._hm.fire(HookEvent(
                    type=EVENT_HONEY_TOKEN_RETRIEVED,
                    tenant_id=tc.tenant_id,
                    trace_id=tc.trace_id,
                    span_id=tc.span_id,
                    data={"honey_token_id": token_id, "document_id": result.document_id},
                ))

    def _search(self, req_bytes: bytes, ctx) -> bytes:
        import time as _time
        tc = _tc_from_context(ctx)
        req = SearchRequest.model_validate(_deserialize(req_bytes))
        self._pub.publish(event_search_started(tc, req.query, "grpc"))
        resp = self._orch.search(req)
        self._pub.publish(event_search_completed(tc, len(resp.results), resp.metadata.filtered_out, resp.processing_time_ms))
        if resp.metadata.filtered_out > 0:
            cats = req.user.allowed_categories if req.user else []
            self._pub.publish(event_permission_filtered(tc, resp.metadata.total_candidates, resp.metadata.filtered_out, cats))
        self._fire_honey_tokens(resp, tc)
        return _serialize(resp)

    def _search_with_permissions(self, req_bytes: bytes, ctx) -> bytes:
        import time as _time
        tc = _tc_from_context(ctx)
        req = SearchRequest.model_validate(_deserialize(req_bytes))
        self._pub.publish(event_search_started(tc, req.query, "permissions"))
        resp = self._orch.search(req)
        self._pub.publish(event_search_completed(tc, len(resp.results), resp.metadata.filtered_out, resp.processing_time_ms))
        if resp.metadata.filtered_out > 0:
            cats = req.user.allowed_categories if req.user else []
            self._pub.publish(event_permission_filtered(tc, resp.metadata.total_candidates, resp.metadata.filtered_out, cats))
        self._fire_honey_tokens(resp, tc)
        return _serialize(resp)

    def _hybrid_search(self, req_bytes: bytes, ctx) -> bytes:
        tc = _tc_from_context(ctx)
        req = SearchRequest.model_validate(_deserialize(req_bytes))
        if req.options is None:
            req.options = SearchOptions()
        req.options.use_hybrid = True
        self._pub.publish(event_search_started(tc, req.query, "hybrid"))
        resp = self._orch.search(req)
        self._pub.publish(event_search_completed(tc, len(resp.results), resp.metadata.filtered_out, resp.processing_time_ms))
        if resp.metadata.filtered_out > 0:
            cats = req.user.allowed_categories if req.user else []
            self._pub.publish(event_permission_filtered(tc, resp.metadata.total_candidates, resp.metadata.filtered_out, cats))
        self._fire_honey_tokens(resp, tc)
        return _serialize(resp)

    def _batch_search(self, req_bytes: bytes, ctx) -> bytes:
        tc = _tc_from_context(ctx)
        data = _deserialize(req_bytes)
        req = BatchSearchRequest.model_validate(data)
        results = [self._orch.search(r) for r in req.queries]
        total = sum(len(r.results) for r in results)
        self._pub.publish(event_batch_search_completed(tc, len(req.queries), total))
        self._hm.fire(HookEvent(
            type=EVENT_BATCH_SEARCH_COMPLETED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"query_count": len(req.queries), "total_results": total},
        ))
        return _serialize({"responses": [r.model_dump() for r in results]})

    def _embed(self, req_bytes: bytes, ctx) -> bytes:
        import time as _time
        tc = _tc_from_context(ctx)
        req = EmbedRequest.model_validate(_deserialize(req_bytes))
        t0 = _time.perf_counter()
        vec = self._orch.embed(req.text)
        dur = (_time.perf_counter() - t0) * 1000.0
        self._pub.publish(event_embed_completed(tc, len(req.text), len(vec), dur))
        self._hm.fire(HookEvent(
            type=EVENT_EMBED_COMPLETED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"text_length": len(req.text), "dim_count": len(vec), "duration_ms": dur},
        ))
        return _serialize({"embedding": vec})

    def _batch_embed(self, req_bytes: bytes, ctx) -> bytes:
        import time as _time
        tc = _tc_from_context(ctx)
        req = BatchEmbedRequest.model_validate(_deserialize(req_bytes))
        t0 = _time.perf_counter()
        vecs = self._orch.embed_batch(req.texts)
        dur = (_time.perf_counter() - t0) * 1000.0
        dim = len(vecs[0]) if vecs else 0
        self._pub.publish(event_batch_embed_completed(tc, len(req.texts), dim, dur))
        self._hm.fire(HookEvent(
            type=EVENT_BATCH_EMBED_COMPLETED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"text_count": len(req.texts), "dim_count": dim, "duration_ms": dur},
        ))
        return _serialize({"embeddings": vecs})

    def _rerank(self, req_bytes: bytes, ctx) -> bytes:
        import time as _time
        tc = _tc_from_context(ctx)
        req = RerankRequest.model_validate(_deserialize(req_bytes))
        t0 = _time.perf_counter()
        results = self._orch.rerank(req.query, req.candidates, req.top_k)
        dur = (_time.perf_counter() - t0) * 1000.0
        self._pub.publish(event_rerank_completed(tc, len(req.candidates), req.top_k or len(results), dur))
        self._hm.fire(HookEvent(
            type=EVENT_RERANK_COMPLETED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"candidate_count": len(req.candidates), "top_k": req.top_k, "duration_ms": dur},
        ))
        return _serialize({"results": [r.model_dump() for r in results]})

    def _get_collections(self, _req: bytes, ctx) -> bytes:
        cols = self._orch.collections()
        return _serialize(CollectionsResponse(collections=cols))

    def _health(self, _req: bytes, ctx) -> bytes:
        return _serialize({"status": "ok"})


class GRPCServer:
    def __init__(self, orch: Orchestrator, pub: Publisher, hm: HookManager) -> None:
        self._orch = orch
        self._pub = pub
        self._hm = hm
        self._server: grpc.Server | None = None

    def hooks(self) -> HookManager:
        return self._hm

    def serve(self, port: int) -> None:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._server.add_generic_rpc_handlers([_NavigatorServiceHandler(self._orch, self._pub, self._hm)])
        self._server.add_insecure_port(f"[::]:{port}")
        self._server.start()
        log.info("[grpc] navigator listening on :%d", port)
        self._server.wait_for_termination()
