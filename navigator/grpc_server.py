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
from .events import Publisher, TraceContext, event_search_started, event_search_completed, event_permission_filtered, event_honey_token_retrieved
from .hooks import HookManager, HookEvent, EVENT_HONEY_TOKEN_RETRIEVED
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
        tc = _tc_from_context(ctx)
        req = SearchRequest.model_validate(_deserialize(req_bytes))
        resp = self._orch.search(req)
        self._fire_honey_tokens(resp, tc)
        return _serialize(resp)

    def _search_with_permissions(self, req_bytes: bytes, ctx) -> bytes:
        return self._search(req_bytes, ctx)

    def _hybrid_search(self, req_bytes: bytes, ctx) -> bytes:
        tc = _tc_from_context(ctx)
        req = SearchRequest.model_validate(_deserialize(req_bytes))
        if req.options is None:
            req.options = SearchOptions()
        req.options.use_hybrid = True
        resp = self._orch.search(req)
        self._fire_honey_tokens(resp, tc)
        return _serialize(resp)

    def _batch_search(self, req_bytes: bytes, ctx) -> bytes:
        data = _deserialize(req_bytes)
        req = BatchSearchRequest.model_validate(data)
        results = [self._orch.search(r) for r in req.requests]
        return _serialize({"responses": [r.model_dump() for r in results]})

    def _embed(self, req_bytes: bytes, ctx) -> bytes:
        req = EmbedRequest.model_validate(_deserialize(req_bytes))
        vec = self._orch.embed(req.text)
        return _serialize({"embedding": vec})

    def _batch_embed(self, req_bytes: bytes, ctx) -> bytes:
        req = BatchEmbedRequest.model_validate(_deserialize(req_bytes))
        vecs = self._orch.embed_batch(req.texts)
        return _serialize({"embeddings": vecs})

    def _rerank(self, req_bytes: bytes, ctx) -> bytes:
        req = RerankRequest.model_validate(_deserialize(req_bytes))
        results = self._orch.rerank(req.query, req.candidates, req.top_k)
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
