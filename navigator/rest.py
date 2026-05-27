"""FastAPI REST server for Navigator."""
from __future__ import annotations

from fastapi import FastAPI, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .config import Config
from .orchestrator import Orchestrator
from .events import (
    Publisher, extract_trace_context,
    event_search_started, event_search_completed, event_permission_filtered,
    event_honey_token_retrieved, event_federation_started, event_federation_completed,
)
from .hooks import HookManager, HookEvent, EVENT_HONEY_TOKEN_RETRIEVED, EVENT_SEARCH_COMPLETED
from .models import (
    AgentGenerateRequest, AgentGenerateResponse,
    BatchEmbedRequest, BatchEmbedResponse, BatchSearchRequest, BatchSearchResponse,
    CollectionsResponse, EmbedRequest, EmbedResponse, HealthStatus, RerankRequest,
    RerankResponse, SearchOptions, SearchRequest, SearchResponse,
)

import logging

log = logging.getLogger(__name__)


def _tc(request: Request, req=None):
    h = request.headers
    return extract_trace_context(
        trace_id=h.get("x-trace-id", ""),
        span_id=h.get("x-span-id", ""),
        parent_span_id=h.get("x-parent-span-id", ""),
        tenant_id=getattr(req, "tenant_id", "") or h.get("x-tenant-id", ""),
        user_id=(req.user.user_id if req and req.user else "") or h.get("x-user-id", ""),
        request_id=getattr(req, "request_id", "") or h.get("x-request-id", ""),
    )


def _fire_honey_token_events(results, tc, req, pub: Publisher, hm: HookManager):
    for result in results:
        if result.metadata.get("is_honey_token") == "true":
            token_id = result.metadata.get("honey_token_id", "")
            pub.publish(event_honey_token_retrieved(tc, token_id, result.document_id))
            hm.fire(HookEvent(
                type=EVENT_HONEY_TOKEN_RETRIEVED,
                tenant_id=tc.tenant_id,
                trace_id=tc.trace_id,
                span_id=tc.span_id,
                data={"honey_token_id": token_id, "document_id": result.document_id},
            ))


def _is_federation_mode(orch) -> bool:
    return type(orch).__name__ == "FederatedOrchestrator"


def build_app(cfg: Config, orch: Orchestrator, pub: Publisher, hm: HookManager) -> FastAPI:
    app = FastAPI(title="Bastion Navigator", version="2.0.0")

    # ── health ──────────────────────────────────────────────────────────────

    @app.get("/v1/health")
    def health():
        return HealthStatus(status="ok", checks={"service": "up"})

    @app.get("/v1/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/v1/health/ready")
    def ready():
        return {"status": "ready"}

    @app.get("/v1/metrics")
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── search ──────────────────────────────────────────────────────────────

    @app.post("/v1/navigator/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request):
        tc = _tc(request, req)
        # Federation mode: extract loop-prevention headers passed by peer callers.
        hop_depth = int(request.headers.get("x-hop-depth", "0"))
        origin_id = request.headers.get("x-origin-id", "")
        pub.publish(event_search_started(tc, req.query, "full"))
        if _is_federation_mode(orch):
            resp = orch.search(req, hop_depth=hop_depth, origin_id=origin_id)
        else:
            resp = orch.search(req)
        pub.publish(event_search_completed(tc, len(resp.results), resp.metadata.filtered_out, resp.processing_time_ms))
        if resp.metadata.filtered_out > 0:
            cats = req.user.allowed_categories if req.user else []
            pub.publish(event_permission_filtered(tc, resp.metadata.total_candidates, resp.metadata.filtered_out, cats))
        _fire_honey_token_events(resp.results, tc, req, pub, hm)
        return resp

    @app.post("/v1/navigator/search/with-permissions", response_model=SearchResponse)
    def search_with_permissions(req: SearchRequest, request: Request):
        if not req.user or not req.user.allowed_categories:
            from fastapi import HTTPException
            raise HTTPException(400, "user.allowed_categories is required")
        if not req.tenant_id:
            from fastapi import HTTPException
            raise HTTPException(400, "tenant_id is required")
        tc = _tc(request, req)
        pub.publish(event_search_started(tc, req.query, "permissions"))
        resp = orch.search(req)
        pub.publish(event_search_completed(tc, len(resp.results), resp.metadata.filtered_out, resp.processing_time_ms))
        if resp.metadata.filtered_out > 0:
            pub.publish(event_permission_filtered(tc, resp.metadata.total_candidates, resp.metadata.filtered_out, req.user.allowed_categories))
        _fire_honey_token_events(resp.results, tc, req, pub, hm)
        return resp

    @app.post("/v1/navigator/search/hybrid", response_model=SearchResponse)
    def hybrid_search(req: SearchRequest, request: Request):
        if req.options is None:
            req.options = SearchOptions()
        req.options.use_hybrid = True
        return search(req, request)

    @app.post("/v1/navigator/search/batch", response_model=BatchSearchResponse)
    def batch_search(req: BatchSearchRequest):
        responses = [orch.search(q) for q in req.queries]
        return BatchSearchResponse(request_id=req.request_id, results=responses)

    # ── embedding ───────────────────────────────────────────────────────────

    @app.post("/v1/navigator/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest):
        vec = orch.embed(req.text)
        return EmbedResponse(request_id=req.request_id, embedding=vec, dim_count=len(vec))

    @app.post("/v1/navigator/embed/batch", response_model=BatchEmbedResponse)
    def embed_batch(req: BatchEmbedRequest):
        vecs = orch.embed_batch(req.texts)
        return BatchEmbedResponse(request_id=req.request_id, embeddings=vecs)

    # ── rerank ──────────────────────────────────────────────────────────────

    @app.post("/v1/navigator/rerank", response_model=RerankResponse)
    def rerank(req: RerankRequest):
        results = orch.rerank(req.query, req.candidates, req.top_k)
        return RerankResponse(request_id=req.request_id, results=results)

    # ── collections ─────────────────────────────────────────────────────────

    @app.get("/v1/navigator/collections", response_model=CollectionsResponse)
    def collections():
        return CollectionsResponse(collections=orch.collections())

    @app.get("/v1/navigator/collections/{name}")
    def collection_info(name: str):
        for c in orch.collections():
            if c.name == name:
                return c
        from fastapi import HTTPException
        raise HTTPException(404, f"collection not found: {name}")

    # ── agent generate (agent mode only, doc 22 §5.1) ───────────────────────

    @app.post("/v1/navigator/agent/generate", response_model=AgentGenerateResponse)
    def agent_generate(req: AgentGenerateRequest):
        """Generate a domain answer using the local LLM (agent mode only)."""
        if cfg.mode != "agent":
            from fastapi import HTTPException
            raise HTTPException(503, "agent mode is not enabled on this Navigator")

        local_llm_cfg = cfg.agent.local_llm
        answer, sources, model_name, confidence = _call_local_llm(
            query=req.query,
            context=req.context,
            llm_cfg=local_llm_cfg,
            max_tokens=req.max_tokens,
        )
        return AgentGenerateResponse(
            answer=answer,
            sources=sources,
            model=model_name,
            confidence=confidence,
        )

    return app


def _call_local_llm(query: str, context, llm_cfg, max_tokens: int):
    """Call the configured local LLM provider and return (answer, sources, model, confidence)."""
    import httpx

    sources = [r.document_id for r in context if r.document_id]
    context_text = "\n\n".join(r.content for r in context if r.content)

    prompt = query
    if context_text:
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    try:
        if llm_cfg.provider == "ollama":
            resp = httpx.post(
                f"{llm_cfg.endpoint}/api/generate",
                json={"model": llm_cfg.model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens}},
                timeout=llm_cfg.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "")
        else:
            # OpenAI-compatible endpoint (llamacpp, custom_http)
            resp = httpx.post(
                f"{llm_cfg.endpoint}/v1/chat/completions",
                json={
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=llm_cfg.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]

        confidence = min(1.0, max(0.0, 1.0 - (1.0 / (1.0 + len(answer) / 100))))
        return answer, sources, llm_cfg.model, confidence

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("agent_generate local LLM call failed: %s", exc)
        return "", sources, llm_cfg.model, 0.0
