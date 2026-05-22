package rest

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/bastion/navigator/internal/models"
	"github.com/bastion/navigator/internal/orchestrator"
)

type handlers struct {
	orch *orchestrator.Orchestrator
}

// --- Health ---

func (h *handlers) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, models.HealthStatus{
		Status:  "ok",
		Version: "1.0.0",
		Checks:  map[string]string{"service": "up"},
	})
}

func (h *handlers) Live(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "alive"})
}

func (h *handlers) Ready(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// --- Search ---

func (h *handlers) Search(w http.ResponseWriter, r *http.Request) {
	var req models.SearchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	resp, err := h.orch.Search(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) HybridSearch(w http.ResponseWriter, r *http.Request) {
	var req models.SearchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Options == nil {
		req.Options = &models.SearchOptions{}
	}
	req.Options.UseHybrid = true
	resp, err := h.orch.Search(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) BatchSearch(w http.ResponseWriter, r *http.Request) {
	var batch models.BatchSearchRequest
	if !decodeJSON(w, r, &batch) {
		return
	}

	responses := make([]models.SearchResponse, 0, len(batch.Queries))
	for _, q := range batch.Queries {
		resp, err := h.orch.Search(r.Context(), q)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		responses = append(responses, resp)
	}
	writeJSON(w, http.StatusOK, models.BatchSearchResponse{
		RequestID: batch.RequestID,
		Results:   responses,
	})
}

// --- Embedding ---

func (h *handlers) Embed(w http.ResponseWriter, r *http.Request) {
	var req models.EmbedRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	vec, err := h.orch.Embed(r.Context(), req.Text)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models.EmbedResponse{
		RequestID: req.RequestID,
		Embedding: vec,
		DimCount:  len(vec),
	})
}

func (h *handlers) BatchEmbed(w http.ResponseWriter, r *http.Request) {
	var req models.BatchEmbedRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	vecs, err := h.orch.EmbedBatch(r.Context(), req.Texts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models.BatchEmbedResponse{
		RequestID:  req.RequestID,
		Embeddings: vecs,
	})
}

// --- Reranking ---

func (h *handlers) Rerank(w http.ResponseWriter, r *http.Request) {
	var req models.RerankRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	results, err := h.orch.Rerank(r.Context(), req.Query, req.Candidates, req.TopK)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models.RerankResponse{
		RequestID: req.RequestID,
		Results:   results,
	})
}

// --- Collections ---

func (h *handlers) Collections(w http.ResponseWriter, r *http.Request) {
	cols, err := h.orch.Collections(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models.CollectionsResponse{Collections: cols})
}

func (h *handlers) CollectionInfo(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	name = strings.TrimSpace(name)
	cols, err := h.orch.Collections(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	for _, c := range cols {
		if c.Name == name {
			writeJSON(w, http.StatusOK, c)
			return
		}
	}
	writeError(w, http.StatusNotFound, "collection not found: "+name)
}

// --- helpers ---

func decodeJSON(w http.ResponseWriter, r *http.Request, dst interface{}) bool {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, body interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, models.ErrorResponse{
		Error: msg,
		Code:  status,
	})
}
