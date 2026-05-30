"""HyDE — Hypothetical Document Embeddings (MR-02-002).

For short factual or procedural queries, generates a hypothetical document
passage and uses its embedding instead of the raw query embedding. This
improves recall because short questions and long document paragraphs occupy
very different regions of the embedding space.

The hypothetical text is NEVER returned to the caller; it is only used to
produce the query vector for vector search. BM25 search still uses the
original (rewritten) query.

Generation hierarchy:
  1. Local LLM via Ollama-compatible endpoint (if cfg.llm_endpoint is set)
  2. Template-based expansion (default — always available, no external call)
  3. Original query (fallback on any error or timeout)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

log = logging.getLogger(__name__)

_MAX_HYPOTHETICAL_CHARS = 400

# Domain-aware passage templates keyed by (intent, domain_hint).
# Variables: {query}
_TEMPLATES: dict[str, str] = {
    "factual": (
        "{query}\n\n"
        "According to the relevant documentation, the requested information is "
        "as follows. The official record states the current value, applicable "
        "conditions, and effective date. Supporting details include the relevant "
        "metric, its historical context, and any associated policy or threshold."
    ),
    "procedural": (
        "{query}\n\n"
        "The official procedure for this request is outlined below. "
        "Step 1: Submit the appropriate form through the designated channel. "
        "Step 2: Obtain required approvals from the responsible department. "
        "Step 3: Confirm completion and retain the reference number for audit. "
        "Eligibility criteria and exceptions are specified in the current policy."
    ),
}

_DEFAULT_TEMPLATE = (
    "{query}\n\n"
    "The following information addresses this query based on the available "
    "documentation. Relevant details, supporting data, and applicable policies "
    "are provided for reference."
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


class HyDETransformer:
    """Generates hypothetical document passages to improve embedding quality."""

    def __init__(self, llm_endpoint: str = "", timeout_ms: float = 2000.0) -> None:
        self._endpoint = llm_endpoint.rstrip("/")
        self._timeout = timeout_ms / 1000.0

    def should_apply(self, query: str, intent: str, max_words: int) -> bool:
        """True when HyDE is appropriate: short factual or procedural query."""
        return (
            intent in ("factual", "procedural", "ambiguous")
            and _word_count(query) <= max_words
        )

    def generate(self, query: str, intent: str) -> str:
        """Return a hypothetical document passage, or the original query on failure."""
        if self._endpoint:
            hyp = self._llm_generate(query, intent)
            if hyp:
                return hyp[:_MAX_HYPOTHETICAL_CHARS]
        return self._template_generate(query, intent)

    def _template_generate(self, query: str, intent: str) -> str:
        template = _TEMPLATES.get(intent, _DEFAULT_TEMPLATE)
        return template.format(query=query)[:_MAX_HYPOTHETICAL_CHARS]

    def _llm_generate(self, query: str, intent: str) -> Optional[str]:
        try:
            import httpx  # type: ignore
            prompt = (
                f"Write a short document passage (2-3 sentences) that directly "
                f"answers the following question. Output only the passage, no preamble.\n\n"
                f"Question: {query}"
            )
            resp = httpx.post(
                f"{self._endpoint}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or None
        except Exception as exc:
            log.debug("[hyde] LLM generation failed (%s), using template", exc)
            return None
