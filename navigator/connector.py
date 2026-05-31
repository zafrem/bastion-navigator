"""Source connector interface and built-in implementations — MR-06 (FR-MR-06-001).

Architecture:
  SourceConnector (ABC)
    └─ JsonlConnector        reads a JSONL file line-by-line
    └─ DirectoryConnector    walks a directory for text/markdown/JSON files
    └─ RestPullConnector     polls a configured HTTP endpoint periodically

All connectors produce Document objects that feed the same indexing pipeline:
    Document → Chunker → Embedder → Qdrant upsert

Security: connector credentials MUST be injected at construction time, never
stored in config files (SC-09). The caller (main.py) is responsible for fetching
them from Vault KMS before instantiating a connector.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# ─── Domain model ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A source document ready to be chunked and indexed."""
    id: str
    content: str
    title: str = ""
    category: str = ""
    metadata: dict = field(default_factory=dict)
    updated_at: Optional[datetime] = None
    source_version: str = ""
    mime_type: str = "text/markdown"

    def content_hash(self) -> str:
        """SHA-256 of the document content (hex, 64 chars)."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# ─── Abstract interface ───────────────────────────────────────────────────────

class SourceConnector(ABC):
    """Abstract base for all source connectors (FR-MR-06-001)."""

    @abstractmethod
    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        """Yield documents. When since is set, yield only docs updated after that time."""

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Fetch a single document by ID, or None if not found."""

    @abstractmethod
    def document_updated_at(self, doc_id: str) -> Optional[datetime]:
        """Return the last-modified timestamp for doc_id, or None if unknown."""


# ─── JSONL connector ──────────────────────────────────────────────────────────

class JsonlConnector(SourceConnector):
    """Reads documents from a JSONL file.

    Each line must be a JSON object with at minimum an ``id`` and ``content``
    field. Optional fields: ``title``, ``category``, ``metadata``,
    ``updated_at`` (ISO-8601 string), ``source_version``.

    This connector is always available — it wraps the existing
    `navigator-cli index` ingestion path.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        try:
            with open(self._path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("[jsonl] line %d: invalid JSON, skipped", lineno)
                        continue
                    doc = _doc_from_dict(obj)
                    if since and doc.updated_at and doc.updated_at <= since:
                        continue
                    yield doc
        except FileNotFoundError:
            log.error("[jsonl] file not found: %s", self._path)

    def get_document(self, doc_id: str) -> Optional[Document]:
        for doc in self.list_documents():
            if doc.id == doc_id:
                return doc
        return None

    def document_updated_at(self, doc_id: str) -> Optional[datetime]:
        doc = self.get_document(doc_id)
        return doc.updated_at if doc else None


# ─── Directory connector ──────────────────────────────────────────────────────

_EXT_MIME = {
    ".md":   "text/markdown",
    ".txt":  "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm":  "text/html",
    ".csv":  "text/csv",
}

class DirectoryConnector(SourceConnector):
    """Walks a directory and yields every file whose extension is in ``extensions``.

    Document ID is the file path relative to the root directory.
    Documents are re-yielded when their mtime is newer than ``since``.
    """

    def __init__(
        self,
        directory: str,
        extensions: Optional[list[str]] = None,
        category: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        self._dir = os.path.abspath(directory)
        self._exts = frozenset(extensions or list(_EXT_MIME.keys()))
        self._category = category
        self._base_meta = metadata or {}

    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        for root, _, files in os.walk(self._dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self._exts:
                    continue
                fpath = os.path.join(root, fname)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
                if since and mtime <= since:
                    continue
                rel = os.path.relpath(fpath, self._dir)
                try:
                    content = open(fpath, encoding="utf-8").read()
                except Exception as exc:
                    log.warning("[dir] could not read %s: %s", fpath, exc)
                    continue
                yield Document(
                    id=rel,
                    content=content,
                    title=os.path.splitext(fname)[0],
                    category=self._category,
                    metadata=dict(self._base_meta),
                    updated_at=mtime,
                    mime_type=_EXT_MIME.get(ext, "text/plain"),
                )

    def get_document(self, doc_id: str) -> Optional[Document]:
        fpath = os.path.join(self._dir, doc_id)
        if not os.path.isfile(fpath):
            return None
        for doc in self.list_documents():
            if doc.id == doc_id:
                return doc
        return None

    def document_updated_at(self, doc_id: str) -> Optional[datetime]:
        fpath = os.path.join(self._dir, doc_id)
        if not os.path.isfile(fpath):
            return None
        return datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)


# ─── REST pull connector ──────────────────────────────────────────────────────

class RestPullConnector(SourceConnector):
    """Polls a REST endpoint for documents.

    Expects the endpoint to return a JSON array of document objects at
    ``GET {endpoint}/documents?since=<ISO-8601>``.

    Credentials are passed in an ``Authorization`` header; the caller must
    obtain the value from Vault KMS before constructing this connector (SC-09).
    """

    def __init__(
        self,
        endpoint: str,
        auth_header: str = "",
        timeout: float = 10.0,
        page_size: int = 100,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._headers = {"Authorization": auth_header} if auth_header else {}
        self._timeout = timeout
        self._page_size = page_size

    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        try:
            import httpx  # type: ignore
        except ImportError:
            log.error("[rest] httpx not installed — REST pull connector unavailable")
            return

        params: dict = {"limit": self._page_size}
        if since:
            params["since"] = since.isoformat()

        try:
            resp = httpx.get(
                f"{self._endpoint}/documents",
                params=params,
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            for obj in resp.json().get("documents", []):
                yield _doc_from_dict(obj)
        except Exception as exc:
            log.error("[rest] list_documents failed: %s", exc)

    def get_document(self, doc_id: str) -> Optional[Document]:
        try:
            import httpx  # type: ignore
            resp = httpx.get(
                f"{self._endpoint}/documents/{doc_id}",
                headers=self._headers,
                timeout=self._timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _doc_from_dict(resp.json())
        except Exception as exc:
            log.error("[rest] get_document(%s) failed: %s", doc_id, exc)
            return None

    def document_updated_at(self, doc_id: str) -> Optional[datetime]:
        doc = self.get_document(doc_id)
        return doc.updated_at if doc else None


# ─── helpers ──────────────────────────────────────────────────────────────────

def _doc_from_dict(obj: dict) -> Document:
    """Construct a Document from a plain dict (JSONL line or REST response body)."""
    updated_at = None
    raw_ts = obj.get("updated_at") or obj.get("last_modified") or obj.get("timestamp")
    if isinstance(raw_ts, str):
        try:
            updated_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            pass

    return Document(
        id=str(obj.get("id") or obj.get("document_id") or ""),
        content=str(obj.get("content", "")),
        title=str(obj.get("title", "")),
        category=str(obj.get("category", "")),
        metadata=dict(obj.get("metadata", {})),
        updated_at=updated_at,
        source_version=str(obj.get("source_version", "")),
        mime_type=str(obj.get("mime_type", "text/markdown")),
    )


def build_connector(cfg) -> Optional[SourceConnector]:
    """Factory: build a connector from a ConnectorConfig object.

    Returns None when the connector type is unknown or disabled.
    """
    if not cfg or not cfg.enabled:
        return None
    t = cfg.type
    if t == "jsonl":
        return JsonlConnector(cfg.path)
    if t == "directory":
        return DirectoryConnector(cfg.path, category=cfg.category)
    if t == "rest":
        return RestPullConnector(cfg.endpoint, auth_header=cfg.auth_header, timeout=cfg.timeout_seconds)
    log.warning("[connector] unknown type %r", t)
    return None
