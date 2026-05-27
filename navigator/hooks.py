"""Lightweight hook/extension-point system for Navigator (doc 01 §2.3)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

EVENT_SEARCH_COMPLETED      = "search_completed"
EVENT_HONEY_TOKEN_RETRIEVED = "honey_token_retrieved"
EVENT_PERMISSION_FILTERED   = "permission_filtered"


@dataclass
class HookEvent:
    type:       str
    tenant_id:  str = ""
    trace_id:   str = ""
    span_id:    str = ""
    request_id: str = ""
    data:       dict[str, Any] = field(default_factory=dict)


# A hook handler is any callable that accepts a HookEvent.
HookHandler = Callable[[HookEvent], None]


class HookManager:
    """Thread-safe hook registry. Registered handlers are called asynchronously."""

    def __init__(self) -> None:
        self._mu: threading.RLock = threading.RLock()
        self._handlers: dict[str, list[HookHandler]] = {}

    def register(self, event_type: str, handler: HookHandler) -> None:
        with self._mu:
            self._handlers.setdefault(event_type, []).append(handler)

    def fire(self, ev: HookEvent) -> None:
        with self._mu:
            handlers = list(self._handlers.get(ev.type, []))
        for h in handlers:
            threading.Thread(target=self._safe_call, args=(h, ev), daemon=True).start()

    @staticmethod
    def _safe_call(h: HookHandler, ev: HookEvent) -> None:
        try:
            h(ev)
        except Exception as e:
            log.warning("[hooks] handler error for %s: %s", type(ev), e)
