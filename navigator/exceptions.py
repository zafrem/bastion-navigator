"""Scoped exceptions: signed exemptions pushed by Tracker, enforced from a local copy.

This file is intentionally identical in navigator and anchor (and mirrors the Go
package used by tracker, sentinel and vault) so each module stays independently
deployable with no shared library. Keep the copies in sync.

Tracker signs every change with Ed25519. This client verifies the signature,
applies the change, persists the signed envelope locally, and answers
``allowed()`` from that local store only. So a module keeps its last known
exceptions when Tracker or NATS is down, and with no public key configured the
feature is off and nothing is ever exempted.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

log = logging.getLogger(__name__)

SUBJECT_UPDATES = "bastion.exceptions.updates"
SUBJECT_SNAPSHOT = "bastion.exceptions.snapshot"
_SIGN_DOMAIN = b"bastion-exceptions-v1\n"
_FRACTION = re.compile(r"(\.\d{1,9})")


class ExceptionsError(Exception):
    """An envelope could not be trusted."""


def _parse_time(value: str) -> datetime:
    """Parse an RFC 3339 time as Go writes it (up to nanoseconds, 'Z' suffix)."""
    v = value.strip().replace("Z", "+00:00")
    v = _FRACTION.sub(lambda m: (m.group(1) + "000000")[:7], v, count=1)
    return datetime.fromisoformat(v)


def parse_public_keys(csv: str) -> list[Ed25519PublicKey]:
    """Decode a comma-separated list of base64 Ed25519 public keys (several during rotation)."""
    keys = []
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            raw = base64.b64decode(part, validate=True)
            if len(raw) != 32:
                raise ValueError("wrong length")
            keys.append(Ed25519PublicKey.from_public_bytes(raw))
        except Exception as exc:  # noqa: BLE001
            raise ExceptionsError("public key must be a base64 Ed25519 public key") from exc
    return keys


def verify(keys: list[Ed25519PublicKey], envelope: dict) -> dict:
    """Check the signature against any key and return the decoded payload."""
    if not keys:
        raise ExceptionsError("no public key configured")
    try:
        payload = base64.b64decode(envelope["payload"], validate=True)
        sig = base64.b64decode(envelope["sig"], validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ExceptionsError("malformed envelope") from exc
    for key in keys:
        try:
            key.verify(sig, _SIGN_DOMAIN + payload)
            break
        except InvalidSignature:
            continue
    else:
        raise ExceptionsError("signature verification failed")
    try:
        return json.loads(payload)
    except ValueError as exc:
        raise ExceptionsError("bad payload") from exc


def _active(e: dict, now: datetime) -> bool:
    return not e.get("revoked") and now < _parse_time(e["expires_at"])


class Client:
    """Module-side exception store. Safe for use from multiple threads."""

    def __init__(self, module: str, public_keys: str = "", path: str = "") -> None:
        self.module = module
        self.path = path
        self._keys = parse_public_keys(public_keys)
        self._lock = threading.RLock()
        self._by_id: dict[str, dict] = {}
        self._envs: list[dict] = []
        self._on_change: Optional[Callable[[], None]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mtime = 0.0
        # Injectable so tests can pin time; expiry is judged here, never trusted from Tracker.
        self._clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
        if self._keys:
            self._load_file()

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def set_on_change(self, fn: Callable[[], None]) -> None:
        self._on_change = fn

    def allowed(self, rule: str, tenant: str, user: str = "") -> Optional[str]:
        """Return the ID of an active exception that exempts ``rule`` for this caller, or None."""
        if not self.enabled or not tenant:
            return None
        now = self._clock()
        with self._lock:
            for e in self._by_id.values():
                if (
                    e["rule"] == rule
                    and e["tenant_id"] == tenant
                    and self.module in e.get("modules", [])
                    and (not e.get("user_id") or e["user_id"] == user)
                    and _active(e, now)
                ):
                    return e["id"]
        return None

    def list(self) -> list[dict]:
        with self._lock:
            return list(self._by_id.values())

    def apply(self, envelope: dict) -> None:
        """Verify, merge and persist an envelope. Raises ExceptionsError if untrusted."""
        if not self.enabled:
            raise ExceptionsError("disabled (no public key)")
        payload = verify(self._keys, envelope)
        with self._lock:
            changed = self._merge(payload)
            if changed:
                self._envs.append(envelope)
                self._prune()
                self._persist()
        if changed:
            self._notify()

    # ── internals ────────────────────────────────────────────────────────────
    def _merge(self, payload: dict) -> bool:
        changed = False
        for e in payload.get("exceptions") or []:
            cur = self._by_id.get(e["id"])
            if cur and cur["version"] >= e["version"]:
                continue
            self._by_id[e["id"]] = e
            changed = True
        return changed

    def _prune(self) -> None:
        now = self._clock()
        kept = []
        for env in self._envs:
            try:
                payload = json.loads(base64.b64decode(env["payload"]))
            except Exception:  # noqa: BLE001
                continue
            for e in payload.get("exceptions") or []:
                cur = self._by_id.get(e["id"])
                if cur and cur["version"] == e["version"] and now < _parse_time(e["expires_at"]):
                    kept.append(env)
                    break
        self._envs = kept
        for eid in [k for k, e in self._by_id.items() if now >= _parse_time(e["expires_at"])]:
            del self._by_id[eid]

    def _persist(self) -> None:
        if not self.path:
            return
        try:
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, mode=0o750, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".exceptions-")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._envs, fh)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("[exceptions] could not persist %s: %s", self.path, exc)

    def _load_file(self) -> None:
        """Merge every envelope in the file whose signature verifies; ignore the rest.

        Editing the file by hand therefore cannot create an exception.
        """
        if not self.path or not os.path.exists(self.path):
            return
        try:
            mtime = os.path.getmtime(self.path)
            with open(self.path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            log.warning("[exceptions] ignoring unreadable %s: %s", self.path, exc)
            return
        envs = data if isinstance(data, list) else [data]  # also accept one exported envelope
        changed = False
        with self._lock:
            for env in envs:
                try:
                    payload = verify(self._keys, env)
                except (ExceptionsError, TypeError, KeyError) as exc:
                    log.warning("[exceptions] ignoring envelope in %s: %s", self.path, exc)
                    continue
                if self._merge(payload):
                    self._envs.append(env)
                    changed = True
            self._prune()
            self._last_mtime = mtime
        if changed:
            self._notify()

    def _notify(self) -> None:
        if self._on_change:
            try:
                self._on_change()
            except Exception:  # noqa: BLE001
                log.exception("[exceptions] on_change callback failed")

    # ── background: NATS push + snapshot, file watch ─────────────────────────
    def start(self, nats_url: str = "", file_poll_seconds: float = 30.0) -> None:
        """Start background sync. Never blocks and never fails startup."""
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, args=(nats_url, file_poll_seconds), daemon=True, name="exceptions-sync"
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self, nats_url: str, poll: float) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._main(nats_url, poll))
        finally:
            loop.close()

    async def _main(self, nats_url: str, poll: float) -> None:
        # The file watcher and the bus are independent: a down bus must never delay
        # picking up an envelope an operator dropped into the file.
        tasks = []
        if self.path and poll:
            tasks.append(asyncio.create_task(self._watch_file(poll)))
        if nats_url:
            tasks.append(asyncio.create_task(self._supervise_nats(nats_url)))
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.2)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch_file(self, poll: float) -> None:
        while True:
            await asyncio.sleep(poll)
            try:
                if os.path.getmtime(self.path) > self._last_mtime:
                    self._load_file()
            except OSError:
                pass

    async def _supervise_nats(self, url: str) -> None:
        nc = None
        try:
            while True:
                if nc is None or nc.is_closed:
                    nc = await self._connect(url)  # sleeps 5s on failure, then retries
                else:
                    await asyncio.sleep(1)
        finally:
            if nc is not None and not nc.is_closed:
                await nc.close()

    async def _connect(self, url: str):
        try:
            import nats  # local import: nats-py is only needed when a bus is used

            async def on_update(msg) -> None:
                try:
                    self.apply(json.loads(msg.data))
                except (ExceptionsError, ValueError, TypeError, KeyError) as exc:
                    log.warning("[exceptions] rejected update: %s", exc)

            holder: dict = {}

            async def on_reconnect() -> None:
                # The bus came back: catch up on anything missed while disconnected.
                if holder.get("nc") is not None:
                    await self._request_snapshot(holder["nc"])

            nc = await nats.connect(
                url,
                connect_timeout=2,
                max_reconnect_attempts=-1,
                reconnect_time_wait=5,
                reconnected_cb=on_reconnect,
            )
            holder["nc"] = nc
            await nc.subscribe(SUBJECT_UPDATES, cb=on_update)
            await self._request_snapshot(nc)
            return nc
        except Exception as exc:  # noqa: BLE001
            log.warning("[exceptions] nats unavailable (%s); using local exceptions only", exc)
            await asyncio.sleep(5)
            return None

    async def _request_snapshot(self, nc) -> None:
        """Ask Tracker for everything, covering what was missed while away."""
        try:
            msg = await nc.request(SUBJECT_SNAPSHOT, b"", timeout=3)
            self.apply(json.loads(msg.data))
        except Exception as exc:  # noqa: BLE001
            log.warning("[exceptions] no snapshot from tracker (%s); keeping local exceptions", exc)
