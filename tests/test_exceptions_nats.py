"""Exceptions over a live NATS server.

Tracker publishes signed exceptions on a bus; the module's client must pick them up. The unit tests
feed the client envelopes directly, so none of the transport is exercised there: subscribing,
asking for a snapshot on start, reconnecting after the bus restarts. These tests run a real
nats-server and stand in for Tracker (sign, answer snapshot requests, publish updates).

They need a ``nats-server`` binary: set NATS_SERVER_BIN, or have it on PATH. Without one they are
skipped, and CI fails the job if any is skipped, so a missing server cannot pass silently.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from navigator.exceptions import SUBJECT_SNAPSHOT, SUBJECT_UPDATES, Client

nats = pytest.importorskip("nats")

MODULE = "navigator"
RULE = "navigator.purpose:marketing"
DOMAIN = b"bastion-exceptions-v1\n"
WAIT = 30  # seconds; a reconnect waits 5 s before trying again

SERVER = os.environ.get("NATS_SERVER_BIN") or shutil.which("nats-server")
pytestmark = pytest.mark.skipif(not SERVER, reason="no nats-server binary (set NATS_SERVER_BIN)")


# ── signing, as Tracker does it ───────────────────────────────────────────────

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _keypair():
    sk = Ed25519PrivateKey.generate()
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, base64.b64encode(pub).decode()


def _envelope(sk, *excs: dict, kind: str = "update") -> dict:
    payload = json.dumps({"kind": kind, "issued_at": _iso(datetime.now(timezone.utc)), "exceptions": list(excs)}).encode()
    return {"payload": base64.b64encode(payload).decode(), "sig": base64.b64encode(sk.sign(DOMAIN + payload)).decode()}


def _exc(version=1, tenant="acme", revoked=False, eid="exc-nats", rule=RULE):
    now = datetime.now(timezone.utc)
    e = {
        "id": eid, "version": version, "rule": rule, "modules": [MODULE], "tenant_id": tenant,
        "justification": "live bus test", "created_by": "admin",
        "created_at": _iso(now), "expires_at": _iso(now + timedelta(hours=1)),
    }
    if revoked:
        e["revoked"] = True
    return e


# ── a real server ─────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, port: int | None = None) -> None:
        self.port = port or _free_port()
        self.url = f"nats://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def start(self) -> "Server":
        self.proc = subprocess.Popen(
            [SERVER, "-a", "127.0.0.1", "-p", str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return self
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("nats-server did not start")

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


class FakeTracker:
    """Answers snapshot requests and publishes updates, on its own thread and event loop."""

    def __init__(self, url: str, snapshot: dict) -> None:
        self.url, self.snapshot = url, snapshot
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.nc = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "FakeTracker":
        self.thread.start()
        assert self.ready.wait(15), "fake tracker could not connect"
        return self

    def _run(self) -> None:
        self.loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self.nc = await nats.connect(self.url, max_reconnect_attempts=-1, reconnect_time_wait=1)

        async def answer(msg) -> None:
            await msg.respond(json.dumps(self.snapshot).encode())

        await self.nc.subscribe(SUBJECT_SNAPSHOT, cb=answer)
        await self.nc.flush()
        self.ready.set()
        while not self._done:
            await asyncio.sleep(0.05)
        await self.nc.close()

    _done = False

    def publish(self, envelope: dict) -> None:
        async def go() -> None:
            await self.nc.publish(SUBJECT_UPDATES, json.dumps(envelope).encode())
            await self.nc.flush()

        asyncio.run_coroutine_threadsafe(go(), self.loop).result(10)

    def stop(self) -> None:
        self._done = True
        self.thread.join(5)


def _wait(cond, what: str, timeout: float = WAIT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for: {what}")


@pytest.fixture
def server():
    s = Server().start()
    yield s
    s.stop()


@pytest.fixture
def keys():
    return _keypair()


@pytest.fixture
def client(keys):
    c = Client(MODULE, public_keys=keys[1])
    yield c
    c.close()


@pytest.fixture
def tracker_factory():
    made = []

    def make(url, snapshot):
        t = FakeTracker(url, snapshot).start()
        made.append(t)
        return t

    yield make
    for t in made:
        t.stop()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_snapshot_is_fetched_on_start(server, keys, client, tracker_factory):
    sk, _ = keys
    tracker_factory(server.url, _envelope(sk, _exc(), kind="snapshot"))
    assert client.allowed(RULE, "acme") is None
    client.start(server.url)
    _wait(lambda: client.allowed(RULE, "acme") == "exc-nats", "the snapshot to arrive")
    assert client.allowed(RULE, "globex") is None, "the exception is scoped to its tenant"


def test_pushed_update_applies_and_revocation_removes_it(server, keys, client, tracker_factory):
    sk, _ = keys
    tracker = tracker_factory(server.url, _envelope(sk, kind="snapshot"))
    client.start(server.url)
    time.sleep(1)  # let the subscription settle
    tracker.publish(_envelope(sk, _exc(version=1)))
    _wait(lambda: client.allowed(RULE, "acme") == "exc-nats", "the pushed exception")
    tracker.publish(_envelope(sk, _exc(version=2, revoked=True)))
    _wait(lambda: client.allowed(RULE, "acme") is None, "the revocation")


def test_an_envelope_signed_with_another_key_is_rejected_over_the_bus(server, keys, client, tracker_factory):
    sk, _ = keys
    other, _ = _keypair()
    tracker = tracker_factory(server.url, _envelope(sk, kind="snapshot"))
    client.start(server.url)
    time.sleep(1)
    tracker.publish(_envelope(other, _exc(eid="exc-forged")))
    time.sleep(1.5)
    assert client.allowed(RULE, "acme") is None, "a forged exception must not be applied"
    tracker.publish(_envelope(sk, _exc(eid="exc-real")))  # and the client must not be wedged by it
    _wait(lambda: client.allowed(RULE, "acme") == "exc-real", "a valid update after a forged one")


def test_a_bus_that_starts_late_is_picked_up(keys, client, tracker_factory):
    sk, _ = keys
    port = _free_port()
    client.start(f"nats://127.0.0.1:{port}")  # nothing is listening yet
    time.sleep(1)
    assert client.allowed(RULE, "acme") is None
    server = Server(port).start()
    try:
        tracker_factory(server.url, _envelope(sk, _exc(), kind="snapshot"))
        _wait(lambda: client.allowed(RULE, "acme") == "exc-nats", "the client to connect and fetch the snapshot")
    finally:
        server.stop()


def test_exceptions_missed_while_the_bus_was_down_are_caught_up(keys, client, tracker_factory):
    sk, _ = keys
    server = Server().start()
    try:
        tracker = tracker_factory(server.url, _envelope(sk, _exc(version=1), kind="snapshot"))
        client.start(server.url)
        _wait(lambda: client.allowed(RULE, "acme") == "exc-nats", "the first snapshot")
        server.stop()
        # While the bus is down the exception is revoked. Nothing can be pushed.
        tracker.snapshot = _envelope(sk, _exc(version=2, revoked=True), kind="snapshot")
        time.sleep(1)
        assert client.allowed(RULE, "acme") == "exc-nats", "without the bus the client cannot know yet"
        server = Server(server.port).start()
        _wait(lambda: client.allowed(RULE, "acme") is None, "the client to reconnect and re-request the snapshot")
    finally:
        server.stop()
