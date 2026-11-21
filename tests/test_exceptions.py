"""Tests for the scoped-exceptions client (shared with anchor and the Go modules)."""
import base64
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from navigator.exceptions import Client, ExceptionsError, parse_public_keys, verify

DOMAIN = b"bastion-exceptions-v1\n"
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "exceptions_go_envelope.json")


def keypair():
    sk = Ed25519PrivateKey.generate()
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, base64.b64encode(pub).decode()


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"  # nanosecond-style, as Go writes it


def exc(eid="e1", version=1, rule="navigator.purpose:marketing", tenant="acme", user="", modules=("navigator",),
        revoked=False, hours=24):
    return {
        "id": eid, "version": version, "rule": rule, "modules": list(modules), "tenant_id": tenant,
        **({"user_id": user} if user else {}), "justification": "needed for the campaign",
        "created_by": "admin", "created_at": iso(datetime.now(timezone.utc)),
        "expires_at": iso(datetime.now(timezone.utc) + timedelta(hours=hours)),
        **({"revoked": True} if revoked else {}),
    }


def sign(sk, *exceptions):
    payload = json.dumps({"kind": "update", "issued_at": iso(datetime.now(timezone.utc)),
                          "exceptions": list(exceptions)}).encode()
    return {"payload": base64.b64encode(payload).decode(),
            "sig": base64.b64encode(sk.sign(DOMAIN + payload)).decode()}


def test_disabled_without_key_never_exempts():
    c = Client("navigator", "")
    assert not c.enabled
    assert c.allowed("navigator.purpose:marketing", "acme") is None
    with pytest.raises(ExceptionsError):
        c.apply({"payload": "", "sig": ""})


def test_scope_tenant_user_module_and_rule():
    sk, pub = keypair()
    c = Client("navigator", pub)
    c.apply(sign(sk, exc(user="alice")))
    assert c.allowed("navigator.purpose:marketing", "acme", "alice") == "e1"
    assert c.allowed("navigator.purpose:marketing", "acme", "bob") is None      # other user
    assert c.allowed("navigator.purpose:marketing", "globex", "alice") is None  # other tenant
    assert c.allowed("navigator.purpose:research", "acme", "alice") is None     # other rule
    other_module = Client("anchor", pub)
    other_module.apply(sign(sk, exc(user="alice")))
    assert other_module.allowed("navigator.purpose:marketing", "acme", "alice") is None


def test_rejects_forged_and_tampered():
    sk, pub = keypair()
    attacker, _ = keypair()
    c = Client("navigator", pub)
    with pytest.raises(ExceptionsError):
        c.apply(sign(attacker, exc()))
    env = sign(sk, exc())
    env["payload"] = base64.b64encode(base64.b64decode(env["payload"]).replace(b"acme", b"evil")).decode()
    with pytest.raises(ExceptionsError):
        c.apply(env)
    assert c.list() == []


def test_replay_cannot_undo_revocation():
    sk, pub = keypair()
    c = Client("navigator", pub)
    created = sign(sk, exc(version=1))
    c.apply(created)
    c.apply(sign(sk, exc(version=2, revoked=True)))
    assert c.allowed("navigator.purpose:marketing", "acme") is None
    c.apply(created)  # attacker replays the old, validly signed message
    assert c.allowed("navigator.purpose:marketing", "acme") is None


def test_expiry_is_judged_locally():
    sk, pub = keypair()
    c = Client("navigator", pub)
    c.apply(sign(sk, exc(hours=1)))
    assert c.allowed("navigator.purpose:marketing", "acme")
    c._clock = lambda: datetime.now(timezone.utc) + timedelta(hours=2)
    assert c.allowed("navigator.purpose:marketing", "acme") is None


def test_persistence_survives_restart_without_tracker(tmp_path):
    sk, pub = keypair()
    f = str(tmp_path / "exc.json")
    Client("navigator", pub, f).apply(sign(sk, exc()))
    assert Client("navigator", pub, f).allowed("navigator.purpose:marketing", "acme") == "e1"


def test_edited_file_grants_nothing_but_exported_envelope_works(tmp_path):
    sk, pub = keypair()
    env = sign(sk, exc())
    f = tmp_path / "exc.json"
    bad = dict(env, payload=base64.b64encode(base64.b64decode(env["payload"]).replace(b"acme", b"evil")).decode())
    f.write_text(json.dumps([bad]))
    assert Client("navigator", pub, str(f)).list() == []
    f.write_text(json.dumps(env))  # a single exported envelope dropped in (air-gapped module)
    assert Client("navigator", pub, str(f)).allowed("navigator.purpose:marketing", "acme") == "e1"


def test_key_rotation_accepts_old_and_new():
    old, old_pub = keypair()
    new, new_pub = keypair()
    only_old = Client("navigator", old_pub)
    with pytest.raises(ExceptionsError):
        only_old.apply(sign(new, exc()))
    both = Client("navigator", f"{new_pub},{old_pub}")
    both.apply(sign(old, exc("a")))
    both.apply(sign(new, exc("b", rule="navigator.purpose:research")))
    assert both.allowed("navigator.purpose:marketing", "acme") == "a"
    assert both.allowed("navigator.purpose:research", "acme") == "b"
    with pytest.raises(ExceptionsError):
        parse_public_keys("not-a-key")


def test_on_change_fires_only_for_real_changes():
    sk, pub = keypair()
    c = Client("navigator", pub)
    calls = []
    c.set_on_change(lambda: calls.append(1))
    env = sign(sk, exc())
    c.apply(env)
    c.apply(env)  # same version again: nothing changed
    assert len(calls) == 1


def test_verifies_envelope_signed_by_the_go_implementation():
    """Cross-language check: an envelope produced by the Go code must verify and parse here."""
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    keys = parse_public_keys(fx["public_key"])
    payload = verify(keys, fx["envelope"])
    by_id = {e["id"]: e for e in payload["exceptions"]}
    active, revoked = by_id[fx["active_id"]], by_id[fx["revoked_id"]]
    assert active["rule"] == "navigator.purpose:marketing" and not active.get("revoked")
    assert "<campaign> & more" in active["justification"].replace("data ", "")  # Go escapes <,> and & in JSON
    assert revoked["revoked"] is True

    # Apply with the clock pinned to issue time, so the fixture never goes stale.
    issued = datetime.fromisoformat(payload["issued_at"].replace("Z", "+00:00")[:26] + "+00:00")
    c = Client("navigator", fx["public_key"])
    c._clock = lambda: issued + timedelta(minutes=1)
    c.apply(fx["envelope"])
    assert c.allowed("navigator.purpose:marketing", "acme") == fx["active_id"]
    assert c.allowed("anchor.bias:gender", "acme", "alice") is None  # revoked, and other module


def test_background_start_never_blocks_or_raises_with_bus_down(tmp_path):
    sk, pub = keypair()
    f = str(tmp_path / "exc.json")
    c = Client("navigator", pub, f)
    t0 = time.monotonic()
    c.start("nats://127.0.0.1:1", file_poll_seconds=0.2)  # nothing listens there
    assert time.monotonic() - t0 < 1.0
    # An operator drops a signed file in while the bus is unreachable: picked up by the poller.
    with open(f, "w") as fh:
        json.dump(sign(sk, exc()), fh)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not c.allowed("navigator.purpose:marketing", "acme"):
        time.sleep(0.1)
    c.close()
    assert c.allowed("navigator.purpose:marketing", "acme") == "e1"


# ── live NATS (set EXCEPTIONS_E2E_NATS=host:port; skipped otherwise) ────────────
def _wait(cond, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not os.environ.get("EXCEPTIONS_E2E_NATS"), reason="EXCEPTIONS_E2E_NATS not set")
def test_live_nats_push_snapshot_and_revocation():
    import asyncio

    import nats

    url = "nats://" + os.environ["EXCEPTIONS_E2E_NATS"]
    sk, pub = keypair()
    state = {"envs": [sign(sk, exc("snap"))]}  # what a Tracker snapshot would hold

    # A stand-in Tracker on a background loop: answers snapshot requests, publishes updates.
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box = {}

    async def tracker():
        nc = await nats.connect(url)

        async def on_snapshot(msg):
            await msg.respond(json.dumps(state["envs"][-1]).encode())

        await nc.subscribe("bastion.exceptions.snapshot", cb=on_snapshot)
        box["nc"] = nc
        ready.set()

    t = threading.Thread(target=lambda: (asyncio.set_event_loop(loop), loop.run_until_complete(tracker()), loop.run_forever()), daemon=True)
    t.start()
    assert ready.wait(5)

    def publish(env):
        asyncio.run_coroutine_threadsafe(box["nc"].publish("bastion.exceptions.updates", json.dumps(env).encode()), loop).result(3)
        asyncio.run_coroutine_threadsafe(box["nc"].flush(), loop).result(3)

    c = Client("navigator", pub)
    c.start(url, file_poll_seconds=0)
    try:
        # Starting up fetches the snapshot.
        assert _wait(lambda: c.allowed("navigator.purpose:marketing", "acme") == "snap"), "snapshot not applied"
        # A pushed update applies live.
        publish(sign(sk, exc("live", rule="navigator.purpose:research")))
        assert _wait(lambda: c.allowed("navigator.purpose:research", "acme") == "live"), "push not applied"
        # A forged push is ignored.
        attacker, _ = keypair()
        publish(sign(attacker, exc("evil", rule="navigator.purpose:evil")))
        time.sleep(0.5)
        assert c.allowed("navigator.purpose:evil", "acme") is None
        # Revocation applies.
        publish(sign(sk, exc("live", version=2, rule="navigator.purpose:research", revoked=True)))
        assert _wait(lambda: c.allowed("navigator.purpose:research", "acme") is None), "revocation not applied"
    finally:
        c.close()
        asyncio.run_coroutine_threadsafe(box["nc"].close(), loop).result(3)
        loop.call_soon_threadsafe(loop.stop)


# ── Navigator integration: the purpose pre-filter ──────────────────────────────
from navigator.config import Config  # noqa: E402
from navigator.events import TraceContext  # noqa: E402
from navigator.models import SearchRequest, SearchResult, UserContext  # noqa: E402
from navigator.orchestrator import Orchestrator  # noqa: E402


class _Pub:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


def _orch(client=None):
    pub = _Pub()
    orch = Orchestrator(Config(), None, None, None, None, publisher=pub)
    orch.set_exceptions(client)
    return orch, pub


def _results():
    return [
        SearchResult(document_id="ok", metadata={"permitted_purposes": "marketing,support"}),
        SearchResult(document_id="hr", metadata={"permitted_purposes": "hr"}),   # not for marketing
        SearchResult(document_id="open"),                                          # no restriction
    ]


def _req(tenant="acme", purpose="marketing"):
    return SearchRequest(request_id="r1", tenant_id=tenant, query="q", purpose=purpose, user=UserContext(user_id="alice"))


def _ids(rs):
    return sorted(r.document_id for r in rs)


def test_purpose_filter_enforced_by_default():
    orch, pub = _orch()
    out = orch._apply_purpose_filter(_results(), _req(), TraceContext())
    assert _ids(out) == ["ok", "open"]
    assert [e.event_type for e in pub.events] == ["purpose_filtered"]


def test_exception_lets_one_purpose_through_for_one_tenant_and_records_use():
    sk, pub_key = keypair()
    client = Client("navigator", pub_key)
    client.apply(sign(sk, exc(rule="navigator.purpose:marketing")))
    orch, pub = _orch(client)

    out = orch._apply_purpose_filter(_results(), _req("acme"), TraceContext())
    assert _ids(out) == ["hr", "ok", "open"]  # the restricted document is now included
    assert [e.event_type for e in pub.events] == ["exception_applied"]
    assert pub.events[0].data["exception_ids"] == ["e1"]

    other_tenant, pub2 = _orch(client)
    assert _ids(other_tenant._apply_purpose_filter(_results(), _req("globex"), TraceContext())) == ["ok", "open"]
    other_purpose, _ = _orch(client)
    assert _ids(other_purpose._apply_purpose_filter(_results(), _req("acme", "research"), TraceContext())) == ["open"]


def test_use_is_not_recorded_when_nothing_was_excluded():
    sk, pub_key = keypair()
    client = Client("navigator", pub_key)
    client.apply(sign(sk, exc(rule="navigator.purpose:marketing")))
    orch, pub = _orch(client)
    orch._apply_purpose_filter([SearchResult(document_id="open")], _req(), TraceContext())
    assert pub.events == []


def test_revocation_restores_the_purpose_filter():
    sk, pub_key = keypair()
    client = Client("navigator", pub_key)
    client.apply(sign(sk, exc(version=1, rule="navigator.purpose:marketing")))
    orch, _ = _orch(client)
    assert "hr" in _ids(orch._apply_purpose_filter(_results(), _req(), TraceContext()))
    client.apply(sign(sk, exc(version=2, rule="navigator.purpose:marketing", revoked=True)))
    assert "hr" not in _ids(orch._apply_purpose_filter(_results(), _req(), TraceContext()))


def test_tenant_isolation_and_permissions_are_not_exemptable_rules():
    """Tracker refuses these rules, so no signed exception can name them; the Go check is KnownRule."""
    sk, pub_key = keypair()
    client = Client("navigator", pub_key)
    client.apply(sign(sk, exc(rule="navigator.tenant_isolation")))  # even if one were somehow signed...
    orch, _ = _orch(client)
    # ...the purpose filter consults only navigator.purpose:<purpose>, never the isolation rule.
    assert _ids(orch._apply_purpose_filter(_results(), _req(), TraceContext())) == ["ok", "open"]
