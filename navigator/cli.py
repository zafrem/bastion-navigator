"""navigator-cli — Bastion Navigator command-line interface (SRS §6.3)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ─── helpers ─────────────────────────────────────────────────────────────────

def _api_get(base_url: str, path: str) -> Any:
    import urllib.request
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=10) as resp:
        return json.loads(resp.read())


def _api_post(base_url: str, path: str, payload: dict) -> Any:
    import urllib.request
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ─── commands ────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace) -> int:
    """Start the Navigator REST + gRPC server."""
    import os
    import sys
    # Re-invoke the main module so the server lifecycle is identical to
    # running `python -m navigator.main` directly.
    argv = ["navigator"]
    if args.config:
        argv += ["--config", args.config]
    sys.argv = argv
    from navigator.main import main as _main
    _main()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Send a search request to a running Navigator server."""
    payload: dict[str, Any] = {
        "query": args.query,
        "top_k": args.top_k,
        "use_hybrid": not args.no_hybrid,
        "use_reranking": not args.no_rerank,
    }
    if args.tenant:
        payload["tenant_id"] = args.tenant
    if args.collection:
        payload["collections"] = [args.collection]
    result = _api_post(args.api, "/v1/navigator/search", payload)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        results = result.get("results", [])
        print(f"Found {len(results)} result(s):")
        for i, r in enumerate(results, 1):
            score = r.get("score", 0.0)
            doc_id = r.get("id", "—")
            text = (r.get("content") or r.get("text") or "")[:120]
            print(f"  [{i}] score={score:.4f}  id={doc_id}")
            if text:
                print(f"       {text}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed a text string via a running Navigator server."""
    payload = {"text": args.text}
    result = _api_post(args.api, "/v1/navigator/embed", payload)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        emb = result.get("embedding", [])
        print(f"Embedding dim={len(emb)}, norm={sum(x**2 for x in emb)**0.5:.6f}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check the health of a running Navigator server."""
    result = _api_get(args.api, "/v1/health")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result.get("status", "unknown")
        version = result.get("version", "?")
        print(f"Navigator  status={status}  version={version}")
    ok = result.get("status") in ("ok", "ready")
    return 0 if ok else 1


def cmd_collections(args: argparse.Namespace) -> int:
    """List available Qdrant collections."""
    result = _api_get(args.api, "/v1/navigator/collections")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for c in result.get("collections", []):
            print(f"  {c.get('name', c)}")
    return 0


# ─── parser ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="navigator-cli",
        description="Bastion Navigator — search, embed, and manage the Navigator service",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # server
    srv = sub.add_parser("server", help="Start the Navigator server")
    srv.add_argument("--config", default="", help="Path to config YAML")
    srv.set_defaults(func=cmd_server)

    # search
    s = sub.add_parser("search", help="Search documents")
    s.add_argument("--query", "-q", required=True, help="Search query")
    s.add_argument("--tenant", "-t", default="", help="Tenant ID")
    s.add_argument("--collection", "-c", default="", help="Restrict to a single collection")
    s.add_argument("--top-k", type=int, default=5, help="Max results to return (default 5)")
    s.add_argument("--no-hybrid", action="store_true", help="Disable hybrid (BM25) search")
    s.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    s.add_argument("--api", default="http://localhost:8082", help="Navigator API base URL")
    s.add_argument("--json", action="store_true", help="Output raw JSON")
    s.set_defaults(func=cmd_search)

    # embed
    e = sub.add_parser("embed", help="Embed a text string")
    e.add_argument("--text", required=True, help="Text to embed")
    e.add_argument("--api", default="http://localhost:8082", help="Navigator API base URL")
    e.add_argument("--json", action="store_true", help="Output raw JSON")
    e.set_defaults(func=cmd_embed)

    # health
    h = sub.add_parser("health", help="Check server health")
    h.add_argument("--api", default="http://localhost:8082", help="Navigator API base URL")
    h.add_argument("--json", action="store_true", help="Output raw JSON")
    h.set_defaults(func=cmd_health)

    # collections
    col = sub.add_parser("collections", help="List available collections")
    col.add_argument("--api", default="http://localhost:8082", help="Navigator API base URL")
    col.add_argument("--json", action="store_true", help="Output raw JSON")
    col.set_defaults(func=cmd_collections)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
