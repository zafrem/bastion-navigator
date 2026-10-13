"""Caller authentication for Navigator's REST and gRPC APIs (shared bearer token).

Navigator trusts the identity a caller asserts (tenant, user, department, roles), so who may call
it is the control that matters. When a token is configured every endpoint except the health and
metrics probes requires ``Authorization: Bearer <token>``. Federation peers present the same kind of
token to each other.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

import grpc

# Environment fallback when server.auth_token is empty.
ENV_TOKEN = "NAVIGATOR_AUTH_TOKEN"

# Reachable without a token so orchestrators and Prometheus can probe the service.
PUBLIC_REST_PATHS = frozenset({"/v1/health", "/v1/health/live", "/v1/health/ready", "/v1/metrics"})
# The service is reachable under the proto's canonical name and the legacy one (see grpc_server).
GRPC_SERVICE_NAMES = ("bastion.navigator.v1.NavigatorService", "navigator.NavigatorService")
PUBLIC_GRPC_METHODS = frozenset(f"/{svc}/Health" for svc in GRPC_SERVICE_NAMES)


def resolve_token(configured: str = "") -> str:
    """The shared token: the configured value, else NAVIGATOR_AUTH_TOKEN, else "" (unauthenticated)."""
    return configured or os.environ.get(ENV_TOKEN, "")


def token_matches(header_value: Optional[str], token: str) -> bool:
    """True if ``header_value`` is ``Bearer <token>``, compared in constant time."""
    if not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(header_value[len("Bearer "):].encode(), token.encode())


class GrpcAuthInterceptor(grpc.ServerInterceptor):
    """Rejects gRPC calls that lack the bearer token (``authorization`` metadata)."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._deny = grpc.unary_unary_rpc_method_handler(self._abort)

    @staticmethod
    def _abort(_request, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid bearer token")

    def intercept_service(self, continuation, handler_call_details):
        if not self._token or handler_call_details.method in PUBLIC_GRPC_METHODS:
            return continuation(handler_call_details)
        for key, value in handler_call_details.invocation_metadata or ():
            if key.lower() == "authorization" and token_matches(value, self._token):
                return continuation(handler_call_details)
        return self._deny
