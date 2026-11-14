"""TLS for the gRPC server: credentials from PEM files, with certificate rotation.

Set ``grpc_tls_cert`` and ``grpc_tls_key`` to serve gRPC over TLS. The files are re-checked on every
new connection, so a rotated certificate is picked up without a restart; if a replacement cannot be
loaded (for example the cert was written but the key not yet), the previous working pair is kept.
Setting ``grpc_tls_client_ca`` as well requires clients to present a certificate signed by it
(mutual TLS). With all three empty TLS is off.
"""
from __future__ import annotations

import logging
import os
import ssl
from typing import Optional

import grpc

log = logging.getLogger(__name__)


class TLSConfigError(ValueError):
    """The gRPC TLS settings are incomplete or unusable."""


def _mtime(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _read(path: str, what: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as exc:
        raise TLSConfigError(f"cannot read {what} {path}: {exc}") from exc


def _validated_pair(cert_file: str, key_file: str) -> tuple[bytes, bytes]:
    """Read a key pair and check that the key belongs to the certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ctx.load_cert_chain(cert_file, key_file)
    except (ssl.SSLError, OSError) as exc:
        raise TLSConfigError(f"invalid grpc_tls_cert / grpc_tls_key: {exc}") from exc
    return _read(key_file, "grpc_tls_key"), _read(cert_file, "grpc_tls_cert")


def server_credentials(
    cert_file: str = "", key_file: str = "", client_ca_file: str = ""
) -> Optional[grpc.ServerCredentials]:
    """gRPC server credentials from PEM files, or ``None`` when TLS is not configured.

    Raises TLSConfigError for incomplete or unusable settings.
    """
    if not (cert_file or key_file or client_ca_file):
        return None
    if not (cert_file and key_file):
        raise TLSConfigError(
            "grpc_tls_cert and grpc_tls_key must be set together (grpc_tls_client_ca also needs both)"
        )

    key, chain = _validated_pair(cert_file, key_file)
    root: Optional[bytes] = None
    if client_ca_file:
        root = _read(client_ca_file, "grpc_tls_client_ca")
        try:
            ssl.create_default_context().load_verify_locations(cadata=root.decode("ascii", "ignore"))
        except ssl.SSLError as exc:
            raise TLSConfigError(f"grpc_tls_client_ca {client_ca_file} has no usable certificate: {exc}") from exc

    initial = grpc.ssl_server_certificate_configuration([(key, chain)], root_certificates=root)
    state = {"mtimes": (_mtime(cert_file), _mtime(key_file))}

    def fetch_config() -> Optional[grpc.ServerCertificateConfiguration]:
        mtimes = (_mtime(cert_file), _mtime(key_file))
        if mtimes == state["mtimes"]:
            return None  # unchanged: keep the current certificate
        state["mtimes"] = mtimes  # remember the attempt: retry only after the next change
        try:
            new_key, new_chain = _validated_pair(cert_file, key_file)
        except TLSConfigError as exc:
            log.warning("[grpc] TLS certificate reload failed, keeping the previous one: %s", exc)
            return None
        log.info("[grpc] TLS certificate reloaded from %s", cert_file)
        return grpc.ssl_server_certificate_configuration([(new_key, new_chain)], root_certificates=root)

    return grpc.dynamic_ssl_server_credentials(
        initial, fetch_config, require_client_authentication=bool(client_ca_file)
    )
