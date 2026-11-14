"""A throwaway PKI for TLS tests, built with the ``openssl`` CLI (tests skip when it is missing)."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

OPENSSL = shutil.which("openssl")
needs_openssl = pytest.mark.skipif(OPENSSL is None, reason="openssl CLI not available")


def _run(*args: str) -> None:
    subprocess.run([OPENSSL, *args], check=True, capture_output=True)


class PKI:
    """A CA plus a server certificate (localhost / 127.0.0.1) and a client certificate."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self._serial = 100
        self.ca_cert = self.dir / "ca.pem"
        self.ca_key = self.dir / "ca.key"
        self.server_cert, self.server_key = self.dir / "server.pem", self.dir / "server.key"
        self.client_cert, self.client_key = self.dir / "client.pem", self.dir / "client.key"
        _run("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
             "-keyout", str(self.ca_key), "-out", str(self.ca_cert), "-subj", "/CN=test-ca", "-days", "2",
             "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        self.issue(self.server_cert, self.server_key, "server-1", server=True)
        self.issue(self.client_cert, self.client_key, "client-1", server=False)

    def issue(self, cert: Path, key: Path, cn: str, *, server: bool,
              ca_cert: Path | None = None, ca_key: Path | None = None) -> None:
        self._serial += 1
        csr, ext = self.dir / f"{cert.stem}.csr", self.dir / f"{cert.stem}.ext"
        ext.write_text(
            "subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n" if server
            else "extendedKeyUsage=clientAuth\n"
        )
        _run("req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
             "-keyout", str(key), "-out", str(csr), "-subj", f"/CN={cn}")
        _run("x509", "-req", "-in", str(csr), "-CA", str(ca_cert or self.ca_cert), "-CAkey", str(ca_key or self.ca_key),
             "-set_serial", str(self._serial), "-out", str(cert), "-days", "2", "-extfile", str(ext))

    def rotate_server(self) -> None:
        """Replace the server key pair on disk with a new one from the same CA."""
        self.issue(self.server_cert, self.server_key, "server-2", server=True)
        later = time.time() + 5  # make sure the mtime differs even on coarse filesystems
        for p in (self.server_cert, self.server_key):
            os.utime(p, (later, later))

    def other_ca(self, name: str = "other-ca") -> tuple[Path, Path]:
        """A second, unrelated CA (cert, key)."""
        cert, key = self.dir / f"{name}.pem", self.dir / f"{name}.key"
        _run("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
             "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={name}", "-days", "2",
             "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        return cert, key

    def rogue_client(self) -> tuple[Path, Path]:
        """A client certificate signed by a CA the server does not trust."""
        ca_cert, ca_key = self.other_ca("rogue-ca")
        cert, key = self.dir / "rogue.pem", self.dir / "rogue.key"
        self.issue(cert, key, "rogue", server=False, ca_cert=ca_cert, ca_key=ca_key)
        return cert, key
