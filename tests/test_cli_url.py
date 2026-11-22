"""The CLI must only talk http(s): urllib would also happily read file: and ftp: URLs."""
import pytest

from navigator import cli


@pytest.mark.parametrize("base", ["file:///etc", "ftp://example.com", "gopher://x", "/just/a/path", "javascript:alert(1)"])
def test_url_refuses_other_schemes(base):
    with pytest.raises(ValueError, match="http"):
        cli._url(base, "/v1/health")


@pytest.mark.parametrize("base,want", [
    ("http://localhost:8083", "http://localhost:8083/v1/health"),
    ("https://example.com/", "https://example.com/v1/health"),
])
def test_url_accepts_http_and_https(base, want):
    assert cli._url(base, "/v1/health") == want


def test_api_get_does_not_open_a_file_url():
    with pytest.raises(ValueError):
        cli._api_get("file:///etc/passwd", "")


def test_server_host_defaults_to_all_interfaces_and_can_be_narrowed():
    from navigator.config import ServerConfig
    assert ServerConfig().host == "0.0.0.0"
    assert ServerConfig(host="127.0.0.1").host == "127.0.0.1"
