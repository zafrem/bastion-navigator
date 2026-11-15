"""Startup safety of navigator.main: the --factory entry point and the mock fallbacks."""
import os

import pytest

from navigator import main as nav_main
from navigator.config import Config


def _boom(*_a, **_k):
    raise OSError("model download failed")


def test_factory_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", os.devnull)
    monkeypatch.delenv("NAVIGATOR_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="refusing to start"):
        nav_main.build_app()


def test_factory_starts_with_a_token(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", os.devnull)
    monkeypatch.setenv("NAVIGATOR_AUTH_TOKEN", "t" * 16)
    assert nav_main.build_app() is not None


def test_failed_embedder_stops_startup_outside_dev_mode(monkeypatch):
    monkeypatch.setattr(nav_main, "build_embedder", _boom)
    cfg = Config()
    cfg.server.auth_token = "t" * 16
    with pytest.raises(RuntimeError, match="embedder unavailable"):
        nav_main._build_runtime(cfg)


def test_failed_searcher_stops_startup_outside_dev_mode(monkeypatch):
    monkeypatch.setattr(nav_main, "QdrantSearcher", _boom)
    cfg = Config()
    cfg.server.auth_token = "t" * 16
    cfg.embedder.type = "mock"
    cfg.reranker.type = "mock"
    with pytest.raises(RuntimeError, match="vector search unavailable"):
        nav_main._build_runtime(cfg)


def test_dev_mode_falls_back_to_mocks(monkeypatch):
    monkeypatch.setattr(nav_main, "build_embedder", _boom)
    monkeypatch.setattr(nav_main, "QdrantSearcher", _boom)
    cfg = Config()
    cfg.server.allow_unauthenticated = True
    cfg.reranker.type = "mock"
    orch, _pub, _hm = nav_main._build_runtime(cfg)
    assert orch is not None
