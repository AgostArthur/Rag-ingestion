from pathlib import Path

import pytest

from chatbot.health import collect_health, llama_models_url


def test_llama_models_url():
    assert llama_models_url("http://localhost:8080/v1") == "http://localhost:8080/v1/models"


def test_collect_health_reports_components(monkeypatch):
    monkeypatch.setattr("chatbot.health.ping_qdrant", lambda settings=None: (True, "chunks"))
    monkeypatch.setattr("chatbot.health.ping_catalog", lambda settings=None: (True, "ok"))
    monkeypatch.setattr(
        "chatbot.health.ping_http",
        lambda url, timeout=2.0, api_key=None: (False, "connection refused"),
    )
    monkeypatch.setattr(
        "chatbot.health.load_chat_settings",
        lambda: type(
            "S",
            (),
            {
                "llama_server_base_url": "http://localhost:8080/v1",
                "llama_server_model": "local",
                "llama_server_api_key": "sk-no-key-required",
                "llm_provider": "local",
            },
        )(),
    )
    monkeypatch.setattr(
        "chatbot.health.load_settings",
        lambda: type("S", (), {"qdrant_url": "http://localhost:6333"})(),
    )
    report = collect_health()
    assert report["status"] == "degraded"
    assert report["qdrant"]["ok"] is True
    assert report["llm"]["ok"] is False
    assert report["llama_server"]["ok"] is False


def test_build_checkpointer_writes_sqlite(tmp_path: Path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from chatbot.graph import build_checkpointer

    path = tmp_path / "chat.sqlite"
    saver = build_checkpointer(path)
    assert path.is_file()
    assert saver is not None
