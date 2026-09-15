from pathlib import Path

import pytest

from chatbot.config import ChatSettings


def _settings() -> ChatSettings:
    return ChatSettings(
        llama_server_base_url="http://localhost:8080/v1",
        llama_server_model="local",
        llama_server_api_key="sk-no-key-required",
        temperature=0.0,
        rag_limit=4,
        rag_hit_max_chars=600,
        rag_table_max_chars=2400,
        max_context_tokens=48000,
        max_tool_calls=2,
        api_host="127.0.0.1",
        api_port=8000,
        checkpoint_db=Path("data/chat_checkpoints.sqlite"),
        prompt_file=Path("config/chatbot/prompt.txt"),
        settings_file=Path("config/chatbot/settings.json"),
        project_root=Path("."),
    )


def test_root_is_not_404(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setattr("chatbot.api.build_graph", lambda settings: object())
    from chatbot.api import create_app

    client = TestClient(create_app(_settings()))
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["health"] == "/health"
    assert body["chat"] == "POST /chat"
    assert client.get("/favicon.ico").status_code == 204
