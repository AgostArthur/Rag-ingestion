from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from chatbot.config import ChatSettings


def _settings(**overrides: object) -> ChatSettings:
    base: dict = dict(
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
    base.update(overrides)
    return ChatSettings(**base)  # type: ignore[arg-type]


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


class _FakeGraph:
    def invoke(self, state, config=None):
        return {"messages": []}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr("chatbot.api.build_graph", lambda settings: _FakeGraph())
    monkeypatch.setattr("chatbot.api.last_message_text", lambda result: "pong")
    monkeypatch.setattr(
        "chatbot.api.envelope_from_result",
        lambda result: {
            "focus": {"document_ids": [], "project_ids": [], "site_ids": []},
            "documents": [],
            "citations": [],
        },
    )
    monkeypatch.setattr(
        "chatbot.api.invoke_turn",
        lambda graph, message, *, thread_id, recursion_limit: (
            {"messages": []},
            {"total_seconds": 0.12, "steps": {"llm": 0.1}},
        ),
    )
    from chatbot.api import create_app

    app = create_app(_settings(checkpoint_db=tmp_path / "ck.sqlite"))
    return TestClient(app)


def test_chat_openapi_uses_json_body_not_query(client: TestClient):
    schema = client.app.openapi()
    post = schema["paths"]["/chat"]["post"]
    assert "requestBody" in post
    params = post.get("parameters") or []
    assert not any(p.get("in") == "query" and p.get("name") in {"body", "payload"} for p in params)


def test_chat_post_json_is_accepted(client: TestClient):
    res = client.post("/chat", json={"message": "hello", "site_id": "lot:2363352"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["reply"] == "pong"
    assert data["thread_id"]
    assert data["focus"]["document_ids"] == []
    assert data["timing"]["total_seconds"] == 0.12
    assert data["timing"]["steps"]["llm"] == 0.1


def test_chat_injects_context_bar_into_the_prompt(monkeypatch, tmp_path: Path):
    from chatbot.api import create_app, message_with_context

    seen: dict[str, str] = {}

    def _capture(graph, message, *, thread_id, recursion_limit):
        seen["message"] = message
        return {"messages": []}, {"total_seconds": 0.01, "steps": {}}

    monkeypatch.setattr("chatbot.api.build_graph", lambda settings: _FakeGraph())
    monkeypatch.setattr("chatbot.api.last_message_text", lambda result: "pong")
    monkeypatch.setattr(
        "chatbot.api.envelope_from_result",
        lambda result: {
            "focus": {"document_ids": [], "project_ids": [], "site_ids": []},
            "documents": [],
            "citations": [],
        },
    )
    monkeypatch.setattr("chatbot.api.invoke_turn", _capture)
    client = TestClient(create_app(_settings(checkpoint_db=tmp_path / "ck.sqlite")))
    res = client.post(
        "/chat",
        json={
            "message": "quels dépassements",
            "context": [
                "@E25 - ÉES phase II.pdf",
                "2025-08-08 Analyse laboratoire",
            ],
        },
    )
    assert res.status_code == 200, res.text
    assert seen["message"] == message_with_context(
        "quels dépassements",
        ["@E25 - ÉES phase II.pdf", "2025-08-08 Analyse laboratoire"],
    )
    assert seen["message"].startswith("Contexte sélectionné dans l'interface:")
    assert "Question: quels dépassements" in seen["message"]


def test_message_with_context_leaves_plain_questions():
    from chatbot.api import message_with_context

    assert message_with_context("bonjour", []) == "bonjour"
    assert message_with_context("bonjour", ["  ", ""]) == "bonjour"


def test_chat_missing_body_is_json_not_query(client: TestClient):
    res = client.post("/chat")
    assert res.status_code == 422
    loc = res.json()["detail"][0]["loc"]
    assert loc[0] == "body"
    assert "query" not in loc
