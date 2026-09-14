from pathlib import Path

import pytest

from chatbot.config import ChatSettings
from chatbot.graph import last_message_text, skip_extra_tool_calls, tool_messages_this_turn
from chatbot.tools import _MAX_RAG_LIMIT, _filters_from_args, apply_ui_focus_filters, format_hits, ui_document_id, ui_site_id


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


def test_format_hits_empty():
    assert format_hits([]) == "No matching chunks found in the knowledge base."


def test_format_hits_includes_source_and_truncates():
    hits = [
        {
            "score": 0.87654,
            "text": "A" * 900,
            "source_path": "/docs/contrat.pdf",
            "document_id": "abc123",
            "page": 4,
            "doc_type": "contrat",
            "heading_path": "Art. 7 Responsabilité",
            "entities": ["ACME"],
        }
    ]
    out = format_hits(hits, max_chars=50)
    assert "### Hit 1 (score=0.8765)" in out
    assert "source: contrat.pdf" in out
    assert "document_id: abc123" in out
    assert "page: 4" in out
    assert "section: Art. 7 Responsabilité" in out
    assert "entities: ACME" in out
    assert "…" in out
    assert "A" * 50 in out
    assert "A" * 51 not in out.replace("…", "")


def test_search_knowledge_caps_limit(monkeypatch):
    pytest.importorskip("langchain_core")

    captured: dict = {}

    def fake_search(query, *, limit=5, filters=None, settings=None):
        captured["limit"] = limit
        return []

    monkeypatch.setattr("rag_ingestion.retrieve.search", fake_search)

    from chatbot.tools import build_search_tool

    tool = build_search_tool(_settings())
    tool.invoke({"query": "x", "limit": 99})
    assert captured["limit"] == _MAX_RAG_LIMIT


def test_trim_messages_for_model_keeps_recent():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from chatbot.graph import _trim_messages_for_model

    messages = []
    for i in range(20):
        messages.append(HumanMessage(content=f"q{i} " + ("word " * 200)))
        messages.append(AIMessage(content=f"a{i}"))
        messages.append(
            ToolMessage(content="hit " * 500, tool_call_id=f"c{i}", name="search_knowledge")
        )
        messages.append(AIMessage(content=f"final{i}"))
    out = _trim_messages_for_model({"messages": messages}, max_tokens=2000)
    trimmed = out["llm_input_messages"]
    assert len(trimmed) < len(messages)
    assert isinstance(trimmed[0], HumanMessage)


def test_filters_from_args_skips_blank():
    assert _filters_from_args() == {}
    assert _filters_from_args(doc_type="  ", entities="", document_id="") == {}
    assert _filters_from_args(
        doc_type="rapport",
        entities="ACME, Hydro-Québec",
        document_id="abc",
    ) == {
        "doc_type": "rapport",
        "entities": "ACME, Hydro-Québec",
        "document_id": "abc",
    }


def test_apply_ui_focus_filters_document_wins_over_site():
    token_s = ui_site_id.set("lot:2363352")
    token_d = ui_document_id.set("aaa")
    try:
        out = apply_ui_focus_filters({"project_id": "4405"})
        assert out["document_id"] == "aaa"
        assert out["project_id"] == "4405"
        assert "site_id" not in out
    finally:
        ui_document_id.reset(token_d)
        ui_site_id.reset(token_s)


def test_search_knowledge_applies_ui_site(monkeypatch):
    pytest.importorskip("langchain_core")
    captured: dict = {}

    def fake_search(query, *, limit=5, filters=None, settings=None):
        captured["filters"] = filters
        return []

    monkeypatch.setattr("rag_ingestion.retrieve.search", fake_search)
    from chatbot.tools import build_search_tool

    token = ui_site_id.set("lot:2363352")
    try:
        tool = build_search_tool(_settings())
        tool.invoke({"query": "contamination"})
    finally:
        ui_site_id.reset(token)
    assert captured["filters"] == {"site_id": "lot:2363352"}


def test_search_knowledge_calls_retrieve(monkeypatch):
    pytest.importorskip("langchain_core")

    captured: dict = {}

    def fake_search(query, *, limit=5, filters=None, settings=None):
        captured["query"] = query
        captured["limit"] = limit
        captured["filters"] = filters
        return [
            {
                "score": 0.5,
                "text": "extrait",
                "source_path": "a.pdf",
                "page": 1,
                "doc_type": "rapport",
                "heading_path": "",
                "entities": [],
            }
        ]

    monkeypatch.setattr("rag_ingestion.retrieve.search", fake_search)

    from chatbot.tools import build_search_tool

    tool = build_search_tool(_settings())
    text = tool.invoke(
        {"query": "clause", "doc_type": "rapport", "limit": 0},
    )
    assert captured["query"] == "clause"
    assert captured["limit"] == 4
    assert captured["filters"] == {"doc_type": "rapport"}
    assert "extrait" in text
    assert "a.pdf" in text


class _Msg:
    def __init__(self, content):
        self.content = content


def test_last_message_text_string_and_blocks():
    assert last_message_text({}) == ""
    assert last_message_text({"messages": [_Msg("bonjour")]}) == "bonjour"
    assert last_message_text(
        {"messages": [_Msg([{"type": "text", "text": "a"}, {"text": "b"}])]}
    ) == "a\nb"


def test_tool_messages_this_turn_counts_after_last_human():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        HumanMessage(content="q1"),
        ToolMessage(content="old", tool_call_id="x", name="search_knowledge"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "search_knowledge", "id": "a", "args": {}}]),
        ToolMessage(content="h1", tool_call_id="a", name="search_knowledge"),
        AIMessage(content="", tool_calls=[{"name": "search_knowledge", "id": "b", "args": {}}]),
        ToolMessage(content="h2", tool_call_id="b", name="search_knowledge"),
    ]
    assert tool_messages_this_turn(messages) == 2
    assert tool_messages_this_turn(messages[:3]) == 0


def test_skip_extra_tool_calls_dummies_over_quota():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="h1", tool_call_id="a", name="search_knowledge"),
        ToolMessage(content="h2", tool_call_id="b", name="search_knowledge"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_knowledge", "id": "c", "args": {"query": "x"}},
            ],
        ),
    ]
    extras = skip_extra_tool_calls(messages, max_tool_calls=2)
    assert len(extras) == 1
    assert extras[0].tool_call_id == "c"
    assert "skipped" in extras[0].content


def test_skip_extra_tool_calls_keeps_remaining_budget():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_knowledge", "id": "a", "args": {}},
                {"name": "search_knowledge", "id": "b", "args": {}},
                {"name": "search_knowledge", "id": "c", "args": {}},
            ],
        ),
    ]
    extras = skip_extra_tool_calls(messages, max_tool_calls=2)
    assert [m.tool_call_id for m in extras] == ["c"]


def test_recursion_limit_for_scales_with_tool_budget():
    from chatbot.graph import recursion_limit_for

    assert recursion_limit_for(2) >= 12
    assert recursion_limit_for(5) > recursion_limit_for(2)

