from pathlib import Path

import pytest

from chatbot.cli import _format_invoke_error
from chatbot.config import load_prompt, load_settings_file


def test_format_invoke_error_context_overflow():
    msg = _format_invoke_error(
        Exception(
            "Error code: 400 - {'error': {'code': 400, "
            "'message': 'request exceeds the available context size', "
            "'type': 'exceed_context_size_error'}}"
        )
    )
    assert "/new" in msg
    assert "context" in msg.lower()


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "config" / "chatbot" / "prompt.txt"
DEFAULT_SETTINGS = ROOT / "config" / "chatbot" / "settings.json"


def test_load_default_prompt():
    text = load_prompt(DEFAULT_PROMPT)
    assert "search_knowledge" in text
    assert "RAG" in text


def test_load_default_settings():
    data = load_settings_file(DEFAULT_SETTINGS)
    assert data["llama_server_base_url"].startswith("http")
    assert int(data["rag_limit"]) >= 1
    assert int(data["max_context_tokens"]) >= 2000
    assert int(data["max_tool_calls"]) >= 1
    assert int(data["rag_hit_max_chars"]) >= 100
    assert int(data["rag_table_max_chars"]) >= 400
    assert data["checkpoint_db"]
    assert "temperature" in data
    assert "top_p" in data
    assert "rag_prefetch" in data
    assert "rag_score_threshold" in data
    assert data["rag_include_catalog"] is True
    assert data["rag_hybrid_text"] is True


def test_load_chat_settings_reads_precision_flags():
    from chatbot.config import load_chat_settings

    settings = load_chat_settings()
    assert settings.rag_include_catalog is True
    assert settings.rag_hybrid_text is True
    assert settings.temperature == 0.0
    assert settings.rag_prefetch >= settings.rag_limit
    assert 0.0 <= settings.top_p <= 1.0


def test_load_prompt_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt(tmp_path / "missing.txt")


def test_load_prompt_empty(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_prompt(path)


def test_load_settings_rejects_bad_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_settings_file(bad)


def test_load_settings_rejects_list(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_settings_file(path)
