import pytest

from rag_ingestion.llm_providers import (
    GEMINI_OPENAI_BASE,
    langextract_extract_kwargs,
    resolve_chat_llm,
    resolve_extract_llm,
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "CHAT_PROVIDER",
        "LANGEXTRACT_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_BASE_URL",
        "LLAMA_SERVER_BASE_URL",
        "LLAMA_SERVER_MODEL",
        "LLAMA_SERVER_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "LANGEXTRACT_MODEL",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_unknown_provider(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_chat_llm()


def test_resolve_chat_gemini(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    llm = resolve_chat_llm(json_model="local")
    assert llm.provider == "gemini"
    assert llm.model == "gemini-3.5-flash-lite"
    assert llm.api_key == "test-gemini-key"
    assert llm.base_url.rstrip("/") == GEMINI_OPENAI_BASE


def test_resolve_chat_openai(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = resolve_chat_llm()
    assert llm.provider == "openai"
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.model == "gpt-4o-mini"
    assert llm.api_key == "sk-test"


def test_chat_provider_overrides_llm_provider(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("CHAT_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    llm = resolve_chat_llm()
    assert llm.provider == "gemini"


def test_resolve_extract_gemini_uses_gemini_model(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    extract = resolve_extract_llm(
        langextract_model="llama3.1:8b",
        ollama_base_url="http://localhost:11434",
    )
    assert extract.provider == "gemini"
    assert extract.model == "gemini-3.5-flash-lite"
    kwargs = langextract_extract_kwargs(extract, timeout_seconds=30)
    assert kwargs["config"].provider == "gemini"
    assert kwargs["use_schema_constraints"] is True


def test_resolve_extract_openai(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LANGEXTRACT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    extract = resolve_extract_llm(
        langextract_model="ignored-when-openai",
        ollama_base_url="http://localhost:11434",
    )
    assert extract.provider == "openai"
    kwargs = langextract_extract_kwargs(extract, timeout_seconds=12)
    assert kwargs["config"].provider == "openai"
    assert kwargs["config"].provider_kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert kwargs["fence_output"] is True


def test_resolve_extract_ollama_keeps_url(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    extract = resolve_extract_llm(
        langextract_model="nemotron-3-nano:4b",
        ollama_base_url="http://localhost:11434",
    )
    assert extract.provider == "ollama"
    assert extract.model == "nemotron-3-nano:4b"
    kwargs = langextract_extract_kwargs(extract, timeout_seconds=9)
    assert kwargs["config"].provider == "ollama"
    assert kwargs["config"].provider_kwargs["model_url"] == "http://localhost:11434"
