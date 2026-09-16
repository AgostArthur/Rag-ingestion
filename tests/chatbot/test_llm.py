"""Tests du client LLM chatbot (branche Gemini vs OpenAI-compat)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_settings(*, provider: str = "local") -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider=provider,
        llama_server_base_url="http://localhost:8080/v1",
        llama_server_model="gemini-3.5-flash-lite",
        llama_server_api_key="test-key",
        temperature=0.0,
        top_p=1.0,
    )


def test_build_chat_model_gemini_uses_native_client():
    from chatbot.llm import build_chat_model

    fake = MagicMock(name="ChatGoogleGenerativeAI")
    with patch.dict("sys.modules", {"langchain_google_genai": MagicMock(ChatGoogleGenerativeAI=fake)}):
        # Re-import path: build_chat_model imports inside the function.
        build_chat_model(_fake_settings(provider="gemini"))

    fake.assert_called_once_with(
        model="gemini-3.5-flash-lite",
        api_key="test-key",
        temperature=0.0,
        top_p=1.0,
    )


def test_build_chat_model_openai_compat_for_local():
    from chatbot.llm import build_chat_model

    fake = MagicMock(name="ChatOpenAI")
    with patch("langchain_openai.ChatOpenAI", fake):
        build_chat_model(_fake_settings(provider="local"))

    fake.assert_called_once_with(
        base_url="http://localhost:8080/v1",
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        top_p=1.0,
    )


def test_build_chat_model_gemini_missing_package():
    from chatbot.llm import build_chat_model

    import builtins

    real_import = builtins.__import__

    def _block_google(name, *args, **kwargs):
        if name == "langchain_google_genai" or name.startswith("langchain_google_genai."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_block_google):
        with pytest.raises(ImportError, match="langchain-google-genai"):
            build_chat_model(_fake_settings(provider="gemini"))
