"""Graphe LangGraph ReAct : le LLM décide d'appeler l'outil RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chatbot.config import ChatSettings, load_chat_settings, load_prompt
from chatbot.llm import build_chat_model
from chatbot.tools import build_search_tool

logger = logging.getLogger(__name__)

_STOP_TOOL_NUDGE = (
    "You have reached the maximum number of searches for this turn. "
    "Answer now using only the tool results already in the conversation. "
    "Do not call any tool."
)


def _is_human(message: Any) -> bool:
    """True si le message est une question utilisateur."""
    t = getattr(message, "type", None)
    return t == "human" or type(message).__name__ in {"HumanMessage", "HumanMessageChunk"}


def _is_tool(message: Any) -> bool:
    """True si le message est un résultat d'outil."""
    t = getattr(message, "type", None)
    return t == "tool" or type(message).__name__ in {"ToolMessage", "ToolMessageChunk"}


def tool_messages_this_turn(messages: list[Any]) -> int:
    """Nombre de ToolMessage depuis le dernier message humain (tour courant)."""
    count = 0
    for message in reversed(messages):
        if _is_human(message):
            break
        if _is_tool(message):
            count += 1
    return count


def skip_extra_tool_calls(
    messages: list[Any], *, max_tool_calls: int
) -> list[Any]:
    """Dummy ToolMessage pour les tool_calls au-delà de `max_tool_calls`.

    Sans ça, llama-server peut émettre 5 appels en parallèle, ou relancer
    search après le quota. Les ids skippés sont marqués résolus pour que
    LangGraph ne les exécute pas et, si le quota est à 0, reboucle vers le LLM.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    if not messages or max_tool_calls < 1:
        return []
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    used = tool_messages_this_turn(messages)
    remaining = max(0, max_tool_calls - used)
    skipped = last.tool_calls[remaining:]
    extras: list[Any] = []
    for call in skipped:
        extras.append(
            ToolMessage(
                content="[skipped] max_tool_calls reached; answer from existing results.",
                tool_call_id=str(call.get("id") or ""),
                name=str(call.get("name") or "search_knowledge"),
            )
        )
    return extras


def _trim_messages_for_model(state: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    """Réduit l'historique envoyé au LLM sans toucher au checkpointer.

    Les résultats d'outils s'accumulent vite (chaque tour rappelle search).
    On garde les messages récents sous `max_tokens` (approx.) pour rester
    sous le `n_ctx` de llama-server.
    """
    from langchain_core.messages.utils import trim_messages

    messages = state.get("messages") or []
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter="approximate",
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed}


def build_checkpointer(path: Path):
    """Checkpointer SQLite (conversations durables) ; repli mémoire si extra absent."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning(
            "langgraph-checkpoint-sqlite is not installed; chat threads are in-memory. "
            "pip install -e '.[chat]'"
        )
        return InMemorySaver()

    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()
    logger.info("Chat checkpointer: %s", path)
    return saver


def recursion_limit_for(max_tool_calls: int) -> int:
    """Plafond LangGraph : hook + LLM + outil par search, plus un tour de réponse."""
    return max(12, int(max_tool_calls) * 6 + 8)


def build_graph(settings: ChatSettings | None = None, checkpointer=None):
    """Construit l'agent ReAct (LLM configurable + outil RAG).

    Args:
        settings: Config chatbot ; `config/chatbot/` si omis.
        checkpointer: Persistence de conversation ; SQLite sous
            `checkpoint_db` si omis (`InMemorySaver` si extra sqlite absent).

    Returns:
        Graphe compilé LangGraph (`invoke` / `stream`).
    """
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent

    s = settings or load_chat_settings()
    model = build_chat_model(s)
    tools = [build_search_tool(s)]
    saver = checkpointer if checkpointer is not None else build_checkpointer(s.checkpoint_db)
    max_tokens = s.max_context_tokens
    max_tool_calls = s.max_tool_calls

    def pre_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        out = _trim_messages_for_model(state, max_tokens=max_tokens)
        messages = list(out["llm_input_messages"])
        if tool_messages_this_turn(state.get("messages") or []) >= max_tool_calls:
            messages.append(HumanMessage(content=_STOP_TOOL_NUDGE))
        out["llm_input_messages"] = messages
        return out

    def post_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        extras = skip_extra_tool_calls(
            list(state.get("messages") or []),
            max_tool_calls=max_tool_calls,
        )
        return {"messages": extras} if extras else {}

    def resolve_model(state: dict[str, Any], _runtime: Any):
        used = tool_messages_this_turn(state.get("messages") or [])
        if used >= max_tool_calls:
            return model
        return model.bind_tools(tools)

    return create_react_agent(
        resolve_model,
        tools,
        prompt=load_prompt(s.prompt_file),
        checkpointer=saver,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
    )


def last_message_text(result: dict) -> str:
    """Extrait le texte de la dernière message du state LangGraph."""
    messages = result.get("messages") or []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", messages[-1])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)
