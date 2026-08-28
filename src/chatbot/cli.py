"""CLI `rag-chat` : REPL et sous-commande `serve` (API HTTP)."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from rag_ingestion.logging_setup import configure_logging

logger = logging.getLogger(__name__)

_CHAT_EXTRA_HINT = (
    "Chat extras are missing. Install with: pip install -e '.[chat]'"
)


def _require_chat() -> None:
    """Vérifie que l'extra `[chat]` est installé."""
    try:
        import fastapi  # noqa: F401
        import langchain_openai  # noqa: F401
        import langgraph  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{_CHAT_EXTRA_HINT}\n({exc})") from exc


def _format_invoke_error(exc: BaseException) -> str:
    """Message utilisateur pour les erreurs LLM / contexte courantes."""
    text = str(exc)
    if "exceed_context_size_error" in text or "exceeds the available context size" in text:
        return (
            "error: context window exceeded (conversation + RAG hits too large). "
            "Type /new to start a fresh thread, or raise llama-server -c / n_ctx."
        )
    return f"error: {exc}"


def cmd_repl(_args: argparse.Namespace) -> int:
    """Boucle interactive : l'agent appelle l'outil RAG si besoin."""
    _require_chat()
    from chatbot.config import load_chat_settings
    from chatbot.graph import build_graph, last_message_text, recursion_limit_for

    settings = load_chat_settings()
    graph = build_graph(settings)
    thread_id = str(uuid.uuid4())
    print("RAG chat (llama-server + Qdrant). Commands: /quit  /new")
    print(f"thread_id={thread_id}")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return 0
        if line == "/new":
            thread_id = str(uuid.uuid4())
            print(f"New conversation. thread_id={thread_id}")
            continue
        try:
            result = graph.invoke(
                {"messages": [{"role": "user", "content": line}]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": recursion_limit_for(settings.max_tool_calls),
                },
            )
        except Exception as exc:
            logger.exception("Chat turn failed")
            print(_format_invoke_error(exc), file=sys.stderr)
            print()
            continue
        print(f"assistant> {last_message_text(result)}")
        print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Lance l'API FastAPI (`POST /chat`, `GET /health`)."""
    _require_chat()
    import uvicorn

    from chatbot.config import load_chat_settings

    s = load_chat_settings()
    host = args.host or s.api_host
    port = args.port if args.port is not None else s.api_port
    logger.info("Serving RAG chat API on http://%s:%s", host, port)
    uvicorn.run("chatbot.api:create_app", factory=True, host=host, port=port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parser argparse : REPL par défaut, `serve` pour l'API."""
    parser = argparse.ArgumentParser(
        prog="rag-chat",
        description="Chatbot LangGraph over the local RAG knowledge base (llama-server).",
    )
    sub = parser.add_subparsers(dest="command")

    parser.set_defaults(func=cmd_repl)

    serve = sub.add_parser("serve", help="HTTP API (POST /chat, GET /health)")
    serve.add_argument("--host", default=None, help="Bind host (default: CHAT_API_HOST)")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: CHAT_API_PORT)")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée `rag-chat` / `python -m chatbot`."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
