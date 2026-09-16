"""Chronométrage d'un tour rag-chat : LLM, outils RAG et sous-étapes."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from rag_ingestion.logging_setup import StepTimer, format_duration

logger = logging.getLogger(__name__)

_active_timer: ContextVar[StepTimer | None] = ContextVar("_active_timer", default=None)


def start_turn_timer() -> StepTimer:
    """Démarre le chronomètre du tour courant (thread-safe via ContextVar)."""
    timer = StepTimer(logger)
    _active_timer.set(timer)
    return timer


def get_turn_timer() -> StepTimer | None:
    """Timer actif pour ce tour, ou None hors invoke rag-chat."""
    return _active_timer.get()


def clear_turn_timer() -> None:
    _active_timer.set(None)


def record_step(timer: StepTimer, name: str, seconds: float) -> None:
    """Accumule une durée (plusieurs appels LLM / search possibles)."""
    timer.steps[name] = timer.steps.get(name, 0.0) + seconds


def invoke_config(base: dict[str, Any], timer: StepTimer) -> dict[str, Any]:
    """Ajoute le callback de chronométrage à la config LangGraph."""
    out = dict(base)
    callbacks = list(out.get("callbacks") or [])
    callbacks.append(ChatTimingHandler(timer))
    out["callbacks"] = callbacks
    return out


def log_turn_summary(timer: StepTimer) -> None:
    """Récapitulatif des étapes sur les logs (INFO)."""
    logger.info("Chat turn finished (%s).", format_duration(timer.total_seconds()))
    for line in timer.summary_lines():
        logger.info("%s", line)


def turn_timing_payload(timer: StepTimer) -> dict[str, Any]:
    """Payload JSON pour l'UI / l'API (`total_seconds` + `steps`)."""
    return {
        "total_seconds": round(timer.total_seconds(), 4),
        "steps": {k: round(v, 4) for k, v in timer.steps.items()},
    }


def invoke_turn(
    graph: Any,
    message: str,
    *,
    thread_id: str,
    recursion_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke le graphe avec chronométrage ; log le récap à la fin."""
    timer = start_turn_timer()
    result: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    try:
        config = invoke_config(
            {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            },
            timer,
        )
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        timing = turn_timing_payload(timer)
        return result, timing
    finally:
        log_turn_summary(timer)
        clear_turn_timer()


class ChatTimingHandler(BaseCallbackHandler):
    """Callback LangChain : durées LLM et outils (ex. search_knowledge)."""

    def __init__(self, timer: StepTimer) -> None:
        self._timer = timer
        self._llm_t0: dict[UUID, float] = {}
        self._tool_t0: dict[UUID, float] = {}
        self._tool_names: dict[UUID, str] = {}
        self._llm_n = 0
        self._tool_n: dict[str, int] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_t0[run_id] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        t0 = self._llm_t0.pop(run_id, None)
        if t0 is None:
            return
        elapsed = time.perf_counter() - t0
        self._llm_n += 1
        record_step(self._timer, "llm", elapsed)
        logger.info(
            "  LLM call %s (%s).",
            self._llm_n,
            format_duration(elapsed),
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        t0 = self._llm_t0.pop(run_id, None)
        if t0 is not None:
            elapsed = time.perf_counter() - t0
            record_step(self._timer, "llm", elapsed)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or "tool")
        self._tool_t0[run_id] = time.perf_counter()
        self._tool_names[run_id] = name

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        t0 = self._tool_t0.pop(run_id, None)
        name = self._tool_names.pop(run_id, "tool")
        if t0 is None:
            return
        elapsed = time.perf_counter() - t0
        record_step(self._timer, name, elapsed)
        self._tool_n[name] = self._tool_n.get(name, 0) + 1
        logger.info(
            "  Tool %s #%s (%s).",
            name,
            self._tool_n[name],
            format_duration(elapsed),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        t0 = self._tool_t0.pop(run_id, None)
        name = self._tool_names.pop(run_id, "tool")
        if t0 is not None:
            record_step(self._timer, name, time.perf_counter() - t0)
