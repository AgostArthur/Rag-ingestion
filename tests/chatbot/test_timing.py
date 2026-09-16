"""Tests du chronométrage rag-chat."""

from __future__ import annotations

import logging
from uuid import uuid4

from chatbot.timing import (
    ChatTimingHandler,
    clear_turn_timer,
    get_turn_timer,
    invoke_turn,
    record_step,
    start_turn_timer,
    turn_timing_payload,
)
from rag_ingestion.logging_setup import StepTimer


def test_record_step_accumulates():
    timer = StepTimer(logging.getLogger("test"))
    record_step(timer, "llm", 0.5)
    record_step(timer, "llm", 0.2)
    assert timer.steps["llm"] == 0.7


def test_turn_timer_contextvar():
    clear_turn_timer()
    assert get_turn_timer() is None
    timer = start_turn_timer()
    assert get_turn_timer() is timer
    clear_turn_timer()
    assert get_turn_timer() is None


def test_chat_timing_handler_llm_and_tool():
    timer = StepTimer(logging.getLogger("test"))
    handler = ChatTimingHandler(timer)
    llm_run = uuid4()
    tool_run = uuid4()

    handler.on_llm_start({}, [], run_id=llm_run)
    handler.on_llm_end(object(), run_id=llm_run)

    handler.on_tool_start({"name": "search_knowledge"}, "{}", run_id=tool_run)
    handler.on_tool_end("ok", run_id=tool_run)

    assert timer.steps["llm"] > 0
    assert timer.steps["search_knowledge"] > 0


class _FakeGraph:
    def __init__(self) -> None:
        self.last_config: dict | None = None

    def invoke(self, state, config=None):
        self.last_config = config
        timer = get_turn_timer()
        if timer is not None:
            record_step(timer, "llm", 0.01)
        return {"messages": []}


def test_invoke_turn_logs_and_clears_timer(caplog):
    caplog.set_level(logging.INFO)
    graph = _FakeGraph()
    clear_turn_timer()

    result, timing = invoke_turn(graph, "hello", thread_id="t1", recursion_limit=12)

    assert result == {"messages": []}
    assert timing["total_seconds"] >= 0
    assert timing["steps"]["llm"] >= 0.01
    assert get_turn_timer() is None
    assert graph.last_config is not None
    assert any(cb.__class__.__name__ == "ChatTimingHandler" for cb in graph.last_config["callbacks"])
    assert any("Chat turn finished" in r.message for r in caplog.records)
    assert any("Runtime:" in r.message for r in caplog.records)


def test_turn_timing_payload_rounds():
    timer = StepTimer(logging.getLogger("test"))
    record_step(timer, "llm", 0.123456)
    payload = turn_timing_payload(timer)
    assert payload["steps"]["llm"] == 0.1235
    assert payload["total_seconds"] >= 0
