"""Send each finished turn to an agent evaluation API, so its answers get scored and diagnosed.

Off unless EVAL_API_URL is set (e.g. http://localhost:8000). The agent never waits for it:
the run is posted from a background thread after the reply is complete, and any failure
is only logged.

What gets sent (one "run" per turn):
  input    the user's question
  output   the agent's final answer
  steps    every model call and tool call, with its arguments, result, error and timing
  context  passages returned by search_knowledge
  instructions  the system prompt and tool descriptions, so what the agent says about itself can be checked
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import requests

from config import EVAL_AGENT_NAME, EVAL_API_URL

log = logging.getLogger("football.eval")


def _parse(content: Any) -> Any:
    """Tool results arrive as text; turn JSON back into data where possible."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def _strip_thinking(text: str) -> str:
    """qwen3 wraps its reasoning in <think> tags; only the answer itself is evaluated."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class RunRecorder:
    """Watches one turn's LangGraph updates and builds the run record from them."""

    def __init__(self, question: str, thread_id: str, model: str, instructions: str | None = None):
        self.question = question
        self.instructions = instructions
        self.thread_id = thread_id
        self.model = model
        self.started = time.perf_counter()
        self.last = self.started
        self.steps: list[dict] = []
        self.context: list[str] = []
        self.calls: dict[str, dict] = {}  # tool_call_id -> {"name", "args"}
        self.answer = ""

    def update(self, data: dict) -> None:
        """Call with each "updates" chunk from agent.stream()."""
        now = time.perf_counter()
        elapsed = int((now - self.last) * 1000)
        self.last = now
        for node, update in data.items():
            messages = update.get("messages") or []
            if not isinstance(messages, list):
                messages = [messages]
            if node == "think":
                self._think(messages[-1], elapsed)
            elif node == "tools":
                for message in messages:
                    self._tool(message, elapsed)

    def _think(self, message: Any, elapsed: int) -> None:
        tool_calls = getattr(message, "tool_calls", None) or []
        # The model's input is the conversation so far (question, instructions, earlier tool results), already sent
        # as `input`, `instructions` and the tool steps, so it isn't repeated here. The output is what the model produced.
        if tool_calls:
            for call in tool_calls:
                self.calls[call["id"]] = {"name": call["name"], "args": call.get("args", {})}
            decided = [{"tool": call["name"], "args": call.get("args", {})} for call in tool_calls]
            self.steps.append({"type": "llm_call", "name": self.model, "output": decided, "latency_ms": elapsed})
        else:
            self.answer = _strip_thinking(getattr(message, "content", "") or "")
            self.steps.append({"type": "llm_call", "name": self.model, "output": self.answer, "latency_ms": elapsed})

    def _tool(self, message: Any, elapsed: int) -> None:
        call = self.calls.get(getattr(message, "tool_call_id", ""), {})
        name = getattr(message, "name", None) or call.get("name", "tool")
        failed = getattr(message, "status", "success") == "error"
        result = _parse(getattr(message, "content", ""))
        step = {
            "type": "retrieval" if name == "search_knowledge" else "tool_call",
            "name": name,
            "input": call.get("args"),
            "latency_ms": elapsed,
        }
        if failed:
            step["error"] = str(result)[:500]
        else:
            step["output"] = result
        self.steps.append(step)
        if name == "search_knowledge" and not failed and isinstance(result, list):
            self.context += [f"[{p['source']}] {p['text']}" for p in result if isinstance(p, dict) and p.get("text")]

    def finish(self) -> None:
        """Post the run in the background. Never raises, never blocks the reply."""
        if not EVAL_API_URL:
            return
        run = {
            "agent_name": EVAL_AGENT_NAME,
            "mode": "live",
            "conversation_id": self.thread_id,
            "input": self.question,
            "output": self.answer,
            "context": self.context,
            "instructions": self.instructions,
            "steps": self.steps,
            "model": self.model,
            "latency_ms": int((time.perf_counter() - self.started) * 1000),
            "metadata": {"framework": "langgraph"},
        }
        threading.Thread(target=_post, args=(run,), daemon=True).start()


def _post(run: dict) -> None:
    url = EVAL_API_URL.rstrip("/") + "/runs"
    try:
        response = requests.post(url, json=run, timeout=10)
        response.raise_for_status()
        log.info("sent run %s to %s", response.json()["run"]["id"], url)
    except Exception as error:  # evaluation must never break the agent
        log.warning("couldn't send run to %s: %s", url, error)
