"""HTTP service wrapping the football agent. Run: uvicorn server:app --reload"""
import json
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import LOG_LEVEL
from agents.football_agent import MODEL, run_events

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("football.api")

app = FastAPI(title="Football Agent API")


class ChatRequest(BaseModel):
    message: str
    # Which stored conversation to use. Per-request, so concurrent users don't
    # share one memory the way a single-process CLI does.
    thread_id: str = "default"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Collect the whole turn, then return it in one JSON body."""
    log.info("chat thread=%s len=%d", req.thread_id, len(req.message))

    reply, tools_used = [], []
    for event in run_events(req.message, req.thread_id):
        if event["type"] == "token":
            reply.append(event["content"])
        elif event["type"] == "tool_call":
            tools_used.extend(event["names"])

    return {"reply": "".join(reply), "tools_used": tools_used, "thread_id": req.thread_id}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same turn, but forwarded token-by-token as Server-Sent Events."""
    log.info("stream thread=%s len=%d", req.thread_id, len(req.message))

    def sse():
        for event in run_events(req.message, req.thread_id):
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    # A sync generator here is fine: Starlette iterates it in a threadpool, so the
    # blocking Ollama call never stalls the event loop.
    return StreamingResponse(sse(), media_type="text/event-stream")
