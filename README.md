# Football Agent

A local-first conversational agent for football data, built to learn agent
engineering end to end: tool-calling, retrieval-augmented generation (RAG),
persistent memory, and HTTP serving — running entirely on a laptop with no cloud
LLM calls.

The agent answers two different *kinds* of question and decides for itself which
to use:

- **Live data** — "Who's the top scorer in the Scottish Premiership?" → calls the
  SportMonks API.
- **Background knowledge** — "How does the Scottish Premiership split work?" →
  retrieves from a local vector store of football documents.

```
you> how does the scottish premiership split work?
  …tool call: ['search_knowledge']
agent> 12 clubs play each other three times (33 rounds), then split into a top
       six and bottom six based on standings… [competition_formats.md]

you> who is the top scorer?
  …tool call: ['top_scorers']
agent> 1. Kasper Waarst Høgh (Celtic) — 3 goals
       2. Killian Phillips (St. Mirren) — 3 goals …
```

## Architecture

```
                    ┌─────────────────────────────────────────┐
   HTTP / CLI  ───▶ │              LangGraph agent             │
                    │                                          │
                    │   ┌────────┐   tool?   ┌────────────┐    │
                    │   │ think  │──────────▶│   tools    │    │
                    │   │ (LLM)  │◀──────────│ (ToolNode) │    │
                    │   └────────┘  result   └────────────┘    │
                    └────────┬──────────────────────┬──────────┘
                             │                       │
              conversation   │                       │  two tool families
              memory         ▼                       ▼
                    ┌──────────────┐      ┌────────────────────────┐
                    │ SQLite       │      │ Live data → SportMonks  │
                    │ checkpointer │      │ Knowledge → Chroma RAG  │
                    │ (per thread) │      └────────────────────────┘
                    └──────────────┘
```

A turn flows: **think** (the LLM, with tools bound) decides whether it needs a
tool. If yes, control routes to the **tools** node, the result is appended to the
conversation, and control returns to **think** to either call another tool or
write the final answer. The loop is a LangGraph state machine; routing is the
built-in `tools_condition`.

## How it works

**The model runs locally.** Chat uses `qwen3:8b` and embeddings use
`nomic-embed-text`, both served by [Ollama](https://ollama.com). No data leaves
the machine and there are no API costs for inference.

**Two tool families, one router.** The `think` node has both the SportMonks tools
and the RAG `search_knowledge` tool bound to it. The model picks per question —
that routing *is* the agent. Live-data questions hit the API; rules/format
questions hit the knowledge base.

**RAG pipeline** (`rag/`):
1. `ingest_chroma.py` — chunk the docs in `rag/docs/`, embed each chunk with
   `nomic-embed-text`, store the vectors in a persistent Chroma collection
   (cosine space), each chunk tagged with its source file.
2. `ask_chroma.py` — embed the query *with the same model*, retrieve the nearest
   chunks, return them. `search_knowledge` wraps this as an agent tool.

**Memory that survives restarts.** A LangGraph `SqliteSaver` checkpoints
conversation state to `football_memory.sqlite`, keyed by `thread_id`. Restart the
process and the conversation is still there. Each HTTP request carries its own
`thread_id`, so concurrent users don't share one memory.

**Served over HTTP** (`server.py`, FastAPI):
- `GET /health` — liveness + which model is loaded.
- `POST /chat` — one question, one JSON reply (with `tools_used`).
- `POST /chat/stream` — the same turn streamed token-by-token as Server-Sent
  Events.

## Design decisions worth calling out

These are the choices that took the most thought — and the ones that show what I
learned:

- **Anti-hallucination lives in the tool's return value, not the prompt.** A small
  local model will ignore a system-prompt rule like "don't invent team ids." So
  when a lookup fails, the tool returns an explicit `{"message": "no team named
  X…"}` rather than an empty list — the *data* tells the model there's nothing to
  report, which it obeys far more reliably than an instruction.

- **The tool retrieves; the agent reasons.** `search_knowledge` returns raw
  passages, not a written answer. The agent already has an LLM (the `think` node)
  — nesting a second LLM inside the tool would double the memory footprint and
  take wording control away from the agent. Tools return data; the graph's model
  turns data into prose.

- **Same-embedder rule.** The query is embedded with the exact model used at
  ingest. Mixing embedding models silently wrecks retrieval — the vectors live in
  different spaces, so "nearest" becomes meaningless.

- **Grounding.** The knowledge answers are instructed to use retrieved context
  only and to say "I don't know" otherwise, so the agent doesn't pad gaps with
  plausible-sounding invention.

## Project structure

```
agents/football_agent.py   The agent: graph, nodes, routing, memory, CLI
tools/football.py          SportMonks tools (standings, scorers, squads, …)
tools/knowledge.py         search_knowledge — the RAG tool
services/football_api.py   Thin SportMonks HTTP client
rag/ingest_chroma.py       Build the vector store from rag/docs/
rag/ask_chroma.py          Embed query → retrieve nearest chunks
server.py                  FastAPI service (/health, /chat, /chat/stream)
config.py                  Single source of runtime config (env)
```

## Running it locally

**Prerequisites:** [Ollama](https://ollama.com) installed and running.

```bash
# 1. Models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Config — create a .env file
echo "FOOTBALL_API_KEY=your_sportmonks_key" > .env

# 4. Build the knowledge base (one-time, re-run when docs change)
python -m rag.ingest_chroma

# 5a. Talk to it in the terminal
python -m agents.football_agent

# 5b. …or serve it over HTTP
uvicorn server:app --reload
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"who is the top scorer?","thread_id":"demo"}'
```

The SportMonks free plan covers the Scottish Premiership and Danish Superliga,
which is what the live-data tools are scoped to.

## Stack

Python · [LangGraph](https://langchain-ai.github.io/langgraph/) ·
[Ollama](https://ollama.com) (qwen3:8b, nomic-embed-text) ·
[Chroma](https://www.trychroma.com/) · [FastAPI](https://fastapi.tiangolo.com/) ·
[SportMonks API](https://www.sportmonks.com/)
