# learning/

Scratch files from building up to the main agent — kept deliberately as a record
of the path from "first LLM call" to "RAG-backed agent served over HTTP." None of
these are imported by the shipped agent (see the root `README.md`); they're
standalone scripts, each isolating one concept.

Run them from the **repo root** as modules, e.g. `python -m learning.tool_demo`.
All of them need [Ollama](https://ollama.com) running; a couple need extra setup
noted below.

## The progression

| # | File | Concept | Run |
|---|------|---------|-----|
| 1 | `messages_demo.py` | System / Human / AI message types — how a chat model is actually prompted | `python -m learning.messages_demo` |
| 2 | `basic_agent.py` | The smallest LangGraph: one node, one LLM call, START → END | `python -m learning.basic_agent` |
| 3 | `structured_output.py` | Forcing typed output with Pydantic + `with_structured_output` (extract a `Person`, a `CryptoToken`, an exchange `Rate`) | `python -m learning.structured_output` |
| 4 | `tool_demo.py` | Tool calling: the model asks for a `calculator`, we run it, feed the `ToolMessage` back | `python -m learning.tool_demo` |
| 5 | `manual_loop.py` | The agent loop **by hand** — no framework — to see what LangGraph automates | `python -m learning.manual_loop` |
| 6 | `langgraph_agent.py` | The same loop as a LangGraph state machine: nodes, edges, conditional routing | (imported by `app.py`) |
| 7 | `app.py` | Tiny runner that invokes the `langgraph_agent` graph on one question | `python -m learning.app` |

## RAG, built up from scratch

| # | File | Concept | Run |
|---|------|---------|-----|
| 8 | `toy_rag.py` | RAG with **no vector library** — embed a corpus, rank by cosine similarity computed by hand, stuff the top matches into the prompt | `python -m learning.toy_rag "when is a player offside?"` |
| 9 | `chroma_play.py` | A sandbox for how a Chroma collection behaves (in-memory, nothing saved) | `python -m learning.chroma_play "what is offside?"` |
| 10 | `ask.py` + `store.json` | The first *persistent* RAG: query side reads pre-computed vectors from `store.json`, applies a relevance threshold, answers from context only | `python -m learning.ask "how does the scottish premiership split work?"` |

`ask.py`/`store.json` were later superseded by the Chroma pipeline in `rag/`
(`ingest_chroma.py` + `ask_chroma.py`), which is what the shipped agent uses. The
from-scratch version is kept here to show the mechanism the library replaced.

## Setup notes

- **All files:** Ollama running, with `qwen3:8b` (chat) and `nomic-embed-text`
  (embeddings) pulled.
- **`manual_loop.py`:** hits the football tools, so it needs `FOOTBALL_API_KEY`
  in `.env` (same as the main app).
- **`toy_rag.py`, `chroma_play.py`, `ask.py`:** use the embedding model; the first
  run after Ollama starts may be slow while the model loads.
