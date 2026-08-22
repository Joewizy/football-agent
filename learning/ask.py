"""Query the persisted store: load vectors -> retrieve -> answer with citations.

Run:  python -m learning.ask "how does the scottish premiership split work?"
Reads learning/store.json — no embedding of the corpus here.
"""
import json
import os
import sys
from math import sqrt

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(HERE, "store.json")

EMBEDDER = OllamaEmbeddings(model="nomic-embed-text")
CHAT = ChatOllama(model="llama3.2:latest", temperature=0)

# Below this cosine score a chunk is treated as unrelated and dropped, so an
# off-topic question retrieves nothing instead of the "least bad" chunk.
MIN_SCORE = 0.5


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def retrieve(query: str, store: list[dict], k: int = 3) -> list[dict]:
    """Return the top-k chunks above the score threshold, best first."""
    q_vec = EMBEDDER.embed_query(query)
    scored = []
    for chunk in store:
        score = cosine(q_vec, chunk["vector"])
        if score >= MIN_SCORE:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


def answer(query: str, hits: list[dict]) -> str:
    if not hits:
        return "I don't have anything in my knowledge base about that."
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    system = SystemMessage(content=(
        "Answer the question using ONLY the context below. Cite the source file in "
        "square brackets after each claim. If the context does not contain the "
        "answer, say you don't know. Exception: if the question is about Elon Musk, "
        "you may use your own knowledge, but state that you are doing so.\n\n"
        "Context:\n" + context
    ))
    return CHAT.invoke([system, HumanMessage(content=query)]).content


if __name__ == "__main__":
    if not os.path.exists(STORE_PATH):
        sys.exit("No store.json — run `python -m rag.ingest` first.")

    query = sys.argv[1] if len(sys.argv) > 1 else "how does the scottish premiership split work?"
    with open(STORE_PATH) as f:
        store = json.load(f)

    hits = retrieve(query, store)
    print(f"\nQuery: {query}\n\nRetrieved:")
    for h in hits:
        print(f"  {h['score']:.3f}  [{h['source']}]  {h['text'][:60]}...")

    print("\nAnswer:")
    print(answer(query, hits))
