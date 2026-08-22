"""Query the persistent Chroma collection built by ingest_chroma, then answer.

Run:  python3 -m rag.ask_chroma "how does the scottish premiership split work?"
"""
import sys

import chromadb
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from rag.ingest_chroma import CHROMA_PATH, COLLECTION, EMBEDDER

CHAT = ChatOllama(model="llama3.2:latest", temperature=0)


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Nearest chunks from the store, best first. Embed with the SAME model used
    at ingest, then hand Chroma the vector — never query_texts (wrong model)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)

    q_vec = EMBEDDER.embed_query(query)
    res = col.query(query_embeddings=[q_vec], n_results=k)

    return [
        {"text": doc, "source": meta["source"], "dist": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def answer(query: str, hits: list[dict]) -> str:
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    system = SystemMessage(content=(
        "Answer using ONLY the context below. If it isn't there, say you don't "
        "know.\n\nContext:\n" + context
    ))
    return CHAT.invoke([system, HumanMessage(content=query)]).content


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "how does the scottish premiership split work?"

    hits = retrieve(query)
    print(f"\nQuery: {query}\n\nRetrieved:")
    for h in hits:
        print(f"  dist={h['dist']:.3f}  [{h['source']}]  {h['text'][:60]}...")

    print("\nAnswer:")
    print(answer(query, hits))
    # Citations come from code, not the model: we know exactly which sources fed the answer.
    print("\nSources:", sorted({h["source"] for h in hits}))
