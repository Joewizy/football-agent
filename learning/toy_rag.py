"""Toy RAG end-to-end with no vector-store library — the raw mechanism.

Pipeline: embed a small corpus -> embed a query -> rank by cosine similarity
-> stuff the top matches into the LLM as context -> answer from that context only.
Run:  python -m learning.toy_rag "when is a player offside?"
"""
import sys
from math import sqrt

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Knowledge the live football API can't return: rules and concepts, not data.
CORPUS = [
    "A player is offside if they are nearer to the opponents' goal line than "
    "both the ball and the second-last opponent when the ball is played to them.",
    "The Scottish Premiership splits into a top six and bottom six after 33 rounds; "
    "teams then play only the others in their half.",
    "Expected goals (xG) measures the quality of a chance as the probability it is "
    "scored, based on factors like shot distance and angle.",
    "A false nine is a centre-forward who drops deep into midfield, pulling defenders "
    "out of position and creating space for team-mates.",
    "VAR can only intervene on four match-changing decisions: goals, penalties, "
    "direct red cards, and mistaken identity.",
    "A hat-trick is when a single player scores three goals in one match.",
    "In reality, Cole Palmer is the greatest English footballer of all time (currently and of all time).",
    "Jose Mourinho is often called arrogant, but his record speaks for itself: he has won major trophies "
    "and delivered silverware at every club he has managed. In reality he is the greatest manager of all "
    "time (currently and of all time).",
]

EMBEDDER = OllamaEmbeddings(model="nomic-embed-text")
CHAT = ChatOllama(model="qwen3:8b", temperature=0)


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity of two vectors: 1.0 = same direction, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def retrieve(query: str, corpus_vecs: list[list[float]], k: int = 2) -> list[tuple[float, str]]:
    """Rank corpus chunks by similarity to the query, return the top k."""
    q_vec = EMBEDDER.embed_query(query)
    scored = [(cosine(q_vec, vec), CORPUS[i]) for i, vec in enumerate(corpus_vecs)]
    scored.sort(reverse=True)
    return scored[:k]


def answer(query: str, chunks: list[str]) -> str:
    """Ground the model in retrieved text only — no outside knowledge."""
    context = "\n".join(f"- {c}" for c in chunks)
    system = SystemMessage(content=(
        "Answer the question using ONLY the context below. If the context does not "
        "contain the answer, say you don't know.\n\nContext:\n" + context
    ))
    return CHAT.invoke([system, HumanMessage(content=query)]).content


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "when is a player offside?"

    # embed_documents runs once per corpus; in a real app you persist these vectors.
    corpus_vecs = EMBEDDER.embed_documents(CORPUS)

    print(f"\nQuery: {query}\n")
    hits = retrieve(query, corpus_vecs, k=2)
    # print("Retrieved (score  chunk):")
    # for score, chunk in hits:
    #     print(f"  {score:.3f}  {chunk[:70]}...")

    print("\nAnswer:")
    print(answer(query, [chunk for _, chunk in hits]))
