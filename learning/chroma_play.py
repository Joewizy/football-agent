"""A sandbox for feeling how a Chroma collection behaves. Edit DOCS, change the
query, re-run. In-memory only — nothing is saved, so every run starts clean.

Run:  python3 -m learning.chroma_play "what is offside?"
"""
import sys
import chromadb

DOCS = [
    "A player is offside if nearer the goal than the second-last defender.",
    "A hat-trick is scoring three goals in one match.",
    "VAR reviews goals, penalties, red cards and mistaken identity.",
    "Offside is not an offence unless the player is involved in active play.",
]

# in-memory scratch store
client = chromadb.Client()                              
# get_or_create avoids the "already exists" error on re-runs within one process.
# Default distance is squared L2 (lower = closer); swap to cosine with the metadata below.
col = client.get_or_create_collection(
    "play",
    # metadata={"hnsw:space": "cosine"},
)

# Chroma embeds the documents for us (its built-in model), then indexes them.
col.add(ids=[str(i) for i in range(len(DOCS))], documents=DOCS)


def search(query: str, k: int = 2) -> None:
    res = col.query(query_texts=[query], n_results=k)
    print(f"\nQuery: {query}\n")
    # [0] unwraps the batch dimension — we only asked one query.
    for doc_id, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
        print(f"  id={doc_id}  dist={dist:.3f}  {doc}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "what is offside?"
    search(query)
