"""Production ingest: chunk docs -> embed with nomic -> store in a persistent
Chroma collection, each chunk tagged with its source. Replaces store.json.

Run (whenever docs change):  python3 -m rag.ingest_chroma
"""
import os

import chromadb
from langchain_ollama import OllamaEmbeddings

from rag.ingest import load_chunks, HERE  # reuse the exact same chunker

CHROMA_PATH = os.path.join(HERE, "chroma_db")
COLLECTION = "football_knowledge"
EMBEDDER = OllamaEmbeddings(model="nomic-embed-text")


if __name__ == "__main__":
    chunks = load_chunks()
    vectors = EMBEDDER.embed_documents([c["text"] for c in chunks])

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Rebuild from scratch so the collection always matches the docs — the same
    # "retake the photo" idea as overwriting store.json.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # cosine space so "closer" means what it meant in your hand-rolled cosine.
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    col.add(
        ids=[f"{c['source']}-{i}" for i, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
        embeddings=vectors,  # our own nomic vectors, not Chroma's built-in model
    )

    print(f"Stored {col.count()} chunks in '{COLLECTION}' at {CHROMA_PATH}")
