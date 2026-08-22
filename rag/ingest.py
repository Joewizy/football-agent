"""Shared document loader + word-window chunker for the RAG pipeline.

`load_chunks` and `chunk_words` are imported by `ingest_chroma.py`, which is the
pipeline the shipped agent uses. The `__main__` block below is the original
from-scratch builder (docs -> chunk -> embed -> store.json), kept as a record of
the mechanism the Chroma store replaced. Run it with `python -m rag.ingest`.
"""
import glob
import json
import os

from langchain_ollama import OllamaEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_GLOB = os.path.join(HERE, "docs", "*.md")
STORE_PATH = os.path.join(HERE, "store.json")

EMBEDDER = OllamaEmbeddings(model="nomic-embed-text")


def chunk_words(text: str, size: int = 60, overlap: int = 12) -> list[str]:
    """Slide a fixed window over the words. Overlap repeats a few words between
    chunks so a sentence split across a boundary still appears whole somewhere."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def load_chunks() -> list[dict]:
    """Read every doc, split into chunks, tag each chunk with its source file."""
    chunks = []
    for path in sorted(glob.glob(DOCS_GLOB)):
        source = os.path.basename(path)
        with open(path) as f:
            for piece in chunk_words(f.read()):
                chunks.append({"text": piece, "source": source})
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()

    # One network call embeds the whole batch; each vector is stored beside its text.
    vectors = EMBEDDER.embed_documents([c["text"] for c in chunks])
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec

    with open(STORE_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"Embedded {len(chunks)} chunks from {DOCS_GLOB} -> {STORE_PATH}")
