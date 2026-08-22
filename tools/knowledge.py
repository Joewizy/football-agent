from langchain_core.tools import tool

from rag.ask_chroma import retrieve

# Cosine DISTANCE (not similarity): lower means closer. Above this a chunk is
# treated as off-topic and dropped — the same MIN_SCORE idea from rag/ask.py,
# expressed in Chroma's distance space (distance ≈ 1 - similarity).
MAX_DIST = 0.5


@tool
def search_knowledge(query: str) -> list:
    """Search the football knowledge base for background facts and rules:
    competition formats (e.g. how the Scottish Premiership split works), the laws
    of the game, and a short bio. Use this for rules/format/background questions,
    NOT for live data like scores, standings, squads or player details — those come
    from the data tools. Returns matching passages, each tagged with its source file.
    """
    hits = [h for h in retrieve(query, k=3) if h["dist"] <= MAX_DIST]
    if not hits:
        return [{"message": "No relevant information found in the knowledge base."}]
    return [{"source": h["source"], "text": h["text"]} for h in hits]
