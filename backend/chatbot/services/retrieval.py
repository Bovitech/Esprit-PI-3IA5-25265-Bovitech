"""
RAG retrieval layer.
- Qdrant client is initialised once (thread-safe flag)
- search() is the only public API
"""
import os
import logging
import threading

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bovin_chunks"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
SCORE_THRESHOLD = 0.5
TOP_K           = 3

# Module-level singletons — created once, reused forever
_qdrant: QdrantClient | None        = None
_embedder: SentenceTransformer | None = None
_init_lock   = threading.Lock()
_initialized = False


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=str(settings.BASE_DIR / "qdrant_data"))
    return _qdrant


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text).tolist()


def _split(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_chunks() -> list[str]:
    knowledge_dir = settings.BASE_DIR / "backend" / "knowledge_base"
    if not knowledge_dir.exists():
        logger.warning("knowledge_base directory not found at %s", knowledge_dir)
        return []
    chunks = []
    for path in knowledge_dir.glob("*.txt"):
        chunks.extend(_split(path.read_text(encoding="utf-8")))
    return chunks


def init() -> None:
    """Idempotent — safe to call multiple times."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:          # double-checked locking
            return
        client     = _get_qdrant()
        collection_names = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            chunks = _load_chunks()
            points = [
                PointStruct(id=i, vector=_embed(c), payload={"text": c})
                for i, c in enumerate(chunks)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("Qdrant indexed with %d chunks", len(points))
        _initialized = True


def search(query: str, top_k: int = TOP_K) -> list[str]:
    """Return relevant text chunks for the query."""
    init()
    results = _get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=_embed(query),
        limit=top_k,
    )
    if not hasattr(results, "points"):
        return []
    return [
        p.payload.get("text", "")
        for p in results.points
        if p.score > SCORE_THRESHOLD and p.payload
    ]