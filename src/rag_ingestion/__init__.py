"""Ingestion RAG locale : LiteParse, LangExtract, embeddings, Qdrant."""

__all__ = ["ingest_path"]


def __getattr__(name: str):
    """Import paresseux de `ingest_path` pour ne pas charger Qdrant/LiteParse au `import`."""
    if name == "ingest_path":
        from rag_ingestion.pipeline import ingest_path

        return ingest_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
