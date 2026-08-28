from pathlib import Path

from rag_ingestion.config import DEFAULT_EMBED_FALLBACK_MODEL, Settings
from rag_ingestion.embed import (
    _fastembed_declared_dims,
    _resolve_fastembed_name,
    embedding_dimension,
)


def _settings(**overrides: object) -> Settings:
    base = dict(
        qdrant_url="http://localhost:6333",
        qdrant_collection="chunks",
        embed_model="BAAI/bge-m3",
        embed_dim=512,
        ollama_base_url="http://localhost:11434",
        langextract_model="x",
        langextract_timeout_seconds=1.0,
        langextract_prompt_file=Path("config/langextract/prompt.txt"),
        langextract_few_shots_file=Path("config/langextract/few_shots.json"),
        ocr_language="fra",
        ocr_server_url=None,
        ocr_heavy_ratio=0.5,
        chunk_size_chars=2400,
        chunk_overlap_chars=300,
        data_dir=Path("data"),
        project_root=Path("."),
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_fallback_model_dim_is_declared_by_fastembed():
    dims = _fastembed_declared_dims()
    assert DEFAULT_EMBED_FALLBACK_MODEL in dims
    assert dims[DEFAULT_EMBED_FALLBACK_MODEL] == 1024


def test_embedding_dimension_uses_fastembed_not_env_embed_dim():
    """EMBED_DIM=512 ne doit plus faire planter un modèle 1024-d."""
    resolved = _resolve_fastembed_name("BAAI/bge-m3")
    dim = embedding_dimension(_settings(embed_model="BAAI/bge-m3", embed_dim=512))
    assert dim == _fastembed_declared_dims()[resolved]
    assert dim != 512
