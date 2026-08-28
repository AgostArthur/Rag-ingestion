"""Embeddings denses (FastEmbed) : texte de chunk → vecteur, pas le JSON.

La dimension Qdrant vient du modèle FastEmbed réellement chargé.
`EMBED_DIM` / `DEFAULT_EMBED_DIM` ne servent que si FastEmbed n'annonce pas `dim`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

from rag_ingestion.config import (
    DEFAULT_EMBED_FALLBACK_MODEL,
    Settings,
    load_settings,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fastembed_declared_dims() -> dict[str, int]:
    """Dimensions publiées par FastEmbed (`model` → `dim`)."""
    from fastembed import TextEmbedding

    out: dict[str, int] = {}
    for meta in TextEmbedding.list_supported_models():
        name = meta.get("model")
        dim = meta.get("dim")
        if name and dim:
            out[str(name)] = int(dim)
    return out


def _supported_fastembed() -> set[str]:
    """Noms de modèles denses connus de FastEmbed."""
    return set(_fastembed_declared_dims())


@lru_cache(maxsize=8)
def _resolve_fastembed_name(requested: str) -> str:
    """Nom FastEmbed utilisable : `requested` ou `DEFAULT_EMBED_FALLBACK_MODEL`."""
    supported = _supported_fastembed()
    if requested in supported:
        return requested
    logger.warning(
        "FastEmbed model « %s » is not available in this version; "
        "falling back to %s. Change DEFAULT_EMBED_FALLBACK_MODEL in config.py "
        "or EMBED_MODEL in .env.",
        requested,
        DEFAULT_EMBED_FALLBACK_MODEL,
    )
    return DEFAULT_EMBED_FALLBACK_MODEL


@lru_cache(maxsize=2)
def _model(name: str):
    """Instance FastEmbed mise en cache (téléchargement au premier appel)."""
    from fastembed import TextEmbedding

    logger.info("Loading embedding model « %s » (first call may download)…", name)
    return TextEmbedding(model_name=name)


def _resolved_model(settings: Settings):
    """Modèle FastEmbed correspondant à `settings.embed_model` (avec repli)."""
    return _model(_resolve_fastembed_name(settings.embed_model))


def embedding_dimension(settings: Settings | None = None) -> int:
    """Dimension à déclarer dans Qdrant : dim FastEmbed du modèle résolu.

    Si FastEmbed n'annonce pas la taille, on utilise `EMBED_DIM` (.env) puis
    `DEFAULT_EMBED_DIM` (config.py).

    Args:
        settings: Config ; `.env` si omis.

    Returns:
        Taille des vecteurs du modèle effectivement utilisé.
    """
    s = settings or load_settings()
    resolved = _resolve_fastembed_name(s.embed_model)
    probed = _fastembed_declared_dims().get(resolved)
    if probed is not None:
        if os.getenv("EMBED_DIM") and s.embed_dim != probed:
            logger.warning(
                "EMBED_DIM=%s ignored: model « %s » is %s-d. "
                "Remove EMBED_DIM from .env (dimension is taken from FastEmbed).",
                s.embed_dim,
                resolved,
                probed,
            )
        return probed
    return s.embed_dim


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Calcule un vecteur dense par texte (chunks à indexer).

    Args:
        texts: Markdown des chunks, dans le même ordre que les payloads.
        settings: Config modèle ; `.env` si omis.

    Returns:
        Liste de vecteurs, même longueur que `texts`.

    Raises:
        RuntimeError: Le modèle n'a pas renvoyé un vecteur par texte.
    """
    if not texts:
        return []
    s = settings or load_settings()
    model = _resolved_model(s)
    vectors: list[list[float]] = []
    for vec in model.embed(texts):
        arr = np.asarray(vec, dtype=np.float32)
        vectors.append(arr.tolist())
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding model did not return one vector per text.")
    expected = embedding_dimension(s)
    actual = len(vectors[0])
    if actual != expected:
        logger.warning(
            "Model produced %s-d vectors (config expected %s). Using %s for Qdrant.",
            actual,
            expected,
            actual,
        )
    return vectors


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Vecteur d'une question (`query_embed` si le modèle le propose).

    Args:
        text: Phrase de recherche.
        settings: Config ; `.env` si omis.

    Returns:
        Un vecteur de la dimension du modèle chargé.
    """
    s = settings or load_settings()
    model = _resolved_model(s)
    if hasattr(model, "query_embed"):
        vec = next(iter(model.query_embed([text])))
    else:
        vec = next(iter(model.embed([text])))
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tolist()
