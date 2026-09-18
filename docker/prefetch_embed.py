"""Télécharge le modèle FastEmbed pendant `docker compose build`.

Même résolution que `rag_ingestion.embed` (repli si le nom n'est pas supporté).
"""

from __future__ import annotations

import os
import sys

from rag_ingestion.config import DEFAULT_EMBED_MODEL
from rag_ingestion.embed import _model, _resolve_fastembed_name


def main() -> int:
    requested = (os.environ.get("EMBED_MODEL") or "").strip() or DEFAULT_EMBED_MODEL
    resolved = _resolve_fastembed_name(requested)
    print(f"Prefetch FastEmbed requested={requested} resolved={resolved}", flush=True)
    model = _model(resolved)
    vector = next(iter(model.embed(["ok"])))
    print(f"Prefetch FastEmbed ok dim={len(vector)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
