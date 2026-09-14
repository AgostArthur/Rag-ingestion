# RAG ingestion

Pipeline : PDF → LiteParse (OCR) → Markdown → chunks + LangExtract → **catalog SQLite** → FastEmbed → Qdrant. L’interface (`ui/`) affiche carte, chronologie et chat.

- `rag_ingestion` — ingest, recherche hybride, catalog (`sites` / `documents` / `events`)
- `chatbot` — agent LangGraph ; outil `search_knowledge`. `POST /chat` renvoie `focus` + citations issus des hits
- `ui/` — carte (un pin = un site), chronologie du site, chat (`document_id` si on clique un rapport)

## Déploiement via Docker

```bash
cp .env.example .env
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
# VITE_GOOGLE_MAPS_API_KEY=...   # rebuild UI after changing this

docker compose up -d --build
```

- Interface : `http://localhost:3000` (nginx reverse-proxy vers l’API)
- PDF → `incoming/` → worker → `data/archive/{sha}/` (doublon SHA-256 : archivé sans ré-ingest). Échecs → `data/failed/`
- API brute : `http://localhost:8000`
- Qdrant : `http://localhost:6333`

Image embeddings préchargée : `PREFETCH_EMBED=1 docker compose build`

LLM hors image : llama-server / Ollama sur l’hôte (`host.docker.internal`), ou `docker compose --profile extract up -d` avec `OLLAMA_BASE_URL=http://ollama:11434`. Sans Ollama : `INGEST_SKIP_EXTRACT=1`.

## Ingest

Le catalog SQLite est le **registre** écrit après ingest (`document_id` = SHA-256). `site_id` = lot (`lot:2363352`) sinon adresse. 4405 et 2259 partagent un site. Nominatim remplit `lat`/`lon` (`GEOCODE_ENABLED=1`).

| Déclencheur | Commande |
|---|---|
| Drop folder | `incoming/` + Compose `worker` / `rag-ingest watch` |
| Manuel | `rag-ingest ingest fichier.pdf` ou un dossier |
| Un scan | `rag-ingest watch --once` |

## Développement

```bash
cp .env.example .env
docker compose up -d qdrant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pip install -e ".[chat]"
rag-ingest ingest chemin/vers/doc.pdf
rag-chat serve
# autre terminal
cd ui && npm install && npm run dev
```

Vite proxy `/chat`, `/sites`, `/documents` vers `:8000`. Clé Maps : `ui/.env.local` avec `VITE_GOOGLE_MAPS_API_KEY=`.

Documentation : [`docs/README.md`](docs/README.md) (pipeline, catalog, API, UI).
