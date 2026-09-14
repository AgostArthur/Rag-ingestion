# RAG ingestion

Pipeline : PDF → LiteParse (OCR) → Markdown → chunks + LangExtract → **catalog SQLite** → FastEmbed → Qdrant.

Deux packages :

- `rag_ingestion` — ingest, recherche hybride, catalog (`sites` / `documents` / `events`)
- `chatbot` — agent LangGraph ; le RAG est l’outil `search_knowledge`. `POST /chat` renvoie `focus` + citations issus des hits, pas du texte du modèle.

## Déploiement via Docker

Le produit est **Compose** : API chat, worker d’ingest, Qdrant. Le LLM n’est **pas** dans l’image (clé API cloud, ou llama-server / Ollama sur la machine hôte).

```bash
cp .env.example .env
# Cloud chat, par exemple :
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini

docker compose up -d --build
```

- Déposer les PDF dans `**incoming/**`. Le service `worker` les ingère, puis les déplace vers `data/archive/{sha}/` (doublon SHA-256 : archivé sans ré-ingest). Échecs → `data/failed/`.
- API : `http://localhost:8000` (`POST /chat`, `GET /health`, catalog).
- Qdrant : `http://localhost:6333`.

Image client (embeddings déjà dans l’image, premier `build` plus long) :

```bash
PREFETCH_EMBED=1 docker compose build
docker compose up -d
```

LLM hors image :

- llama-server **sur l’hôte** : `LLAMA_SERVER_BASE_URL=http://host.docker.internal:8080/v1`
- Ollama **sur l’hôte** : `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- Ollama **dans Compose** : `docker compose --profile extract up -d` et `OLLAMA_BASE_URL=http://ollama:11434`
- Sans Ollama : `INGEST_SKIP_EXTRACT=1` (chunks + embeddings seulement)

`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` priment sur `LLAMA_SERVER_*` s’ils sont définis.

Le venv Python reste utile **en développement** (`pip install -e ".[dev]"`), pas pour un poste client.

## Ingest : ce qui déclenche quoi

Le catalog SQLite **n’est pas** la source des fichiers. C’est le **registre** écrit **après** un ingest réussi (`document_id` = SHA-256 du PDF).


| Déclencheur        | Commande / service                                               |
| ------------------ | ---------------------------------------------------------------- |
| Drop folder (prod) | Fichier dans `incoming/` → `rag-ingest watch` (Compose `worker`) |
| Manuel             | `rag-ingest ingest chemin.pdf` ou `rag-ingest ingest dossier/`   |
| Un scan            | `rag-ingest watch --once` (cron)                                 |


Rien ne se passe si on copie un PDF ailleurs que dans `incoming/` sans lancer `ingest` / `watch`. Un SHA déjà au catalog est archivé sans ré-ingest ; `--force` pour réessayer.

```bash
rag-ingest watch --once
rag-ingest query "contamination 4405" --filter project_id=4405
```

Artefacts : `data/parsed/`, `data/extractions/`, `data/documents/*.json`, `data/catalog.sqlite`, `data/archive/`.

Le `site_id` privilégie le lot (`lot:2363352`). Deux rapports du même site (4405 et 2259) partagent un site, restent deux documents. Pas de géocodage (`geocode_status=skipped`).

Recherche **hybride** : cosine + n° de projet dans le payload **ou** le texte du chunk.

## Développement (venv)

```bash
cp .env.example .env
docker compose up -d qdrant
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[chat]"
rag-ingest ingest chemin/vers/doc.pdf
rag-chat serve
```

## Embeddings / LangExtract

Changer `EMBED_MODEL` dans `.env` (dimension lue chez FastEmbed). Ré-ingérer après un changement de modèle.

Si le nom n’est pas dans FastEmbed, repli `intfloat/multilingual-e5-large`.

Schéma LangExtract : `config/langextract/prompt.txt` + `few_shots.json` (`LANGEXTRACT_PROMPT_FILE`, `LANGEXTRACT_FEW_SHOTS_FILE`). Chaque `extraction_text` de few-shot doit apparaître tel quel dans `text`. Pour les ÉES : `location` (site étudié), `date.role` (`report`  `fieldwork`), `firm` / `client` / `project_id`.

Détail du pipeline : `[docs/ingestion.md](docs/ingestion.md)`.

## Chat

Le LLM n’a pas les chunks dans le prompt. Compatible **OpenAI HTTP** (`OPENAI_`* ou `LLAMA_SERVER_*`). llama-server local : `llama-server --jinja -fa -m modele.gguf --port 8080`.

```bash
rag-chat          # REPL ; JSON focus sous la réponse
rag-chat serve    # POST /chat → { reply, thread_id, focus, documents, citations }
```

`GET /documents/{id}`, `GET /sites/{id}`, `GET /sites/{id}/timeline`. Checkpoints : `data/chat_checkpoints.sqlite`.