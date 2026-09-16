# RAG ingestion

Pipeline : PDF → LiteParse (OCR) → Markdown → chunks + LangExtract → **catalog SQLite** → FastEmbed → Qdrant. L’interface (`ui/`) affiche carte, chronologie et chat.

- `rag_ingestion` — ingest, recherche hybride, catalog (`sites` / `documents` / `events`)
- `chatbot` — agent LangGraph ; outil `search_knowledge`. `POST /chat` renvoie `focus` + citations issus des hits
- `ui/` — carte (un pin = un site), chronologie du site, chat (`document_id` si on clique un rapport)

## Déploiement via Docker

```bash
cp .env.example .env
# LLM_PROVIDER=gemini          # ou openai | ollama | local
# GEMINI_API_KEY=...           # ou OPENAI_API_KEY + OPENAI_MODEL
# VITE_GOOGLE_MAPS_API_KEY=...   # rebuild UI after changing this

docker compose up -d --build
```

- Interface : `http://localhost:3000` (nginx reverse-proxy vers l’API)
- PDF → `incoming/` → worker → `data/archive/{sha}/` (doublon SHA-256 : archivé sans ré-ingest). Échecs → `data/failed/`
- API brute : `http://localhost:8000`
- Qdrant : `http://localhost:6333`

Image embeddings préchargée : `PREFETCH_EMBED=1 docker compose build`

LLM hors image (chat + LangExtract). Un switch dans `.env` :

| `LLM_PROVIDER` | Chat | LangExtract |
|---|---|---|
| `local` (défaut) | llama-server (`LLAMA_SERVER_*`) | Ollama (`OLLAMA_BASE_URL`) |
| `ollama` | Ollama `/v1` | Ollama |
| `openai` | API OpenAI (ou tout `/v1`) | LangExtract OpenAI |
| `gemini` | SDK natif (`langchain-google-genai`) | LangExtract Gemini |

Overrides : `CHAT_PROVIDER`, `LANGEXTRACT_PROVIDER`. Embeddings FastEmbed **inchangés** (locaux). Sans LLM local : `INGEST_SKIP_EXTRACT=1` ou une clé cloud. Ollama dans Compose : `docker compose --profile extract up -d` et `OLLAMA_BASE_URL=http://ollama:11434`.

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

## Embeddings / LangExtract

Changer `EMBED_MODEL` dans `.env` (dimension lue chez FastEmbed). Ré-ingérer après un changement de modèle.

Si le nom n’est pas dans FastEmbed, repli `intfloat/multilingual-e5-large`.

Schéma LangExtract : `config/langextract/profiles.json` (prompt de base + addendum + few-shots par profil). Le nom du PDF choisit le profil (motifs dans `profiles.json`) ; sinon `default` (warning). Chaque ingest logue `type=` et `fichier=`. L’identifiant est le même que `doc_type` Qdrant : `ees_phase_1` / `ees_phase_2`. Override : `rag-ingest ingest fichier.pdf --profile ees_phase_1`. Chaque `extraction_text` de few-shot doit apparaître tel quel dans `text`.

Détail du pipeline : `[docs/ingestion.md](docs/ingestion.md)`.

## Chat

Le LLM n’a pas les chunks dans le prompt. Provider : `LLM_PROVIDER` (`local` / `ollama` / `openai` / `gemini`). llama-server local : `llama-server --jinja -fa -m modele.gguf --port 8080`.

```bash
rag-chat          # REPL ; JSON focus sous la réponse
rag-chat serve    # POST /chat → { reply, thread_id, focus, documents, citations }
```
Vite proxy `/chat`, `/sites`, `/documents` vers `:8000`. Clé Maps : `ui/.env.local` avec `VITE_GOOGLE_MAPS_API_KEY=`.

Documentation : [`docs/README.md`](docs/README.md) (pipeline, catalog, API, UI).
