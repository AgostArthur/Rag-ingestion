# RAG ingestion

Pipeline : PDF → LiteParse (OCR) → Markdown → chunks + LangExtract → **catalog SQLite** → FastEmbed → Qdrant. L’interface (`ui/`) affiche carte, chronologie et chat.

- `rag_ingestion` — ingest, recherche hybride, catalog (`sites` / `documents` / `events`)
- `chatbot` — agent LangGraph ; outil `search_knowledge`. `POST /chat` renvoie `focus` + citations issus des hits
- `ui/` — carte à gauche (65 %), chat à droite (35 %). Aucun site n’est sélectionné au chargement : la carte ouvre le sud du Québec autour de Montréal. Un clic sur une épingle charge le site.

## Déploiement via Docker

```bash
cp .env.example .env
# LLM_PROVIDER=gemini          # ou openai | ollama | local
# GEMINI_API_KEY=...           # ou OPENAI_API_KEY + OPENAI_MODEL

docker compose up -d --build
```

Le premier build télécharge le modèle FastEmbed (`EMBED_MODEL`). Pour sauter ça : `PREFETCH_EMBED=0 docker compose build`.

- Interface : `http://localhost:3000` (nginx reverse-proxy vers l’API)
- PDF → `incoming/` → `ingest-worker` → `data/archive/{sha}/` (même SHA déjà au catalog **et** encore en archive : pas de ré-ingest). Échecs → `data/failed/`
- API brute : `http://localhost:8000`
- Qdrant : `http://localhost:6333`

Image embeddings préchargée : c’est le défaut (`PREFETCH_EMBED=1`).

LLM hors image (chat + LangExtract). Un switch dans `.env` :

| `LLM_PROVIDER` | Chat | LangExtract |
|---|---|---|
| `local` (défaut) | llama-server (`LLAMA_SERVER_*`) | Ollama (`OLLAMA_BASE_URL`) |
| `ollama` | Ollama `/v1` | Ollama |
| `openai` | API OpenAI (ou tout `/v1`) | LangExtract OpenAI |
| `gemini` | SDK natif (`langchain-google-genai`) | LangExtract Gemini |

Overrides : `CHAT_PROVIDER`, `LANGEXTRACT_PROVIDER`. Embeddings FastEmbed **inchangés** (locaux). Sans LLM local : `INGEST_SKIP_EXTRACT=1` ou une clé cloud. Ollama dans Compose : `docker compose --profile extract up -d` et `OLLAMA_BASE_URL=http://extract-llm:11434`.

## Ingest

Le catalog SQLite est le **registre** écrit après ingest (`document_id` = SHA-256). `site_id` = lot (`lot:2363352`) sinon adresse. 4405 et 2259 partagent un site. Le cadastre QC fournit le polygone du lot ; Nominatim n’est qu’un repli sans lot (`GEOCODE_ENABLED=1`).

| Déclencheur | Commande |
|---|---|
| Drop folder | `incoming/` + Compose `ingest-worker` / `rag-ingest watch` |
| Manuel | `rag-ingest ingest fichier.pdf` ou un dossier |
| Un scan | `rag-ingest watch --once` |

## Développement

```bash
cp .env.example .env
docker compose up -d vector-store
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

Schéma LangExtract : `config/langextract/profiles.json` (prompt de base + addendum + few-shots par profil). Le nom du PDF choisit le profil (motifs dans `profiles.json`) ; sinon les trois premiers titres Markdown ; sinon `default` (warning). Chaque ingest logue `type=` et `fichier=`. L’identifiant est le même que `doc_type` Qdrant : `ees_phase_1` / `ees_phase_2`. Override : `rag-ingest ingest fichier.pdf --profile ees_phase_1`. Chaque `extraction_text` de few-shot doit apparaître tel quel dans `text`.

Détail du pipeline : [`docs/INGESTION.md`](docs/INGESTION.md).

## Chat

Le LLM n’a pas les chunks dans le prompt. Provider : `LLM_PROVIDER` (`local` / `ollama` / `openai` / `gemini`). llama-server local : `llama-server --jinja -fa -m modele.gguf --port 8080`.

```bash
rag-chat          # REPL ; JSON focus sous la réponse
rag-chat serve    # POST /chat → { reply, thread_id, focus, documents, citations }
```
Vite proxy `/chat`, `/sites`, `/documents` vers `:8000`. Carte : Leaflet / OpenStreetMap (pas de clé). Au chargement, aucun site n’est actif. Un clic d’épingle remplit la barre de contexte (`@fichier`) sans écrire dans le champ. Une date de la chronologie (panneau repliable en haut à droite de la carte) ajoute la journée. Les réponses du chat rendent le Markdown et le LaTeX ; les sources groupent les pages sous un seul nom de fichier.

Vue d'ensemble : [`docs/OVERVIEW.md`](docs/OVERVIEW.md). Pipeline : [`docs/INGESTION.md`](docs/INGESTION.md). API et UI : [`docs/API.md`](docs/API.md).
