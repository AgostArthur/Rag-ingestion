# RAG ingestion (local)

Pipeline : PDF → LiteParse (OCR plugin) → Markdown → chunks + LangExtract → embeddings FastEmbed → Qdrant.

Deux packages sous `src/` :

- `rag_ingestion` — ingest + recherche (`retrieve.search`)
- `chatbot` — agent LangGraph ; le RAG n’est qu’un outil (`search_knowledge`)

## Démarrage

```bash
cp .env.example .env
docker compose up -d
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
# Chatbot LangGraph :
pip install -e ".[chat]"
```

Config : changer `.env` (prioritaire) ou les fichiers sous `config/` — `config/langextract/` (ingest) et `config/chatbot/` (chat).

## Changer de modèle d'embeddings

1. Lister les modèles FastEmbed disponibles :

```bash
python -c "from fastembed import TextEmbedding; print('\n'.join(m['model']+' ('+str(m['dim'])+'d)' for m in TextEmbedding.list_supported_models()))"
```

1. Dans `.env`, fixer le nom exact (la dimension Qdrant est lue automatiquement chez FastEmbed — pas besoin de `EMBED_DIM`) :

```env
EMBED_MODEL=intfloat/multilingual-e5-large
```

Exemples multilingues : `intfloat/multilingual-e5-large` (1024-d), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `nomic-ai/nomic-embed-text-v1.5`.

1. Ré-ingérer les documents (les vecteurs d'un modèle ne sont pas comparables à ceux d'un autre).

Si `EMBED_MODEL` n'est pas dans FastEmbed (ex. `BAAI/bge-m3` sous FastEmbed 0.8), repli automatique vers `DEFAULT_EMBED_FALLBACK_MODEL` dans `config.py` (par défaut `intfloat/multilingual-e5-large`).

## Changer le schéma LangExtract (projet)

Prompts et few-shots sont trouve dans le dossier : /config.

```env
LANGEXTRACT_PROMPT_FILE=config/langextract/prompt.txt
LANGEXTRACT_FEW_SHOTS_FILE=config/langextract/few_shots.json
```

Pour un autre domaine (contrats, factures, …) : dupliquer les fichiers sous `config/langextract/` (ex. `prompt_contrats.txt`, `few_shots_contrats.json`) et mettre à jour ces deux variables. Alias accepté pour les few-shots : `LANGEXTRACT_FEW_SHOTS_EXAMPLE`.

Format JSON des few-shots :

```json
[
  {
    "text": "…extrait d'exemple…",
    "extractions": [
      {
        "extraction_class": "title",
        "extraction_text": "…",
        "attributes": { "normalized": "…" }
      }
    ]
  }
]
```

Chaque `extraction_text` doit apparaître tel quel dans le champ `text` (alignement LangExtract).

## Usage

Démarche d’ingestion détaillée (étapes, offsets, LangExtract, payload Qdrant, cas d’échec) : [`docs/ingestion.md`](docs/ingestion.md).

```bash
rag-ingest ingest chemin/vers/doc.pdf
rag-ingest ingest chemin/vers/doc.pdf --skip-extract
rag-ingest query "clause de responsabilité" --filter doc_type=rapport --limit 5
```

Artefacts : `data/parsed/{id}.md`, `data/extractions/{id}.jsonl` + `.html`, `data/documents/{id}.json`.

## Chatbot (LangGraph + llama-server)

Le LLM n’a pas les chunks dans le prompt : il appelle l’outil `search_knowledge`, qui passe par `rag_ingestion.retrieve.search` (FastEmbed + Qdrant).

Prérequis : Qdrant avec des documents ingérés, extra `[chat]`, et **llama-server** (llama.cpp) avec tool-calling. Ce n’est pas l’API cloud OpenAI : llama-server expose le protocole HTTP `/v1/chat/completions`. Le client LangChain `ChatOpenAI` pointe vers `LLAMA_SERVER_BASE_URL`.

```bash
pip install -e ".[chat]"
llama-server --jinja -fa -m /chemin/vers/modele.gguf --port 8080
# Modèle avec template d’outils (ex. Qwen2.5-Instruct). Vérifier http://localhost:8080/props
```

```env
CHAT_PROMPT_FILE=config/chatbot/prompt.txt
CHAT_SETTINGS_FILE=config/chatbot/settings.json
```

Le prompt système est dans `[config/chatbot/prompt.txt](config/chatbot/prompt.txt)` (même idée que `[config/langextract/prompt.txt](config/langextract/prompt.txt)` / `[few_shots.json](config/langextract/few_shots.json)`). URL llama-server, température, `rag_limit` et l’API HTTP sont dans `[config/chatbot/settings.json](config/chatbot/settings.json)`. Les clés `.env` du même nom restent des surcharges optionnelles.

```bash
rag-chat
# /quit  /new

rag-chat serve
# POST /chat  { "message": "…", "thread_id": "optionnel" }
# GET  /health
```

