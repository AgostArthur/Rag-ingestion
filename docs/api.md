# API et interface

Le service Compose `chat` expose FastAPI (`src/chatbot/api.py`, `rag-chat serve`). L’UI Vite (`ui/`) parle à cette API : en Docker via nginx (`ui/nginx.conf`, port **3000**), en local via le proxy Vite vers `127.0.0.1:8000`.

Le modèle **n’a pas** les chunks dans le prompt. Il appelle l’outil `search_knowledge` (`src/chatbot/tools.py` → `retrieve.search`). Si `rag_include_catalog` est vrai, l’outil préfixe aussi la fiche catalog (client, firme, adresse).

## Lecture catalog

| Méthode | Usage |
|---|---|
| `GET /health` | Qdrant, SQLite, LLM (`provider`, `/v1/models`). Alias `llama_server` = `llm`. |
| `GET /sites` | Liste des sites. `?bbox=min_lon,min_lat,max_lon,max_lat`. `?geojson=true` → FeatureCollection |
| `GET /sites/{site_id}` | Fiche + `document_ids` (`site_id` à encoder, ex. `lot%3A2363352`) |
| `GET /sites/{site_id}/timeline` | Events triés (`iso_date`, `role`, titre / firme du PDF). Repli `report_date` s’il n’y a pas d’events |
| `GET /documents/{document_id}` | Fiche rapport + champs du site |

Pas d’écriture catalog depuis ces routes. CORS : origines `localhost:3000` (dev Vite). En Compose, nginx est same-origin : pas besoin de CORS.

## Chat

`POST /chat`

```json
{
  "message": "contamination 4405",
  "thread_id": null,
  "site_id": "lot:2363352",
  "document_id": null
}
```

- `thread_id` omis → UUID côté serveur (checkpoint SQLite `data/chat_checkpoints.sqlite`).
- `site_id` / `document_id` : focus UI. Ils **filtrent** `search_knowledge` (le document prime sur le site) même si le LLM omet l’argument.

Réponse :

```json
{
  "reply": "…",
  "thread_id": "…",
  "focus": {
    "document_ids": ["…"],
    "project_ids": ["4405"],
    "site_ids": ["lot:2363352"]
  },
  "documents": [ ],
  "citations": [
    {
      "document_id": "…",
      "source": "fichier.pdf",
      "page": 4,
      "heading_path": "Conclusions",
      "score": 0.91,
      "project_id": "4405",
      "site_id": "lot:2363352",
      "chunk_id": "…:3"
    }
  ]
}
```

`focus`, `documents` et `citations` viennent des **ToolMessage** de ce tour + jointure catalog (`src/chatbot/focus.py`), **pas** du texte généré.

REPL : `rag-chat` (JSON `focus` sous la réponse). LLM : `OPENAI_*` ou `LLAMA_SERVER_*` (OpenAI-compatible).

## Interface (`ui/`)

Trois panneaux, un état `activeId` = `site_id` catalog :

1. **Carte** — `GET /sites`, un pin par site avec `lat`/`lon` (`AdvancedMarker` + `mapId`, défaut `DEMO_MAP_ID`). Clic → sélection du site. Clé `VITE_GOOGLE_MAPS_API_KEY` (rebuild Compose si elle change). Sans clé : liste cliquable.
2. **Chronologie** — `GET /sites/{id}/timeline`. Clic sur un rapport → chip `@fichier` et `document_id` pour le prochain `POST /chat`.
3. **Chat** — `POST /chat` avec `site_id` et éventuellement `document_id`. Si `focus.site_ids[0]` revient, la carte suit.

Pas de pin par forage / puits : le catalog n’a qu’un point par **site**.

## Succès métier

Question « contamination 4405 » → `focus.project_ids` contient `4405`, `site_ids` le lot → pin L’Épiphanie → timeline : forages 2019-08-17, rapport Enviro-Experts 2023 **et** rapport Géosphère 2259 2024 sur le **même** site.
