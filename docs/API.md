# API et interface

Le service Compose `chat-api` expose FastAPI (`src/chatbot/api.py`, `rag-chat serve`). L’UI Vite (`ui/`) parle à cette API : en Docker via nginx (`web`, `ui/nginx.conf`, port **3000**), en local via le proxy Vite vers `127.0.0.1:8000`.

Le modèle **n’a pas** les chunks dans le prompt. Il appelle l’outil `search_knowledge` (`src/chatbot/tools.py` → `retrieve.search`). Si `rag_include_catalog` est vrai, l’outil préfixe aussi la fiche catalog (client, firme, adresse).

## Lecture catalog

| Méthode | Usage |
|---|---|
| `GET /health` | Qdrant, SQLite, LLM (`provider`, `/v1/models`). Alias `llama_server` = `llm`. |
| `GET /sites` | Liste. Chaque site inclut `lot_geometry` (polygone GeoJSON) et `lat`/`lon` (centroïde). `?bbox=min_lon,min_lat,max_lon,max_lat`. `?geojson=true` → FeatureCollection (polygone si connu, sinon Point) |
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
  "document_id": null,
  "context": ["@E25 - ÉES phase II.pdf", "2025-08-08 Analyse laboratoire"]
}
```

- `thread_id` omis → UUID côté serveur (checkpoint SQLite `data/chat_checkpoints.sqlite`).
- `site_id` / `document_id` : focus UI. Ils **filtrent** `search_knowledge` (le document prime sur le site) même si le LLM omet l’argument.
- `context` : libellés de la barre de contexte (fichier `@…`, puis la journée). Ils sont préfixés au message envoyé au modèle (`Contexte sélectionné dans l'interface:`). Le texte affiché dans le fil reste la question seule.
- **Résolution automatique du lot** : si la question contient un n° de lot cadastral (ex. « Lot 1668054 » ou « 1 668 054 »), `search_knowledge` le normalise et interroge SQLite (`get_site_by_lot`) avant d’appeler Qdrant. Si le lot est connu, `site_id=lot:1668054` est injecté comme filtre — sans dépendre du LLM. Le prompt (`config/chatbot/prompt.txt`) enseigne aussi la règle `lot:{chiffres}` pour que le modèle la construise lui-même.

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

Écran coupé **65 % carte / 35 % chat**. Aucun `site_id` n’est choisi au chargement : la carte ouvre le sud du Québec autour de Montréal (centre ~45.55°N, 73.4°O, zoom 8). `activeId` reste vide jusqu’à un clic d’épingle, ou jusqu’à ce que `focus.site_ids[0]` revienne d’un tour de chat.

1. **Carte** — `GET /sites`, polygone du lot cadastral (Cadastre QC) + pin au centroïde. Sans lot : point Nominatim. Clic → sélection du site, zoom sur le lot, et chips `@fichier` dans la barre de contexte (le champ de saisie n’est pas modifié).
2. **Fiche + chronologie** — carte flottante en haut à droite de la carte. Le lot, l’adresse, la phase (I/II), le client, les fichiers et les contaminants restent affichés. **Chronologie** déplie les jours (`iso_date`, du plus récent au plus ancien, rôles en français). Clic sur une date → second chip (`2025-08-08 Analyse laboratoire`). La phase vient de `doc_type`, sinon du nom de fichier puis du titre (mêmes motifs que `profiles.json`).
3. **Chat** — `POST /chat` avec `site_id` (épingle), `document_id` (un seul fichier chargé, ou le PDF de la journée si ce fichier est encore dans la barre) et `context` (libellés des chips, injectés dans le prompt). Chaque chip se retire par sa croix. La réponse rend le Markdown (gras, listes, tableaux) et le LaTeX (`$...$`, `$$...$$`). Les `citations` sont groupées : un nom de fichier, puis les pages (`p.4`, `p.12`).

Pas de pin par forage / puits : le catalog a un polygone (ou un point) par **site**.

## Succès métier

Question « contamination 4405 » → `focus.project_ids` contient `4405`, `site_ids` le lot → pin L’Épiphanie → timeline : forages 2019-08-17, rapport Enviro-Experts 2023 **et** rapport Géosphère 2259 2024 sur le **même** site.
