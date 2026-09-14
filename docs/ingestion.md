# Ingestion

Transformer un PDF (souvent un scan d’ÉES / ESA Québec) en chunks cherchables **et** en fiches métier. Deux rapports du **même site** (ex. dossiers **4405** Enviro-Experts 2023 et **2259** Géosphère 2024, 619 Route 341, L’Épiphanie, forages du 17 août 2019) doivent **fusionner** sur la carte et **rester distincts** en timeline.

Le LLM de chat ne fait pas ce travail. LangExtract + normalisation + SQLite le font **à l’ingest**.

## Couches

| Couche | Rôle | Qui écrit |
|---|---|---|
| Qdrant | « Quel passage ressemble à la question ? » (vecteur + payload) | Ingest |
| SQLite `data/catalog.sqlite` | « Où, quand, quel dossier ? » (`sites` / `documents` / `events`) | Ingest |
| `data/parsed/`, `extractions/`, `documents/*.json` | Markdown, brut LangExtract, journal d’ingest | Ingest |

Le chatbot appelle `rag_ingestion.retrieve.search`. L’UI lit le catalog via l’API ([api.md](api.md)).

## Déclencheurs

Le catalog n’est **pas** la source des fichiers. Un PDF n’entre dans le corpus que via ingest / watch.

| Déclencheur | Commande |
|---|---|
| Drop folder (Compose `worker`) | Fichier dans `incoming/` → `rag-ingest watch` |
| Manuel | `rag-ingest ingest fichier.pdf` ou un dossier |
| Un scan (cron) | `rag-ingest watch --once` |

`watch` attend que la copie soit stable, calcule le SHA-256, **saute** si le hash est déjà dans le catalog (sauf `--force`), déplace le PDF vers `data/archive/{sha16}/` **puis** ingère depuis ce chemin. Échec → `data/failed/`.

## Identité

`document_id` = SHA-256 des **octets** du PDF (`src/rag_ingestion/parse.py`). Même contenu = même id. Ré-ingest : delete Qdrant par `document_id` puis upsert ; events SQL de ce document remplacés.

Artefacts : `data/parsed/{id}.md`, `data/extractions/{id}.jsonl`, `data/documents/{id}.json`, ligne `documents` dans SQLite.

`site_id` : `lot:{chiffres}` si le lot est connu, sinon `addr:{hash}` de l’adresse normalisée (`src/rag_ingestion/normalize.py`). Deux PDF avec le même lot (ou la même adresse rattachée ensuite) partagent un site.

## Chaîne (`ingest_path`, 7 étapes)

Orchestration : `src/rag_ingestion/pipeline.py`.

```
PDF
  → 1. LiteParse inspect + parse (Markdown + offsets de pages)
  → 2. chunks (fenêtres sur ce Markdown)
  → 3. LangExtract (Ollama) : entités ancrées dans le même Markdown
  → 4. alignement extractions ↔ chunks
  → 5. catalog SQLite + géocodage Nominatim si besoin
  → 6. embeddings FastEmbed (texte du chunk seulement)
  → 7. Qdrant (1 point = 1 vecteur + payload)
```

Quality gates : Markdown vide ou 0 chunk → skip (CLI code `2`), pas d’upsert. LangExtract en exception → warning et **poursuite** (chunks + embeddings sans métadonnées). `--skip-extract` idem sans appeler Ollama.

### 1. Parse

`src/rag_ingestion/parse.py`. OCR = plugin LiteParse, pas une étape amont. `parse_quality` : `ok` / `ocr_partial` / `ocr_heavy` selon `OCR_HEAVY_RATIO` (défaut 0,5). `ocr_heavy` n’arrête pas l’ingest.

Tout le reste (chunks, LangExtract, offsets) travaille sur **une** chaîne Markdown. Images : placeholders, pas de pixels indexés.

### 2. Chunks

`src/rag_ingestion/chunk.py`. Invariant : `markdown[start:end] == chunk.text`. Défauts : 2400 caractères, overlap 300. Recalage en fin de paragraphe / tableau Markdown. `chunk_id = {document_id}:{index}`. `heading_path` et `page` (page PDF au plus grand recouvrement).

Le vecteur = ce texte. Pas de JSON LangExtract dans l’embedding.

### 3. LangExtract

`src/rag_ingestion/extract.py`. Schéma **uniquement** dans `config/langextract/prompt.txt` + `few_shots.json` (autre domaine = autres fichiers via `.env`). Chaque `extraction_text` de few-shot doit apparaître tel quel dans l’exemple `text`.

Classes utiles ÉES : `title`, `doc_type`, `entity` (`project_id`, `firm`, `client`, …), `date` (`normalized` ISO, `role` : `report` / `fieldwork`), `topic`, `location` (`address`, `lot`, `city` = **site étudié**, pas le bureau de la firme).

### 4. Alignement

`src/rag_ingestion/align.py`. Étiquettes locales (entité, date, topic) seulement si l’intervalle LangExtract **recouvre** le chunk. `doc_type` et tous les `project_id` sont recopiés sur **chaque** chunk (`entities` + plus tard `project_id` payload). `location` n’est pas un champ de chunk : elle alimente la fiche document / le catalog.

### 5. Catalog + géocode

`src/rag_ingestion/document_meta.py` agrège au niveau document. `src/rag_ingestion/catalog.py` upsert :

- **sites** — `site_id`, lot, adresse, `lat`/`lon`, `geocode_status`
- **documents** — un PDF = une ligne (`project_id`, firme, titre, `site_id`, …)
- **events** — dates typées (`iso_date`, `role`, `label`) pour la timeline

Géocodage Nominatim **à l’ingest seulement**, si `GEOCODE_ENABLED=1` et le site n’est pas déjà `ok` (`src/rag_ingestion/geocode.py`). Échec → site créé, pin absent (`geocode_status=failed` / `skipped`).

`data/documents/{id}.json` = journal (timings, chemins) **plus** un dump de la fiche métier. L’UI lit SQLite, pas ce JSON.

### 6–7. Embeddings et Qdrant

`embed.py` : `EMBED_MODEL` (repli FastEmbed `intfloat/multilingual-e5-large`). Dimension lue chez FastEmbed ; changer de modèle ⇒ autre collection ou ré-ingest complet.

`qdrant_store.py` : cosine, UUID v5 déterministe par chunk, delete-then-upsert. Payload : `text`, `document_id`, `page`, `heading_path`, `doc_type`, `entities`, `dates`, `topics`, `site_id`, `project_id`, `parse_quality`, offsets. Indexes KEYWORD + index TEXT sur `text` (branche hybride).

## Recherche

`src/rag_ingestion/retrieve.py` (CLI `rag-ingest query` et outil chat) :

1. embedding de la question + recherche dense (filtres payload optionnels) ;
2. si un n° de projet apparaît dans la question ou les filtres : deuxième requête **mot-clé** (`entities` / `project_id` / MatchText dans `text`), fusion des hits ;
3. chatbot (`rag_hybrid_text`) : même branche mot-clé pour les termes d'identité (`client`, `firme`, `adresse`, …) ;
4. chatbot (`rag_include_catalog`) : les fiches SQLite (client, firme, adresse, lot) sont **préfixées** au markdown de l'outil — c'est la source d'identité, pas uniquement les chunks.

Le chatbot lit ces réglages dans `config/chatbot/settings.json` (`rag_limit`, `rag_prefetch`, `rag_score_threshold`, `temperature`, `top_p`).

```bash
rag-ingest query "contamination 4405" --filter project_id=4405
rag-ingest query "contamination" --filter site_id=lot:2363352
```

## Configuration ingest

`.env` à la racine, sinon `DEFAULT_*` dans `src/rag_ingestion/config.py`. Le plus utile :

`QDRANT_URL`, `QDRANT_COLLECTION`, `EMBED_MODEL`, `OLLAMA_BASE_URL`, `LANGEXTRACT_*`, `OCR_*`, `CHUNK_*`, `DATA_DIR`, `INCOMING_DIR` / `ARCHIVE_DIR` / `FAILED_DIR`, `INGEST_SKIP_EXISTING`, `INGEST_SKIP_EXTRACT`, `GEOCODE_ENABLED`.

Compose force `QDRANT_URL=http://qdrant:6333` et `DATA_DIR=/data` dans les conteneurs.

## CLI

| Code | `rag-ingest ingest` |
|---|---|
| 0 | Au moins un upsert Qdrant |
| 1 | Erreur (fichier, Qdrant, …) |
| 2 | Skip (Markdown / chunks vides) |
| 130 | Ctrl-C |

Après un changement de prompt LangExtract, de few-shots, ou de modèle d’embedding : **ré-ingérer** (éventuellement `watch --force` pour un SHA déjà au catalog).
