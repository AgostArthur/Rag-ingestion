# Démarche d’ingestion RAG

## Contexte

Ce dépôt construit un **corpus interrogable** à partir de PDF métier, pour un usage local (pas d’API cloud d’embeddings ni de LLM d’extraction). Le domaine visé aujourd’hui est celui des **études environnementales de site (ÉES / ESA) au Québec**, en particulier les rapports de **phase I** et **phase II** : lettre d’accompagnement, sommaire, historique d’usage, campagnes de forages, tableaux d’analyses (HAM, HAP, métaux, etc.), conclusions et recommandations. Les PDF sont souvent des **scans** : LiteParse déclenche l’OCR page par page. Le texte indexé n’est donc pas le calque PDF « propre », mais un **Markdown bruité** (en-têtes de page recopiés, tableaux cassés, parfois un n° de projet parasite collé d’un autre dossier).

Le besoin n’est pas « coller le PDF dans un LLM ». Un rapport fait souvent des dizaines de pages ; deux rapports du **même site** se ressemblent au mot près (même gabarit de firme, mêmes 9 forages, même date de campagne). Un modèle qui lit tout le fil de conversation **mélange** alors les n° de dossier. Exemple réel déjà observé dans ce projet : le rapport **4405** (Enviro-Experts, site 619 Route 341, L’Épiphanie, forages du 17 août 2019, rapport ~3 janvier 2023) et le rapport **2259** (Géosphère, **même adresse, même lot, même campagne**, rapport ~3 janvier 2024). La recherche sémantique les rapproche ; sans filtre structuré (`entities=4405`) et sans fiche **site** plus tard, le chat attribue au mauvais projet les analyses, les dates et même « l’absence de documents ».

D’où deux produits dans le même repo, volontairement séparés :

| Package | Rôle |
|---|---|
| `rag_ingestion` | Transformer un PDF en Markdown, chunks, métadonnées ancrées, vecteurs, points Qdrant. Commande `rag-ingest`. |
| `chatbot` | Agent LangGraph (`rag-chat`) : le modèle **n’a pas** les chunks dans le prompt système. Il n’y accède que via l’outil `search_knowledge`, qui appelle la même recherche que `rag-ingest query` (FastEmbed + Qdrant). |

L’ingest est donc la **seule** écriture du corpus. Le chatbot **lit** Qdrant ; il ne doit pas inventer ni persister des entités (n° de projet, adresse, dates). Les réponses utilisateur sont en français ; le prompt LangExtract est bilingue EN/FR parce que les rapports québécois mélangent souvent les deux.

La vision produit (carte + timeline à côté du chat, **§ 16**) repose sur le même constat : 2259 et 4405 doivent **fusionner** sur une carte (un lieu) et **rester distincts** dans un historique (deux travaux, deux firmes, deux dates de remise). Ça n’est possible que si l’identité (projet, lot, dates typées) est écrite **à l’ingest**, pas générée par le LLM en fin de tour.

Ce document décrit **exactement** le pipeline d’ingestion tel qu’il est implémenté : entrée, étapes, structures de données, fichiers produits, index Qdrant, cas d’échec, et ré-ingestion. Il ne décrit pas le graphe LangGraph ni les prompts chat, sauf là où l’ingest alimente la recherche. La **§ 16** est le **plan** convenu (SQLite `sites` / `documents` / `events`, géocodage, enveloppe `focus`) : **rien de cette section n’est dans le code** à la date de rédaction.

Point d’entrée : `rag-ingest ingest <chemin.pdf>` → `rag_ingestion.cli.cmd_ingest` → `rag_ingestion.pipeline.ingest_path`.

Chaîne réelle :

```
PDF (octets)
  → SHA-256 = document_id
  → LiteParse inspect (besoin d’OCR par page)
  → LiteParse parse (Markdown + offsets de pages)
  → découpage en chunks (offsets stables dans ce Markdown)
  → LangExtract (Ollama) : métadonnées ancrées dans le même Markdown
  → alignement extractions ↔ chunks
  → embeddings FastEmbed (texte du chunk uniquement)
  → Qdrant : 1 point = 1 vecteur + payload plat
  → fiche parent JSON dans data/documents/
```

Orchestration : `src/rag_ingestion/pipeline.py`. Chaque étape est chronométrée (`StepTimer`) et journalisée.

---

## 1. Intention du pipeline

L’ingest ne « met pas le PDF dans un LLM ». Il construit **deux représentations complémentaires** du même fichier :

1. **Recherche sémantique** — le Markdown est découpé en fenêtres (chunks). Chaque fenêtre devient un vecteur dense dans Qdrant. Une question future est elle-même embeddée ; Qdrant renvoie les chunks les plus proches (cosine).
2. **Filtres structurés** — LangExtract lit le Markdown et en tire des étiquettes (n° de projet, type de document, dates, sujets, etc.). Ces étiquettes sont recopiées dans le **payload** Qdrant. Un filtre `entities=4405` ne cherche pas dans le vecteur : il restreint les points dont le payload contient cette valeur **exacte**.

Ces deux mécanismes ne se substituent pas l’un à l’autre. Un n° de projet court (`4405`) est un mauvais signal pour un embedding ; il est un excellent signal pour un filtre mot-clé. D’où l’insistance, plus bas, à recopier le `project_id` sur **tous** les chunks du document, y compris ceux qui ne contiennent pas le numéro dans le texte (conclusions, tableaux d’analyses).

Ce que l’ingest **ne fait pas** (état actuel du code) :

- pas de base SQL (sites, lots, lat/lon) ;
- pas de géocodage ;
- la classe LangExtract `location` est extraite mais **n’est pas** écrite dans le payload Qdrant ;
- le JSON parent `data/documents/{id}.json` n’est pas une fiche métier : chemins, compteurs, timings, qualité OCR.

---

## 2. Prérequis d’exécution

Avant `rag-ingest ingest` :

| Service | Rôle | Comment |
|---|---|---|
| Qdrant | Stockage vecteurs + payload | `docker compose up -d` — image `qdrant/qdrant:v1.19.0`, HTTP `6333` |
| Ollama | LLM pour LangExtract | `OLLAMA_BASE_URL` (défaut `http://localhost:11434`) et modèle `LANGEXTRACT_MODEL` |
| FastEmbed | Embeddings locaux | Premier appel : téléchargement du modèle `EMBED_MODEL` |

Sans Qdrant, l’étape 6 échoue. Sans Ollama, LangExtract lève une exception : l’ingest **continue** tout de même (chunks + embeddings sans métadonnées), avec un warning. Voir § 8.

La configuration est lue par `load_settings()` (`src/rag_ingestion/config.py`) :

- fichier `.env` à la racine du dépôt, **prioritaire** (`load_dotenv(..., override=True)`) ;
- sinon constantes `DEFAULT_*` dans `config.py`.

Paramètres qui changent réellement le comportement d’ingest :

| Variable | Défaut code | Effet |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Client HTTP Qdrant |
| `QDRANT_COLLECTION` | `chunks` | Nom de la collection (ex. `Test_ingest`) |
| `EMBED_MODEL` | `BAAI/bge-m3` | Nom FastEmbed ; si inconnu de cette version FastEmbed, repli `intfloat/multilingual-e5-large` |
| `EMBED_DIM` | `1024` | **Ignoré** dès que FastEmbed publie la `dim` du modèle résolu |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL LangExtract |
| `LANGEXTRACT_MODEL` | `nemotron-3-nano:4b` | `model_id` passé à `langextract.extract` |
| `LANGEXTRACT_TIMEOUT_SECONDS` | `300` | Timeout Ollama |
| `LANGEXTRACT_PROMPT_FILE` | `config/langextract/prompt.txt` | Schéma d’extraction |
| `LANGEXTRACT_FEW_SHOTS_FILE` | `config/langextract/few_shots.json` | Exemples d’alignement (alias `LANGEXTRACT_FEW_SHOTS_EXAMPLE`) |
| `OCR_LANGUAGE` | `fra+eng` | Langue Tesseract / serveur OCR LiteParse |
| `OCR_SERVER_URL` | vide | Si vide : Tesseract intégré ; sinon URL d’un serveur OCR |
| `OCR_HEAVY_RATIO` | `0.5` | Seuil : fraction de pages `needs_ocr` à partir de laquelle `parse_quality = ocr_heavy` |
| `CHUNK_SIZE_CHARS` | `2400` | Taille cible d’une fenêtre de chunk |
| `CHUNK_OVERLAP_CHARS` | `300` | Chevauchement entre deux fenêtres successives |
| `DATA_DIR` | `data` | Racine des artefacts (`parsed/`, `extractions/`, `documents/`) |

---

## 3. Identité du document (`document_id`)

Le `document_id` n’est **pas** le nom de fichier, ni un UUID aléatoire.

Dans `parse_pdf` :

1. lecture binaire complète du PDF (`path.read_bytes()`) ;
2. `document_id = SHA-256 hexadécimal de ces octets` (`document_id_from_bytes`).

Conséquences :

- deux chemins différents vers le **même contenu** produisent le même id ;
- modifier un seul octet du PDF produit un nouvel id (nouveau document pour Qdrant) ;
- ré-ingérer **le même fichier inchangé** réutilise le même id : l’étape 6 **supprime** d’abord tous les points Qdrant de cet id, puis réécrit (ingest idempotent).

Le même hash est recopié dans la fiche parent sous `content_sha256` (égal à `document_id`).

Les artefacts disque portent ce hash :

- `data/parsed/{document_id}.md`
- `data/extractions/{document_id}.jsonl`
- `data/extractions/{document_id}.html`
- `data/documents/{document_id}.json`

---

## 4. Étape 1/6 — Inspect puis parse PDF → Markdown

Fichier : `src/rag_ingestion/parse.py`.  
Bibliothèque : **LiteParse**. L’OCR n’est pas un préprocesseur séparé : c’est un **plugin** de LiteParse, déclenché page par page.

### 4.1 Inspect (`inspect_pdf`)

Appel : `LiteParse.is_complex(chemin)`.

Pour chaque page, LiteParse indique si le calque texte est insuffisant (`needs_ocr`) et des `reasons` (ex. page scannée). Le pipeline construit un `InspectReport` :

- `n_pages`, `n_needs_ocr` ;
- une liste de `PageSpan` (numéro de page, `needs_ocr`, raisons) — à ce stade `start`/`end` sont encore à 0 ;
- `parse_quality` selon le ratio `n_needs_ocr / n_pages` et `OCR_HEAVY_RATIO` :
  - `ok` si ratio = 0 (aucune page OCR) ;
  - `ocr_heavy` si ratio ≥ seuil (défaut 50 %) ;
  - `ocr_partial` sinon.

`ocr_heavy` n’arrête **pas** l’ingest. Il produit un warning du type `Heavy OCR: 39/39 pages` recopié dans la fiche parent. L’OCR sera plus lent ; le Markdown peut être bruité (tableaux cassés, en-têtes parasites).

### 4.2 Parse Markdown (`parse_pdf`)

Appel : `LiteParse.parse(...)` avec notamment :

- `output_format="markdown"`
- `ocr_enabled=True`
- `ocr_language` issu du `.env`
- `image_mode="placeholder"` (les images ne sont pas des pixels indexés : des placeholders Markdown)
- `extract_links=True`
- `include_complexity=True`

Les pages sont **concaténées** dans l’ordre. Entre deux pages non vides : `"\n\n"`. Pour chaque page on mémorise un `PageSpan` **dans le Markdown concaténé** :

- `start` : offset caractère du début de la page dans la grande chaîne ;
- `end` : offset de fin (exclus) ;
- `page_num` : numéro de page PDF ;
- `needs_ocr` / `reasons` : repris de l’inspect, éventuellement corrigés par `page.complexity` si LiteParse le fournit au parse.

Invariant important pour la suite : **tout le reste du pipeline (chunks, LangExtract, alignement) travaille sur cette unique chaîne Markdown**. Les offsets de chunk et les `char_interval` LangExtract sont des indices dans **cette** chaîne, pas dans le PDF.

Le fichier `data/parsed/{document_id}.md` est écrit à ce moment.

### 4.3 Quality gate : Markdown vide

Si `parsed.markdown` est vide (après `strip`) : ingest **interrompu** (`skipped=True`), aucun chunk, aucun upsert Qdrant. Code de sortie CLI : `2`. Le `document_id` est quand même connu (le hash a déjà été calculé).

---

## 5. Étape 2/6 — Chunking Markdown (offsets stables)

Fichier : `src/rag_ingestion/chunk.py`.  
Fonction : `chunk_markdown(text, document_id=..., max_chars=..., overlap_chars=..., page_spans=...)`.

### 5.1 Pourquoi des offsets et pas « N tokens »

Chaque `Chunk` vérifie :

```text
markdown[chunk.start:chunk.end] == chunk.text
```

Sans ça, LangExtract (qui ancre des extraits dans le Markdown) ne pourrait pas être recollé sur les bons chunks : il faut le **même** système de coordonnées caractères.

### 5.2 Fenêtre glissante

Paramètres (défauts) : `max_chars=2400`, `overlap_chars=300`.

Algorithme, en partant de `i = 0` :

1. Sauter les espaces en tête.
2. Poser une fin brute `raw_end = min(i + max_chars, longueur)`.
3. **Recaler** `end` avec `_snap_end` :
   - si `end` tombe **à l’intérieur d’un tableau Markdown** (lignes `| ... |`) : étendre jusqu’à la fin du tableau si le tableau n’est pas énorme (`t_end - start ≤ 2 × taille de fenêtre`), sinon reculer au début du tableau pour ne pas le couper au milieu ;
   - sinon chercher un saut de paragraphe `\n\n` dans le dernier tiers de la fenêtre, à défaut un `\n` dans la seconde moitié ;
   - sinon garder `raw_end`.
4. Extraire la tranche, `strip` des espaces de bord : le `start` / `end` réels suivent ce strip (le texte indexé ne commence/finit pas par du blanc).
5. Attribuer :
   - `chunk_id = "{document_id}:{index}"` (`index` 0, 1, 2, …) ;
   - `heading_path` : fil d’Ariane des titres `#` … `######` visibles **à la position `start`** (ex. `SOMMAIRE DE L'ÉTUDE` ou `9. CONCLUSIONS ET RECOMMANDATIONS`) ;
   - `page` : numéro de page PDF qui **recouvre le plus** l’intervalle `[start, end)` parmi les `page_spans` (`page_for_span`). Un chunk à cheval sur deux pages reçoit la page au plus grand recouvrement.
6. Avancer : `next_i = real_end - overlap_chars`. Si ça ne progresse pas, avancer d’au moins `max(1, max_chars // 8)` caractères (anti-boucle).

Les chunks dont le texte est vide après strip sont jetés (deuxième filet dans `ingest_path`).

### 5.3 Quality gate : aucun chunk

Si la liste est vide : ingest skip, pas d’upsert. Même logique que Markdown vide.

### 5.4 Ce qui n’est pas embeddé à ce stade

Le chunk ne contient **que** du Markdown. Pas de JSON LangExtract, pas de n° de projet injecté en préfixe. L’identité projet arrive plus tard via le **payload** (`entities`), pas via le vecteur — sauf si le numéro figure déjà dans le texte de la fenêtre.

---

## 6. Étape 3/6 — Extraction structurée (LangExtract + Ollama)

Fichier : `src/rag_ingestion/extract.py`.  
Saut possible : `rag-ingest ingest fichier.pdf --skip-extract` (alors `extractions = []`, timings `langextract = 0`).

### 6.1 Entrée

On passe **tout** le Markdown du document (`parsed.markdown`), pas les chunks un par un. LangExtract voit le rapport entier (lettre, sommaire, tableaux, conclusions). C’est coûteux (souvent plusieurs minutes) mais nécessaire pour trouver un `Projet nº` en page 1 et des dates de forage plus loin.

### 6.2 Schéma (prompt + few-shots)

Le code ne durcit **aucune** classe Python. Le schéma est 100 % dans :

- `config/langextract/prompt.txt`
- `config/langextract/few_shots.json`

Changer de domaine (contrats, factures) = pointer d’autres fichiers via `.env`, sans toucher `extract.py`.

Règle LangExtract : chaque `extraction_text` d’un few-shot doit apparaître **tel quel** comme sous-chaîne du champ `text` de l’exemple (alignement caractère).

Classes actuellement autorisées par le prompt (ÉES / ESA Québec) :

| Classe | Rôle | Attributs typiques |
|---|---|---|
| `title` | Titre principal | — |
| `doc_type` | Type de document | `normalized` : `rapport`, `ees_phase_1`, `ees_phase_2`, `contrat`, `facture`, `note`, `guide`, `autre` |
| `entity` | Personne, org, n° de projet, lieu nommé | `type` : `person`, `org`, `place`, `project_id`, `client`, `firm` ; pour un projet `normalized` = chiffres (`4405`) |
| `date` | Date mentionnée | `normalized` ISO `YYYY-MM-DD` ; parfois `role` dans les few-shots (`report`) |
| `topic` | Thème (phase, contaminants, reco…) | — |
| `location` | Adresse, lot, ville | `type` : `address`, `lot`, `region`, `city`, … |

Consigne métier importante : un n° de projet (`Projet nº`, `No/Réf.`) doit être extrait en `entity` / `type=project_id` avec `extraction_text` = les **chiffres** tels qu’écrits (`2259`, `4405`), pas une année (`2019`). Les few-shots incluent des exemples ÉES (4405, 2259) en plus d’un exemple générique « rapport d’audit ».

### 6.3 Appel

```text
langextract.extract(
  text_or_documents=markdown,
  prompt_description=prompt,
  examples=few-shots,
  model_id=LANGEXTRACT_MODEL,
  model_url=OLLAMA_BASE_URL,
  language_model_params={ timeout: LANGEXTRACT_TIMEOUT_SECONDS },
)
```

LangExtract demande au modèle de recopier des extraits **présents dans le source** et, si possible, renvoie un `char_interval` (`start_pos`, `end_pos`) dans le Markdown. Une extraction **ancrée** (`grounded`) a `start` et `end` non nuls. Une extraction non ancrée a `start = end = None`.

Conversion interne : `GroundedExtraction` (`extraction_id`, `extraction_class`, `extraction_text`, `start`, `end`, `attributes`).

### 6.4 Artefacts

Si la sauvegarde réussit :

- `data/extractions/{document_id}.jsonl` — document annoté LangExtract (brut) ;
- `data/extractions/{document_id}.html` — visualisation (`langextract.visualize`).

Un échec de sauvegarde HTML/JSONL est loggé mais **n’annule pas** l’ingest : les objets `GroundedExtraction` en mémoire restent utilisables pour l’alignement.

### 6.5 Échec LangExtract

Dans `ingest_path`, tout exception autour de `extract_structured` est **avalée** :

- warning `LangExtract failed: …` ;
- `extractions` reste `[]` ;
- **les embeddings et Qdrant s’exécutent quand même**.

Si LangExtract réussit mais ne trouve rien : warning `0 entities extracted by LangExtract`. Les chunks sont indexés sans étiquettes utiles (sauf `doc_type`/`entities` vides).

---

## 7. Étape 4/6 — Alignement extractions → chunks

Fichier : `src/rag_ingestion/align.py`.  
Fonction : `align_extractions(chunks, extractions, source_path=..., parse_quality=...)`.

Produit une liste de `ChunkPayload` : **un par chunk**, même ordre. C’est cette structure qui partira en Qdrant (plus le vecteur).

### 7.1 Recouvrement d’intervalles

Pour une extraction ancrée `[e.start, e.end)` et un chunk `[c.start, c.end)` :

```text
chevauchement ⇔ min(c.end, e.end) > max(c.start, e.start)
```

Les étiquettes « locales » (entité non-projet, date, topic) ne sont collées que si ce test est vrai. Une date citée uniquement en page 12 n’apparaît pas sur le chunk de la page 3.

### 7.2 Niveau document (recopié sur **tous** les chunks)

Deux cas particuliers, volontairement globaux :

**`doc_type`**  
On prend le premier `doc_type` dont le label ( `attributes.normalized` si présent, sinon `extraction_text`) est non vide, mis en minuscules. Ce `doc_type` unique est recopié sur chaque `ChunkPayload`. Le `title` est dans `_DOC_LEVEL` mais n’est **pas** stocké dans le payload Qdrant (il n’est pas un champ de `ChunkPayload`).

**N° de projet (`project_id`)**  
`project_id_label(ext)` accepte :

1. `entity` avec `attributes.type == project_id`, puis dans l’ordre : `normalized` si c’est uniquement des chiffres, sinon `extraction_text` s’il est numérique, sinon le seul groupe `\d{3,8}` du texte s’il n’est pas une année `19xx`/`20xx` ;
2. repli : une `entity` dont le label est uniquement des chiffres, longueur 3 à 8, **et** qui n’est pas une année `19xx`/`20xx` (pour ne pas indexer `2019` comme projet).

Tous les n° ainsi obtenus sont dédupliqués (casse ignorée) puis **préfixés** dans `entities` de **chaque** chunk. Même un chunk de conclusions qui ne contient pas « 4405 » dans son texte portera `entities: ["4405", …]`. C’est ce qui rend le filtre Qdrant `entities=4405` utilisable sur tout le rapport.

Les `project_id` **non ancrés** (`start is None`) sont quand même recopiés au niveau document. Les autres entités non ancrées (`Ghost` sans intervalle) **ne sont pas** copiées : on refuse de polluer tous les chunks avec une invention non localisée, sauf le n° de projet (identifiant du dossier).

### 7.3 Niveau local (seulement si recouvrement)

| Classe | Champ payload | Label |
|---|---|---|
| `entity` (hors project_id) | `entities` | `normalized` ou texte |
| `date` | `dates` | idem |
| `topic` | `topics` | idem |
| `location` | **aucun** | extraite par LangExtract, **ignorée** à l’alignement |

Déduplication par label minuscule à l’intérieur d’un chunk.

`extraction_ids` accumule les ids d’extractions rattachées (y compris `doc_type` et tous les `project_id` sur chaque chunk).

`source_path` = chemin absolu du PDF d’origine. `parse_quality` = `ok` / `ocr_partial` / `ocr_heavy`.

### 7.4 Compteur logué

Le log `N chunk(s) have at least one local label` compte les payloads où `entities` **ou** `dates` **ou** `topics` est non vide. Un chunk qui n’a que le `project_id` recopié compte comme labellisé (il a des `entities`). Un chunk sans projet ni date ni topic local reste « nu » hormis `doc_type`.

---

## 8. Étape 5/6 — Embeddings (texte du chunk, pas le JSON)

Fichier : `src/rag_ingestion/embed.py`.

On embed **uniquement** `[payload.chunk.text for payload in payloads]`. Le JSON LangExtract, le `doc_type`, les `entities` **n’entrent pas** dans le vecteur. Ils voyagent à côté, dans le payload Qdrant.

Résolution du modèle :

1. nom demandé = `EMBED_MODEL` ;
2. s’il n’est pas dans `TextEmbedding.list_supported_models()` de **cette** version FastEmbed → warning et repli `DEFAULT_EMBED_FALLBACK_MODEL` (`intfloat/multilingual-e5-large`) ;
3. dimension Qdrant = `dim` déclarée par FastEmbed pour le modèle **résolu**. `EMBED_DIM` dans `.env` est ignoré dans ce cas (warning).

L’instance FastEmbed est mise en cache (`lru_cache`). Premier ingest : téléchargement possible.

`embed_texts` itère `model.embed(texts)` et convertit chaque vecteur en `list[float]` (float32). Une longueur différente du nombre de textes lève `RuntimeError`.

Pour la **recherche** (hors ingest), `embed_query` utilise `query_embed` si le modèle l’expose, sinon `embed` — certains modèles (E5, etc.) distinguent passage et requête.

---

## 9. Étape 6/6 — Écriture Qdrant

Fichier : `src/rag_ingestion/qdrant_store.py`.

Distance : **cosine**. Un point = un chunk.

### 9.1 Collection et indexes

`ensure_collection(nom, vector_size)` :

- si la collection n’existe pas : création avec `VectorParams(size=vector_size, distance=COSINE)` ;
- si elle existe : lecture de la dimension ; **incompatibilité** → `ValueError` (il faut une autre collection ou ré-embedder tout après un changement de modèle) ;
- dans tous les cas, création (idempotente) d’indexes payload pour le filtrage :

| Champ | Type d’index |
|---|---|
| `document_id` | KEYWORD |
| `source_path` | KEYWORD |
| `page` | INTEGER |
| `heading_path` | KEYWORD |
| `doc_type` | KEYWORD |
| `entities` | KEYWORD |
| `dates` | KEYWORD |
| `parse_quality` | KEYWORD |

`topics` est **stocké** dans le payload mais **n’a pas** d’index dédié dans `_INDEXED_FIELDS` (filtre `topics` non exposé dans `build_filter`).

### 9.2 Ré-ingest : delete puis upsert

Avant d’écrire : `delete_by_document_id(collection, document_id)` — tous les points dont le payload `document_id` égale le SHA. Ensuite upsert.

Id de point Qdrant : **UUID v5 déterministe**

```text
uuid5(namespace=6ba7b810-9dad-11d1-80b4-00c04fd430c8, name="{document_id}:{chunk_index}")
```

Ré-upsert du même chunk (même document, même index) écrase le même id, pas de doublon d’id même si le delete avait échoué partiellement.

Écriture par lots de 64 points (`wait=True`).

### 9.3 Contenu exact du payload

Pour chaque chunk :

```text
document_id      SHA-256 du PDF
source_path      chemin absolu du PDF
page             int ou null
heading_path     fil d’Ariane Markdown
doc_type         str ou null (minuscule, normalisé)
entities         liste de str (project_id en tête, puis entités locales)
dates            liste de str (labels de dates qui recouvrent ce chunk)
topics           liste de str
extraction_ids   ids LangExtract rattachés
text             Markdown du chunk (c’est ce qui a été embeddé)
chunk_index      0..n-1
chunk_id         "{document_id}:{chunk_index}"
parse_quality    ok | ocr_partial | ocr_heavy
char_start       offset début dans le Markdown
char_end         offset fin dans le Markdown
```

Ce JSON LangExtract **n’est pas** dans Qdrant. Il reste dans `data/extractions/{id}.jsonl`.

---

## 10. Fiche parent JSON

Après un ingest **non skippé**, `_write_parent` écrit `data/documents/{document_id}.json` :

```text
document_id, source_path, content_sha256
n_pages, n_pages_needs_ocr, parse_quality
n_chunks, n_extractions
markdown_path
extraction_jsonl, extraction_html
warnings[]
step_seconds { parse, chunk, langextract, align, embed, qdrant }
total_seconds
```

Ce n’est **pas** encore la fiche métier (lot cadastral, lat/lon, dates typées `report` vs `fieldwork`). C’est un journal d’ingest + pointeurs vers les artefacts.

Si l’ingest est skippé (Markdown/chunks vides), cette fiche n’est **pas** écrite par `_write_parent` (sortie anticipée dans `ingest_path`).

---

## 11. Codes de sortie et journalisation

CLI `rag-ingest ingest` :

| Code | Signification |
|---|---|
| `0` | Upsert Qdrant effectué |
| `1` | Exception (fichier absent, Qdrant down, dimension incompatible, …) |
| `2` | Skip (Markdown vide ou 0 chunk) |
| `130` | Ctrl-C |

Stdout succès :

```text
OK document_id=… chunks=N extractions=M quality=… collection=…
warning: …
total time: …
```

Les durées par étape sont aussi dans les logs (`HH:MM:SS  message`) et dans `step_seconds` de la fiche.

---

## 12. Recherche après ingest (`rag-ingest query`)

Hors ingest à proprement parler, mais c’est le contrat de sortie.

`rag_ingestion.retrieve.search` :

1. `embed_query(question)` — **même modèle** que l’ingest (sinon les vecteurs ne sont pas comparables) ;
2. `search_similar` : `query_points` cosine, `limit`, `with_payload=True` ;
3. filtre optionnel `--filter clé=valeur` (répétable).

Filtres autorisés : `document_id`, `source_path`, `doc_type`, `entities`, `dates`, `heading_path`, `parse_quality`, `page`.

Sémantique :

- `page` : égalité entière ;
- une seule valeur : `MatchValue` ;
- plusieurs valeurs séparées par des virgules : `MatchAny` (OR) ;
- plusieurs `--filter` différents : `must` (AND) — ex. `entities=4405` **et** `doc_type=ees_phase_2`.

Exemple :

```bash
rag-ingest query "niveaux de contamination" --filter entities=4405 --limit 8
```

Cela ne retourne **que** des chunks dont le payload `entities` contient exactement `4405`. Si LangExtract n’a pas produit ce n° (ou si le document n’a pas été ré-ingéré depuis le correctif d’alignement), le résultat est vide **même si** le Markdown parle du projet 4405 dans le vecteur. D’où l’obligation de ré-ingérer après un changement de schéma LangExtract / align.

---

## 13. Invariants à retenir

1. **Une chaîne Markdown**, trois systèmes de coordonnées alignés : pages (`PageSpan`), chunks (`start`/`end`), LangExtract (`char_interval`).
2. **Le vecteur encode le texte** ; **le payload encode l’identité** (projet, type, dates locales).
3. **Le n° de projet est une propriété du document**, pas du chunk : recopié partout pour le filtre.
4. **`location` n’est pas dans Qdrant** aujourd’hui : une carte / un lot cadastral exigera un autre magasin (fiche SQL prévue) ou une extension d’alignement.
5. **Changer `EMBED_MODEL`** impose une nouvelle collection ou un ré-ingest total : cosine entre deux espaces vectoriels différents n’a pas de sens.
6. **LangExtract peut échouer** sans tuer l’indexation sémantique ; les filtres `entities=` seront alors inopérants.
7. **OCR heavy** : le Markdown peut coller des colonnes de tableaux ou mélanger des en-têtes (`Projet n°: 5370` parasite). LangExtract et les embeddings héritent de ce bruit.

---

## 14. Fichiers source (carte)

| Fichier | Responsabilité |
|---|---|
| `src/rag_ingestion/cli.py` | `ingest` / `query` |
| `src/rag_ingestion/pipeline.py` | Enchaînement 6 étapes, skip, fiche parent |
| `src/rag_ingestion/config.py` | `.env` + défauts |
| `src/rag_ingestion/parse.py` | SHA-256, LiteParse inspect/parse, `page_for_span` |
| `src/rag_ingestion/chunk.py` | Fenêtres, tableaux, `heading_path` |
| `src/rag_ingestion/extract.py` | Prompt, few-shots, Ollama, JSONL/HTML |
| `src/rag_ingestion/align.py` | Recouvrement, `project_id` global, payload |
| `src/rag_ingestion/embed.py` | FastEmbed, dimension, query vs passage |
| `src/rag_ingestion/qdrant_store.py` | Collection, indexes, delete, upsert, filtres |
| `src/rag_ingestion/retrieve.py` | Façade embed + search (CLI et chatbot) |
| `src/rag_ingestion/models.py` | Dataclasses |
| `config/langextract/prompt.txt` | Schéma d’extraction |
| `config/langextract/few_shots.json` | Exemples (audit générique + ÉES 4405 / 2259) |

---

## 15. Commandes de référence

```bash
docker compose up -d
# Ollama : modèle LANGEXTRACT_MODEL déjà pull

rag-ingest ingest chemin/vers/rapport.pdf
rag-ingest ingest chemin/vers/rapport.pdf --skip-extract

rag-ingest query "contamination des sols projet 4405" --limit 5
rag-ingest query "contamination" --filter entities=4405 --filter doc_type=ees_phase_2
```

Après modification du prompt LangExtract, des few-shots, ou de `align.py` (recopie des `project_id`), **ré-ingérer** les PDF concernés : l’ancien payload Qdrant ne se met pas à jour tout seul.

---

## 16. Prochaines étapes (plan, non implémenté)

Cette section reprend la vision produit et le plan technique déjà discutés. Elle n’est **pas** un état du code. Tant qu’un item n’est pas livré, le comportement réel reste celui des § 1–15.

### 16.1 Vision produit

L’interface cible a **trois panneaux** qui partagent un même état `focus` :

| Panneau | Question à laquelle il répond | Source de vérité |
|---|---|---|
| **Chatbot** | Quel passage du corpus répond à la question ? | Qdrant (chunks + vecteurs + payload) |
| **Carte** | D’où vient ce rapport ? Quelle zone géographique ? | Fiche **site** (lot / adresse / lat-lon) |
| **Historique (timeline)** | Quels travaux ont déjà été faits **sur cette zone** ? | Table **events** (dates typées) + documents du même `site_id` |

Les sorties du chat **mettent à jour** la carte et la timeline. Un clic sur la carte ou un événement de timeline **resserre** le prochain tour RAG (filtre `document_id` / `project_id` / plus tard `site_id`).

Cas métier qui justifie tout le reste : les rapports **2259** (Géosphère, 2024) et **4405** (Enviro-Experts, 2023) parlent du **même site** (même adresse, même lot cadastral, même campagne de forages 17 août 2019) mais sont **deux dossiers distincts**. Sur la carte ils doivent **fusionner** (un pin). En timeline ils doivent **rester distincts** (deux rapports, deux firmes, deux dates de remise). Un LLM qui « dump un JSON d’entités » à la fin de la réponse mélange déjà ces deux n° dans le fil de conversation : on ne lui confie **pas** cette vérité.

### 16.2 Principe : trois couches, une seule vérité

| Couche | Rôle | Qui l’écrit | Qui la lit |
|---|---|---|---|
| **Fiche document** | Identité du PDF : projet, firme, client, type, dates typées, lien vers le site | **Ingest** (LangExtract + normalisation), jamais le chatbot | API documents, enveloppe `POST /chat`, timeline |
| **Fiche site** | Zone géographique + liste des documents de ce lieu | **Ingest** (lot / adresse → `site_id` + géocodage **une fois**) | Carte, `GET /sites`, jointure timeline |
| **Focus de tour** | Quels docs / projets / sites le chat vient d’utiliser **dans ce tour** | API chat, dérivé des **hits Qdrant** (ToolMessage), **pas** du texte généré par le LLM | UI : zoom carte, filtre timeline, filtre RAG du tour suivant |

Règle d’or : le JSON d’entités que l’UI affiche **n’est jamais généré par le modèle de chat**. Il est lu dans la DB (ou, en attendant, dans une fiche disque enrichie) après jointure sur les `document_id` des hits.

### 16.3 Pourquoi une base SQL à l’ingest (et pas « tout dans Qdrant »)

Qdrant répond à : *quel passage ressemble à la question ?*  
La carte et la timeline répondent à : *où est ce rapport, quand a-t-on travaillé sur ce lot, quels autres rapports sont sur la même zone ?*

Un vecteur ne fait pas bien :

- une **jointure** par lot cadastral (2259 et 4405 → un seul site) ;
- un **tri** par date typée (`report` vs `fieldwork`) ;
- une requête **bbox** / rayon pour peupler la carte.

Aujourd’hui :

- `data/documents/{id}.json` = journal d’ingest (chemins, compteurs, timings) — **pas** une fiche métier ;
- `data/extractions/{id}.jsonl` = brut LangExtract, pas interrogeable par l’UI ;
- `location` est extraite puis **jetée** à l’alignement (§ 7.3) ;
- les dates dans le payload Qdrant n’ont **pas** de `role` (rapport vs forage).

La DB **matérialise** ce que LangExtract a déjà (ou devrait) extraire. Ce n’est **pas** un second RAG.

Ce qu’il **ne faut pas** y mettre :

- le texte des chunks (ça reste Qdrant) ;
- des écritures provenant du chatbot (hallucinations, mélange 2259/4405) ;
- un géocodage à **chaque** question (une fois à l’ingest, puis lecture).

Flux cible :

```
PDF → parse → chunks → LangExtract → align
  → upsert fiche SQL (sites / documents / events) + géocodage si besoin
  → embeddings → Qdrant (inchangé dans son rôle)
```

Si LangExtract rate le lot, la fiche peut être incomplète ; les chunks restent cherchables. La qualité de la DB = qualité OCR + LangExtract + règles de normalisation.

### 16.4 Quelle base

Pour ce dépôt **local** : **SQLite**, un fichier sous `data/` (ex. `data/catalog.sqlite`), zéro service Docker de plus.

Plus tard, si la carte exige rayon, polygone de lot, index spatial : **Postgres + PostGIS**. Inutile pour le premier jet (un point `lat`/`lon` par site).

Éviter de « tout mettre dans le payload Qdrant » : payload = **filtre RAG** ; DB = **vérité métier + UI**. On pourra **aussi** recopier `site_id` dans le payload Qdrant plus tard, uniquement pour filtrer `search_knowledge` par zone — ce n’est pas un substitut aux tables.

### 16.5 Schéma SQL minimal

Trois tables, pas une mega-row par PDF.

#### `sites` — la zone (clé carte + timeline)

Un site = un lieu physique. Deux rapports sur le même lot = **une** ligne.

| Colonne | Rôle |
|---|---|
| `site_id` | Clé primaire. Convention : `lot:{lot_normalisé}` si le lot est connu, sinon hash stable de l’adresse normalisée (minuscules, accents, ponctuation). |
| `lot_cadastral` | Chiffres seuls (`2363352`), jamais `2 363 352` ni `lot 2 363 352`. |
| `address` | Adresse du **site étudié**, pas celle du bureau de la firme. |
| `city` | Ville / municipalité si extraite. |
| `lat`, `lon` | Géocodage à l’ingest ; `NULL` si échec ou adresse trop pauvre. |
| `geocode_status` | Ex. `ok` / `failed` / `skipped` / `pending`. |
| `updated_at` | Dernier upsert. |

Règle de fusion : après normalisation du lot (ou de l’adresse), `INSERT … ON CONFLICT(site_id)` : on **réutilise** le site existant. On ne géocode **que** si `lat`/`lon` sont encore nuls (éviter de rappeler l’API de géocodage à chaque ré-ingest).

#### `documents` — le rapport

| Colonne | Rôle |
|---|---|
| `document_id` | SHA-256 actuel du PDF (même id que Qdrant). PK. |
| `project_id` | N° dossier (`4405`). Un document = un projet principal ; si plusieurs n° extraits, documenter la règle (premier / liste séparée) au moment de l’implémentation. |
| `title` | Titre LangExtract (`title`), aujourd’hui **perdu** pour Qdrant. |
| `doc_type` | Même normalisation qu’aujourd’hui (`ees_phase_2`, …). |
| `firm` | Organisme rédacteur (`entity` type `firm`). |
| `client` | Client (`entity` type `client`). |
| `site_id` | FK vers `sites`. |
| `source_path` | Chemin du PDF. |
| `parse_quality` | `ok` / `ocr_partial` / `ocr_heavy`. |
| `report_date` | Date de **remise** du rapport (`role=report`), si extraite. |
| `ingested_at` | Horodatage d’ingest. |

2259 et 4405 → **deux** lignes `documents`, **un** `site`.

Ré-ingest du même PDF : `ON CONFLICT(document_id) DO UPDATE` (symétrique du `delete_by_document_id` Qdrant). Les `events` de ce `document_id` sont remplacés (delete puis insert, ou upsert par clé composite).

#### `events` — la timeline

| Colonne | Rôle |
|---|---|
| `id` | PK interne (autoincrement ou UUID). |
| `document_id` | FK. |
| `site_id` | FK (dénormalisé pour `GET /sites/{id}/timeline` sans jointure lourde). |
| `iso_date` | `YYYY-MM-DD`. |
| `role` | `report` \| `fieldwork` \| `phase1` \| `sampling` \| … — **obligatoire**. Une date sans rôle est inutilisable en timeline (mélange « 3 jan 2024 » et « 17 août 2019 »). |
| `label` | Libellé optionnel (ex. « Campagne de forages »). |

Sans `role` dans LangExtract (prompt + few-shots), cette table n’a pas de sens. C’est un prérequis d’extraction, pas seulement de SQL.

### 16.6 Travail LangExtract / align à faire **avant** ou **avec** la DB

Aujourd’hui le prompt extrait `location` et parfois `date.role` dans les few-shots, mais l’alignement **ignore** `location` et ne type pas les dates dans le payload.

À ajouter côté ingest (code + prompt) :

1. **Adresse et lot** : classes `location` (`address`, `lot`) agrégées au **niveau document** (comme `project_id`), pas seulement sur le chunk qui contient la phrase. Choisir le site étudié, pas le siège social de la firme (règle prompt + few-shots ÉES : « 619, route 341 » vs adresse Enviro-Experts / Géosphère).
2. **`site_id`** calculé après normalisation (lot prioritaire).
3. **Dates avec `role`** : `report`, `fieldwork`, etc. Recopiées dans `events`, pas seulement une liste plate `dates[]` dans Qdrant.
4. **`title`, `firm`, `client`** : aujourd’hui `title` est dans `_DOC_LEVEL` mais **pas** dans `ChunkPayload` ; `firm`/`client` sont des `entity` locales. Il faut un agrégat **document** (première occurrence fiable, ou vote) pour remplir `documents`.
5. Optionnel plus tard : recopier `site_id` dans le payload Qdrant + index KEYWORD, pour `search_knowledge` filtré par zone.

Tant que 2259 et 4405 n’ont **pas** le même `site_id` après ré-ingest, **ne pas** enchaîner l’UI carte.

### 16.7 Géocodage

- Appelé **uniquement à l’ingest**, et **uniquement** si le site n’a pas encore de coordonnées.
- Entrée : adresse du site (+ ville / province si besoin). Le lot seul ne géocode souvent pas ; l’adresse oui.
- Sortie : `lat`/`lon` + `geocode_status`. Échec → site quand même créé, pin absent ou fallback (ville).
- Ne pas géocoder depuis le chatbot.

Le fournisseur (Nominatim, service interne, etc.) sera choisi à l’implémentation ; le contrat est : **un point par `site_id`**, pas un point par document.

### 16.8 Chaîne d’ingest cible (étape supplémentaire)

Entre l’alignement actuel (étape 4/6) et les embeddings (5/6), ou juste après l’alignement :

1. Agréger les extractions **niveau document** (projet, titre, type, firme, client, adresse, lot, dates+rôles).
2. Normaliser le lot / l’adresse → trouver ou créer `sites`.
3. Géocoder si `lat`/`lon` absents.
4. Upsert `documents`.
5. Remplacer les `events` de ce `document_id`.
6. Continuer embeddings + Qdrant comme aujourd’hui.

La fiche `data/documents/{id}.json` peut rester un **journal d’ingest** (timings, chemins) **ou** être enrichie pour coller au schéma ci-dessus en attendant que l’API lise SQLite. La DB est la source pour l’UI ; le JSON disque ne doit pas diverger (soit il devient un dump de la ligne SQL, soit l’API ignore le JSON métier).

### 16.9 Lien avec le chat (enveloppe HTTP)

`POST /chat` **ne remplit pas** la DB. Après `graph.invoke` :

1. Lire les ToolMessage / hits **retenus** de ce tour (respecter le plafond d’outils : union des hits du tour, pas 80 recherches).
2. Extraire les `document_id`.
3. **Joindre** SQLite → fiches + `site_id`.
4. Répondre :

```json
{
  "reply": "…texte du modèle…",
  "thread_id": "…",
  "focus": {
    "document_ids": ["f2138…", "f95b…"],
    "project_ids": ["4405", "2259"],
    "site_ids": ["lot:2363352"]
  },
  "documents": [ { "fiches lues en DB" } ]
}
```

L’UI : le chat affiche `reply` ; map et timeline se branchent sur `focus` + `documents`. Un clic carte/timeline met à jour `focus` et le prochain `search_knowledge` (filtres).

Le REPL `rag-chat` peut afficher le JSON `focus` sous `assistant>` pour debug, sans attendre l’UI.

### 16.10 API lecture (sans passer par le chat)

Une fois la DB peuplée :

| Endpoint | Usage |
|---|---|
| `GET /documents/{document_id}` | Fiche rapport |
| `GET /sites/{site_id}` | Géométrie + `document_ids` |
| `GET /sites/{site_id}/timeline` | `events` triés par `iso_date` (tous les travaux de la zone) |
| `GET /sites?bbox=` | GeoJSON pour peupler la carte |

Clic pin → `site_id` → timeline + filtre chat.  
Clic événement → `document_id` / `project_id` → même filtre.

Pas de second index vectoriel « pour la carte ».

### 16.11 Ordre d’implémentation (le plus sûr)

1. **Fiche document + `site_id` + dates typées** (SQLite `sites` / `documents` / `events`), **sans UI**. Ré-ingest des deux ÉES. Vérification manuelle : **même** `site_id`, **deux** `project_id`, deux dates de rapport distinctes, même date de forage si c’est le cas dans les PDF.
2. **Géocodage à l’ingest** + `GET /sites` (point lat/lon).
3. **Enveloppe `POST /chat`** (`focus` + fiches). REPL : JSON sous la réponse.
4. **`GET /sites/{id}/timeline`**.
5. **UI** trois panneaux branchée sur `focus`.

Tant que l’étape 1 est fausse, la carte collera deux adresses de bureaux ou ignorera le lot, et l’historique mélangera date de rapport et date de chantier.

### 16.12 Critère de succès mental (déjà testable une fois l’étape 1 livrée)

Question « contamination 4405 » → `focus.project_ids = ["4405"]`, `site_ids = ["lot:…"]` → carte sur L’Épiphanie → timeline : forages 17 août 2019, rapport Enviro-Experts 3 jan 2023 — **et**, si l’UI affiche « tous les travaux du site » plutôt que « seulement le projet de la question », le 2259 Géosphère 3 jan 2024 sur la **même** zone.

### 16.13 Rappel : état actuel vs plan

| Élément | Aujourd’hui (code) | Cible |
|---|---|---|
| Qdrant chunks + `entities` (project_id global) | Oui | Conservé |
| `location` dans payload / DB | Extraite, **ignorée** | Niveau document + `sites` |
| Dates avec `role` | Rarement dans few-shots, **pas** en payload | Table `events` |
| `data/documents/*.json` | Journal d’ingest | + fiche métier **ou** remplacé par SQLite |
| SQLite `sites` / `documents` / `events` | **Non** | Écrit à l’ingest |
| Géocodage | **Non** | Une fois par site |
| `POST /chat` → `focus` | `{ reply, thread_id }` seulement | Hits → jointure DB |
| UI carte / timeline | **Non** | Après API lecture |
