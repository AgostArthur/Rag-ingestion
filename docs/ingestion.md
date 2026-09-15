# Ingestion

Transformer un PDF (souvent un scan d’ÉES / ESA Québec) en chunks cherchables **et** en fiches métier. Deux rapports du **même site** (ex. dossiers **4405** Enviro-Experts 2023 et **2259** Géosphère 2024, 619 Route 341, L’Épiphanie, forages du 17 août 2019) doivent **fusionner** sur la carte et **rester distincts** en timeline.

Le LLM de chat ne fait pas ce travail. LangExtract + normalisation + SQLite le font **à l’ingest**.

## Couches

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

- fichier `.env` à la racine du dépôt (complète les variables absentes) ;
- variables déjà posées par le shell ou Compose **non écrasées** ;
- `localhost` / `host.docker.internal` pour Ollama sont réécrits selon que le process est dans Docker ou non ;
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
| `LANGEXTRACT_PROFILES_FILE` | `config/langextract/profiles.json` | Routage des schémas (phase I / II / default) |
| `LANGEXTRACT_PROMPT_FILE` | `config/langextract/prompt.base.txt` | Repli si `profiles.json` est absent |
| `LANGEXTRACT_FEW_SHOTS_FILE` | `config/langextract/profiles/default/few_shots.json` | Repli si `profiles.json` est absent |
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

### 6.2 Schéma (prompt + few-shots par profil)

Le code ne durcit **aucune** classe Python. Le schéma est 100 % dans `config/langextract/` :

- `prompt.base.txt` — classes autorisées, ancrage verbatim, règle `project_id` (partagé)
- `profiles.json` — ordre des profils et motifs sur le **nom de fichier** (puis le premier heading Markdown)
- `profiles/<id>/addendum.txt` — consigne métier du profil
- `profiles/<id>/few_shots.json` — exemples d’alignement

À l’ingest, `resolve_extract_schema` concatène base + addendum et charge les few-shots du profil. Priorité : `--profile` > nom du PDF > premier heading > `default`. Chaque résolution est loguée (`type=` + `fichier=`) ; un nom sans motif connu déclenche un **warning** et le schéma générique.

Ordre dans `profiles.json` : **ees_phase_2 avant ees_phase_1**, sinon `phase II` serait lu comme `phase I`. Les ids de profil (`--profile`, log `type=`, filtre Qdrant `doc_type`) sont **`ees_phase_1` / `ees_phase_2` / `default`**. Les dossiers restent `profiles/phase_1/` et `profiles/phase_2/`.

Changer de domaine = ajouter un dossier sous `profiles/` et une entrée dans `profiles.json`, sans toucher `extract.py`.

Règle LangExtract : chaque `extraction_text` d’un few-shot doit apparaître **tel quel** comme sous-chaîne du champ `text` de l’exemple (alignement caractère).

Classes actuellement autorisées par le prompt (ÉES / ESA Québec) :

| Classe | Rôle | Attributs typiques |
|---|---|---|
| `title` | Titre principal | — |
| `doc_type` | Type de document | `normalized` : `rapport`, `ees_phase_1`, `ees_phase_2`, `contrat`, `facture`, `note`, `guide`, `autre` |
| `entity` | Personne, org, n° de projet, lieu nommé | `type` : `person`, `org`, `place`, `project_id`, `client`, `firm` ; pour un projet `normalized` = chiffres (`4405`) |
| `date` | Date mentionnée | `normalized` ISO `YYYY-MM-DD` ; `role` : `contract` \| `fieldwork` \| `report` |
| `topic` | Thème (phase, reco…) ; contamination **détectée** | `kind=contaminant` → persisté dans `contaminants` (document), pas seulement dans `topics` |
| `location` | Adresse et lot du site analysé | `type` : `address`, `lot` |

Consigne métier importante : un n° de projet (`Projet nº`, `No/Réf.`) doit être extrait en `entity` / `type=project_id` avec `extraction_text` = les **chiffres** tels qu’écrits (`2259`, `4405`), pas une année (`2019`). Les few-shots phase II illustrent 4405 / 2259 ; phase I a ses propres exemples ; le profil `default` garde un rapport d’audit générique.

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

Les `project_id` **non ancrés** (`start is None`) sont quand même recopiés au niveau document. Les contaminations détectées (`topic` + `kind=contaminant`) suivent la même règle : liste unique recopiée sur **tous** les chunks (`contaminants`), et **pas** mélangées dans `topics`. Les autres entités non ancrées (`Ghost` sans intervalle) **ne sont pas** copiées.

### 7.3 Niveau local (seulement si recouvrement)

| Classe | Champ payload | Label |
|---|---|---|
| `entity` (hors project_id) | `entities` | `normalized` ou texte |
| `date` | `dates` | idem (rôle **non** recopié ici ; voir catalog `events`) |
| `topic` (hors contaminant) | `topics` | idem |
| `topic` + `kind=contaminant` | `contaminants` | niveau document, tous les chunks |
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

```text
uuid5(namespace=6ba7b810-9dad-11d1-80b4-00c04fd430c8, name="{document_id}:{chunk_index}")
```

Ré-upsert du même chunk (même document, même index) écrase le même id, pas de doublon d’id même si le delete avait échoué partiellement.

Écriture par lots de 64 points (`wait=True`).

### 9.3 Contenu exact du payload

Pour chaque chunk :

```text
`document_id`    SHA-256 des **octets** du PDF (`src/rag_ingestion/parse.py`). Même contenu = même id. Ré-ingest : delete Qdrant par `document_id` puis upsert ; events SQL de ce document remplacés.
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
site_id          lot:… ou addr:… ou null
project_id       n° de projet principal ou null
contaminants     liste de str (contaminations détectées, niveau document)
```

Artefacts : `data/parsed/{id}.md`, `data/extractions/{id}.jsonl`, `data/documents/{id}.json`, ligne `documents` dans SQLite.

`site_id` : `lot:{chiffres}` si le lot est connu, sinon `addr:{hash}` de l’adresse normalisée (`src/rag_ingestion/normalize.py`). Deux PDF avec le même lot (ou la même adresse rattachée ensuite) partagent un site.

## Chaîne (`ingest_path`, 7 étapes)

Orchestration : `src/rag_ingestion/pipeline.py`.

```

Quand LangExtract a tourné, `_write_parent` recopie aussi l’identité catalog : `site_id`, `project_id(s)`, `title`, `doc_type`, firme/client, adresse/lot, `report_date`, `contaminants`, `events` (`contract` / `fieldwork` / `report`). La vérité métier reste SQLite (`catalog.sqlite`) ; ce JSON est un journal d’ingest + copie locale.

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

Quality gates : Markdown vide ou 0 chunk → skip (CLI code `2`), pas d’upsert. LangExtract en exception → warning et **poursuite** (chunks + embeddings sans métadonnées). `--skip-extract` idem sans appeler Ollama.

| Fichier | Responsabilité |
|---|---|
| `src/rag_ingestion/cli.py` | `ingest` / `query` |
| `src/rag_ingestion/pipeline.py` | Enchaînement 6 étapes, skip, fiche parent |
| `src/rag_ingestion/config.py` | `.env` + défauts |
| `src/rag_ingestion/parse.py` | SHA-256, LiteParse inspect/parse, `page_for_span` |
| `src/rag_ingestion/chunk.py` | Fenêtres, tableaux, `heading_path` |
| `src/rag_ingestion/extract.py` | Chargement prompt/few-shots, Ollama, JSONL/HTML |
| `src/rag_ingestion/extract_profile.py` | Routage du profil LangExtract (nom de fichier / heading / `--profile`) |
| `src/rag_ingestion/align.py` | Recouvrement, `project_id` global, payload |
| `src/rag_ingestion/embed.py` | FastEmbed, dimension, query vs passage |
| `src/rag_ingestion/qdrant_store.py` | Collection, indexes, delete, upsert, filtres |
| `src/rag_ingestion/retrieve.py` | Façade embed + search (CLI et chatbot) |
| `src/rag_ingestion/models.py` | Dataclasses |
| `config/langextract/prompt.base.txt` | Socle du prompt (classes, `project_id`) |
| `config/langextract/profiles.json` | Routage phase I / II / default |
| `config/langextract/profiles/*/addendum.txt` | Consigne métier du profil |
| `config/langextract/profiles/*/few_shots.json` | Exemples du profil |

---

## 15. Commandes de référence

```bash
docker compose up -d
# Ollama : modèle LANGEXTRACT_MODEL déjà pull

rag-ingest ingest chemin/vers/rapport.pdf
rag-ingest ingest chemin/vers/rapport.pdf --profile ees_phase_1
rag-ingest ingest chemin/vers/rapport.pdf --skip-extract

rag-ingest query "contamination des sols projet 4405" --limit 5
rag-ingest query "contamination" --filter entities=4405 --filter doc_type=ees_phase_2
rag-ingest query "HAM" --filter contaminants=HAM --filter doc_type=ees_phase_2
```

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
| `contaminants` | JSON des contaminations **détectées** (`topic` + `kind=contaminant`). |
| `ingested_at` | Horodatage d’ingest. |

2259 et 4405 → **deux** lignes `documents`, **un** `site`.

Ré-ingest du même PDF : `ON CONFLICT(document_id) DO UPDATE` (symétrique du `delete_by_document_id` Qdrant). Les `events` de ce `document_id` sont remplacés (delete puis insert, ou upsert par clé composite).

| Code | `rag-ingest ingest` |
|---|---|
| 0 | Au moins un upsert Qdrant |
| 1 | Erreur (fichier, Qdrant, …) |
| 2 | Skip (Markdown / chunks vides) |
| 130 | Ctrl-C |

Après un changement de prompt LangExtract, de few-shots, ou de modèle d’embedding : **ré-ingérer** (éventuellement `watch --force` pour un SHA déjà au catalog).
