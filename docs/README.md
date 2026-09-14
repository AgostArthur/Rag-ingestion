# Documentation

Le [README](../README.md) explique comment lancer le produit. Ici : comment le système est câblé.

| Fichier | Contenu |
|---|---|
| [ingestion.md](ingestion.md) | Pipeline PDF → catalog SQLite → Qdrant, drop folder, recherche hybride |
| [api.md](api.md) | HTTP `rag-chat`, enveloppe `focus`, interface `ui/` |

**Règle :** l’ingest est la seule écriture du corpus (Qdrant + SQLite). Le chat et l’UI **lisent**. Le modèle de chat n’invente pas les sites, dates, ni coordonnées.
