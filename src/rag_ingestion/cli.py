"""CLI `rag-ingest` : sous-commandes `ingest` et `query`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_ingestion.config import load_settings
from rag_ingestion.logging_setup import StepTimer, configure_logging, format_duration
from rag_ingestion.pipeline import ingest_path
from rag_ingestion.retrieve import search as retrieve_search

logger = logging.getLogger(__name__)


def _parse_filters(raw: list[str]) -> dict[str, str]:
    """Parse `--filter clé=valeur` répétés en dictionnaire.

    Raises:
        SystemExit: Syntaxe invalide (pas de `=`).
    """
    out: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid filter « {item} » (expected key=value)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid filter « {item} »")
        out[key] = value.strip()
    return out


def cmd_ingest(args: argparse.Namespace) -> int:
    """Sous-commande `ingest` : indexe un PDF et affiche le résumé.

    Returns:
        0 si OK, 2 si l'ingest a été ignoré (Markdown/chunks vides).
    """
    settings = load_settings()
    result = ingest_path(
        Path(args.path),
        settings=settings,
        skip_extract=args.skip_extract,
    )
    if result.skipped:
        print(f"Skipped document_id={result.document_id}: {result.skip_reason}")
        for w in result.warnings:
            print(f"warning: {w}")
        print(f"total time: {format_duration(result.total_seconds)}")
        return 2
    print(
        f"OK document_id={result.document_id} "
        f"chunks={result.n_chunks} extractions={result.n_extractions} "
        f"quality={result.parse_quality} collection={settings.qdrant_collection}"
    )
    for w in result.warnings:
        print(f"warning: {w}")
    print(f"total time: {format_duration(result.total_seconds)}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Sous-commande `query` : recherche dense + filtres, impression des hits.

    Returns:
        Toujours 0 (y compris si aucun résultat).
    """
    settings = load_settings()
    filters = _parse_filters(args.filter or [])
    timer = StepTimer(logger)
    logger.info("Search in « %s »: %s", settings.qdrant_collection, args.text[:120])
    if filters:
        logger.info("  filters: %s", filters)
    with timer.step(
        "retrieve",
        f"Embedding the query and retrieving the {args.limit} closest chunks…",
    ):
        hits = retrieve_search(
            args.text,
            limit=args.limit,
            filters=filters,
            settings=settings,
        )
    logger.info("  %s hit(s) (%s).", len(hits), timer.took("retrieve"))
    for line in timer.summary_lines():
        logger.info("%s", line)
    if not hits:
        print("No results.")
        return 0
    for i, hit in enumerate(hits, start=1):
        heading = hit.get("heading_path") or ""
        print(
            f"[{i}] score={hit['score']:.4f} page={hit.get('page')} "
            f"doc_type={hit.get('doc_type')} entities={hit.get('entities')}"
        )
        if heading:
            print(f"    section: {heading}")
        print(f"    source: {hit.get('source_path')}")
        text = (hit.get("text") or "").replace("\n", " ")
        print(f"    {text[:400]}{'…' if len(text) > 400 else ''}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parser argparse des sous-commandes `ingest` et `query`."""
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Local RAG ingest (LiteParse → chunks → LangExtract + embeddings → Qdrant).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse a PDF and index it in Qdrant")
    ingest.add_argument("path", help="Path to the PDF")
    ingest.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip LangExtract (chunks + embeddings only)",
    )
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Dense search with payload filters")
    query.add_argument("text", help="Question or search phrase")
    query.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="key=value",
        help="Payload filter (doc_type, entities, dates, document_id, page). Repeatable.",
    )
    query.add_argument("--limit", type=int, default=5)
    query.set_defaults(func=cmd_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée `rag-ingest` / `python -m rag_ingestion`.

    Args:
        argv: Arguments CLI ; `sys.argv[1:]` si omis.

    Returns:
        Code de sortie (0 OK, 1 erreur, 2 skip ingest, 130 Ctrl-C).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("Failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
