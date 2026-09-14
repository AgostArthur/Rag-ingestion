"""CLI `rag-ingest` : sous-commandes `ingest`, `query` et `watch`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_ingestion.config import load_settings
from rag_ingestion.inbox import watch_inbox
from rag_ingestion.logging_setup import StepTimer, configure_logging, format_duration
from rag_ingestion.pipeline import collect_pdf_paths, ingest_path
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


def _print_ingest_result(result, collection: str) -> None:
    if result.error:
        print(f"error: {result.error} document_id={result.document_id or '?'}")
        return
    if result.skipped:
        print(f"Skipped document_id={result.document_id}: {result.skip_reason}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        print(f"total time: {format_duration(result.total_seconds)}")
        return
    extras = []
    if result.site_id:
        extras.append(f"site_id={result.site_id}")
    if result.project_id:
        extras.append(f"project_id={result.project_id}")
    extra = (" " + " ".join(extras)) if extras else ""
    print(
        f"OK document_id={result.document_id} "
        f"chunks={result.n_chunks} extractions={result.n_extractions} "
        f"quality={result.parse_quality} collection={collection}{extra}"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"total time: {format_duration(result.total_seconds)}")


def cmd_ingest(args: argparse.Namespace) -> int:
    """Sous-commande `ingest` : indexe un PDF ou un dossier de PDF.

    Returns:
        0 si au moins un upsert Qdrant, 2 si tout a été ignoré, 1 si une
        erreur fatale (fichier absent) ou un item du lot a levé.
    """
    settings = load_settings()
    try:
        paths = collect_pdf_paths(Path(args.path))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    any_ok = False
    any_error = False
    any_skipped = False
    for index, path in enumerate(paths, start=1):
        if len(paths) > 1:
            print(f"[{index}/{len(paths)}] {path}")
        try:
            result = ingest_path(
                path,
                settings=settings,
                skip_extract=args.skip_extract,
                extract_profile=args.profile,
            )
        except Exception as exc:
            logger.exception("Failed ingesting %s", path)
            print(f"error: {exc}", file=sys.stderr)
            any_error = True
            continue
        _print_ingest_result(result, settings.qdrant_collection)
        if result.error:
            any_error = True
        elif result.skipped:
            any_skipped = True
        else:
            any_ok = True
    if any_error:
        return 1
    if any_ok:
        return 0
    if any_skipped:
        return 2
    return 1


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
            f"doc_type={hit.get('doc_type')} project_id={hit.get('project_id')} "
            f"site_id={hit.get('site_id')} entities={hit.get('entities')}"
        )
        if heading:
            print(f"    section: {heading}")
        print(f"    source: {hit.get('source_path')}")
        text = (hit.get("text") or "").replace("\n", " ")
        print(f"    {text[:400]}{'…' if len(text) > 400 else ''}")
        print()
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Sous-commande `watch` : drop folder `incoming/` → ingest → archive."""
    from dataclasses import replace

    settings = load_settings()
    if args.incoming:
        incoming = Path(args.incoming).expanduser().resolve()
        settings = replace(settings, incoming_dir=incoming)
    skip_extract = True if args.skip_extract else None
    watch_inbox(
        settings=settings,
        skip_extract=skip_extract,
        force=args.force,
        poll_seconds=args.poll,
        once=args.once,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parser argparse des sous-commandes `ingest`, `query` et `watch`."""
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Local RAG ingest (LiteParse → chunks → LangExtract + embeddings → Qdrant).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse a PDF or a directory of PDFs and index in Qdrant")
    ingest.add_argument("path", help="Path to a PDF file or a directory of PDFs")
    ingest.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip LangExtract (chunks + embeddings only)",
    )
    ingest.add_argument(
        "--profile",
        default=None,
        metavar="ID",
        help="LangExtract profile (ees_phase_1, ees_phase_2, default). "
        "Same id as Qdrant filter doc_type. Overrides filename detection.",
    )
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Dense search with payload filters")
    query.add_argument("text", help="Question or search phrase")
    query.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="key=value",
        help="Payload filter (doc_type, entities, dates, document_id, page, site_id, project_id, contaminants). Repeatable.",
    )
    query.add_argument("--limit", type=int, default=5)
    query.set_defaults(func=cmd_query)

    watch = sub.add_parser(
        "watch",
        help="Poll the incoming drop folder, ingest new PDFs, archive them",
    )
    watch.add_argument(
        "--once",
        action="store_true",
        help="Single scan then exit (cron). Default: loop.",
    )
    watch.add_argument(
        "--incoming",
        default=None,
        help="Override INCOMING_DIR",
    )
    watch.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip LangExtract (same as ingest --skip-extract)",
    )
    watch.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the SHA-256 is already in the catalog",
    )
    watch.add_argument(
        "--poll",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Poll interval (default: INGEST_POLL_SECONDS)",
    )
    watch.set_defaults(func=cmd_watch)
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
