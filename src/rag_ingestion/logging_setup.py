"""Logging stdout et chronométrage des étapes du pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

_LOG_FORMAT = "%(asctime)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Active le logging lisible (heure + message).

    Args:
        level: Niveau racine (INFO par défaut).
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
    # SDK Gemini (via LangExtract) et HTTP : WARNING+ seulement.
    for name in ("google", "google_genai", "google.genai", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


def format_duration(seconds: float) -> str:
    """Durée lisible : millisecondes, secondes ou minutes."""
    if seconds < 0.001:
        return "<1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.1f} s"


class StepTimer:
    """Chronomètre les étapes et journalise le début / la fin de chacune."""

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger
        self.steps: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def step(self, name: str, start: str) -> Iterator[None]:
        """Log le début, enregistre la durée. L'appelant logue le résultat."""
        self._log.info("%s", start)
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - t0
            self.steps[name] = elapsed
            self._log.error("Failed — %s (%s)", name, format_duration(elapsed))
            raise
        self.steps[name] = time.perf_counter() - t0

    def took(self, name: str) -> str:
        """Durée formatée d'une étape déjà mesurée."""
        return format_duration(self.steps.get(name, 0.0))

    def total_seconds(self) -> float:
        """Secondes écoulées depuis la création du timer."""
        return time.perf_counter() - self._t0

    def summary_lines(self) -> list[str]:
        """Lignes d'un récapitulatif « étape → durée » + total."""
        width = max((len(k) for k in self.steps), default=8)
        width = max(width, len("total"))
        lines = ["Runtime:"]
        for name, seconds in self.steps.items():
            lines.append(f"  {name:<{width}}  {format_duration(seconds)}")
        lines.append(f"  {'total':<{width}}  {format_duration(self.total_seconds())}")
        return lines
