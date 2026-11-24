"""Job sources. Each module exposes NAME and `async fetch(client, store, settings, query, location)`."""
from __future__ import annotations

from types import ModuleType

from ..config import Settings
from . import adzuna, greenhouse, lever, remotive


def enabled_sources(settings: Settings) -> list[ModuleType]:
    sources = [remotive, greenhouse, lever]
    if settings.adzuna_enabled:
        sources.append(adzuna)
    return sources
