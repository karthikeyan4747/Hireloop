"""Shared helpers for job sources: HTTP + cache, HTML stripping, ids, matching."""
from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx

from ..config import Settings
from ..models import Job
from ..store import Store

log = logging.getLogger("hireloop.sources")

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|br|li|ul|ol|div|h[1-6]|tr)[^>]*>", re.I)
_WS = re.compile(r"[ \t\r\f\v]+")
_STOP = {"a", "an", "and", "the", "of", "in", "for", "at", "to", "or", "with", "job", "jobs"}


class SourceError(Exception):
    """A source could not be queried. Message is shown to the user."""


def make_job_id(source: str, source_id: str) -> str:
    return hashlib.sha256(f"{source}:{source_id}".encode()).hexdigest()[:16]


def strip_html(raw: str | None) -> str:
    """HTML (possibly entity-escaped, as Greenhouse returns it) -> plain text."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _BLOCK.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    lines = (_WS.sub(" ", ln).strip() for ln in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_dt(value: Any) -> datetime | None:
    """ISO strings or epoch milliseconds -> aware datetime; None if unparseable."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        if isinstance(value, str) and value:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        pass
    return None


def query_tokens(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", query.lower()) if t not in _STOP and len(t) > 1]


def relevance(title: str, description: str, tokens: list[str]) -> int:
    """Crude keyword relevance used to filter per-company boards. 0 = not relevant."""
    if not tokens:
        return 1
    t, d = title.lower(), description.lower()
    title_hits = sum(tok in t for tok in tokens)
    desc_hits = sum(tok in d for tok in tokens)
    if title_hits:
        return 10 * title_hits + desc_hits
    return desc_hits if desc_hits * 2 >= len(tokens) else 0


async def get_json(
    client: httpx.AsyncClient,
    settings: Settings,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET JSON. Raises SourceError with a readable message."""
    try:
        resp = await client.get(url, params=params, timeout=settings.http_timeout)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        raise SourceError("timed out") from None
    except httpx.HTTPStatusError as exc:
        raise SourceError(f"HTTP {exc.response.status_code}") from None
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceError(f"request failed ({type(exc).__name__})") from None
    return data


async def cached(store: Store, settings: Settings, key: str, producer: Callable[[], Awaitable[Any]]) -> Any:
    """6-hour SQLite cache around an async producer returning JSON-serialisable data.

    We cache the already-trimmed jobs rather than the raw payload: Greenhouse boards
    with content=true can be several MB each.
    """
    hit = store.cache_get(key, settings.cache_ttl_hours)
    if hit is not None:
        return hit
    data = await producer()
    store.cache_put(key, data)
    return data


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and corporate suffixes for dedupe keys."""
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    t = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|pvt|gmbh|co)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


MAX_DESC = 8000  # characters kept per description


def clip(text: str) -> str:
    return text[:MAX_DESC]


async def cached_jobs(
    store: Store, settings: Settings, key: str, producer: Callable[[], Awaitable[list[Job]]]
) -> list[Job]:
    """`cached`, but for lists of Job models."""

    async def _dump() -> list[dict]:
        return [j.model_dump(mode="json") for j in await producer()]

    return [Job.model_validate(d) for d in await cached(store, settings, key, _dump)]
