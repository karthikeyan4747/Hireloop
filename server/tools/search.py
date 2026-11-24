"""search_jobs: one query across all enabled public sources, deduplicated."""
from __future__ import annotations

import asyncio
import logging

import httpx

from ..deps import get_store, settings, validate_input
from ..models import Job, JobSummary, SearchJobsInput, SearchJobsOutput
from ..policy import Tier, tiered
from ..sources import enabled_sources
from ..sources.base import normalise, query_tokens, relevance

log = logging.getLogger("hireloop.search")

LAST_SEARCH_KEY = "last_search_job_ids"
KEEP_FOR_RANKING = 60  # how many results from the latest search rank_fit will consider


def dedupe(jobs: list[Job]) -> tuple[list[Job], int]:
    """Collapse jobs with the same normalised (company, title, location).

    Keeps the entry with the longest description (most useful for fit scoring).
    Order of first appearance is preserved. Returns (unique_jobs, removed_count).
    """
    best: dict[tuple[str, str, str], Job] = {}
    for j in jobs:
        key = (normalise(j.company), normalise(j.title), normalise(j.location))
        cur = best.get(key)
        if cur is None or len(j.description) > len(cur.description):
            best[key] = j
    return list(best.values()), len(jobs) - len(best)


def apply_filters(jobs: list[Job], location: str | None, remote_only: bool) -> list[Job]:
    out = jobs
    if remote_only:
        out = [j for j in out if j.remote]
    elif location:
        loc = location.lower().strip()
        out = [j for j in out if j.remote or loc in j.location.lower()]
    return out


@tiered(Tier.READ)
async def search_jobs(
    query: str,
    location: str | None = None,
    remote_only: bool = False,
    limit: int = 20,
) -> SearchJobsOutput:
    """Search several public job sources at once (Greenhouse and Lever company boards,
    Remotive, and Adzuna if configured) and return a de-duplicated, compact list.

    No LinkedIn/Naukri/Indeed scraping. Results are cached locally for 6 hours and saved so
    that rank_fit can score them next. Example query: "python backend engineer".
    """
    inp = validate_input(SearchJobsInput, query=query, location=location,
                         remote_only=remote_only, limit=limit)
    cfg, store = settings(), get_store()
    sources = enabled_sources(cfg)

    async with httpx.AsyncClient(
        follow_redirects=True, headers={"User-Agent": "HireLoop/0.1 (personal job search MCP)"}
    ) as client:
        async def run(src):
            try:
                return src.NAME, await src.fetch(client, store, cfg, inp.query, inp.location), None
            except Exception as exc:  # one broken source must never sink the search
                log.warning("source %s failed: %r", src.NAME, exc)
                return src.NAME, [], str(exc) or type(exc).__name__

        results = await asyncio.gather(*(run(s) for s in sources))

    ok = [n for n, _, err in results if err is None]
    failed = {n: err for n, _, err in results if err is not None}
    raw = [j for _, jobs, _ in results for j in jobs]

    unique, removed = dedupe(raw)
    unique = apply_filters(unique, inp.location, inp.remote_only)

    tokens = query_tokens(inp.query)
    unique.sort(
        key=lambda j: (relevance(j.title, j.description, tokens), j.posted_at.timestamp() if j.posted_at else 0),
        reverse=True,
    )

    store.upsert_jobs(unique)
    store.set_meta(LAST_SEARCH_KEY, [j.id for j in unique[:max(inp.limit, KEEP_FOR_RANKING)]])

    shown = unique[: inp.limit]
    if not unique:
        msg = "No matching jobs found. Try broader keywords or drop the location filter."
    else:
        msg = f"Showing {len(shown)} of {len(unique)} matches. Next: rank_fit to score them against your profile."
    if failed:
        msg += " Some sources were unavailable: " + ", ".join(f"{k} ({v})" for k, v in failed.items()) + "."
    return SearchJobsOutput(
        query=inp.query, total_found=len(unique), duplicates_removed=removed,
        returned=len(shown), sources_ok=ok, sources_failed=failed, message=msg,
        jobs=[
            JobSummary(id=j.id, title=j.title, company=j.company, location=j.location or "n/a",
                       url=j.url, source=j.source)
            for j in shown
        ],
    )
