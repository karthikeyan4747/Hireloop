"""Adzuna job search API. Optional: only enabled when ADZUNA_APP_ID / ADZUNA_APP_KEY are set."""
from __future__ import annotations

import httpx

from ..config import Settings
from ..models import Job
from ..store import Store
from .base import cached_jobs, clip, get_json, make_job_id, parse_dt, strip_html

NAME = "adzuna"
API = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


def parse_results(payload: dict) -> list[Job]:
    out: list[Job] = []
    for item in payload.get("results", []):
        sid = str(item.get("id", ""))
        if not sid:
            continue
        loc = (item.get("location") or {}).get("display_name", "") or ""
        title = strip_html(item.get("title"))
        out.append(
            Job(
                id=make_job_id(NAME, sid), source=NAME, source_id=sid,
                company=(item.get("company") or {}).get("display_name", "") or "Unknown",
                title=title, location=loc, remote="remote" in f"{loc} {title}".lower(),
                url=item.get("redirect_url", ""), description=clip(strip_html(item.get("description"))),
                posted_at=parse_dt(item.get("created")),
            )
        )
    return out


async def fetch(client: httpx.AsyncClient, store: Store, settings: Settings,
                query: str, location: str | None) -> list[Job]:
    params = {
        "app_id": settings.adzuna_app_id, "app_key": settings.adzuna_app_key,
        "what": query, "results_per_page": 30, "content-type": "application/json",
    }
    if location:
        params["where"] = location

    async def produce() -> list[Job]:
        url = API.format(country=settings.adzuna_country)
        return parse_results(await get_json(client, settings, url, params=params))

    # Cache key deliberately excludes the API key.
    key = f"adzuna:{settings.adzuna_country}:{query.lower()}:{(location or '').lower()}"
    return await cached_jobs(store, settings, key, produce)
