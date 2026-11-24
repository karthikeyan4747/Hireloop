"""Remotive public remote-jobs API."""
from __future__ import annotations

import httpx

from ..config import Settings
from ..models import Job
from ..store import Store
from .base import cached_jobs, clip, get_json, make_job_id, parse_dt, strip_html

NAME = "remotive"
API = "https://remotive.com/api/remote-jobs"


def parse_jobs(payload: dict) -> list[Job]:
    out: list[Job] = []
    for item in payload.get("jobs", []):
        sid = str(item.get("id", ""))
        if not sid:
            continue
        out.append(
            Job(
                id=make_job_id(NAME, sid), source=NAME, source_id=sid,
                company=item.get("company_name", ""), title=(item.get("title") or "").strip(),
                location=item.get("candidate_required_location") or "Remote", remote=True,
                url=item.get("url", ""), description=clip(strip_html(item.get("description"))),
                posted_at=parse_dt(item.get("publication_date")),
            )
        )
    return out


async def fetch(client: httpx.AsyncClient, store: Store, settings: Settings,
                query: str, location: str | None) -> list[Job]:
    async def produce() -> list[Job]:
        return parse_jobs(await get_json(client, settings, API, params={"search": query}))

    return await cached_jobs(store, settings, f"remotive:{query.lower().strip()}", produce)
