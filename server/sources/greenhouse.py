"""Greenhouse public job board API (per company)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import yaml

from ..config import Settings
from ..models import Job
from ..store import Store
from .base import (SourceError, cached_jobs, clip, get_json, log, make_job_id, parse_dt,
                   query_tokens, relevance, strip_html)

NAME = "greenhouse"
API = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def load_companies(kind: str) -> list[dict]:
    data = yaml.safe_load((Path(__file__).parent / "companies.yaml").read_text(encoding="utf-8"))
    return list(data.get(kind, []))


def parse_board(company: str, payload: dict) -> list[Job]:
    jobs: list[Job] = []
    for item in payload.get("jobs", []):
        sid = str(item.get("id", ""))
        if not sid:
            continue
        loc = (item.get("location") or {}).get("name", "") or ""
        jobs.append(
            Job(
                id=make_job_id(NAME, sid), source=NAME, source_id=sid, company=company,
                title=item.get("title", "").strip(), location=loc,
                remote="remote" in loc.lower(), url=item.get("absolute_url", ""),
                description=clip(strip_html(item.get("content"))),
                posted_at=parse_dt(item.get("updated_at") or item.get("first_published")),
            )
        )
    return jobs


async def fetch(client: httpx.AsyncClient, store: Store, settings: Settings,
                query: str, location: str | None) -> list[Job]:
    companies = load_companies("greenhouse")
    sem = asyncio.Semaphore(8)
    errors: list[str] = []

    async def one(c: dict) -> list[Job]:
        async def produce() -> list[Job]:
            data = await get_json(client, settings, API.format(token=c["token"]), params={"content": "true"})
            return parse_board(c["name"], data)

        async with sem:
            try:
                return await cached_jobs(store, settings, f"greenhouse:{c['token']}", produce)
            except SourceError as exc:
                errors.append(f"{c['token']}: {exc}")
                return []

    boards = await asyncio.gather(*(one(c) for c in companies))
    if errors:
        log.info("greenhouse: %d/%d boards failed (%s)", len(errors), len(companies), "; ".join(errors[:3]))
    if len(errors) == len(companies):
        raise SourceError("all boards unreachable")
    tokens = query_tokens(query)
    return [j for board in boards for j in board if relevance(j.title, j.description, tokens)]
