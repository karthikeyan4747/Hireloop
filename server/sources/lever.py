"""Lever public postings API (per company)."""
from __future__ import annotations

import asyncio

import httpx

from ..config import Settings
from ..models import Job
from ..store import Store
from .base import (SourceError, cached_jobs, clip, get_json, log, make_job_id, parse_dt,
                   query_tokens, relevance, strip_html)
from .greenhouse import load_companies

NAME = "lever"
API = "https://api.lever.co/v0/postings/{token}"


def parse_postings(company: str, payload: list[dict]) -> list[Job]:
    jobs: list[Job] = []
    for item in payload:
        sid = str(item.get("id", ""))
        if not sid:
            continue
        cats = item.get("categories") or {}
        loc = cats.get("location") or ""
        remote = (item.get("workplaceType") == "remote") or "remote" in loc.lower()
        desc = item.get("descriptionPlain") or strip_html(item.get("description"))
        jobs.append(
            Job(
                id=make_job_id(NAME, sid), source=NAME, source_id=sid, company=company,
                title=(item.get("text") or "").strip(), location=loc, remote=remote,
                url=item.get("hostedUrl", ""), description=clip(desc),
                posted_at=parse_dt(item.get("createdAt")),
            )
        )
    return jobs


async def fetch(client: httpx.AsyncClient, store: Store, settings: Settings,
                query: str, location: str | None) -> list[Job]:
    companies = load_companies("lever")
    errors: list[str] = []

    async def one(c: dict) -> list[Job]:
        async def produce() -> list[Job]:
            data = await get_json(client, settings, API.format(token=c["token"]), params={"mode": "json"})
            return parse_postings(c["name"], data if isinstance(data, list) else [])

        try:
            return await cached_jobs(store, settings, f"lever:{c['token']}", produce)
        except SourceError as exc:
            errors.append(f"{c['token']}: {exc}")
            return []

    boards = await asyncio.gather(*(one(c) for c in companies))
    if errors:
        log.info("lever: %d/%d boards failed (%s)", len(errors), len(companies), "; ".join(errors[:3]))
    if len(errors) == len(companies):
        raise SourceError("all boards unreachable")
    tokens = query_tokens(query)
    return [j for board in boards for j in board if relevance(j.title, j.description, tokens)]
