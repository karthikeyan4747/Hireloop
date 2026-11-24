"""rank_fit (two-stage fit scoring) and skill_gap (what the best-fitting jobs need)."""
from __future__ import annotations

import asyncio
import logging

from .. import llm
from ..deps import get_store, validate_input
from ..models import (FitBatch, FitResult, Job, Profile, RankedJob, RankFitInput, RankFitOutput,
                      SkillCount, SkillGapOutput)
from ..policy import Tier, ToolFailure, tiered
from .search import LAST_SEARCH_KEY

log = logging.getLogger("hireloop.fit")

SHORTLIST_EXTRA = 5     # LLM scores top_n + this many, then we keep top_n by score
LLM_BATCH = 6           # jobs per LLM call
DESC_CHARS = 1800       # job description characters sent to the LLM

_ALIASES = {
    "js": "javascript", "ts": "typescript", "golang": "go", "k8s": "kubernetes",
    "postgres": "postgresql", "ml": "machine learning", "nodejs": "node.js", "node": "node.js",
    "gcp": "google cloud", "amazon web services": "aws",
}


def canon_skill(s: str) -> str:
    k = " ".join(s.lower().strip().split())
    return _ALIASES.get(k, k)


def profile_text(p: Profile) -> str:
    return (f"{p.headline}. Roles: {', '.join(p.target_roles)}. Skills: {', '.join(p.skills)}.\n"
            f"{p.resume_text[:3000]}")


def job_text(j: Job) -> str:
    return f"{j.title} at {j.company}. {j.location}.\n{j.description[:2000]}"


# --------------------------------------------------------------------------- rank_fit
EMBED_BACKEND_KEY = "embed_backend"


async def _ensure_embeddings(jobs: list[Job]) -> dict[str, list[float]]:
    """Job vectors from the current backend, embedding any that are missing.

    Vectors from different backends (Bedrock Titan, fastembed, hashing) live in different
    spaces and can even share a dimension, so when the backend changes all stored job vectors
    are discarded and recomputed rather than compared across spaces.
    """
    store = get_store()
    backend = llm.embed_backend()
    if store.get_meta(EMBED_BACKEND_KEY) != backend:
        log.info("Embedding backend changed to %s; recomputing job embeddings.", backend)
        store.clear_embeddings()
        store.set_meta(EMBED_BACKEND_KEY, backend)
    have = store.get_embeddings([j.id for j in jobs])
    missing = [j for j in jobs if j.id not in have]
    if missing:
        vecs = await llm.embed([job_text(j) for j in missing])
        if llm.embed_backend() != backend:  # backend fell back mid-run: start over consistently
            store.clear_embeddings()
            store.set_meta(EMBED_BACKEND_KEY, llm.embed_backend())
            have, missing = {}, list(jobs)
            vecs = await llm.embed([job_text(j) for j in missing])
        new = {j.id: v for j, v in zip(missing, vecs, strict=True)}
        store.set_embeddings(new)
        have.update(new)  # type: ignore[arg-type]
    return have  # type: ignore[return-value]


async def _score_batch(profile: Profile, jobs: list[Job]) -> list[FitResult]:
    listing = "\n\n".join(
        f"JOB_ID: {j.id}\nTITLE: {j.title}\nCOMPANY: {j.company}\nLOCATION: {j.location}\n"
        f"DESCRIPTION: {j.description[:DESC_CHARS]}"
        for j in jobs
    )
    prompt = (
        "You are a candid career advisor scoring how well a candidate fits each job.\n\n"
        f"CANDIDATE\nHeadline: {profile.headline}\nYears of experience: {profile.years_experience}\n"
        f"Target roles: {', '.join(profile.target_roles) or 'not specified'}\n"
        f"Skills: {', '.join(profile.skills)}\n\n"
        f"JOBS\n{listing}\n\n"
        "For EVERY job return one result with:\n"
        "- job_id: copied exactly\n"
        "- score: integer 0-100 (90+ = near-perfect, 70-89 = strong, 50-69 = stretch, <50 = poor)\n"
        "- matched_skills: skills the job asks for that the candidate demonstrably has\n"
        "- missing_skills: skills/requirements the job asks for that the candidate lacks (short names, "
        "most important first, max 6)\n"
        "- explanation: exactly 2 plain-language sentences for a non-expert; no jargon, no hype.\n"
        "Only use skills actually named in the job description. Be honest, do not inflate scores."
    )
    batch = await llm.complete(prompt, json_schema=FitBatch)
    valid = {j.id for j in jobs}
    out: list[FitResult] = []
    for r in batch.results:
        if r.job_id in valid:
            out.append(FitResult(job_id=r.job_id, score=r.score, matched_skills=r.matched_skills,
                                 missing_skills=r.missing_skills, explanation=r.explanation.strip()))
    return out


@tiered(Tier.READ)
async def rank_fit(job_ids: list[str] | None = None, top_n: int = 10) -> RankFitOutput:
    """Score and rank jobs against your saved profile, with a plain-language explanation.

    Stage 1 shortlists by embedding similarity between your profile and each job; stage 2 has
    an LLM score only the shortlist (0-100) and list matched and missing skills. Defaults to
    the jobs from your most recent search_jobs. Requires a saved profile (build_profile).
    """
    inp = validate_input(RankFitInput, job_ids=job_ids, top_n=top_n)
    store = get_store()
    profile = store.get_profile()
    if profile is None:
        raise ToolFailure("No profile yet. Call build_profile with your resume text first.")

    ids = inp.job_ids or store.get_meta(LAST_SEARCH_KEY, [])
    if not ids:
        raise ToolFailure("No jobs to rank. Run search_jobs first (or pass job_ids).")
    jobs = store.get_jobs(ids)
    if not jobs:
        raise ToolFailure("Those job ids are not in the local database. Run search_jobs again.")

    # Stage 1: cosine similarity shortlist (profile embedded first so the backend tag is current)
    pvec = (await llm.embed([profile_text(profile)]))[0]
    jvecs = await _ensure_embeddings(jobs)
    sims = {j.id: llm.cosine(pvec, jvecs[j.id]) for j in jobs}

    shortlist = sorted(jobs, key=lambda j: sims[j.id], reverse=True)[: inp.top_n + SHORTLIST_EXTRA]

    # Stage 2: LLM scoring (reuse scores newer than the profile; they'd be identical inputs)
    cached = {f.job_id: f for f in store.get_fits([j.id for j in shortlist])
              if profile.updated_at and f.scored_at >= profile.updated_at}
    todo = [j for j in shortlist if j.id not in cached]
    fresh: list[FitResult] = []
    if todo:
        batches = [todo[i : i + LLM_BATCH] for i in range(0, len(todo), LLM_BATCH)]
        for res in await asyncio.gather(*(_score_batch(profile, b) for b in batches)):
            fresh.extend(res)
        for f in fresh:
            f.similarity = round(sims.get(f.job_id, 0.0), 4)
        store.save_fits(fresh)

    all_fits = {**cached, **{f.job_id: f for f in fresh}}
    by_id = {j.id: j for j in shortlist}
    ranked = sorted(all_fits.values(), key=lambda f: f.score, reverse=True)[: inp.top_n]
    skipped = len(todo) - len(fresh)
    msg = (f"Ranked {len(ranked)} of {len(jobs)} jobs ({len(shortlist)} shortlisted by similarity). "
           "Next: skill_gap to see what is holding your scores back.")
    if skipped:
        msg += f" {skipped} job(s) could not be scored this time."
    return RankFitOutput(
        considered=len(jobs), message=msg,
        ranked=[
            RankedJob(job_id=f.job_id, title=by_id[f.job_id].title, company=by_id[f.job_id].company,
                      url=by_id[f.job_id].url, score=f.score, matched_skills=f.matched_skills,
                      missing_skills=f.missing_skills, explanation=f.explanation,
                      similarity=f.similarity)
            for f in ranked
        ],
    )


# --------------------------------------------------------------------------- skill_gap
def aggregate_gaps(fits: list[FitResult], jobs: dict[str, Job]) -> list[SkillCount]:
    """Count how many of the given jobs need each missing skill; most-needed first.

    Ties break on the summed fit score of the jobs that need it (closer-to-fit jobs matter more).
    """
    counts: dict[str, dict] = {}
    for f in fits:
        seen_in_job: set[str] = set()
        for raw in f.missing_skills:
            key = canon_skill(raw)
            if not key or key in seen_in_job:
                continue
            seen_in_job.add(key)
            e = counts.setdefault(key, {"label": raw.strip(), "jobs": [], "score": 0})
            j = jobs.get(f.job_id)
            e["jobs"].append(f"{j.title} @ {j.company}" if j else f.job_id)
            e["score"] += f.score
    n = max(len(fits), 1)
    ranked = sorted(counts.values(), key=lambda e: (len(e["jobs"]), e["score"]), reverse=True)
    return [SkillCount(skill=e["label"], jobs_needing=len(e["jobs"]),
                       share_of_jobs=round(len(e["jobs"]) / n, 2), example_jobs=e["jobs"][:3])
            for e in ranked]


@tiered(Tier.READ)
async def skill_gap(top_n_jobs: int = 10) -> SkillGapOutput:
    """Find the skills you are missing, ranked by how many of your best-fitting jobs need them.

    Uses the scores from rank_fit, so run that first. Learning the top item here lifts the
    most fit scores at once.
    """
    if not 1 <= top_n_jobs <= 50:
        raise ToolFailure("top_n_jobs must be between 1 and 50.")
    store = get_store()
    fits = store.get_fits()[:top_n_jobs]  # already ordered by score desc
    if not fits:
        raise ToolFailure("No fit scores yet. Run search_jobs and rank_fit first.")
    jobs = {j.id: j for j in store.get_jobs([f.job_id for f in fits])}
    gaps = aggregate_gaps(fits, jobs)
    msg = (f"Across your {len(fits)} best-fitting jobs, '{gaps[0].skill}' is the most common gap."
           if gaps else "No skill gaps found across your best-fitting jobs. Nice.")
    return SkillGapOutput(jobs_analysed=len(fits), gaps=gaps, message=msg)
