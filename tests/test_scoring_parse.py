import pytest

from server import llm
from server.models import ExtractedProfile, FitBatch, FitResult
from server.policy import ToolFailure
from server.tools.fit import aggregate_gaps, canon_skill

from .conftest import make_job


def test_extract_json_strips_fences_and_prose():
    assert llm._extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert llm._extract_json('Sure! {"a": 1} hope that helps') == '{"a": 1}'


async def test_complete_retries_once_then_succeeds(monkeypatch):
    replies = iter(["not json at all", '{"name":"A","headline":"h","years_experience":3,"skills":["py"]}'])

    async def fake(prompt, *, system, json_mode):
        return next(replies)

    monkeypatch.setattr(llm, "_groq_complete", fake)
    out = await llm.complete("x", json_schema=ExtractedProfile)
    assert out.skills == ["py"]


async def test_complete_gives_up_after_two_bad_replies(monkeypatch):
    async def fake(prompt, *, system, json_mode):
        return '{"results": [{"job_id": "1", "score": 500}]}'  # score out of range

    monkeypatch.setattr(llm, "_groq_complete", fake)
    with pytest.raises(ToolFailure):
        await llm.complete("x", json_schema=FitBatch)


def test_score_bounds_validated():
    with pytest.raises(ValueError):
        FitResult(job_id="1", score=101, explanation="x")


def test_canon_skill_aliases():
    assert canon_skill(" K8s ") == "kubernetes"
    assert canon_skill("Golang") == "go"


def test_aggregate_gaps_ranks_by_job_count_and_merges_aliases():
    jobs = {f"j{i}": make_job(sid=str(i), title=f"T{i}") for i in range(3)}
    fits = [
        FitResult(job_id="j0", score=90, explanation="", missing_skills=["Kubernetes", "Rust"]),
        FitResult(job_id="j1", score=80, explanation="", missing_skills=["k8s", "k8s"]),  # dup in one job
        FitResult(job_id="j2", score=70, explanation="", missing_skills=["Rust", "Go"]),
    ]
    gaps = aggregate_gaps(fits, jobs)
    by = {g.skill: g for g in gaps}
    assert by["Kubernetes"].jobs_needing == 2   # k8s merged, counted once per job
    assert by["Rust"].jobs_needing == 2
    assert by["Go"].jobs_needing == 1
    assert gaps[0].skill == "Kubernetes"        # tie with Rust broken by higher fit score (90+80 > 90+70)
    assert gaps[0].share_of_jobs == 0.67
