import json
from datetime import date, datetime, timedelta, timezone

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from server import llm, voice
from server.deps import get_store
from server.main import create_app
from server.models import (FitResult, GradeResult, PipelineEntry, PlanDay, PlanDraft, Profile,
                           QuestionDraft, QuestionSet, RubricScores, TailorDraft)
from server.tools.interview import question_mix, start_mock_interview, submit_answer
from server.tools.pipeline import pipeline_summary, summarise, track_pipeline
from server.tools.plan import build_plan, collect_weak_topics, prioritise
from server.tools.tailor import tailor_application

from .conftest import make_job


@pytest.fixture
def seeded(tmp_db, monkeypatch):
    """Profile + two jobs + fit scores in a temp DB, ElevenLabs unconfigured."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    s = get_store()
    s.save_profile(Profile(name="Asha", headline="Backend eng", skills=["python"], resume_text="Built APIs in Python."))
    j1, j2 = make_job(sid="1", title="Backend Engineer"), make_job(sid="2", title="SRE", company="Beta")
    s.upsert_jobs([j1, j2])
    s.save_fits([FitResult(job_id=j1.id, score=80, explanation="x", missing_skills=["Kubernetes"]),
                 FitResult(job_id=j2.id, score=70, explanation="y", missing_skills=["k8s", "Go"])])
    return s, j1, j2


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def fake(prompt, *, quality="fast", json_schema=None, system=None):
        calls.append((quality, json_schema))
        if json_schema is TailorDraft:
            return TailorDraft(bullets=["Built APIs"] * 4, cover_letter="Dear team, " + "x" * 80, gaps=["Kubernetes"])
        if json_schema is QuestionSet:
            return QuestionSet(questions=[QuestionDraft(kind="technical", question=f"Question number {i}?") for i in range(9)])
        if json_schema is GradeResult:
            return GradeResult(scores=RubricScores(technical=3, structure=2, clarity=4, relevance=5),
                               feedback="Solid but missing a result.", weak_topics=["STAR results", "star results", "indexing"])
        if json_schema is PlanDraft:
            return PlanDraft(days=[PlanDay(day=i + 5, focus=f"F{i}", why="because", tasks=["t"]) for i in range(7)])
        raise AssertionError(json_schema)

    monkeypatch.setattr(llm, "complete", fake)
    return calls


# ------------------------------------------------------------------ pipeline
async def test_track_pipeline_refuses_without_confirm(seeded):
    _, j1, _ = seeded
    with pytest.raises(ToolError, match="confirm=true"):
        await track_pipeline(j1.id, "applied")
    assert get_store().list_pipeline() == []  # nothing written


async def test_track_pipeline_validates_status_and_job(seeded):
    _, j1, _ = seeded
    with pytest.raises(ToolError, match="Invalid status"):
        await track_pipeline(j1.id, "hired", confirm=True)
    with pytest.raises(ToolError, match="Unknown job"):
        await track_pipeline("nope", "saved", confirm=True)


async def test_applied_sets_followup_7_days_and_clears_on_reply(seeded):
    _, j1, _ = seeded
    out = await track_pipeline(j1.id, "applied", notes="via referral", confirm=True)
    assert out.entry.follow_up_on == (date.today() + timedelta(days=7)).isoformat()
    assert out.entry.applied_at is not None and out.entry.notes == "via referral"
    out2 = await track_pipeline(j1.id, "interviewing", confirm=True)
    assert out2.entry.follow_up_on is None and out2.entry.notes == "via referral"  # notes kept
    assert out2.entry.applied_at == out.entry.applied_at


def test_summary_counts_due_and_silent():
    now = datetime.now(timezone.utc)
    old = PipelineEntry(job_id="a", status="applied", applied_at=now - timedelta(days=9),
                        follow_up_on=(date.today() - timedelta(days=2)).isoformat())
    fresh = PipelineEntry(job_id="b", status="applied", applied_at=now - timedelta(days=1),
                          follow_up_on=(date.today() + timedelta(days=6)).isoformat())
    saved = PipelineEntry(job_id="c", status="saved")
    out = summarise([old, fresh, saved], date.today(), now)
    assert out.by_status["applied"] == 2 and out.by_status["saved"] == 1 and out.total == 3
    assert [e.job_id for e in out.follow_ups_due] == ["a"]
    assert [e.job_id for e in out.no_reply_after_7_days] == ["a"]


async def test_pipeline_summary_empty(seeded):
    assert (await pipeline_summary()).total == 0


# ------------------------------------------------------------------ tailor
async def test_tailor_returns_draft_with_gaps(seeded, fake_llm):
    _, j1, _ = seeded
    out = await tailor_application(j1.id)
    assert 4 <= len(out.bullets) <= 6 and out.gaps_flagged == ["Kubernetes"]


async def test_tailor_needs_profile(tmp_db):
    j = make_job()
    get_store().upsert_jobs([j])
    with pytest.raises(ToolError, match="No profile"):
        await tailor_application(j.id)


# ------------------------------------------------------------------ interview
def test_question_mix_adds_up():
    for n in range(1, 11):
        for gaps in (True, False):
            mix = question_mix(n, gaps)
            assert sum(mix.values()) == n
    assert question_mix(5, True) == {"technical": 3, "behavioural": 1, "gap": 1}
    assert question_mix(5, False)["gap"] == 0


async def test_interview_text_only_when_voice_unconfigured(seeded, fake_llm):
    _, j1, _ = seeded
    out = await start_mock_interview(j1.id, num_questions=5, voice=True)
    assert len(out.questions) == 5 and all(q.audio_path is None for q in out.questions)
    assert "text only" in out.message
    assert get_store().get_interview(out.interview_id)["job_id"] == j1.id


async def test_interview_voice_failure_degrades_gracefully(seeded, fake_llm, monkeypatch, tmp_path):
    _, j1, _ = seeded
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setenv("HIRELOOP_DB_PATH", str(get_store().path))

    async def synth(text, path):
        if "number 1" in text:
            raise voice.VoiceError("boom")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")
        return path

    monkeypatch.setattr(voice, "synthesize", synth)
    out = await start_mock_interview(j1.id, num_questions=3)
    with_audio = [q for q in out.questions if q.audio_path]
    assert len(out.questions) == 3 and len(with_audio) == 2


async def test_interview_arg_validation(seeded):
    _, j1, _ = seeded
    with pytest.raises(ToolError):
        await start_mock_interview(j1.id, num_questions=0)


async def test_submit_answer_grades_stores_and_uses_high_quality(seeded, fake_llm):
    _, j1, _ = seeded
    iv = await start_mock_interview(j1.id, num_questions=3, voice=False)
    out = await submit_answer(iv.interview_id, 1, answer_text="I rebuilt our API and cut latency by 40 percent.")
    assert out.average == 3.5 and out.weak_topics == ["STAR results", "indexing"]
    assert ("high", GradeResult) in fake_llm  # quality-critical path
    rows = get_store().list_answers(iv.interview_id)
    assert len(rows) == 1 and rows[0]["scores"]["structure"] == 2
    await submit_answer(iv.interview_id, 1, answer_text="A different second attempt answer.")
    assert len(get_store().list_answers(iv.interview_id)) == 1  # replaced, not duplicated


async def test_submit_answer_input_errors(seeded, fake_llm):
    _, j1, _ = seeded
    iv = await start_mock_interview(j1.id, num_questions=2, voice=False)
    with pytest.raises(ToolError, match="Unknown interview"):
        await submit_answer("zzz", 0, answer_text="hello there")
    with pytest.raises(ToolError, match="between 0 and 1"):
        await submit_answer(iv.interview_id, 5, answer_text="hello there")
    with pytest.raises(ToolError, match="Provide answer"):
        await submit_answer(iv.interview_id, 0)


def test_audio_file_checks(tmp_path):
    with pytest.raises(voice.VoiceError, match="not found"):
        voice.check_audio_file(str(tmp_path / "missing.mp3"))
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    with pytest.raises(voice.VoiceError, match="Unsupported"):
        voice.check_audio_file(str(secret))  # arbitrary files are never uploaded
    ok = tmp_path / "a.mp3"
    ok.write_bytes(b"x")
    assert voice.check_audio_file(str(ok)) == ok


# ------------------------------------------------------------------ plan
def test_weak_topics_and_prioritise():
    answers = [{"weak_topics": ["Indexing", "STAR results"]}, {"weak_topics": ["indexing"]}, {"weak_topics": []}]
    topics = collect_weak_topics(answers)
    assert topics[0] == {"topic": "Indexing", "times": 2}
    ranked = prioritise([{"skill": "Kubernetes", "jobs_needing": 3}], topics, jobs_analysed=4, answers_graded=3)
    # impacts: Kubernetes 3/4 = .75, Indexing 2/3 = .67, STAR results 1/3 = .33
    assert [r["item"] for r in ranked] == ["Kubernetes", "Indexing", "STAR results"]
    assert ranked[0]["impact"] >= ranked[1]["impact"] >= ranked[2]["impact"]


async def test_plan_needs_input(tmp_db):
    get_store().save_profile(Profile(name="A"))
    with pytest.raises(ToolError, match="Nothing to plan"):
        await build_plan()


async def test_plan_preview_then_confirm_saves_shown_draft(seeded, fake_llm):
    store, _, _ = seeded
    draft = await build_plan(days=7)
    assert draft.saved is False and store.latest_plan() is None
    assert [d.day for d in draft.days] == [1, 2, 3, 4, 5, 6, 7]  # renumbered
    assert draft.based_on.gap_skills[0]["skill"] == "Kubernetes"  # k8s merged into one gap
    n_calls = len(fake_llm)
    saved = await build_plan(days=7, confirm=True)
    assert saved.saved and len(fake_llm) == n_calls  # no second LLM call: saved what was shown
    assert store.latest_plan()["id"] == saved.plan_id


async def test_plan_uses_interview_weak_topics(seeded, fake_llm):
    store, j1, _ = seeded
    iv = await start_mock_interview(j1.id, num_questions=2, voice=False)
    await submit_answer(iv.interview_id, 0, answer_text="An answer that is long enough.")
    out = await build_plan(days=3)
    assert {t["topic"] for t in out.based_on.weak_topics} == {"STAR results", "indexing"}


async def test_plan_arg_validation(seeded):
    with pytest.raises(ToolError):
        await build_plan(days=0)


# ------------------------------------------------------------------ resources + prompt
async def test_resources_and_prompt(seeded):
    app = create_app()
    uris = {str(r.uri) for r in await app.list_resources()}
    assert uris == {"hireloop://profile", "hireloop://pipeline", "hireloop://plan/latest"}
    prof = json.loads((await app.read_resource("hireloop://profile"))[0].content)
    assert prof["name"] == "Asha" and "resume_text" not in prof  # resume body not exposed
    assert json.loads((await app.read_resource("hireloop://pipeline"))[0].content) == []
    assert "plan" in json.loads((await app.read_resource("hireloop://plan/latest"))[0].content)
    prompt = await app.get_prompt("weekly_job_review", {"query": "data engineer", "location": "Pune"})
    text = prompt.messages[0].content.text
    assert "data engineer" in text and "Pune" in text and "build_plan" in text


async def test_all_ten_tools_registered_with_tiers(tmp_db):
    from server.policy import TOOL_TIERS, Tier
    names = {t.name for t in await create_app().list_tools()}
    assert len(names) == 10
    assert {n for n, t in TOOL_TIERS.items() if t is Tier.ACT} >= {"build_profile", "track_pipeline", "submit_answer"}
    assert names <= set(TOOL_TIERS)  # every registered tool declares a tier


async def test_missing_elevenlabs_package_is_a_clean_voice_error(monkeypatch, tmp_path):
    import sys

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setitem(sys.modules, "elevenlabs.client", None)  # makes the import raise ImportError
    with pytest.raises(voice.VoiceError, match="not installed"):
        await voice.synthesize("hello", tmp_path / "a.mp3")
