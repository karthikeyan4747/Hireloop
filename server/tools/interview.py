"""Mock interview: start_mock_interview (draft) and submit_answer (act)."""
from __future__ import annotations

import asyncio
import logging
import uuid

from .. import llm
from .. import voice as voice_mod  # the tool's `voice` parameter shadows the module name
from ..deps import get_store, need_job, need_profile, settings
from ..models import (GradeResult, InterviewQuestion, QuestionSet, StartInterviewOutput,
                      SubmitAnswerOutput)
from ..policy import Tier, ToolFailure, tiered

log = logging.getLogger("hireloop.interview")


def question_mix(n: int, has_gaps: bool) -> dict[str, int]:
    """How many technical / behavioural / gap-targeted questions for an n-question interview."""
    gap = (2 if n >= 6 else 1 if n >= 3 else 0) if has_gaps else 0
    behavioural = max(1, n // 3) if n >= 2 else 0
    technical = max(n - gap - behavioural, 0)
    return {"technical": technical, "behavioural": behavioural, "gap": gap}


@tiered(Tier.DRAFT)
async def start_mock_interview(job_id: str, num_questions: int = 5, voice: bool = True) -> StartInterviewOutput:
    """Start a mock interview for one job: a mix of technical, behavioural and gap-targeted
    questions built from the job description and your profile.

    If voice=true and ElevenLabs is configured, each question is also spoken: audio files are
    saved locally and their paths returned (falls back to text if voice is unavailable). Answer
    with submit_answer, one question at a time.
    """
    if not 1 <= num_questions <= 10:
        raise ToolFailure("num_questions must be between 1 and 10.")
    job, profile, store = need_job(job_id), need_profile(), get_store()
    fit = next(iter(store.get_fits([job_id])), None)
    gaps = fit.missing_skills[:5] if fit else []
    mix = question_mix(num_questions, bool(gaps))

    qs = await llm.complete(
        f"Write a realistic interview for the role below. Produce exactly {num_questions} questions: "
        f"{mix['technical']} technical, {mix['behavioural']} behavioural (answerable with a STAR story), "
        f"{mix['gap']} gap-targeted (probe a skill the job needs that the candidate lacks, fairly, "
        "so they can show how they would learn or approximate it).\n"
        "Each question must be specific to this job, one or two sentences, and speakable aloud "
        "(no code blocks, no lists). Order them from easier to harder.\n"
        "kind must be one of: technical, behavioural, gap.\n\n"
        f"ROLE: {job.title} at {job.company}\nJOB DESCRIPTION:\n{job.description[:4000]}\n\n"
        f"CANDIDATE: {profile.headline}; skills: {', '.join(profile.skills)}\n"
        f"GAPS vs this job: {', '.join(gaps) or 'none identified'}",
        json_schema=QuestionSet,
    )
    drafts = qs.questions[:num_questions]
    interview_id = uuid.uuid4().hex[:8]
    questions = [InterviewQuestion(index=i, kind=d.kind, question=d.question.strip())
                 for i, d in enumerate(drafts)]

    note = ""
    if voice:
        if not voice_mod.tts_configured():
            note = " Voice is off (ElevenLabs not configured), so questions are text only."
        else:
            audio_dir = settings().audio_dir
            sem = asyncio.Semaphore(3)

            async def speak(q: InterviewQuestion) -> None:
                async with sem:
                    try:
                        path = await voice_mod.synthesize(q.question, audio_dir / f"{interview_id}_q{q.index}.mp3")
                        q.audio_path = str(path)
                    except voice_mod.VoiceError as exc:  # one failed clip must not lose the interview
                        log.warning("TTS failed for q%d: %s", q.index, exc)

            await asyncio.gather(*(speak(q) for q in questions))
            made = sum(q.audio_path is not None for q in questions)
            note = (f" Audio saved for {made}/{len(questions)} questions in {audio_dir}."
                    if made else " Voice failed, so questions are text only.")

    store.create_interview(interview_id, job_id, [q.model_dump() for q in questions])
    return StartInterviewOutput(
        interview_id=interview_id, job_id=job_id, title=job.title, company=job.company,
        questions=questions,
        message=("Interview ready. Ask the first question, then call submit_answer with "
                 f"interview_id='{interview_id}' and question_index=0 (answer as text or an audio file)." + note),
    )


@tiered(Tier.ACT)
async def submit_answer(
    interview_id: str,
    question_index: int,
    answer_text: str | None = None,
    answer_audio_path: str | None = None,
) -> SubmitAnswerOutput:
    """Submit your answer to one interview question and get it graded. Saves the result.

    Provide answer_text, or answer_audio_path (a local audio file, transcribed with Whisper).
    The answer is graded 1-5 on technical accuracy, structure (STAR), clarity and relevance,
    with specific feedback and weak topics. Weak topics feed build_plan. Resubmitting the same
    question replaces the earlier grade. The act is explicit: you are submitting your own answer.
    """
    store = get_store()
    interview = store.get_interview(interview_id)
    if interview is None:
        raise ToolFailure(f"Unknown interview id '{interview_id}'. Call start_mock_interview first.")
    questions = interview["questions"]
    if not 0 <= question_index < len(questions):
        raise ToolFailure(f"question_index must be between 0 and {len(questions) - 1}.")
    if not (answer_text and answer_text.strip()) and not answer_audio_path:
        raise ToolFailure("Provide answer_text or answer_audio_path.")

    transcript = answer_text.strip() if answer_text and answer_text.strip() else await voice_mod.transcribe(answer_audio_path or "")
    if len(transcript) < 5:
        raise ToolFailure("The answer is too short to grade.")

    q = questions[question_index]
    job = need_job(interview["job_id"])
    profile = need_profile()
    grade = await llm.complete(
        "You are a strict but fair interviewer grading ONE answer. Be specific and concrete.\n\n"
        "RUBRIC (integers 1-5 each; 1 = poor, 3 = acceptable, 5 = excellent)\n"
        "- technical: correctness and depth for the role's level\n"
        "- structure: for behavioural questions, STAR (Situation, Task, Action, Result); "
        "otherwise a logical order of ideas\n"
        "- clarity: concise, easy to follow when spoken\n"
        "- relevance: answers what was asked, tied to this role\n"
        "Do not give 5 unless the answer is truly strong. A vague or very short answer scores low.\n\n"
        "feedback: 3-5 sentences: what was good, what was missing, and one concrete way to improve "
        "(quote or point at part of the answer). Address the candidate as 'you'.\n"
        "weak_topics: 0-3 short topic names (e.g. 'database indexing', 'STAR results') that the answer "
        "shows need work; empty if none.\n\n"
        f"ROLE: {job.title} at {job.company}\nQUESTION ({q['kind']}): {q['question']}\n"
        f"CANDIDATE LEVEL: {profile.headline}, {profile.years_experience} years\n"
        f"ANSWER (may be a speech transcript):\n{transcript[:6000]}",
        quality="high", json_schema=GradeResult,
    )
    seen: dict[str, str] = {}
    for t in grade.weak_topics:
        if t.strip():
            seen.setdefault(t.strip().lower(), t.strip())  # case-insensitive dedupe, first spelling wins
    weak = list(seen.values())
    store.save_answer(interview_id, question_index, q["question"], transcript,
                      grade.scores.model_dump(), grade.feedback.strip(), weak)

    s = grade.scores
    avg = round((s.technical + s.structure + s.clarity + s.relevance) / 4, 2)
    nxt = question_index + 1
    more = (f"Next: question {nxt}." if nxt < len(questions)
            else "That was the last question. Next: build_plan to turn weak spots into a prep plan.")
    return SubmitAnswerOutput(
        interview_id=interview_id, question_index=question_index, question=q["question"],
        transcript=transcript, scores=s, average=avg, feedback=grade.feedback.strip(),
        weak_topics=weak, graded_by=_grader_name(), message=f"Graded and saved. {more}",
    )


def _grader_name() -> str:
    """Which model actually graded (Bedrock, or the Groq fallback)."""
    return llm.last_provider() or "unknown"
