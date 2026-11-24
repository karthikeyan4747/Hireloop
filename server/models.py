"""Pydantic models: domain objects and tool inputs/outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SourceName = Literal["greenhouse", "lever", "remotive", "adzuna"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- domain
class Job(BaseModel):
    """Common shape every source module returns."""

    id: str  # sha256(source + ":" + source_id)[:16]
    source: SourceName
    source_id: str
    company: str
    title: str
    location: str = ""
    remote: bool = False
    url: str
    description: str = ""  # HTML already stripped
    posted_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utcnow)


class Profile(BaseModel):
    """Single-row user profile. Stays on the user's machine."""

    name: str = ""
    headline: str = ""
    years_experience: float = 0
    skills: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    remote_ok: bool = True
    resume_text: str = ""
    updated_at: datetime | None = None


class FitResult(BaseModel):
    """LLM-produced fit assessment for one job (stage 2 of rank_fit)."""

    job_id: str
    score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str
    similarity: float | None = None  # stage-1 cosine score, shown for transparency
    scored_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- LLM output schemas
class ExtractedProfile(BaseModel):
    """What the LLM must return when parsing a resume."""

    name: str = ""
    headline: str = ""
    years_experience: float = 0
    skills: list[str] = Field(default_factory=list)


class _FitItem(BaseModel):
    job_id: str
    score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str


class FitBatch(BaseModel):
    """LLM output for a batch of jobs."""

    results: list[_FitItem]


# --------------------------------------------------------------------------- tool I/O
class BuildProfileInput(BaseModel):
    resume_text: str = Field(min_length=20)
    target_roles: list[str] | None = None
    preferred_locations: list[str] | None = None
    confirm: bool = False


class BuildProfileOutput(BaseModel):
    saved: bool
    message: str
    profile: Profile


class SearchJobsInput(BaseModel):
    query: str = Field(min_length=1)
    location: str | None = None
    remote_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)


class JobSummary(BaseModel):
    id: str
    title: str
    company: str
    location: str
    url: str
    source: str


class SearchJobsOutput(BaseModel):
    query: str
    total_found: int
    duplicates_removed: int
    returned: int
    sources_ok: list[str]
    sources_failed: dict[str, str]  # source -> reason
    jobs: list[JobSummary]
    message: str = ""


class RankFitInput(BaseModel):
    job_ids: list[str] | None = None
    top_n: int = Field(default=10, ge=1, le=25)


class RankedJob(BaseModel):
    job_id: str
    title: str
    company: str
    url: str
    score: int
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str
    similarity: float | None = None


class RankFitOutput(BaseModel):
    ranked: list[RankedJob]
    considered: int
    message: str = ""


class SkillCount(BaseModel):
    skill: str
    jobs_needing: int
    share_of_jobs: float  # 0-1
    example_jobs: list[str]  # "Title @ Company"


class SkillGapOutput(BaseModel):
    jobs_analysed: int
    gaps: list[SkillCount]
    message: str = ""


# =========================================================================== Tailor, interview, plan, pipeline
PipelineStatus = Literal["saved", "applied", "interviewing", "offer", "rejected", "ghosted"]
PIPELINE_STATUSES: tuple[str, ...] = ("saved", "applied", "interviewing", "offer", "rejected", "ghosted")


# --------------------------------------------------------------------------- tailor
class TailorDraft(BaseModel):
    """LLM output for tailor_application."""

    bullets: list[str] = Field(min_length=4, max_length=6)
    cover_letter: str = Field(min_length=50)
    gaps: list[str] = Field(default_factory=list)


class TailorOutput(BaseModel):
    job_id: str
    title: str
    company: str
    bullets: list[str]
    cover_letter: str
    gaps_flagged: list[str]  # requirements the profile does not evidence, stated honestly
    message: str


# --------------------------------------------------------------------------- interview
QuestionKind = Literal["technical", "behavioural", "gap"]


class QuestionDraft(BaseModel):
    kind: QuestionKind
    question: str = Field(min_length=10)


class QuestionSet(BaseModel):
    questions: list[QuestionDraft] = Field(min_length=1)


class InterviewQuestion(BaseModel):
    index: int
    kind: QuestionKind
    question: str
    audio_path: str | None = None


class StartInterviewOutput(BaseModel):
    interview_id: str
    job_id: str
    title: str
    company: str
    questions: list[InterviewQuestion]
    message: str


class RubricScores(BaseModel):
    technical: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)  # STAR for behavioural answers
    clarity: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


class GradeResult(BaseModel):
    """LLM output for grading one answer."""

    scores: RubricScores
    feedback: str = Field(min_length=10)
    weak_topics: list[str] = Field(default_factory=list)


class SubmitAnswerOutput(BaseModel):
    interview_id: str
    question_index: int
    question: str
    transcript: str
    scores: RubricScores
    average: float
    feedback: str
    weak_topics: list[str]
    graded_by: str
    message: str


# --------------------------------------------------------------------------- plan
class PlanDay(BaseModel):
    day: int = Field(ge=1)
    focus: str
    why: str  # one line: why this item is here
    tasks: list[str] = Field(min_length=1)
    resources: list[str] = Field(default_factory=list)


class PlanDraft(BaseModel):
    days: list[PlanDay] = Field(min_length=1)


class PlanBasis(BaseModel):
    gap_skills: list[dict]  # [{"skill": str, "jobs_needing": int}]
    weak_topics: list[dict]  # [{"topic": str, "times": int}]


class PlanOutput(BaseModel):
    saved: bool
    plan_id: str | None
    days: list[PlanDay]
    based_on: PlanBasis
    message: str


# --------------------------------------------------------------------------- pipeline
class PipelineEntry(BaseModel):
    job_id: str
    title: str = ""
    company: str = ""
    status: PipelineStatus
    applied_at: datetime | None = None
    follow_up_on: str | None = None  # ISO date
    notes: str = ""
    updated_at: datetime | None = None


class TrackPipelineOutput(BaseModel):
    entry: PipelineEntry
    message: str


class PipelineSummaryOutput(BaseModel):
    total: int
    by_status: dict[str, int]
    follow_ups_due: list[PipelineEntry]
    no_reply_after_7_days: list[PipelineEntry]
    message: str
