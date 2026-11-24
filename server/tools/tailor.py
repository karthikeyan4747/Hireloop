"""tailor_application: resume bullets + cover letter for one job (draft tier)."""
from __future__ import annotations

from .. import llm
from ..deps import get_store, need_job, need_profile
from ..models import TailorDraft, TailorOutput
from ..policy import Tier, tiered


@tiered(Tier.DRAFT)
async def tailor_application(job_id: str) -> TailorOutput:
    """Draft 4-6 resume bullets and a short cover letter tailored to one job.

    Uses only experience that is in your saved profile and resume: nothing is invented, and
    requirements you do not evidence are listed honestly in gaps_flagged instead of being
    glossed over. Output is text for you to review and edit; nothing is sent or saved.
    """
    job, profile = need_job(job_id), need_profile()
    fit = next(iter(get_store().get_fits([job_id])), None)
    known_gaps = ", ".join(fit.missing_skills) if fit else "unknown (run rank_fit for a fit analysis)"

    draft = await llm.complete(
        "You are an honest resume coach. Tailor the candidate's materials to the job below.\n\n"
        "HARD RULES\n"
        "- Use ONLY facts present in the RESUME. Never invent employers, projects, metrics, "
        "tools, degrees or years of experience.\n"
        "- You may reword, reorder and emphasise real experience to match the job's language.\n"
        "- If the job needs something the resume does not show, do NOT claim it. List it in "
        "`gaps` (short phrases) and keep the cover letter honest about it, e.g. by showing "
        "eagerness to learn, never by pretending.\n\n"
        "OUTPUT\n"
        "- bullets: 4 to 6 resume bullets, each starting with a strong verb, concise, "
        "quantified only where the resume already gives numbers.\n"
        "- cover_letter: under 220 words, plain and specific, addressed to the hiring team at the "
        "company, no cliches, ends with a short call to action.\n"
        "- gaps: requirements from the job not evidenced in the resume.\n\n"
        f"JOB: {job.title} at {job.company} ({job.location})\n"
        f"DESCRIPTION:\n{job.description[:5000]}\n\n"
        f"CANDIDATE: {profile.name or 'the candidate'}; {profile.headline}\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Previously identified gaps for this job: {known_gaps}\n"
        f"RESUME:\n{profile.resume_text[:8000]}",
        json_schema=TailorDraft,
    )
    return TailorOutput(
        job_id=job.id, title=job.title, company=job.company,
        bullets=[b.strip() for b in draft.bullets], cover_letter=draft.cover_letter.strip(),
        gaps_flagged=draft.gaps,
        message=("Draft only: review every line against your real experience before using it. "
                 "Next: start_mock_interview to practise for this role."),
    )
