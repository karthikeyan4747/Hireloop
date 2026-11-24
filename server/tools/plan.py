"""build_plan: skill gaps + weak interview topics -> a day-by-day prep plan (draft tier)."""
from __future__ import annotations

import hashlib
import json
import uuid

from .. import llm
from ..deps import get_store, need_profile
from ..models import PlanBasis, PlanDay, PlanDraft, PlanOutput
from ..policy import Tier, ToolFailure, tiered
from .fit import aggregate_gaps, canon_skill

GAP_JOBS = 10  # best-fitting jobs considered for skill gaps


def collect_weak_topics(answers: list[dict]) -> list[dict]:
    """Count weak topics across graded answers, most frequent first."""
    counts: dict[str, dict] = {}
    for a in answers:
        for t in a.get("weak_topics", []):
            key = canon_skill(t)
            if key:
                e = counts.setdefault(key, {"topic": t.strip(), "times": 0})
                e["times"] += 1
    return sorted(counts.values(), key=lambda e: e["times"], reverse=True)


def prioritise(gaps: list[dict], topics: list[dict], jobs_analysed: int, answers_graded: int) -> list[dict]:
    """Merge skill gaps and weak topics into one list, highest impact first.

    impact = share of target jobs that need the skill (gaps), or share of graded answers that
    exposed the topic (weak topics). Both are 0-1 so they can be compared directly.
    """
    items: list[dict] = []
    for g in gaps:
        items.append({"item": g["skill"], "source": "skill gap", "impact": g["jobs_needing"] / max(jobs_analysed, 1),
                      "evidence": f"missing from {g['jobs_needing']} of {jobs_analysed} best-fit jobs"})
    for t in topics:
        items.append({"item": t["topic"], "source": "weak interview topic",
                      "impact": t["times"] / max(answers_graded, 1),
                      "evidence": f"weak in {t['times']} of {answers_graded} graded interview answers"})
    return sorted(items, key=lambda i: i["impact"], reverse=True)


@tiered(Tier.DRAFT)
async def build_plan(days: int = 7, hours_per_day: float = 2.0, confirm: bool = False) -> PlanOutput:
    """Build a day-by-day prep plan from your skill gaps (rank_fit/skill_gap) and the weak
    topics from your mock interviews (submit_answer).

    Highest-impact items come first, each day has concrete tasks and free resources, and a
    one-line reason says why it is there. Without confirm this is a draft for you to review;
    call again with the same arguments and confirm=true to save it as your latest plan.
    """
    if not 1 <= days <= 30:
        raise ToolFailure("days must be between 1 and 30.")
    if not 0.25 <= hours_per_day <= 12:
        raise ToolFailure("hours_per_day must be between 0.25 and 12.")
    store = get_store()
    profile = need_profile()

    fits = store.get_fits()[:GAP_JOBS]
    jobs = {j.id: j for j in store.get_jobs([f.job_id for f in fits])}
    gaps = [{"skill": g.skill, "jobs_needing": g.jobs_needing} for g in aggregate_gaps(fits, jobs)][:8]
    answers = store.list_answers()
    topics = collect_weak_topics(answers)[:6]
    if not gaps and not topics:
        raise ToolFailure("Nothing to plan from yet. Run rank_fit (for skill gaps) and/or take a mock "
                          "interview (for weak topics) first.")

    basis = PlanBasis(gap_skills=gaps, weak_topics=topics)
    ranked = prioritise(gaps, topics, len(fits), len(answers))
    fp = hashlib.sha256(json.dumps([days, hours_per_day, ranked], sort_keys=True).encode()).hexdigest()[:16]
    key = f"pending_plan:{fp}"

    stash = store.get_meta(key)
    if stash is not None:
        plan_days = [PlanDay.model_validate(d) for d in stash]  # exactly what the user was shown
    else:
        listing = "\n".join(f"{i + 1}. {r['item']} ({r['source']}; {r['evidence']})" for i, r in enumerate(ranked))
        draft = await llm.complete(
            f"Create a {days}-day preparation plan for a job seeker with {hours_per_day} hours available "
            "per day.\n\n"
            "PRIORITISED ITEMS (highest impact first; schedule in this order, spending more time on the top ones):\n"
            f"{listing}\n\n"
            f"CANDIDATE: {profile.headline}; current skills: {', '.join(profile.skills)}\n\n"
            "RULES\n"
            f"- Return exactly {days} entries with day numbers 1..{days}.\n"
            "- focus: short title. why: ONE sentence that cites the evidence for this item "
            "(e.g. which jobs need it or which answer exposed it).\n"
            "- tasks: 2-4 concrete, checkable tasks that fit in the day's hours (build X, solve N "
            "problems, rehearse one STAR story aloud and record it).\n"
            "- resources: 1-3 FREE resources by name (official docs, well-known free courses, "
            "MDN, freeCodeCamp, YouTube channels). Include a URL only when it is a site's main "
            "domain you are sure of. Never invent links.\n"
            "- Use the final day for a mock interview run-through covering the top items.",
            json_schema=PlanDraft,
        )
        plan_days = sorted(draft.days, key=lambda d: d.day)[:days]
        for i, d in enumerate(plan_days, start=1):
            d.day = i  # normalise numbering
        store.set_meta(key, [d.model_dump() for d in plan_days])

    short = f" (the model returned {len(plan_days)} of {days} days)" if len(plan_days) < days else ""
    if not confirm:
        return PlanOutput(saved=False, plan_id=None, days=plan_days, based_on=basis,
                          message=f"DRAFT, not saved{short}. Review with the user; to keep it call "
                                  "build_plan again with the same arguments and confirm=true.")
    plan_id = uuid.uuid4().hex[:8]
    store.save_plan(plan_id, [d.model_dump() for d in plan_days], basis.model_dump())
    store.set_meta(key, None)
    return PlanOutput(saved=True, plan_id=plan_id, days=plan_days, based_on=basis,
                      message=f"Plan saved{short}. After practising, re-run rank_fit: an updated profile changes your fit scores.")
