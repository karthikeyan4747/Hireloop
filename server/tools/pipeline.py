"""track_pipeline (act) and pipeline_summary (read)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from ..deps import get_store, need_job
from ..models import (PIPELINE_STATUSES, PipelineEntry, PipelineSummaryOutput, TrackPipelineOutput)
from ..policy import Tier, ToolFailure, require_confirm, tiered

FOLLOW_UP_DAYS = 7


@tiered(Tier.ACT)
async def track_pipeline(
    job_id: str, status: str, notes: str | None = None, confirm: bool = False
) -> TrackPipelineOutput:
    """Record where an application stands. Writes to your local pipeline; needs confirm=true.

    status is one of: saved, applied, interviewing, offer, rejected, ghosted. Setting
    'applied' stamps today's date and schedules a follow-up reminder 7 days later. HireLoop
    never applies for you: use this to log applications you submitted yourself. Ask the user to
    confirm, then call with confirm=true.
    """
    status = status.strip().lower()
    if status not in PIPELINE_STATUSES:
        raise ToolFailure(f"Invalid status '{status}'. Use one of: {', '.join(PIPELINE_STATUSES)}.")
    job = need_job(job_id)
    require_confirm(confirm, f"set '{job.title}' at {job.company} to '{status}' in your pipeline")

    store = get_store()
    cur = store.get_pipeline_entry(job_id)
    applied_at = cur.applied_at if cur else None
    follow_up = cur.follow_up_on if cur else None
    if status == "applied" and (cur is None or cur.status != "applied"):
        applied_at = datetime.now(timezone.utc)
        follow_up = (date.today() + timedelta(days=FOLLOW_UP_DAYS)).isoformat()
    elif status != "applied" and status != "saved":
        follow_up = None  # no reminder once there is a reply or a final outcome
    merged_notes = (notes.strip() if notes else (cur.notes if cur else ""))
    entry = store.save_pipeline_entry(PipelineEntry(
        job_id=job_id, status=status, applied_at=applied_at, follow_up_on=follow_up, notes=merged_notes,
    ))
    msg = f"Marked '{job.title}' at {job.company} as {status}."
    if entry.follow_up_on and status == "applied":
        msg += f" Follow-up reminder set for {entry.follow_up_on}."
    return TrackPipelineOutput(entry=entry, message=msg)


@tiered(Tier.READ)
async def pipeline_summary() -> PipelineSummaryOutput:
    """Summarise your applications: counts by status, follow-ups due, and applications that
    have had no reply for 7+ days.
    """
    entries = get_store().list_pipeline()
    return summarise(entries, date.today(), datetime.now(timezone.utc))


def summarise(entries: list[PipelineEntry], today: date, now: datetime) -> PipelineSummaryOutput:
    by_status = {s: 0 for s in PIPELINE_STATUSES}
    for e in entries:
        by_status[e.status] += 1
    applied = [e for e in entries if e.status == "applied"]
    due = [e for e in applied if e.follow_up_on and e.follow_up_on <= today.isoformat()]
    silent = [e for e in applied if e.applied_at and now - e.applied_at >= timedelta(days=FOLLOW_UP_DAYS)]
    if not entries:
        msg = "Your pipeline is empty. Use track_pipeline to log jobs you save or apply to."
    else:
        msg = (f"{len(entries)} tracked: {len(due)} follow-up(s) due, "
               f"{len(silent)} application(s) with no reply after 7 days.")
    return PipelineSummaryOutput(total=len(entries), by_status=by_status, follow_ups_due=due,
                                 no_reply_after_7_days=silent, message=msg)
