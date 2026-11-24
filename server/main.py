"""HireLoop MCP server entry point.

    python -m server.main                       # stdio (Claude Desktop)
    python -m server.main --transport streamable-http --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import threading

from mcp.server.fastmcp import FastMCP

from . import llm
from .config import load_settings
from .deps import get_store
from .http_app import AuthError, serve
from .tools.fit import rank_fit, skill_gap
from .tools.interview import start_mock_interview, submit_answer
from .tools.pipeline import pipeline_summary, track_pipeline
from .tools.plan import build_plan
from .tools.profile import build_profile
from .tools.search import search_jobs
from .tools.tailor import tailor_application

log = logging.getLogger("hireloop")

INSTRUCTIONS = """HireLoop runs the whole job-hunt loop: Find -> Fit -> Prepare -> Practise -> Improve.
Typical order: build_profile (preview, then confirm) -> search_jobs -> rank_fit -> skill_gap.
Never apply to jobs on the user's behalf: HireLoop finds, ranks and drafts; the human applies.
Tools that change stored data (tier 'act') need confirm=true after the user agrees."""

# Tiers (see server/policy.py):
#   read : search_jobs, rank_fit, skill_gap, pipeline_summary
#   draft: tailor_application, start_mock_interview, build_plan (saving needs confirm)
#   act  : build_profile, track_pipeline (confirm), submit_answer (explicit: user submits own answer)
TOOLS = [
    build_profile, search_jobs, rank_fit, skill_gap,
    tailor_application, start_mock_interview, submit_answer,
    build_plan, track_pipeline, pipeline_summary,
]

WEEKLY_REVIEW = """Run my weekly job review with HireLoop. Work through these steps in order and pause after each to show me the result:

1. Profile: check hireloop://profile. If it is missing or older than a few weeks, ask me for my resume and use build_profile (preview first, save only after I confirm).
2. Search: use search_jobs for "{query}"{location_clause}. Report how many unique jobs were found and which sources responded.
3. Rank: use rank_fit on those results. Show the top matches with score, matched skills and the two-sentence explanation.
4. Gaps: use skill_gap. Tell me the top 3 missing skills and how many of my best-fit jobs need each.
5. Plan: use build_plan (include any weak topics from past mock interviews). Show the draft and ask before saving it with confirm=true.
6. Pipeline: use pipeline_summary and remind me of follow-ups that are due.

Never apply to anything for me. Finish with a three-line summary: best role to pursue this week, the one skill to learn first, and the next action I should take."""


def create_app(host: str = "127.0.0.1", port: int = 8000, stateless: bool = False) -> FastMCP:
    # stateless_http: each request stands alone, so any instance behind a load balancer can serve it.
    app = FastMCP("HireLoop", instructions=INSTRUCTIONS, host=host, port=port, stateless_http=stateless)
    for fn in TOOLS:
        app.tool()(fn)

    @app.resource("hireloop://profile", mime_type="application/json",
                  description="Your saved job-search profile (resume text omitted). Read-only.")
    def profile_resource() -> str:
        p = get_store().get_profile()
        if p is None:
            return json.dumps({"profile": None, "hint": "Call build_profile first."})
        return p.model_dump_json(exclude={"resume_text"}, indent=2)

    @app.resource("hireloop://pipeline", mime_type="application/json",
                  description="Your tracked applications and their status. Read-only.")
    def pipeline_resource() -> str:
        entries = get_store().list_pipeline()
        return json.dumps([e.model_dump(mode="json") for e in entries], indent=2)

    @app.resource("hireloop://plan/latest", mime_type="application/json",
                  description="Your most recently saved prep plan. Read-only.")
    def plan_resource() -> str:
        plan = get_store().latest_plan()
        return json.dumps(plan or {"plan": None, "hint": "Call build_plan and confirm to save one."}, indent=2)

    @app.prompt(name="weekly_job_review",
                description="Guided weekly review: search, rank, skill gaps, prep plan, pipeline follow-ups.")
    def weekly_job_review(query: str = "software engineer", location: str = "") -> str:
        clause = f" in {location} (remote roles also welcome)" if location else ""
        return WEEKLY_REVIEW.format(query=query, location_clause=clause)

    return app


def main() -> None:
    cfg = load_settings()
    parser = argparse.ArgumentParser(description="HireLoop MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=cfg.transport)
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    args = parser.parse_args()

    log.info("Starting HireLoop (transport=%s, db=%s, bedrock=%s)", args.transport, cfg.db_path,
             "on" if cfg.use_bedrock else "off")
    threading.Thread(target=llm.warmup, name="embed-warmup", daemon=True).start()

    if args.transport == "streamable-http":
        try:
            serve(create_app(args.host, args.port, stateless=True), args.host, args.port, cfg.auth_token)
        except AuthError as exc:
            raise SystemExit(f"error: {exc}") from None
    else:
        create_app().run(transport="stdio")


if __name__ == "__main__":
    main()
