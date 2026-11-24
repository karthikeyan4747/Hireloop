"""build_profile: resume -> structured profile (preview first, save on confirm)."""
from __future__ import annotations

import hashlib

from .. import llm
from ..deps import get_store, validate_input
from ..models import BuildProfileInput, BuildProfileOutput, ExtractedProfile, Profile
from ..policy import Tier, require_confirm, tiered


def _clean(items: list[str] | None) -> list[str]:
    seen: dict[str, str] = {}
    for it in items or []:
        it = it.strip()
        if it:
            seen.setdefault(it.lower(), it)
    return list(seen.values())


def _fingerprint(inp: BuildProfileInput) -> str:
    raw = "|".join([inp.resume_text, ",".join(inp.target_roles or []), ",".join(inp.preferred_locations or [])])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@tiered(Tier.ACT)
async def build_profile(
    resume_text: str,
    target_roles: list[str] | None = None,
    preferred_locations: list[str] | None = None,
    confirm: bool = False,
) -> BuildProfileOutput:
    """Build your job-search profile from resume text. Writes to your local profile.

    Two-step: call first WITHOUT confirm to extract skills, experience level and a headline
    and preview them (nothing is saved). Show the preview to the user, let them correct it,
    then call again with the same inputs and confirm=true to save it. Your resume stays on
    this machine. Optionally pass target_roles (e.g. ["Backend Engineer"]) and
    preferred_locations (e.g. ["Bangalore"]).
    """
    inp = validate_input(
        BuildProfileInput, resume_text=resume_text, target_roles=target_roles,
        preferred_locations=preferred_locations, confirm=confirm,
    )
    store = get_store()
    key = f"pending_profile:{_fingerprint(inp)}"

    pending = store.get_meta(key)
    if pending is not None:
        profile = Profile.model_validate(pending)  # exactly what the user was shown
    else:
        extracted = await llm.complete(
            "Extract a job-seeker profile from this resume.\n"
            "- name: full name, or empty string\n"
            "- headline: one line, e.g. 'Backend engineer, 4 yrs, Python/AWS'\n"
            "- years_experience: total years of professional experience (number, estimate if needed)\n"
            "- skills: concrete technical and professional skills actually evidenced in the resume "
            "(languages, frameworks, tools, domains). Short canonical names, max 40, no duplicates. "
            "Do NOT invent skills that are not in the resume.\n\n"
            f"RESUME:\n{inp.resume_text[:12000]}",
            json_schema=ExtractedProfile,
        )
        existing = store.get_profile()
        profile = Profile(
            name=extracted.name.strip(),
            headline=extracted.headline.strip(),
            years_experience=max(0.0, extracted.years_experience),
            skills=_clean(extracted.skills),
            target_roles=_clean(inp.target_roles) or (existing.target_roles if existing else []),
            preferred_locations=_clean(inp.preferred_locations)
            or (existing.preferred_locations if existing else []),
            remote_ok=existing.remote_ok if existing else True,
            resume_text=inp.resume_text,
        )
        store.set_meta(key, profile.model_dump(mode="json"))

    if not inp.confirm:
        return BuildProfileOutput(
            saved=False, profile=profile,
            message=(
                "PREVIEW ONLY - nothing saved. Show this profile to the user; if it looks right, "
                "call build_profile again with the same inputs and confirm=true."
            ),
        )

    require_confirm(inp.confirm, "save this profile to your local HireLoop database")
    saved = store.save_profile(profile)
    store.set_meta(key, None)
    return BuildProfileOutput(
        saved=True, profile=saved,
        message="Profile saved locally. Next: search_jobs, then rank_fit.",
    )
