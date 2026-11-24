"""Lazily-created shared singletons (settings + store)."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings, load_settings
from .models import Job, Profile
from .policy import ToolFailure
from .store import Store

_store: Store | None = None
_store_path = None


def settings() -> Settings:
    return load_settings()


def get_store() -> Store:
    """One Store per DB path (path is re-read from env so tests can redirect it)."""
    global _store, _store_path
    path = load_settings().db_path
    if _store is None or _store_path != path:
        _store, _store_path = Store(path), path
    return _store


M = TypeVar("M", bound=BaseModel)


def validate_input(model: type[M], **kwargs: Any) -> M:
    """Build an input model, turning pydantic errors into a readable ToolFailure."""
    try:
        return model(**kwargs)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ToolFailure(f"Invalid input: {problems}") from None


def need_profile() -> Profile:
    p = get_store().get_profile()
    if p is None:
        raise ToolFailure("No profile yet. Call build_profile with your resume text first.")
    return p


def need_job(job_id: str) -> Job:
    jobs = get_store().get_jobs([job_id])
    if not jobs:
        raise ToolFailure(f"Unknown job id '{job_id}'. Use an id returned by search_jobs or rank_fit.")
    return jobs[0]
