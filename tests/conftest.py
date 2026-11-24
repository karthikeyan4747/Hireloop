import pytest

from server.models import Job
from server.sources.base import make_job_id


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HIRELOOP_DB_PATH", str(tmp_path / "test.db"))
    return tmp_path / "test.db"


@pytest.fixture(autouse=True)
def _reset_llm_state(monkeypatch):
    """Provider caches are module-global; isolate every test and default Bedrock/auth to off."""
    from server import llm

    llm._cred_cache.clear()
    llm._clients.clear()
    llm._logged_reasons.clear()
    monkeypatch.setattr(llm, "_embed_backend", "")
    monkeypatch.delenv("USE_BEDROCK", raising=False)
    monkeypatch.delenv("HIRELOOP_AUTH_TOKEN", raising=False)
    yield


def make_job(source="greenhouse", sid="1", company="Acme", title="Backend Engineer",
             location="Bangalore", description="", remote=False) -> Job:
    return Job(id=make_job_id(source, sid), source=source, source_id=sid, company=company,
               title=title, location=location, url=f"https://x/{sid}",
               description=description, remote=remote)
