import sqlite3

import numpy as np

from server.models import FitResult, Profile
from server.store import MIGRATIONS, Store

from .conftest import make_job


def test_schema_created_with_all_tables(tmp_db):
    Store(tmp_db)
    with sqlite3.connect(tmp_db) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = c.execute("PRAGMA user_version").fetchone()[0]
    assert {"profile", "jobs", "fit_scores", "pipeline", "interviews", "interview_answers",
            "plans", "source_cache", "meta"} <= names
    assert version == len(MIGRATIONS)


def test_reopening_is_idempotent(tmp_db):
    Store(tmp_db).save_profile(Profile(name="A"))
    assert Store(tmp_db).get_profile().name == "A"


def test_profile_roundtrip_and_single_row(tmp_db):
    s = Store(tmp_db)
    assert s.get_profile() is None
    s.save_profile(Profile(name="Asha", skills=["python", "sql"], target_roles=["Backend"], remote_ok=False))
    s.save_profile(Profile(name="Asha K", skills=["python"]))
    p = s.get_profile()
    assert p.name == "Asha K" and p.skills == ["python"] and p.updated_at is not None
    with sqlite3.connect(tmp_db) as c:
        assert c.execute("SELECT COUNT(*) FROM profile").fetchone()[0] == 1


def test_jobs_embeddings_and_fits(tmp_db):
    s = Store(tmp_db)
    j1, j2 = make_job(sid="1"), make_job(sid="2", title="SRE")
    s.upsert_jobs([j1, j2])
    s.upsert_jobs([j1])  # upsert, not duplicate
    assert [j.id for j in s.get_jobs([j2.id, j1.id])] == [j2.id, j1.id]  # order preserved

    s.set_embeddings({j1.id: [0.1, 0.2, 0.3]})
    emb = s.get_embeddings([j1.id, j2.id])
    assert set(emb) == {j1.id} and np.allclose(emb[j1.id], [0.1, 0.2, 0.3])

    s.save_fits([FitResult(job_id=j1.id, score=40, explanation="x", missing_skills=["go"]),
                 FitResult(job_id=j2.id, score=80, explanation="y")])
    fits = s.get_fits()
    assert [f.job_id for f in fits] == [j2.id, j1.id]  # best first
    assert fits[1].missing_skills == ["go"]
    assert [f.job_id for f in s.get_fits([j1.id])] == [j1.id]


def test_cache_ttl(tmp_db):
    s = Store(tmp_db)
    s.cache_put("k", {"a": 1})
    assert s.cache_get("k", ttl_hours=6) == {"a": 1}
    assert s.cache_get("k", ttl_hours=0) is None  # expired
    assert s.cache_get("missing", ttl_hours=6) is None


def test_meta(tmp_db):
    s = Store(tmp_db)
    assert s.get_meta("x", "default") == "default"
    s.set_meta("x", ["a", "b"])
    assert s.get_meta("x") == ["a", "b"]
