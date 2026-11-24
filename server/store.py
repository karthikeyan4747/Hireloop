"""Thin SQLite storage layer. Local only: your career data stays on your machine.

All SQL lives here so a DynamoDB implementation can replace this class later.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .models import FitResult, Job, PipelineEntry, Profile

# --------------------------------------------------------------------------- migrations
# One place for schema changes. Append new entries; never edit shipped ones.
MIGRATIONS: list[str] = [
    # v1: full schema
    """
    CREATE TABLE profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        name TEXT, headline TEXT, years_experience REAL,
        skills TEXT, target_roles TEXT, preferred_locations TEXT,
        remote_ok INTEGER, resume_text TEXT, updated_at TEXT
    );
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY, source TEXT, source_id TEXT, company TEXT, title TEXT,
        location TEXT, remote INTEGER, url TEXT, description TEXT,
        posted_at TEXT, fetched_at TEXT, embedding BLOB
    );
    CREATE TABLE fit_scores (
        job_id TEXT PRIMARY KEY REFERENCES jobs(id),
        score INTEGER, matched_skills TEXT, missing_skills TEXT,
        explanation TEXT, similarity REAL, scored_at TEXT
    );
    CREATE TABLE pipeline (
        job_id TEXT PRIMARY KEY REFERENCES jobs(id),
        status TEXT CHECK (status IN
            ('saved','applied','interviewing','offer','rejected','ghosted')),
        applied_at TEXT, follow_up_on TEXT, notes TEXT, updated_at TEXT
    );
    CREATE TABLE interviews (
        id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id),
        started_at TEXT, questions TEXT
    );
    CREATE TABLE interview_answers (
        interview_id TEXT REFERENCES interviews(id), question_index INTEGER,
        question TEXT, transcript TEXT, scores TEXT, feedback TEXT, weak_topics TEXT,
        PRIMARY KEY (interview_id, question_index)
    );
    CREATE TABLE plans (
        id TEXT PRIMARY KEY, created_at TEXT, days TEXT, based_on TEXT
    );
    CREATE TABLE source_cache (
        key TEXT PRIMARY KEY, fetched_at TEXT, payload TEXT
    );
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def vec_to_blob(vec: Sequence[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(sql)
                conn.execute(f"PRAGMA user_version = {i}")

    # ------------------------------------------------------------------ profile
    def get_profile(self) -> Profile | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM profile WHERE id = 1").fetchone()
        if r is None:
            return None
        return Profile(
            name=r["name"] or "",
            headline=r["headline"] or "",
            years_experience=r["years_experience"] or 0,
            skills=json.loads(r["skills"] or "[]"),
            target_roles=json.loads(r["target_roles"] or "[]"),
            preferred_locations=json.loads(r["preferred_locations"] or "[]"),
            remote_ok=bool(r["remote_ok"]),
            resume_text=r["resume_text"] or "",
            updated_at=_dt(r["updated_at"]),
        )

    def save_profile(self, p: Profile) -> Profile:
        p = p.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        with self._conn() as c:
            c.execute(
                """INSERT INTO profile (id, name, headline, years_experience, skills,
                       target_roles, preferred_locations, remote_ok, resume_text, updated_at)
                   VALUES (1,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       name=excluded.name, headline=excluded.headline,
                       years_experience=excluded.years_experience, skills=excluded.skills,
                       target_roles=excluded.target_roles,
                       preferred_locations=excluded.preferred_locations,
                       remote_ok=excluded.remote_ok, resume_text=excluded.resume_text,
                       updated_at=excluded.updated_at""",
                (
                    p.name, p.headline, p.years_experience, json.dumps(p.skills),
                    json.dumps(p.target_roles), json.dumps(p.preferred_locations),
                    int(p.remote_ok), p.resume_text, p.updated_at.isoformat(),
                ),
            )
        return p

    # ------------------------------------------------------------------ jobs
    def upsert_jobs(self, jobs: Sequence[Job]) -> None:
        with self._conn() as c:
            for j in jobs:
                c.execute(
                    """INSERT INTO jobs (id, source, source_id, company, title, location,
                           remote, url, description, posted_at, fetched_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                           company=excluded.company, title=excluded.title,
                           location=excluded.location, remote=excluded.remote,
                           url=excluded.url, description=excluded.description,
                           posted_at=excluded.posted_at, fetched_at=excluded.fetched_at""",
                    (
                        j.id, j.source, j.source_id, j.company, j.title, j.location,
                        int(j.remote), j.url, j.description,
                        j.posted_at.isoformat() if j.posted_at else None,
                        j.fetched_at.isoformat(),
                    ),
                )

    @staticmethod
    def _job(r: sqlite3.Row) -> Job:
        return Job(
            id=r["id"], source=r["source"], source_id=r["source_id"], company=r["company"],
            title=r["title"], location=r["location"] or "", remote=bool(r["remote"]),
            url=r["url"], description=r["description"] or "",
            posted_at=_dt(r["posted_at"]), fetched_at=_dt(r["fetched_at"]) or datetime.now(timezone.utc),
        )

    def get_jobs(self, ids: Sequence[str]) -> list[Job]:
        """Return jobs in the order of `ids` (missing ids skipped)."""
        if not ids:
            return []
        with self._conn() as c:
            q = ",".join("?" * len(ids))
            rows = {r["id"]: r for r in c.execute(f"SELECT * FROM jobs WHERE id IN ({q})", list(ids))}
        return [self._job(rows[i]) for i in ids if i in rows]

    def set_embeddings(self, vecs: dict[str, Sequence[float]]) -> None:
        with self._conn() as c:
            for job_id, v in vecs.items():
                c.execute("UPDATE jobs SET embedding = ? WHERE id = ?", (vec_to_blob(v), job_id))

    def clear_embeddings(self) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET embedding = NULL")

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        with self._conn() as c:
            q = ",".join("?" * len(ids))
            rows = c.execute(
                f"SELECT id, embedding FROM jobs WHERE id IN ({q}) AND embedding IS NOT NULL",
                list(ids),
            ).fetchall()
        return {r["id"]: blob_to_vec(r["embedding"]) for r in rows}

    # ------------------------------------------------------------------ fit scores
    def save_fits(self, fits: Sequence[FitResult]) -> None:
        with self._conn() as c:
            for f in fits:
                c.execute(
                    """INSERT INTO fit_scores (job_id, score, matched_skills, missing_skills,
                           explanation, similarity, scored_at) VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(job_id) DO UPDATE SET score=excluded.score,
                           matched_skills=excluded.matched_skills,
                           missing_skills=excluded.missing_skills,
                           explanation=excluded.explanation, similarity=excluded.similarity,
                           scored_at=excluded.scored_at""",
                    (
                        f.job_id, f.score, json.dumps(f.matched_skills),
                        json.dumps(f.missing_skills), f.explanation, f.similarity,
                        f.scored_at.isoformat(),
                    ),
                )

    def get_fits(self, ids: Sequence[str] | None = None) -> list[FitResult]:
        """All fit scores (best first), optionally restricted to `ids`."""
        sql, args = "SELECT * FROM fit_scores", []
        if ids is not None:
            if not ids:
                return []
            sql += f" WHERE job_id IN ({','.join('?' * len(ids))})"
            args = list(ids)
        sql += " ORDER BY score DESC"
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [
            FitResult(
                job_id=r["job_id"], score=r["score"],
                matched_skills=json.loads(r["matched_skills"] or "[]"),
                missing_skills=json.loads(r["missing_skills"] or "[]"),
                explanation=r["explanation"] or "", similarity=r["similarity"],
                scored_at=_dt(r["scored_at"]) or datetime.now(timezone.utc),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ source cache
    def cache_get(self, key: str, ttl_hours: int) -> Any | None:
        with self._conn() as c:
            r = c.execute("SELECT fetched_at, payload FROM source_cache WHERE key = ?", (key,)).fetchone()
        if r is None:
            return None
        if datetime.now(timezone.utc) - datetime.fromisoformat(r["fetched_at"]) > timedelta(hours=ttl_hours):
            return None
        return json.loads(r["payload"])

    def cache_put(self, key: str, payload: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO source_cache (key, fetched_at, payload) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET fetched_at=excluded.fetched_at, payload=excluded.payload",
                (key, _now(), json.dumps(payload)),
            )

    # ------------------------------------------------------------------ meta (small key/value)
    def set_meta(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            r = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    # ------------------------------------------------------------------ pipeline
    @staticmethod
    def _entry(r: sqlite3.Row) -> PipelineEntry:
        return PipelineEntry(
            job_id=r["job_id"], title=r["title"] or "", company=r["company"] or "",
            status=r["status"], applied_at=_dt(r["applied_at"]), follow_up_on=r["follow_up_on"],
            notes=r["notes"] or "", updated_at=_dt(r["updated_at"]),
        )

    _PIPE_SQL = ("SELECT p.*, j.title AS title, j.company AS company FROM pipeline p "
                 "LEFT JOIN jobs j ON j.id = p.job_id")

    def get_pipeline_entry(self, job_id: str) -> PipelineEntry | None:
        with self._conn() as c:
            r = c.execute(self._PIPE_SQL + " WHERE p.job_id = ?", (job_id,)).fetchone()
        return self._entry(r) if r else None

    def list_pipeline(self) -> list[PipelineEntry]:
        with self._conn() as c:
            rows = c.execute(self._PIPE_SQL + " ORDER BY p.updated_at DESC").fetchall()
        return [self._entry(r) for r in rows]

    def save_pipeline_entry(self, e: PipelineEntry) -> PipelineEntry:
        now = datetime.now(timezone.utc)
        with self._conn() as c:
            c.execute(
                """INSERT INTO pipeline (job_id, status, applied_at, follow_up_on, notes, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
                       applied_at=excluded.applied_at, follow_up_on=excluded.follow_up_on,
                       notes=excluded.notes, updated_at=excluded.updated_at""",
                (e.job_id, e.status, e.applied_at.isoformat() if e.applied_at else None,
                 e.follow_up_on, e.notes, now.isoformat()),
            )
        return self.get_pipeline_entry(e.job_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------ interviews
    def create_interview(self, interview_id: str, job_id: str, questions: list[dict]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO interviews (id, job_id, started_at, questions) VALUES (?,?,?,?)",
                      (interview_id, job_id, _now(), json.dumps(questions)))

    def get_interview(self, interview_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "job_id": r["job_id"], "started_at": r["started_at"],
                "questions": json.loads(r["questions"] or "[]")}

    def save_answer(self, interview_id: str, index: int, question: str, transcript: str,
                    scores: dict, feedback: str, weak_topics: list[str]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO interview_answers (interview_id, question_index, question,
                       transcript, scores, feedback, weak_topics) VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(interview_id, question_index) DO UPDATE SET
                       question=excluded.question, transcript=excluded.transcript,
                       scores=excluded.scores, feedback=excluded.feedback,
                       weak_topics=excluded.weak_topics""",
                (interview_id, index, question, transcript, json.dumps(scores), feedback,
                 json.dumps(weak_topics)),
            )

    def list_answers(self, interview_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM interview_answers", []
        if interview_id:
            sql, args = sql + " WHERE interview_id = ?", [interview_id]
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY interview_id, question_index", args).fetchall()
        return [{"interview_id": r["interview_id"], "question_index": r["question_index"],
                 "question": r["question"], "transcript": r["transcript"],
                 "scores": json.loads(r["scores"] or "{}"), "feedback": r["feedback"],
                 "weak_topics": json.loads(r["weak_topics"] or "[]")} for r in rows]

    # ------------------------------------------------------------------ plans
    def save_plan(self, plan_id: str, days: list[dict], based_on: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO plans (id, created_at, days, based_on) VALUES (?,?,?,?)",
                      (plan_id, _now(), json.dumps(days), json.dumps(based_on)))

    def latest_plan(self) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM plans ORDER BY created_at DESC LIMIT 1").fetchone()
        if r is None:
            return None
        return {"id": r["id"], "created_at": r["created_at"], "days": json.loads(r["days"]),
                "based_on": json.loads(r["based_on"])}
