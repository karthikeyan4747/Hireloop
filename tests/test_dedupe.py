from server.sources.base import make_job_id, normalise, strip_html
from server.tools.search import apply_filters, dedupe

from .conftest import make_job


def test_dedupe_across_sources_ignores_case_punctuation_and_suffix():
    a = make_job("greenhouse", "1", "Acme, Inc.", "Backend Engineer", "Bangalore", "short")
    b = make_job("lever", "9", "ACME", "backend  engineer!", "bangalore", "a much longer description")
    c = make_job("remotive", "5", "Other Co", "Backend Engineer", "Bangalore")
    unique, removed = dedupe([a, b, c])
    assert removed == 1
    assert len(unique) == 2
    kept = next(j for j in unique if normalise(j.company) == "acme")
    assert kept.description == "a much longer description"  # richer copy wins


def test_dedupe_keeps_different_locations():
    a = make_job(sid="1", location="Bangalore")
    b = make_job(sid="2", location="London")
    assert dedupe([a, b])[1] == 0


def test_job_id_is_stable_and_source_scoped():
    assert make_job_id("lever", "1") == make_job_id("lever", "1")
    assert make_job_id("lever", "1") != make_job_id("greenhouse", "1")


def test_strip_html_handles_escaped_greenhouse_content():
    raw = "&lt;p&gt;Build &amp;amp; ship&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;"
    text = strip_html(raw)
    assert "<" not in text and "Build & ship" in text and "Python" in text


def test_filters_location_and_remote():
    blr = make_job(sid="1", location="Bangalore, India")
    lon = make_job(sid="2", location="London")
    rem = make_job(sid="3", location="Anywhere", remote=True)
    assert {j.id for j in apply_filters([blr, lon, rem], "bangalore", False)} == {blr.id, rem.id}
    assert [j.id for j in apply_filters([blr, lon, rem], None, True)] == [rem.id]
