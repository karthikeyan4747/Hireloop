import io
import json

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from server import llm
from server.deps import get_store
from server.http_app import AuthError, BearerAuthMiddleware, check_startup, is_loopback
from server.models import ExtractedProfile
from server.tools.fit import _ensure_embeddings

from .conftest import make_job

GOOD = '{"name":"A","headline":"h","years_experience":1,"skills":["py"]}'


@pytest.fixture
def bedrock_on(monkeypatch):
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "test.model-v1")
    monkeypatch.setenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setattr(llm, "_has_aws_credentials", lambda: True)


@pytest.fixture
def groq_stub(monkeypatch):
    calls = []

    async def fake(prompt, *, system, json_mode):
        calls.append(prompt)
        return GOOD

    monkeypatch.setattr(llm, "_groq_complete", fake)
    return calls


# ------------------------------------------------------------------ availability
def test_unavailable_reasons(monkeypatch):
    assert "off" in llm.bedrock_unavailable_reason("m")
    monkeypatch.setenv("USE_BEDROCK", "true")
    assert "model id" in llm.bedrock_unavailable_reason("")
    monkeypatch.setattr(llm, "_has_aws_credentials", lambda: False)
    assert "credentials" in llm.bedrock_unavailable_reason("m")
    monkeypatch.setattr(llm, "_has_aws_credentials", lambda: True)
    assert llm.bedrock_unavailable_reason("m") is None


# ------------------------------------------------------------------ completion routing
async def test_high_quality_uses_bedrock_when_enabled(bedrock_on, groq_stub, monkeypatch):
    async def bedrock(prompt, *, system):
        return GOOD

    monkeypatch.setattr(llm, "_bedrock_complete", bedrock)
    out = await llm.complete("x", quality="high", json_schema=ExtractedProfile)
    assert out.skills == ["py"] and llm.last_provider() == "bedrock:test.model-v1"
    assert groq_stub == []  # Groq untouched


async def test_fast_quality_never_uses_bedrock(bedrock_on, groq_stub, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("bedrock must not be called for quality=fast")

    monkeypatch.setattr(llm, "_bedrock_complete", boom)
    await llm.complete("x", quality="fast", json_schema=ExtractedProfile)
    assert llm.last_provider().startswith("groq:")


async def test_bedrock_error_falls_back_to_groq(bedrock_on, groq_stub, monkeypatch):
    async def bedrock(prompt, *, system):
        raise RuntimeError("AccessDeniedException")

    monkeypatch.setattr(llm, "_bedrock_complete", bedrock)
    out = await llm.complete("x", quality="high", json_schema=ExtractedProfile)
    assert out.name == "A" and llm.last_provider().startswith("groq:") and len(groq_stub) == 1


async def test_disabled_or_no_credentials_falls_back_to_groq(groq_stub, monkeypatch):
    await llm.complete("x", quality="high", json_schema=ExtractedProfile)  # USE_BEDROCK off
    assert llm.last_provider().startswith("groq:")
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "m")
    monkeypatch.setattr(llm, "_has_aws_credentials", lambda: False)
    await llm.complete("x", quality="high", json_schema=ExtractedProfile)
    assert llm.last_provider().startswith("groq:") and len(groq_stub) == 2


async def test_converse_request_and_response_shape(bedrock_on, monkeypatch):
    seen = {}

    class Client:
        def converse(self, **kw):
            seen.update(kw)
            return {"output": {"message": {"content": [{"text": "hel"}, {"text": "lo"}]}}}

    monkeypatch.setattr(llm, "_bedrock_client", lambda: Client())
    assert await llm._bedrock_complete("question", system="be strict") == "hello"
    assert seen["modelId"] == "test.model-v1"
    assert seen["messages"] == [{"role": "user", "content": [{"text": "question"}]}]
    assert seen["system"] == [{"text": "be strict"}]


# ------------------------------------------------------------------ embeddings
class _Titan:
    def __init__(self):
        self.bodies = []

    def invoke_model(self, *, modelId, body, contentType, accept):
        self.bodies.append((modelId, json.loads(body)))
        return {"body": io.BytesIO(json.dumps({"embedding": [0.1, 0.2]}).encode())}


async def test_titan_v2_request_includes_dimensions(bedrock_on, monkeypatch):
    client = _Titan()
    monkeypatch.setattr(llm, "_bedrock_client", lambda: client)
    vecs = await llm.embed(["a", "b"])
    assert vecs == [[0.1, 0.2], [0.1, 0.2]] and llm.embed_backend() == "bedrock:amazon.titan-embed-text-v2:0"
    assert client.bodies[0][1] == {"inputText": "a", "dimensions": llm.TITAN_DIM, "normalize": True}


async def test_titan_v1_request_has_no_v2_params(bedrock_on, monkeypatch):
    monkeypatch.setenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
    client = _Titan()
    monkeypatch.setattr(llm, "_bedrock_client", lambda: client)
    await llm.embed(["a"])
    assert client.bodies[0][1] == {"inputText": "a"}


async def test_embed_falls_back_to_local_on_bedrock_error(bedrock_on, monkeypatch):
    async def boom(texts):
        raise RuntimeError("throttled")

    monkeypatch.setattr(llm, "_bedrock_embed", boom)
    monkeypatch.setattr(llm, "_fastembed_failed", True)  # skip the model download in tests
    vecs = await llm.embed(["python developer"])
    assert len(vecs[0]) == llm.HASH_DIM and llm.embed_backend() == "hash"


async def test_backend_change_discards_stored_job_vectors(tmp_db, monkeypatch):
    store = get_store()
    j = make_job()
    store.upsert_jobs([j])
    store.set_embeddings({j.id: [1.0] * llm.HASH_DIM})  # same dimension as the new backend!
    store.set_meta("embed_backend", "fastembed:other")
    monkeypatch.setattr(llm, "_embed_backend", "hash")

    async def fake_embed(texts):
        return [[0.5] * llm.HASH_DIM for _ in texts]

    monkeypatch.setattr(llm, "embed", fake_embed)
    out = await _ensure_embeddings([j])
    assert out[j.id][0] == pytest.approx(0.5)  # recomputed, not the stale vector
    assert store.get_meta("embed_backend") == "hash"


# ------------------------------------------------------------------ HTTP auth
def _client(token="s3cret") -> TestClient:
    inner = Starlette(routes=[Route("/mcp", lambda r: PlainTextResponse("in"), methods=["GET", "POST"])])
    return TestClient(BearerAuthMiddleware(inner, token))


def test_health_is_open_and_mcp_requires_token():
    c = _client()
    assert c.get("/health").status_code == 200
    r = c.post("/mcp")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.post("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/mcp", headers={"Authorization": "Basic s3cret"}).status_code == 401
    ok = c.post("/mcp", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.text == "in"


def test_startup_guard():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.0.0.5") and not is_loopback("example.com")
    with pytest.raises(AuthError):
        check_startup("0.0.0.0", "")
    check_startup("0.0.0.0", "token")  # fine with a token
    check_startup("127.0.0.1", "")  # fine for local dev
