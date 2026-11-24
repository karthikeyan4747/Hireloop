"""The only module that talks to LLM / embedding providers.

    complete(prompt, *, quality="fast"|"high", json_schema=None)
    embed(texts)

Routing:
  * quality="fast" -> Groq, always.
  * quality="high" -> Amazon Bedrock (Converse API) when USE_BEDROCK=true, a model id is set and
                      AWS credentials exist; otherwise Groq. A Bedrock runtime error also falls
                      back to Groq. Every fallback is logged.
  * embed          -> Bedrock Titan under the same conditions; otherwise local fastembed;
                      otherwise a dependency-free hashing fallback so nothing hard-fails.

`last_provider()` and `embed_backend()` report what actually served the most recent call.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from contextvars import ContextVar
from typing import Any, Literal, TypeVar, overload

import numpy as np
from pydantic import BaseModel, ValidationError

from .config import load_settings
from .policy import ToolFailure

log = logging.getLogger("hireloop.llm")

T = TypeVar("T", bound=BaseModel)
Quality = Literal["fast", "high"]

LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 384
TITAN_DIM = 1024


class LLMError(ToolFailure):
    """LLM/provider problem, phrased for the end user."""


# --------------------------------------------------------------------------- provider bookkeeping
_provider: ContextVar[str] = ContextVar("hireloop_llm_provider", default="")
_embed_backend = ""
_logged_reasons: set[str] = set()


def last_provider() -> str:
    """Provider that served the latest complete() in this task, e.g. 'bedrock:<model>' or 'groq:<model>'."""
    return _provider.get()


def embed_backend() -> str:
    """Backend that produced the latest embeddings: 'bedrock:<id>', 'fastembed:<model>' or 'hash'."""
    return _embed_backend


def _note_fallback(reason: str, what: str) -> None:
    """Log each distinct fallback reason once (not on every call)."""
    key = f"{what}:{reason}"
    if key not in _logged_reasons:
        _logged_reasons.add(key)
        log.info("%s: using %s instead of Bedrock (%s).", what, "Groq" if what == "completion" else "local embeddings", reason)


# --------------------------------------------------------------------------- Bedrock plumbing
_cred_cache: dict[str, bool] = {}
_clients: dict[str, Any] = {}


def _has_aws_credentials() -> bool:
    """True if boto3 can find credentials (env, profile, or an instance/task role). Cached."""
    region = load_settings().aws_region
    if region not in _cred_cache:
        try:
            import boto3

            _cred_cache[region] = boto3.Session(region_name=region).get_credentials() is not None
        except Exception as exc:
            log.warning("AWS credential lookup failed: %r", exc)
            _cred_cache[region] = False
    return _cred_cache[region]


def bedrock_unavailable_reason(model_id: str) -> str | None:
    """None if Bedrock can be used for `model_id`, else a short human-readable reason."""
    s = load_settings()
    if not s.use_bedrock:
        return "USE_BEDROCK is off"
    if not model_id:
        return "no Bedrock model id configured"
    try:
        import boto3  # noqa: F401
    except ImportError:
        return "boto3 is not installed (pip install 'hireloop[aws]')"
    if not _has_aws_credentials():
        return "no AWS credentials found"
    return None


def _bedrock_client() -> Any:
    region = load_settings().aws_region
    if region not in _clients:
        import boto3
        from botocore.config import Config

        s = load_settings()
        _clients[region] = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(connect_timeout=5, read_timeout=int(s.http_timeout * 3),
                          retries={"max_attempts": 2, "mode": "standard"}),
        )
    return _clients[region]


async def _bedrock_complete(prompt: str, *, system: str | None) -> str:
    s = load_settings()
    kwargs: dict[str, Any] = {
        "modelId": s.bedrock_model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"temperature": 0.2, "maxTokens": 4096},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = await asyncio.to_thread(_bedrock_client().converse, **kwargs)
    parts = resp["output"]["message"]["content"]
    return "".join(p.get("text", "") for p in parts)


async def _bedrock_embed(texts: list[str]) -> list[list[float]]:
    s = load_settings()
    model = s.bedrock_embed_model_id
    client = _bedrock_client()
    sem = asyncio.Semaphore(8)  # Titan embeds one text per request

    def one(text: str) -> list[float]:
        body: dict[str, Any] = {"inputText": text}
        if "titan-embed-text-v2" in model:
            body.update({"dimensions": TITAN_DIM, "normalize": True})
        resp = client.invoke_model(modelId=model, body=json.dumps(body),
                                   contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["embedding"]

    async def guarded(text: str) -> list[float]:
        async with sem:
            return await asyncio.to_thread(one, text)

    return list(await asyncio.gather(*(guarded(t) for t in texts)))


# --------------------------------------------------------------------------- completion
async def _groq_complete(prompt: str, *, system: str | None, json_mode: bool) -> str:
    s = load_settings()
    if not s.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set. Add it to the .env file and restart the server.")
    from groq import AsyncGroq  # imported lazily so tests don't need the network stack

    client = AsyncGroq(api_key=s.groq_api_key, timeout=s.http_timeout * 3, max_retries=1)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kwargs: dict = {"model": s.groq_model, "messages": messages, "temperature": 0.2}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as exc:  # network, auth, rate-limit, model errors
        log.warning("Groq call failed: %r", exc)
        raise LLMError(f"The language model request failed ({type(exc).__name__}). Please try again shortly.") from None
    return resp.choices[0].message.content or ""


async def _generate(prompt: str, *, system: str | None, json_mode: bool, quality: Quality) -> str:
    """Route one generation to a provider and record which one served it."""
    s = load_settings()
    if quality == "high":
        reason = bedrock_unavailable_reason(s.bedrock_model_id)
        if reason is None:
            try:
                text = await _bedrock_complete(prompt, system=system)
                _provider.set(f"bedrock:{s.bedrock_model_id}")
                return text
            except Exception as exc:  # throttling, access denied, timeout, bad model id...
                log.warning("Bedrock completion failed (%r); falling back to Groq.", exc)
        else:
            _note_fallback(reason, "completion")
    text = await _groq_complete(prompt, system=system, json_mode=json_mode)
    _provider.set(f"groq:{s.groq_model}")
    return text


def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


@overload
async def complete(prompt: str, *, quality: Quality = ..., json_schema: None = ..., system: str | None = ...) -> str: ...
@overload
async def complete(prompt: str, *, quality: Quality = ..., json_schema: type[T], system: str | None = ...) -> T: ...
async def complete(
    prompt: str,
    *,
    quality: Quality = "fast",
    json_schema: type[T] | None = None,
    system: str | None = None,
) -> str | T:
    """Single entry point for text generation.

    With `json_schema` (a pydantic model class) the model is asked for JSON, the reply is
    validated, and on failure we retry once with the validation error fed back.
    """
    if json_schema is None:
        return await _generate(prompt, system=system, json_mode=False, quality=quality)

    schema_hint = json.dumps(json_schema.model_json_schema(), indent=None)
    full = (
        f"{prompt}\n\nRespond with ONLY a JSON object matching this JSON Schema "
        f"(no prose, no markdown):\n{schema_hint}"
    )
    last_err = ""
    for attempt in range(2):
        text = await _generate(
            full if attempt == 0 else f"{full}\n\nYour previous reply was invalid: {last_err}\nFix it.",
            system=system, json_mode=True, quality=quality,
        )
        try:
            return json_schema.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as exc:
            last_err = str(exc)[:500]
            log.warning("LLM JSON invalid (attempt %d): %s", attempt + 1, last_err)
    raise LLMError("The model returned malformed output twice. Please try again.")


# --------------------------------------------------------------------------- embeddings
_fastembed_model = None
_fastembed_failed = False
_load_lock = threading.Lock()


def _hash_embed(text: str) -> list[float]:
    """Last-resort bag-of-words hashing embedding. Crude but deterministic and offline."""
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9+#.]{2,}", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % HASH_DIM] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def _local_embed_sync(texts: list[str]) -> list[list[float]]:
    global _fastembed_model, _fastembed_failed, _embed_backend
    with _load_lock:  # warmup thread and first request must not both load the model
        if not _fastembed_failed:
            try:
                if _fastembed_model is None:
                    from fastembed import TextEmbedding

                    _fastembed_model = TextEmbedding(model_name=LOCAL_EMBED_MODEL)
                    log.info("Loaded local embedding model %s", LOCAL_EMBED_MODEL)
                vecs = [np.asarray(v, dtype=np.float32).tolist() for v in _fastembed_model.embed(texts)]
                _embed_backend = f"fastembed:{LOCAL_EMBED_MODEL}"
                return vecs
            except Exception as exc:  # not installed, offline first run, etc.
                _fastembed_failed = True
                log.warning("fastembed unavailable (%r); using hashing fallback embeddings.", exc)
    _embed_backend = "hash"
    return [_hash_embed(t) for t in texts]


def warmup() -> None:
    """Load the local embedding model in the background so the first rank_fit is fast.

    Skipped when Bedrock embeddings will be used. The first run downloads ~90 MB.
    """
    if bedrock_unavailable_reason(load_settings().bedrock_embed_model_id) is not None:
        _local_embed_sync(["warmup"])


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts (Bedrock Titan if enabled, else local). Compare vectors with cosine().

    Vectors from different backends are NOT comparable; check embed_backend() before mixing.
    """
    global _embed_backend
    if not texts:
        return []
    clipped = [t[:6000] for t in texts]  # keep within model limits
    model = load_settings().bedrock_embed_model_id
    reason = bedrock_unavailable_reason(model)
    if reason is None:
        try:
            vecs = await _bedrock_embed(clipped)
            _embed_backend = f"bedrock:{model}"
            return vecs
        except Exception as exc:
            log.warning("Bedrock embeddings failed (%r); falling back to local embeddings.", exc)
    else:
        _note_fallback(reason, "embeddings")
    return await asyncio.to_thread(_local_embed_sync, clipped)


def cosine(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0
