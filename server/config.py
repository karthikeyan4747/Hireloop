"""Environment-driven configuration. All secrets come from env vars / .env."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env next to the project root (works regardless of the client's cwd).
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# IMPORTANT: stdio transport uses stdout for the protocol. Log to stderr only.
logging.basicConfig(
    level=os.getenv("HIRELOOP_LOG_LEVEL", "INFO"),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    audio_dir: Path
    transport: str
    host: str
    port: int
    groq_api_key: str
    groq_model: str
    adzuna_app_id: str
    adzuna_app_key: str
    adzuna_country: str
    use_bedrock: bool
    aws_region: str
    bedrock_model_id: str
    bedrock_embed_model_id: str
    auth_token: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_turbo_v2_5"
    whisper_model: str = "whisper-large-v3-turbo"
    http_timeout: float = 20.0
    cache_ttl_hours: int = 6

    @property
    def adzuna_enabled(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)


def load_settings() -> Settings:
    """Read settings fresh from the environment (tests can monkeypatch env)."""
    db = Path(os.getenv("HIRELOOP_DB_PATH", "~/.hireloop/hireloop.db")).expanduser()
    return Settings(
        db_path=db,
        audio_dir=db.parent / "audio",
        transport=os.getenv("HIRELOOP_TRANSPORT", "stdio").strip().lower(),
        host=os.getenv("HIRELOOP_HOST", "127.0.0.1"),
        port=int(os.getenv("HIRELOOP_PORT", "8000")),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        adzuna_app_id=os.getenv("ADZUNA_APP_ID", ""),
        adzuna_app_key=os.getenv("ADZUNA_APP_KEY", ""),
        adzuna_country=os.getenv("ADZUNA_COUNTRY", "in"),
        use_bedrock=_flag("USE_BEDROCK", False),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", ""),
        auth_token=os.getenv("HIRELOOP_AUTH_TOKEN", ""),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo"),
        bedrock_embed_model_id=os.getenv(
            "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
        ),
    )

for _noisy in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
