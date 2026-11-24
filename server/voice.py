"""Voice: ElevenLabs text-to-speech (the interviewer) and Groq Whisper speech-to-text."""
from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path

from .config import load_settings
from .policy import ToolFailure

log = logging.getLogger("hireloop.voice")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq's file limit


class VoiceError(ToolFailure):
    """Voice feature problem, phrased for the end user."""


def tts_configured() -> bool:
    s = load_settings()
    return bool(s.elevenlabs_api_key and s.elevenlabs_voice_id)


async def synthesize(text: str, out_path: Path) -> Path:
    """Render `text` to an mp3 at `out_path` with ElevenLabs. Raises VoiceError."""
    s = load_settings()
    if not tts_configured():
        raise VoiceError("ElevenLabs is not configured (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID).")
    try:
        from elevenlabs.client import AsyncElevenLabs
    except ImportError:
        raise VoiceError("The 'elevenlabs' package is not installed (pip install 'hireloop[voice]').") from None

    client = AsyncElevenLabs(api_key=s.elevenlabs_api_key, timeout=s.http_timeout * 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = client.text_to_speech.convert(
            s.elevenlabs_voice_id, text=text, model_id=s.elevenlabs_model,
            output_format="mp3_44100_64",
        )
        if inspect.isawaitable(stream):
            stream = await stream
        chunks = [c async for c in stream if c]
    except Exception as exc:
        log.warning("ElevenLabs TTS failed: %r", exc)
        raise VoiceError(f"Text-to-speech failed ({type(exc).__name__}).") from None
    if not chunks:
        raise VoiceError("Text-to-speech returned no audio.")
    out_path.write_bytes(b"".join(chunks))
    return out_path


def check_audio_file(path_str: str) -> Path:
    """Only plain audio files of sane size may be sent to the transcription service."""
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise VoiceError(f"Audio file not found: {path_str}")
    if path.suffix.lower() not in AUDIO_EXTS:
        raise VoiceError(f"Unsupported audio type '{path.suffix}'. Use one of: {', '.join(sorted(AUDIO_EXTS))}.")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise VoiceError("Audio file is larger than 25 MB. Record a shorter answer.")
    return path


async def transcribe(audio_path: str) -> str:
    """Speech to text with Groq Whisper. Raises VoiceError."""
    s = load_settings()
    path = check_audio_file(audio_path)
    if not s.groq_api_key:
        raise VoiceError("GROQ_API_KEY is not set, so audio cannot be transcribed. Send the answer as text instead.")
    from groq import AsyncGroq

    client = AsyncGroq(api_key=s.groq_api_key, timeout=s.http_timeout * 3, max_retries=1)
    data = await asyncio.to_thread(path.read_bytes)
    try:
        result = await client.audio.transcriptions.create(
            model=s.whisper_model, file=(path.name, data), response_format="text", language="en",
        )
    except Exception as exc:
        log.warning("Whisper failed: %r", exc)
        raise VoiceError(f"Transcription failed ({type(exc).__name__}). Send the answer as text instead.") from None
    text = (result if isinstance(result, str) else getattr(result, "text", "")).strip()
    if not text:
        raise VoiceError("No speech detected in the audio.")
    return text
