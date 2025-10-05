# core/audio_io.py
from __future__ import annotations
import io, os
import uuid
from pathlib import Path
import streamlit as st

# ---- Config from secrets/env via utils/config.py ----
from utils.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY

# Optional local recording (disabled on Cloud unless explicitly enabled)
HAVE_SD = False
try:
    if os.getenv("ENABLE_NATIVE_AUDIO", "0") == "1":
        import sounddevice as sd  # type: ignore
        import soundfile as sf    # type: ignore
        HAVE_SD = True
except Exception:
    HAVE_SD = False

import speech_recognition as sr
r = sr.Recognizer()

DEFAULT_RACHEL_ID = "21m00Tcm4TlvDq8ikWAM"

# ---------- ALL AUDIO SAVED HERE ----------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _voice(v: str | None) -> str:
    if not v or len(v.strip()) < 10:
        return DEFAULT_RACHEL_ID
    bad = ("_get_secret", "st.secrets", "os.getenv", "default", "VOICE_ID")
    if any(b in v for b in bad):
        return DEFAULT_RACHEL_ID
    return v.strip()

def _save_bytes_to_uploads(audio_bytes: bytes, ext: str = "wav") -> Path:
    """Persist audio bytes to data/uploads/<uuid>.<ext> and return the Path."""
    fname = f"{uuid.uuid4().hex}.{ext.lstrip('.')}"
    out_path = UPLOAD_DIR / fname
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    return out_path

# ============================ TTS ============================

def speak_text_bytes(text: str) -> bytes | None:
    """
    Return MP3 bytes (preferred for st.audio). Tries ElevenLabs, then falls back to OpenAI TTS.
    """
    if not text:
        return None

    # Try ElevenLabs first
    if ELEVENLABS_API_KEY:
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=str(ELEVENLABS_API_KEY).strip())
            stream = client.text_to_speech.convert(
                voice_id=_voice(ELEVENLABS_VOICE_ID),
                optimize_streaming_latency="0",
                output_format="mp3_44100_128",
                text=text,
            )
            buf = io.BytesIO()
            for chunk in stream:
                if chunk:
                    buf.write(chunk)
            return buf.getvalue()
        except Exception as e:
            st.warning(f"TTS (ElevenLabs) error: {e}")

    # Fallback: OpenAI TTS
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=str(OPENAI_API_KEY).strip())
            resp = client.audio.speech.create(
                model="gpt-5-nano",   # or "gpt-4o-tts" if you have access
                voice="alloy",
                input=text,
                response_format="mp3",
            )
            return resp.read()  # bytes
        except Exception as e:
            st.warning(f"TTS (OpenAI) error: {e}")

    return None

# Back-compat alias
def speak_text(text: str) -> bytes | None:
    return speak_text_bytes(text)

# ============================ STT ============================

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Persist bytes to data/uploads/*.wav and send that file to OpenAI STT.
    This avoids Windows temp-file locks.
    """
    if not OPENAI_API_KEY:
        return ""
    try:
        # Save to uploads (keep file as requested)
        wav_path = _save_bytes_to_uploads(audio_bytes, ext="wav")

        from openai import OpenAI
        client = OpenAI(api_key=str(OPENAI_API_KEY).strip())

        # Open the saved file ONLY within a context manager, so Windows releases the lock immediately after.
        with open(wav_path, "rb") as f:
            try:
                resp = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f,
                )
            except Exception:
                f.seek(0)
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        st.warning(f"Transcription error: {e}")
        return ""

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an existing file path in data/uploads (or anywhere readable).
    Opens once in a context manager to avoid PermissionError on Windows.
    """
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=str(OPENAI_API_KEY).strip())
        with open(file_path, "rb") as f:
            try:
                resp = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f,
                )
            except Exception:
                f.seek(0)
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        st.warning(f"Transcription error: {e}")
        return ""

# ======================= (Optional) Local Recording =======================

def record_audio(seconds: int = 10, samplerate: int = 16000, channels: int = 1) -> bytes:
    """
    LOCAL dev only: records from system mic. On Cloud this raises a clear error.
    """
    if not HAVE_SD:
        raise RuntimeError(
            "Native mic recording is unavailable on this host. "
            "Use the browser mic (streamlit-mic-recorder) or file upload."
        )
    import numpy as np  # local-only
    frames = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, frames, samplerate, format="WAV")
    return buf.getvalue()

__all__ = [
    "speak_text_bytes", "speak_text",
    "transcribe_audio_bytes", "transcribe_audio",
    "record_audio",
]
