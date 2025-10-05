# core/audio_io.py
from __future__ import annotations
import io
import os
import tempfile

import streamlit as st

# ---- Config (from Streamlit secrets or env via utils/config.py) ----
from utils.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY

# -------------------------------------------------------------------
# Optional native recording for LOCAL development only (PortAudio).
# Will NOT load on Streamlit Cloud unless ENABLE_NATIVE_AUDIO=1.
# -------------------------------------------------------------------
HAVE_SD = False
try:
    if os.getenv("ENABLE_NATIVE_AUDIO", "0") == "1":
        import sounddevice as sd  # type: ignore
        import soundfile as sf    # type: ignore
        HAVE_SD = True
except Exception:
    HAVE_SD = False


# ============================ TTS ============================
# core/audio_io.py (top)
from utils.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY

# Known safe public voice ID for "Rachel" (from ElevenLabs docs)

def _resolve_voice_id(v: str | None) -> str:
    """
    Returns a valid voice_id (UUID-like). If the provided value is missing or
    looks like a pasted code snippet / placeholder, fall back to Rachel.
    """
    if not v:
        return DEFAULT_RACHEL_ID
    bad_patterns = ('_get_secret', 'st.secrets', 'os.getenv', 'default', '""', "''", 'VOICE_ID')
    if any(p in v for p in bad_patterns):
        return DEFAULT_RACHEL_ID
    # crude UUID-ish check (ElevenLabs voice IDs are 22 chars base58; this len check is lenient)
    if len(v.strip()) < 10:
        return DEFAULT_RACHEL_ID
    return v.strip()
# core/audio_io.py
from __future__ import annotations
import io, tempfile
import streamlit as st
from utils.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY

DEFAULT_RACHEL_ID = "21m00Tcm4TlvDq8ikWAM"

def _voice(v: str | None) -> str:
    if not v or len(v.strip()) < 10:
        return DEFAULT_RACHEL_ID
    return v.strip()

def speak_text_bytes(text: str) -> bytes | None:
    if not text:
        return None

    # ---- Try ElevenLabs first ----
    if ELEVENLABS_API_KEY:
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY.strip())
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

    # ---- Fallback: OpenAI TTS ----
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Use text-to-speech (model names may vary; tts-1 is common)
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                format="mp3"
            )
            return speech.read()  # bytes
        except Exception as e:
            st.warning(f"TTS (OpenAI) error: {e}")

    # No TTS available
    return None

# ============================ STT ============================

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Cloud-safe transcription using OpenAI. Tries 'gpt-4o-transcribe', falls back to 'whisper-1'.
    Returns a plain string (possibly empty on failure).
    """
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            try:
                resp = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=open(tmp.name, "rb"),
                )
            except Exception:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(tmp.name, "rb"),
                )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        st.warning(f"Transcription error: {e}")
        return ""


def transcribe_audio(file_path: str) -> str:
    """
    Convenience wrapper to transcribe from a file path (used in some parts of the app).
    """
    try:
        with open(file_path, "rb") as f:
            return transcribe_audio_bytes(f.read())
    except Exception:
        return ""


# ======================= (Optional) Local Recording =======================

def record_audio(seconds: int = 10, samplerate: int = 16000, channels: int = 1) -> bytes:
    """
    LOCAL dev only: record from the system mic to WAV bytes.
    On Cloud this raises a clear error because PortAudio is not present.
    """
    if not HAVE_SD:
        raise RuntimeError(
            "Native mic recording is unavailable on this host. "
            "Use the browser mic component (streamlit-mic-recorder) or file upload."
        )
    import numpy as np  # local-only
    frames = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, frames, samplerate, format="WAV")
    return buf.getvalue()

def speak_text(*args, **kwargs):
    """Alias to the new speak_text_bytes; returns MP3 bytes or None."""
    return speak_text_bytes(*args, **kwargs)
