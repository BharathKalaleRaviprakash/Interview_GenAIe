# core/audio_io.py
from __future__ import annotations
import io, os, tempfile
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

DEFAULT_RACHEL_ID = "21m00Tcm4TlvDq8ikWAM"

def _voice(v: str | None) -> str:
    if not v or len(v.strip()) < 10:
        return DEFAULT_RACHEL_ID
    bad = ("_get_secret", "st.secrets", "os.getenv", "default", "VOICE_ID")
    if any(b in v for b in bad):
        return DEFAULT_RACHEL_ID
    return v.strip()

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
    # ---- Fallback: OpenAI TTS ----
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=str(OPENAI_API_KEY).strip())
            resp = client.audio.speech.create(
                model="gpt-4o-mini-tts",   # or "gpt-4o-tts" if you have access
                voice="alloy",
                input=text,
                response_format="mp3",    # <-- was 'format', must be 'response_format'
            )
            return resp.read()  # bytes
        except Exception as e:
            st.warning(f"TTS (OpenAI) error: {e}")


        return None

# Back-compat alias (some modules may still call speak_text)
def speak_text(text: str) -> bytes | None:
    return speak_text_bytes(text)

# ============================ STT ============================

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes using OpenAI. Tries 'gpt-4o-transcribe', falls back to 'whisper-1'.
    """
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=str(OPENAI_API_KEY).strip())
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
    """Convenience wrapper for path-based calls."""
    try:
        with open(file_path, "rb") as f:
            return transcribe_audio_bytes(f.read())
    except Exception:
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
