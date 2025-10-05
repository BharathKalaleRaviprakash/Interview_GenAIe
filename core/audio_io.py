import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import numpy as np
import time
import os, io
import streamlit as st 

from utils.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    RECORDING_SAMPLE_RATE,
    RECORDING_CHANNELS,
    RECORDING_DURATION_SECONDS,
    TEMP_AUDIO_FILENAME,
)

# ---------- ElevenLabs client (robust init) ----------
try:
    el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
except Exception as e:
    print(f"[WARN] ElevenLabs init failed: {e}")
    el_client = None

r = sr.Recognizer()

def _fallback_say(text: str):
    text = text or ""
    print(f"Interviewer: {text}")
    # ~3 wps reading speed; avoid zero/negatives
    est = max(0.5, len(text.split()) / 3.0)
    time.sleep(est)

import io, time
from typing import Optional

# --- TTS bytes + browser autoplay helpers ---

import base64
import streamlit as st
import streamlit.components.v1 as components

def _ensure_bytes(audio_obj) -> bytes:
    if isinstance(audio_obj, (bytes, bytearray)):
        return bytes(audio_obj)
    chunks = []
    for ch in audio_obj:
        if not ch:
            continue
        chunks.append(bytes(ch) if not isinstance(ch, (bytes, bytearray)) else ch)
    return b"".join(chunks)

def _resolve_voice_id(cli, hint: str | None) -> str | None:
    hint = (hint or "").strip()
    if hint and hint.isalnum() and len(hint) >= 18:
        return hint
    try:
        res = cli.voices.search(query=hint or "Rachel")
        if getattr(res, "voices", None):
            exact = next((v for v in res.voices if v.name.lower() == (hint or "Rachel").lower()), None)
            return (exact or res.voices[0]).voice_id
    except Exception as e:
        print(f"[TTS] Voice search failed: {e}")
    return None

def speak_text_bytes(text: str, voice_hint: str | None = None) -> bytes | None:
    """Return MP3 bytes for the given text (no local playback)."""
    text = (text or "").strip()
    if not text or not el_client:
        return None
    try:
        # free device if busy
        try:
            sd.stop()
        except Exception:
            pass

        vid = _resolve_voice_id(el_client, voice_hint or ELEVENLABS_VOICE_ID or "Rachel")
        if not vid:
            return None
        stream = el_client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        return _ensure_bytes(stream)
    except Exception as e:
        print(f"[TTS] speak_text_bytes failed: {e}")
        return None

def autoplay_audio_in_browser(audio_bytes: bytes, key: str = "tts") -> None:
    """Try to autoplay; if blocked, arm the next click/keypress to play it."""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(
        f"""
        <audio id="{key}" src="data:audio/mp3;base64,{b64}" autoplay></audio>
        <script>
          (function() {{
            const el = document.getElementById("{key}");
            function arm() {{
              const go = () => {{
                el.play().catch(()=>{{}});
                window.removeEventListener('pointerdown', go);
                window.removeEventListener('keydown', go);
              }};
              window.addEventListener('pointerdown', go, {{ once: true }});
              window.addEventListener('keydown', go, {{ once: true }});
            }}
            el.play().catch(arm);
          }})();
        </script>
        """,
        height=0
    )

def speak_text(
    text: str,
    *,
    voice_id: str | None = None,
    show_in_streamlit: bool = True,
    also_play_locally: bool = False  # plays on server/host speakers
) -> bool:
    text = (text or "").strip()
    if not text:
        return False

    if not el_client or not ELEVENLABS_API_KEY:
        print("[INFO] ElevenLabs not configured; printing instead.")
        _fallback_say(text)
        if st: st.info("TTS fallback (no API key).")
        return False

    # Free any busy audio device
    try: sd.stop()
    except Exception: pass

    vid = _resolve_voice_id(el_client, voice_id or ELEVENLABS_VOICE_ID or "Rachel")
    if not vid:
        if st: st.warning("Could not resolve ElevenLabs voice. Set ELEVENLABS_VOICE_ID to a valid id or a name like 'Rachel'.")
        _fallback_say(text)
        return False

    try:
        # Your SDK returns a streaming generator here:
        audio_stream = el_client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        # 🔧 Convert generator → bytes once
        audio_bytes = _ensure_bytes(audio_stream)
        if not audio_bytes:
            raise RuntimeError("TTS returned empty audio")

        # Browser playback (Streamlit)
        if show_in_streamlit and st:
            st.audio(io.BytesIO(audio_bytes), format="audio/mp3")

        # Optional: server/desktop speakers
        if also_play_locally:
            try:
                play(audio_bytes)
            except Exception as pe:
                print(f"[TTS] Local playback error: {pe}")

        print(f"[TTS] OK: {len(text.split())} words, {len(audio_bytes)} bytes, voice_id={vid}")
        return True

    except Exception as e:
        print(f"[TTS] ElevenLabs error: {e}")
        _fallback_say(text)
        if st: st.error(f"ElevenLabs TTS failed: {e}")
        return False
   
def record_audio(
    duration: int = RECORDING_DURATION_SECONDS,
    filename: str = TEMP_AUDIO_FILENAME) -> str | None:
    """Record a short clip and save to WAV."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except Exception:
        # If dirname is empty (e.g., just "file.wav"), that's fine
        pass

    print(f"Recording for {duration} seconds...")
    try:
        frames = int(duration * RECORDING_SAMPLE_RATE)
        if frames <= 0:
            raise ValueError("Duration * sample rate must be > 0")

        recording = sd.rec(
            frames,
            samplerate=RECORDING_SAMPLE_RATE,
            channels=RECORDING_CHANNELS,
            dtype="float32"
        )
        sd.wait()  # block until done

        if recording is None or recording.size == 0:
            print("[ERROR] Empty audio buffer.")
            return None

        peak = np.max(np.abs(recording))
        if peak > 0:
            recording = recording / peak  # normalize to -1..1

        # If downstream tools prefer int16:
        # recording_i16 = (recording * 32767).astype(np.int16)
        # sf.write(filename, recording_i16, RECORDING_SAMPLE_RATE, subtype="PCM_16")

        sf.write(filename, recording, RECORDING_SAMPLE_RATE)  # float32 WAV
        print(f"Recording saved to {filename}")
        return filename
    except Exception as e:
        print("[ERROR] Recording failed:", e)
        return None

def transcribe_audio(filename: str = TEMP_AUDIO_FILENAME) -> str | None:
    """Transcribe a WAV file with Google Web Speech; returns None on failure."""
    print(f"Transcribing audio from {filename}...")
    if not filename or not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        return None

    try:
        with sr.AudioFile(filename) as source:
            # Improve robustness in noisy rooms
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)

        # Add language for better accuracy; set show_all=True for debugging
        text = r.recognize_google(audio_data, language="en-US")
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("[INFO] Could not understand the audio.")
    except sr.RequestError as e:
        print(f"[ERROR] Google SR request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Transcription error: {e}")
    return None

# --------- Example flow (comment out in production) ---------
# speak_text("Hi, tell me about your last project.")
# path = record_audio(6, TEMP_AUDIO_FILE)
# if path:
#     answer = transcribe_audio(path)
#     print("You said:", answer)
