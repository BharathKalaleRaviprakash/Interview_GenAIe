import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from elevenlabs import ElevenLabs, play, VoiceSettings  # add Voice import *only if* you really need it
# from elevenlabs import Voice  # uncomment if you prefer constructing Voice objects
import numpy as np
import time
import os

from utils.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    RECORDING_SAMPLE_RATE,
    RECORDING_CHANNELS,
    RECORDING_DURATION_SECONDS,
    TEMP_AUDIO_FILE,
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

def speak_text(text: str):
    """TTS via ElevenLabs; graceful console fallback."""
    if not text:
        return
    if not el_client or not ELEVENLABS_API_KEY:
        print("[INFO] ElevenLabs not configured; printing instead.")
        return _fallback_say(text)

    try:
        print("Generating speech...")
        # EITHER pass a Voice object (needs `from elevenlabs import Voice`)
        # voice_obj = Voice(
        #     voice_id=ELEVENLABS_VOICE_ID,
        #     settings=VoiceSettings(stability=0.75, similarity_boost=0.85, style=0.1, use_speaker_boost=True)
        # )
        # audio = el_client.generate(text=text, voice=voice_obj, model="eleven_multilingual_v2")

        # OR simpler: pass voice_id + settings directly (works on current SDKs)
        audio = el_client.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.75, similarity_boost=0.85, style=0.1, use_speaker_boost=True
            ),
        )
        print("Speaking...")
        play(audio)
        print("Done speaking.")
    except Exception as e:
        print(f"[WARN] TTS error: {e}")
        _fallback_say(text)

def record_audio(
    duration: int = RECORDING_DURATION_SECONDS,
    filename: str = TEMP_AUDIO_FILE,
    samplerate: int = RECORDING_SAMPLE_RATE,
    channels: int = RECORDING_CHANNELS,
    device: int | None = None,
):
    """Record a short clip and save to WAV."""
    duration = max(1, int(duration))
    channels = 1 if channels <= 0 else channels
    print(f"Recording {duration}s @ {samplerate} Hz, {channels} ch...")

    try:
        # Optional: verify device supports the requested format
        if device is not None:
            sd.default.device = device

        # Pre-allocate and record
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
        sd.wait()  # block until done

        # Normalize path & dir
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)
        sf.write(filename, recording, samplerate)
        print(f"Recording saved to {filename}")
        return filename
    except Exception as e:
        print(f"[ERROR] Audio recording failed: {e}")

def transcribe_audio(filename: str = TEMP_AUDIO_FILE) -> str | None:
    """Transcribe a WAV file with Google Web Speech; returns None on failure."""
    print(f"Transcribing audio from {filename}...")
    if not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        return None

    try:
        with sr.AudioFile(filename) as source:
            # Improve robustness in noisy rooms
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)  # for long files, consider r.listen with phrase_time_limit
        text = r.recognize_google(audio_data)  # requires internet
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("[INFO] Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"[ERROR] Google SR request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Transcription error: {e}")
    return None

# --------- Example flow (comment out in production) ---------
# speak_text("Hi, tell me about your last project.")
# record_audio(6, TEMP_AUDIO_FILE)
# answer = transcribe_audio(TEMP_AUDIO_FILE)
# print("You said:", answer)
