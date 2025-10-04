import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import play, Voice, VoiceSettings
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
    if not el_client:
        print("[INFO] ElevenLabs not configured; printing instead.")
        _fallback_say(text)
        return

    try:
        print("Generating speech...")
        voice_obj = Voice(
            voice_id=ELEVENLABS_VOICE_ID,
            settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.85,
                style=0.1,
                use_speaker_boost=True
            )
        )
        audio = el_client.generate(
            text=text,
            voice=voice_obj,
            model="eleven_multilingual_v2"
        )
        print("Speaking...")
        play(audio)
        print("Done speaking.")
    except Exception as e:
        print(f"[ERROR] ElevenLabs TTS: {e}")
        _fallback_say(text)

def record_audio(
    duration: int = RECORDING_DURATION_SECONDS,
    filename: str = TEMP_AUDIO_FILE
) -> str | None:
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

def transcribe_audio(filename: str = TEMP_AUDIO_FILE) -> str | None:
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
