import os, streamlit as st

def _get(name, default=None):
    v = st.secrets.get(name, os.getenv(name, default))
    return v.strip() if isinstance(v, str) else v

OPENAI_API_KEY = "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"
ELEVENLABS_API_KEY = "sk_0fd7e1629d8562731149313cb5a8bcfcaf7cd8b4615dfce1"

ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"


RECORDING_SAMPLE_RATE = 44100
RECORDING_CHANNELS = 1
RECORDING_DURATION_SECONDS = 10
TEMP_AUDIO_FILENAME = "data/recordings/temp_user_response.wav"
RESUME_MIN_TEXT_CHARS = 500