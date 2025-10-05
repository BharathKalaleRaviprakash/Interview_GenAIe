import os, streamlit as st

def _get(name, default=None):
    v = st.secrets.get(name, os.getenv(name, default))
    return v.strip() if isinstance(v, str) else v

ELEVENLABS_API_KEY = _get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
OPENAI_API_KEY = _get("OPENAI_API_KEY")


RECORDING_SAMPLE_RATE = 44100
RECORDING_CHANNELS = 1
RECORDING_DURATION_SECONDS = 10
TEMP_AUDIO_FILENAME = "data/recordings/temp_user_response.wav"
RESUME_MIN_TEXT_CHARS = 500