import streamlit as st
import os
import time
import traceback
import hashlib
import uuid
from pathlib import Path

# --- Core / Agents ---
from core.resume_parser import parse_resume
from agent.round_manger import AVAILABLE_ROUNDS   # <-- fix: was round_manger
from agent.interview_agent import InterviewAgent
from utils.config import TEMP_AUDIO_FILENAME

# --- Audio I/O (TTS + STT) ---
from core.audio_io import speak_text, transcribe_audio
# (we won't use record_audio anymore because we implement async start/stop here)

# --- Feedback ---
from core.feedback_generator import generate_feedback_and_scores

# --- Config ---
from utils import config

# Prefer the single constant your audio layer actually uses
TEMP_AUDIO_FILE = TEMP_AUDIO_FILENAME

# ================== Local async audio recorder (Windows-safe) ==================
# You initially had this inside the same file; keeping it here for convenience.
import queue
import wave
import threading
import time as _time
import sounddevice as sd

# Globals for async recording
_record_q = None
_record_stream = None
_record_thread = None
_record_wf = None
_record_lock = threading.Lock()
_record_running = False

def start_recording(filename: str, samplerate: int = 16000, channels: int = 1) -> str:
    """
    Non-blocking recording to a WAV file. Call stop_recording() to finish.
    Returns the path where audio will be written.
    """
    global _record_q, _record_stream, _record_thread, _record_wf, _record_running

    with _record_lock:
        if _record_running:
            raise RuntimeError("A recording is already in progress. Stop it first.")

        # Ensure target folder exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        _record_q = queue.Queue()
        _record_wf = wave.open(filename, "wb")
        _record_wf.setnchannels(channels)
        _record_wf.setsampwidth(2)  # 16-bit PCM
        _record_wf.setframerate(samplerate)

        def _callback(indata, frames, time_info, status):
            if status:
                # Could log: print(f"[InputStream status] {status}")
                pass
            _record_q.put(indata.copy())

        _record_stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            callback=_callback,
        )

        def _writer():
            # Flush audio from queue into the WAV until stopped
            while _record_running:
                try:
                    chunk = _record_q.get(timeout=0.1)
                    _record_wf.writeframes(chunk.tobytes())
                except queue.Empty:
                    continue

        _record_stream.start()
        _record_running = True
        _record_thread = threading.Thread(target=_writer, daemon=True)
        _record_thread.start()

        return filename

def stop_recording() -> str | None:
    """
    Stop active recording and close WAV cleanly (Windows-safe).
    Returns path to the recorded file or None if nothing was recorded.
    """
    global _record_q, _record_stream, _record_thread, _record_wf, _record_running

    with _record_lock:
        if not _record_running:
            return None
        _record_running = False

    # Stop and close audio stream
    try:
        if _record_stream:
            _record_stream.stop()
            _record_stream.close()
    except Exception:
        pass
    finally:
        _record_stream = None

    # Wait for writer thread to flush
    if _record_thread:
        _record_thread.join(timeout=2.0)
        _record_thread = None

    path = None
    if _record_wf:
        try:
            path = getattr(_record_wf._file, "name", None)
        except Exception:
            path = None
        try:
            _record_wf.close()
        except Exception:
            pass
        _record_wf = None

    _record_q = None

    # Give Windows a moment to fully release the file handle
    _time.sleep(0.25)
    return path

# ---------------------- Windows-safe file deletion helpers ---------------------
def safe_delete(path: str, retries: int = 5, delay: float = 0.3):
    """
    Try to delete a file, retrying if Windows keeps it locked (WinError 32).
    """
    if not path:
        return
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError as e:
            # WinError 32: The process cannot access the file because it is being used by another process
            if i < retries - 1:
                _time.sleep(delay)
                continue
            else:
                st.warning(f"Could not remove temporary file {path}: {e}")
                return
        except Exception as e:
            st.warning(f"Could not remove temporary file {path}: {e}")
            return

def cleanup_temp_file(file_path, retries=5, delay=0.3):
    """
    Backwards-compatible wrapper using safe_delete with retries.
    """
    safe_delete(file_path, retries=retries, delay=delay)

# ============================== Streamlit UI ==================================
st.set_page_config(page_title="AceBot Interviewer", layout="wide")
st.title("AceBot Interviewer")
st.markdown("Upload your resume, choose **an** interview round, and practice with an AI interviewer!")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses AI (OpenAI & ElevenLabs) to conduct a mock interview based on your resume. "
    "Upload your resume, select a round, answer questions by voice, and receive feedback/scores."
)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _parse_resume_cached(path: str, file_hash: str) -> str | None:
    return parse_resume(path)

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

UPLOAD_DIR = Path("data/uploads")

def save_uploaded_file(uploaded_file) -> str | None:
    """Save a Streamlit UploadedFile to disk (with basic safety) and return the path."""
    if not uploaded_file:
        return None
    try:
        # Size guard (e.g., 10 MB)
        max_mb = 10
        if uploaded_file.size and uploaded_file.size > max_mb * 1024 * 1024:
            st.error(f"File too large (> {max_mb} MB).")
            return None

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Derive a safe name and avoid collisions
        orig_name = Path(uploaded_file.name).name
        safe_stem = Path(orig_name).stem[:64] or "resume"
        ext = Path(orig_name).suffix.lower()
        if ext not in {".pdf", ".docx"}:
            st.error("Unsupported file type. Please upload a PDF or DOCX.")
            return None

        dest = UPLOAD_DIR / f"{safe_stem}-{uuid.uuid4().hex[:8]}{ext}"
        uploaded_file.seek(0)
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(dest)
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

# ---------- Session ----------
ss = st.session_state
ss.setdefault("stage", "upload")
ss.setdefault("resume_text", None)
ss.setdefault("interview_agent", None)
ss.setdefault("selected_round_key", None)
ss.setdefault("questions", None)
ss.setdefault("current_question_index", 0)
ss.setdefault("interview_history", [])
ss.setdefault("feedback", None)
ss.setdefault("temp_resume_path", None)
# recording state
ss.setdefault("is_recording", False)
ss.setdefault("recording_path", None)

# ---------- Keys / Feature flags ----------
missing = []
if not getattr(config, "OPENAI_API_KEY", None):
    missing.append("OpenAI")
if not getattr(config, "ELEVENLABS_API_KEY", None):
    st.info("ElevenLabs key not found — voice will fall back to on-screen text.")
keys_blocking = "OpenAI" in missing
if keys_blocking:
    st.error("OpenAI API key not found. Please set it in your .env and reload.")
    st.stop()

# ---------- Stage 1: Upload ----------
if ss.stage == "upload":
    st.header("1) Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        ss.temp_resume_path = save_uploaded_file(uploaded_file)
        if ss.temp_resume_path:
            file_hash = _hash_file(ss.temp_resume_path)
            with st.spinner("Parsing resume..."):
                ss.resume_text = _parse_resume_cached(ss.temp_resume_path, file_hash)

            if ss.resume_text:
                st.success("Resume parsed successfully!")
                try:
                    ss.interview_agent = InterviewAgent(ss.resume_text)
                    ss.stage = "select_round"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize interview agent: {e}")
                    ss.resume_text = None
                    ss.interview_agent = None
                    cleanup_temp_file(ss.temp_resume_path)
                    ss.temp_resume_path = None
            else:
                st.error("Could not extract text from the resume. Try a different file.")
                cleanup_temp_file(ss.temp_resume_path)
                ss.temp_resume_path = None

# ---------- Stage 2: Select Round ----------
if ss.stage == "select_round":
    st.header("2) Select Interview Round")

    if not ss.interview_agent:
        st.error("Interview agent not initialized. Please upload the resume first.")
        ss.stage = "upload"
        cleanup_temp_file(ss.temp_resume_path)
        ss.temp_resume_path = None
        st.rerun()

    round_options = {key: info["name"] for key, info in AVAILABLE_ROUNDS.items()}
    ss.selected_round_key = st.selectbox(
        "Choose the type of interview round:",
        options=list(round_options.keys()),
        format_func=lambda key: round_options[key],
    )

    if st.button("Start Interview Round", key="start_interview"):
        if ss.selected_round_key:
            info = AVAILABLE_ROUNDS[ss.selected_round_key]
            ss.current_question_index = 0
            ss.interview_history = []
            ss.feedback = None

            agent = ss.interview_agent
            with st.spinner(f"Generating questions for the {info['name']} round..."):
                try:
                    if hasattr(agent, "generate_questions"):
                        ss.questions = agent.generate_questions(info["name"], info["num_questions"])
                    else:
                        ss.questions = agent._generate_questions(info["name"], info["num_questions"])  # noqa
                except Exception as e:
                    st.error(f"Error generating questions: {e}")
                    st.error(traceback.format_exc())
                    ss.questions = []

            if ss.questions:
                ss.stage = "interviewing"
                st.success(f"Questions ready. Starting the {info['name']} round!")
                try:
                    speak_text(
                        f"Welcome to the {info['name']} round. I will ask you {len(ss.questions)} questions. Let's begin."
                    )
                except Exception as e:
                    st.warning(f"Could not play welcome audio: {e}.")
                st.rerun()
            else:
                st.error("Failed to generate questions. Please try again.")
        else:
            st.warning("Please select a round first.")

# ---------- Stage 3: Interviewing ----------
if ss.stage == "interviewing":
    round_name = AVAILABLE_ROUNDS[ss.selected_round_key]["name"]
    st.header(f"Interviewing: {round_name}")

    if not ss.questions:
        st.error("No questions loaded for this round. Please go back and select again.")
        if st.button("Go Back"):
            # best-effort stop if recording
            if ss.get("is_recording", False):
                try:
                    stop_recording()
                except Exception:
                    pass
                ss.is_recording = False
                ss.recording_path = None
            ss.stage = "select_round"
            st.rerun()
        st.stop()

    q_idx = ss.current_question_index
    if q_idx < len(ss.questions):
        question = ss.questions[q_idx]
        st.subheader(f"Question {q_idx + 1}/{len(ss.questions)}")
        st.markdown(f"**Interviewer:** {question}")

        # Speak only once per question
        spoken_flag = f"spoken_q{q_idx}"
        if not ss.get(spoken_flag, False):
            try:
                speak_text(question)
            except Exception as e:
                st.warning(f"Could not play question audio: {e}")
            ss[spoken_flag] = True

        st.markdown("**Record your answer:**")

        # Unique temp filename per question; force .wav for consistency
        q_audio_path = str(
            Path(TEMP_AUDIO_FILE).with_suffix("").with_name(
                f"{Path(TEMP_AUDIO_FILE).stem}_q{q_idx}.wav"
            )
        )

        c1, c2, c3 = st.columns(3)

        # START
        with c1:
            has_prior_take = bool(ss.get(f"audio_path_q{q_idx}") or ss.get(f"transcript_q{q_idx}"))
            start_disabled = ss.get("is_recording", False) or has_prior_take
            if st.button("▶️ Start Recording", disabled=start_disabled, key=f"start_q{q_idx}"):
                try:
                    # If somehow a prev recording was hanging, stop it
                    if ss.get("is_recording", False):
                        try:
                            stop_recording()
                        except Exception:
                            pass
                        ss.is_recording = False
                        ss.recording_path = None

                    start_recording(q_audio_path, samplerate=16000, channels=1)
                    ss.is_recording = True
                    ss.recording_path = q_audio_path
                    st.toast("Recording started")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not start recording: {e}")

        # STOP
        with c2:
            stop_disabled = not ss.get("is_recording", False)
            if st.button("⏹ Stop Recording", disabled=stop_disabled, key=f"stop_q{q_idx}"):
                try:
                    final_path = stop_recording()  # may return None on failure
                except Exception as e:
                    final_path = None
                    st.error(f"Could not stop recording: {e}")

                ss.is_recording = False

                # Prefer the returned path; else fall back to the known path
                if final_path and os.path.exists(final_path):
                    ss[f"audio_path_q{q_idx}"] = final_path
                elif ss.get("recording_path") and os.path.exists(ss["recording_path"]):
                    ss[f"audio_path_q{q_idx}"] = ss["recording_path"]
                else:
                    ss[f"audio_path_q{q_idx}"] = None

                ss.recording_path = None

                # Small delay before any operation on the file (Windows handle release)
                time.sleep(0.25)

                # Auto-transcribe on stop if we have a file
                if ss.get(f"audio_path_q{q_idx}"):
                    with st.spinner("Transcribing..."):
                        try:
                            txt = transcribe_audio(ss[f"audio_path_q{q_idx}"]) or ""
                        except Exception as e:
                            st.warning(f"Transcription error: {e}")
                            txt = ""
                        ss[f"transcript_q{q_idx}"] = txt
                        if not txt:
                            st.warning("Could not transcribe that recording. You can re-record.")
                else:
                    st.warning("No audio captured. Try recording again.")
                st.rerun()

        # SUBMIT
        with c3:
            submit_disabled = ss.get("is_recording", False)
            if st.button("✅ Submit Answer", disabled=submit_disabled, key=f"submit_q{q_idx}"):
                answer_text = ss.get(f"transcript_q{q_idx}", "") or "[No response recorded]"
                ss.interview_history.append({"question": question, "answer": answer_text})

                # Clean last temp audio eagerly (Windows-safe)
                safe_delete(ss.get(f"audio_path_q{q_idx}"))
                for k in (f"audio_path_q{q_idx}", f"transcript_q{q_idx}"):
                    if k in ss:
                        del ss[k]

                ss.current_question_index += 1
                if ss.current_question_index < len(ss.questions):
                    try:
                        speak_text("Okay, thank you. Next question.")
                    except Exception as e:
                        st.warning(f"Audio notification error: {e}")
                st.rerun()

        # Show current state / controls below
        if ss.get("is_recording", False):
            st.info("🎙️ Recording... Click **Stop Recording** when you're done.")
        elif has_prior_take:
            st.info("A recording exists for this question. You can **Submit Answer** or **Re-record**.")

        # Playback + transcript (if any)
        if ss.get(f"audio_path_q{q_idx}"):
            st.audio(ss[f"audio_path_q{q_idx}"])

        transcript = ss.get(f"transcript_q{q_idx}", "")
        if transcript:
            st.markdown("**Transcript (auto-generated):**")
            st.write(transcript)

        # Re-record button
        c_rr, _ = st.columns([1, 5])
        with c_rr:
            if st.button("↺ Re-record", key=f"rerecord_q{q_idx}"):
                # If currently recording, stop first to avoid leaks
                if ss.get("is_recording", False):
                    try:
                        stop_recording()
                    except Exception:
                        pass
                    ss.is_recording = False

                # Remove any existing audio/transcript for this question (Windows-safe)
                safe_delete(ss.get(f"audio_path_q{q_idx}"))
                for k in (f"audio_path_q{q_idx}", f"transcript_q{q_idx}"):
                    if k in ss:
                        del ss[k]
                ss.recording_path = None
                st.rerun()

    else:
        # best-effort stop if recording (avoid dangling handles)
        if ss.get("is_recording", False):
            try:
                stop_recording()
            except Exception:
                pass
            ss.is_recording = False
            ss.recording_path = None

        st.success("All questions for this round are completed.")
        ss.stage = "feedback"
        try:
            speak_text("Thank you. That concludes this round. I will now prepare your feedback.")
        except Exception as e:
            st.warning(f"Audio notification error: {e}")
        st.rerun()

# ---------- Stage 4: Feedback ----------
if ss.stage == "feedback":
    st.header("Interview Complete — Feedback")
    agent = ss.interview_agent
    round_name = AVAILABLE_ROUNDS[ss.selected_round_key]["name"]

    if not ss.feedback and agent and ss.interview_history:
        with st.spinner("Generating feedback..."):
            try:
                ss.feedback = generate_feedback_and_scores(
                    resume_text=ss.resume_text,
                    round_name=round_name,
                    qa_pairs=ss.interview_history,
                )
            except Exception as e:
                st.error(f"Failed to generate feedback: {e}")
                st.error(traceback.format_exc())

    if ss.feedback:
        data = ss.feedback
        st.subheader("Overall Feedback")
        st.markdown(data.get("overall_feedback", "N/A"))

        st.subheader("Suggestions for Improvement")
        suggestions = data.get("suggestions", "N/A")
        if isinstance(suggestions, list):
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.markdown(suggestions)

        st.subheader("Scores per Question")
        scores = data.get("scores_per_question", [])
        total_score = int(data.get("total_score", 0))
        max_score = len(ss.interview_history) * 10

        if scores and len(scores) == len(ss.interview_history):
            for i, sc in enumerate(scores):
                st.markdown(f"- **Q{i+1}:** {int(sc)}/10")
        elif scores:
            st.warning(
                f"Note: Number of scores ({len(scores)}) doesn't match number of questions "
                f"({len(ss.interview_history)}). Displaying raw scores: {scores}"
            )
        else:
            st.markdown("Scores could not be determined.")

        st.subheader("Total Score for Round")
        st.markdown(f"**{total_score} / {max_score or 1}**")

        with st.expander("Show Raw Feedback Data (debug)"):
            st.json(data)
    else:
        st.warning("Feedback is not available for this round.")

    st.markdown("---")
    if st.button("Start Another Round"):
        # Reset state for a new round, keeping resume and agent
        ss.stage = "select_round"
        ss.selected_round_key = None
        ss.questions = []
        ss.current_question_index = 0
        ss.interview_history = []
        ss.feedback = None
        # Clear spoken flags
        for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith("spoken_q")]:
            del ss[k]
        st.rerun()

    if st.button("Upload New Resume"):
        # Best-effort stop if recording
        if ss.get("is_recording", False):
            try:
                stop_recording()
            except Exception:
                pass
            ss.is_recording = False
            ss.recording_path = None

        # Full reset (also cleans prior resume file)
        cleanup_temp_file(ss.get("temp_resume_path"))
        for key in list(ss.keys()):
            del ss[key]
        st.session_state.stage = "upload"
        st.rerun()
