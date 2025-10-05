import streamlit as st
import os
import time
import traceback
import hashlib
import uuid
from pathlib import Path
import io

# --- Core / Agents ---
from core.resume_parser import parse_resume
from agent.round_manger import AVAILABLE_ROUNDS   # <-- fix: was round_manger
from agent.interview_agent import InterviewAgent
from utils.config import TEMP_AUDIO_FILENAME

# --- Audio I/O (TTS + STT) ---
from core.audio_io import speak_text, transcribe_audio, speak_text_bytes
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
# ===== Suggested answers (improved/ideal responses) =====
def _suggest_answers_offline(resume_text: str, round_name: str, questions: list[str], user_answers: list[str]) -> list[str]:
    """No-API fallback: STAR-style suggested answers with light coaching."""
    resume_hint = (resume_text or "").split("\n")[0][:120]
    out = []
    for i, q in enumerate(questions):
        ua = (user_answers[i] if i < len(user_answers) else "").strip()
        critique = ""
        if ua:
            critique = (
                " Your original answer could be improved by being more specific on metrics, "
                "explicit trade-offs, and closing with the measurable result."
            )
        sug = (
            f"**Situation:** Worked on {resume_hint or 'a recent project'} where {q[:120]}.\n"
            f"**Task:** Clearly defined the problem, success metrics, and constraints (latency, cost, quality).\n"
            f"**Action:** Evaluated options, explained trade-offs, and chose an approach. "
            f"Implemented iteratively, validated with experiments, and monitored with dashboards.\n"
            f"**Result:** Achieved a quantifiable impact (e.g., +28% accuracy, -35% latency). "
            f"Reflected on risks, next steps, and how to generalize.{critique}"
        )
        out.append(sug)
    return out

def _suggest_answers_llm(resume_text: str, round_name: str, questions: list[str], user_answers: list[str]) -> list[str]:
    """OpenAI-powered suggested answers. Falls back to offline if not available."""
    try:
        import os, json
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("no OPENAI_API_KEY")

        # Prefer new OpenAI Responses API
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            qa_block = "\n".join(
                [f"{i+1}) Q: {q}\n   A: {(user_answers[i] if i < len(user_answers) else '').strip()}"
                 for i, q in enumerate(questions)]
            )
            prompt = (
                "You are an interview coach. For each question, produce a concise, high-quality suggested answer "
                "that follows STAR and includes specific metrics/trade-offs when relevant. "
                "Return JSON: {\"suggested\": [\"...\", \"...\"]} with the same length as the number of questions.\n\n"
                f"ROUND: {round_name}\n\nRESUME EXCERPT:\n{(resume_text or '')[:2000]}\n\n"
                f"Q&A (user answers may be empty):\n{qa_block}"
            )
            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=prompt,
                temperature=0.25,
                max_output_tokens=1800,
                response_format={"type": "json_object"},
            )
            raw = (resp.output_text or "").strip()
        except Exception:
            # Legacy
            import openai as legacy_openai
            legacy_openai.api_key = api_key
            qa_block = "\n".join(
                [f"{i+1}) Q: {q}\n   A: {(user_answers[i] if i < len(user_answers) else '').strip()}"
                 for i, q in enumerate(questions)]
            )
            msgs = [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content":
                    "You are an interview coach. For each question, produce a concise, high-quality suggested answer "
                    "that follows STAR and includes specific metrics/trade-offs when relevant. "
                    "Return JSON: {\"suggested\": [\"...\", \"...\"]} with the same length as the number of questions.\n\n"
                    f"ROUND: {round_name}\n\nRESUME EXCERPT:\n{(resume_text or '')[:2000]}\n\n"
                    f"Q&A (user answers may be empty):\n{qa_block}"
                },
            ]
            raw = legacy_openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msgs,
                temperature=0.25,
                max_tokens=1800,
            )["choices"][0]["message"]["content"]

        data = json.loads(raw)
        suggested = data.get("suggested", [])
        if not isinstance(suggested, list) or not suggested:
            raise ValueError("LLM returned empty/invalid suggestions")

        # match length
        return (suggested + [""] * len(questions))[:len(questions)]
    except Exception:
        return _suggest_answers_offline(resume_text, round_name, questions, user_answers)

# ============================== Streamlit UI ==================================
st.set_page_config(page_title="Interview GenAIe", layout="wide")
st.title("Interview GenAIe")
st.markdown("Upload your resume, choose **an** interview round, and practice with an AI interviewer!")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses AI (OpenAI & ElevenLabs) to conduct a mock interview based on your resume. "
    "Upload your resume, select a round, answer questions by voice, and receive feedback/scores."
)


# ===== Demo helpers (no mic needed) =====
def _demo_generate_answers_offline(resume_text: str, questions: list[str]) -> list[str]:
    """
    No-API, deterministic STAR-ish answers so you can test UI quickly.
    Uses resume_text lightly to personalize.
    """
    who = "I"  # keep it neutral
    resume_hint = (resume_text or "").split("\n")[0][:120]
    answers = []
    for i, q in enumerate(questions, start=1):
        ans = (
            f"**Situation:** {who} worked on a project related to {resume_hint or 'my recent role'}.\n"
            f"**Task:** The goal was to address the problem implied by the question: '{q[:120]}'.\n"
            f"**Action:** {who} scoped requirements, evaluated trade-offs, and implemented a clear solution with measurable checkpoints.\n"
            f"**Result:** Reduced errors by ~25% and improved speed by ~30%. Key learning: communicate assumptions early and quantify impact."
        )
        answers.append(ans)
    return answers

def _demo_generate_answers_llm(resume_text: str, questions: list[str]) -> list[str]:
    """
    Optional OpenAI-powered answers. If OPENAI isn't set, falls back offline.
    """
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("no OPENAI_API_KEY")

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are simulating a candidate in a mock interview.\n"
            "Write concise, STAR-structured answers (120-180 words each), "
            "using the resume excerpt provided. Return JSON: {\"answers\": [\"...\", \"...\"]}\n\n"
            f"RESUME EXCERPT:\n{(resume_text or '')[:2000]}\n\n"
            f"QUESTIONS:\n" + "\n".join([f'{i+1}. {q}' for i, q in enumerate(questions)])
        )
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
            temperature=0.3,
            max_output_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = (resp.output_text or "").strip()
        import json
        data = json.loads(raw)
        answers = data.get("answers", [])
        if not isinstance(answers, list) or not answers:
            raise ValueError("LLM returned empty/invalid answers")
        return answers[:len(questions)]
    except Exception:
        return _demo_generate_answers_offline(resume_text, questions)

# ---------- Helpers ----------
def cleanup_all_recordings():
    for f in Path("data/recordings").glob("temp_user_response_*.wav"):
        safe_delete(f)

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
ss.setdefault("questions", 1)
ss.setdefault("current_question_index", 0)
ss.setdefault("interview_history", [])
ss.setdefault("feedback", None)
ss.setdefault("temp_resume_path", None)
# recording state
ss.setdefault("is_recording", False)
ss.setdefault("recording_path", None)
ss.setdefault("rec_audio", {})       # {qid: bytes of the last recording}
ss.setdefault("rec_transcript", {})  # {qid: str transcript}


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
        st.markdown("— or —")

    if st.button("🧪 Run Demo (auto-generate Q&A + feedback)", key="run_demo"):
        info = AVAILABLE_ROUNDS.get(ss.selected_round_key) if ss.selected_round_key else None
        if not info:
            st.warning("Please select a round first.")
        else:
            # 1) Generate questions via your existing agent
            try:
                if hasattr(ss.interview_agent, "generate_questions"):
                    demo_questions = ss.interview_agent.generate_questions(info["name"], info["num_questions"])
                else:
                    demo_questions = ss.interview_agent._generate_questions(info["name"], info["num_questions"])  # noqa
            except Exception as e:
                st.error(f"Error generating questions: {e}")
                demo_questions = []

            # 2) Use resume (or a tiny built-in fallback) to generate demo answers
            demo_resume = ss.resume_text or (
                "Software/Data professional with experience in ML pipelines, GNN/LSTM models, and "
                "analytics dashboards. Led projects improving accuracy and speed, collaborating with cross-functional teams."
            )

            # Prefer LLM answers if OPENAI is set; otherwise offline STAR-ish stubs
            use_llm = bool(os.getenv("OPENAI_API_KEY"))
            demo_answers = _demo_generate_answers_llm(demo_resume, demo_questions) if use_llm else _demo_generate_answers_offline(demo_resume, demo_questions)

            # 3) Build Q/A pairs and jump straight to feedback
            ss.questions = demo_questions
            ss.interview_history = [
                {"question": q, "answer": (demo_answers[i] if i < len(demo_answers) else "")}
                for i, q in enumerate(demo_questions)
            ]
            ss.current_question_index = len(ss.questions)  # skip interviewing screen
            ss.stage = "feedback"

            # 4) Generate feedback immediately
            round_name = info["name"]
            with st.spinner("Generating demo feedback..."):
                try:
                    ss.feedback = generate_feedback_and_scores(
                        resume_text=demo_resume,
                        round_name=round_name,
                        qa_pairs=ss.interview_history,
                    )
                except Exception as e:
                    st.error(f"Failed to generate feedback: {e}")
                    st.error(traceback.format_exc())
            st.rerun()


# ---------- Stage 3: Interviewing ----------
# ---------- Stage 3: Interviewing ----------
if ss.stage == "interviewing":
    round_name = AVAILABLE_ROUNDS[ss.selected_round_key]["name"]
    st.header(f"Interviewing: {round_name}")

    # Guard: no questions / end of interview
    if not ss.questions:
        st.error("No questions available. Please start a round again.")
        st.stop()
    if ss.current_question_index >= len(ss.questions):
        ss.stage = "feedback"
        st.rerun()

    # Current question
    question_text = ss.questions[ss.current_question_index]
    st.markdown(f"**Interviewer:** {question_text}")

    # --- TTS per question (generate once) ---
    qid = f"hr-q-{ss.current_question_index}"
    if "tts_audio" not in ss:
        ss["tts_audio"] = {}
    if "tts_done" not in ss:
        ss["tts_done"] = {}

    if not ss["tts_done"].get(qid):
        tts_bytes = speak_text_bytes(question_text)  # MP3 bytes
        if tts_bytes:
            ss["tts_audio"][qid] = tts_bytes
        ss["tts_done"][qid] = True

    tts_bytes = ss["tts_audio"].get(qid)
    if tts_bytes:
        st.audio(io.BytesIO(tts_bytes), format="audio/mp3")
    else:
        st.warning("TTS not available (check API key/voice/quota).")

    # --- Recording controls ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Start
        if not ss.is_recording and st.button("🎙️ Start recording"):
            try:
                path = start_recording(TEMP_AUDIO_FILE, samplerate=16000, channels=1)
                ss.is_recording = True
                ss.recording_path = path
                st.toast("Recording started…", icon="🎙️")
            except Exception as e:
                st.error(f"Failed to start recording: {e}")
        elif ss.is_recording:
            st.info("Recording… click Stop when done.")

    with col2:
        # Stop -> auto-transcribe
        if ss.is_recording and st.button("⏹ Stop (auto-transcribe)"):
            try:
                path = stop_recording()
            except Exception as e:
                path = None
                st.error(f"Failed to stop recording: {e}")
            ss.is_recording = False
            path = path or ss.recording_path

            # Read bytes for preview BEFORE deleting
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        audio_bytes = f.read()
                    ss.rec_audio[qid] = audio_bytes

                    # Auto-transcribe directly from the file
                    text = transcribe_audio(path)
                    if text:
                        ss.rec_transcript[qid] = text
                        st.success("Transcription captured ✅")
                    else:
                        st.warning("No transcription captured (silence/network/API).")
                finally:

                    ss.recording_path = None
            else:
                st.warning("No audio file found after stop.")

    with col3:
        # Re-record: flush & start fresh immediately
        if st.button("🔁 Re-record"):
            ss.rec_audio.pop(qid, None)
            ss.rec_transcript.pop(qid, None)
            # stop any stray stream
            if ss.is_recording:
                try:
                    stop_recording()
                except Exception:
                    pass
                ss.is_recording = False
            # start a fresh recording
            try:
                path = start_recording(TEMP_AUDIO_FILE, samplerate=16000, channels=1)
                ss.is_recording = True
                ss.recording_path = path
                st.toast("Recording restarted…", icon="🔁")
            except Exception as e:
                st.error(f"Failed to start new recording: {e}")

    with col4:
        # Submit -> save + next (or feedback)
        disabled = not bool(ss.rec_transcript.get(qid))
        if st.button("✅ Submit", disabled=disabled):
            text = ss.rec_transcript.get(qid)
            if not text:
                st.warning("Please record (and let it auto-transcribe) before submitting.")
            else:
                ss.interview_history.append({"question": question_text, "answer": text})
                # clear per-Q buffers
                ss.rec_audio.pop(qid, None)
                ss.rec_transcript.pop(qid, None)
                ss["tts_done"].pop(qid, None)
                ss["tts_audio"].pop(qid, None)
                # advance or finish
                ss.current_question_index += 1
                if ss.current_question_index >= len(ss.questions):
                    ss.stage = "feedback"
                st.rerun()

    # --- Preview (if any) ---
    if ss.rec_audio.get(qid):
        st.audio(io.BytesIO(ss.rec_audio[qid]), format="audio/wav")
    if ss.rec_transcript.get(qid):
        st.markdown("**Transcript:**")
        st.write(ss.rec_transcript[qid])

# ---------- Stage 4: Feedback ----------
if ss.stage == "feedback":
    st.header("Interview Complete — Feedback")
    agent = ss.interview_agent
    round_name = AVAILABLE_ROUNDS[ss.selected_round_key]["name"]

    if not ss.feedback and agent and ss.interview_history:
        with st.spinner("Generating feedback..."):
            try:
                ss.feedback = generate_feedback_and_scores(
                    resume_text=ss.resume_text or "",
                    round_name=round_name,
                    qa_pairs=ss.interview_history,
                )
            except Exception as e:
                st.error(f"Failed to generate feedback: {e}")
                st.error(traceback.format_exc())

    # ---- Suggested answers (generate once and cache in session) ----
    if "suggested_answers" not in ss and ss.interview_history:
        questions = [qa["question"] for qa in ss.interview_history]
        user_answers = [qa.get("answer", "") for qa in ss.interview_history]
        with st.spinner("Preparing suggested answers..."):
            ss.suggested_answers = _suggest_answers_llm(
                ss.resume_text or "", round_name, questions, user_answers
            )

    # ===== Tabs: Feedback | Suggested Answers =====
    tab_feedback, tab_suggested = st.tabs(["Feedback", "Suggested answers"])

    with tab_feedback:
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

    with tab_suggested:
        st.subheader("Suggested answers to your questions")
        if ss.interview_history and ss.get("suggested_answers"):
            for i, qa in enumerate(ss.interview_history):
                q = qa["question"]
                user_a = qa.get("answer", "")
                sug_a = ss.suggested_answers[i] if i < len(ss.suggested_answers) else ""
                with st.expander(f"Q{i+1}. {q}"):
                    if user_a:
                        st.markdown("**Your answer:**")
                        st.markdown(user_a)
                    st.markdown("**Suggested answer:**")
                    st.markdown(sug_a)
                    # copy-friendly text box
                    st.text_area("Copy/edit:", value=sug_a, height=200, key=f"copy_sug_{i}")
        else:
            st.info("Suggested answers will appear here once questions and feedback are available.")

    st.markdown("---")
    if st.button("Start Another Round"):
        ss.stage = "select_round"
        ss.selected_round_key = None
        ss.questions = []
        ss.current_question_index = 0
        ss.interview_history = []
        ss.feedback = None
        ss.suggested_answers = None
        for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith("spoken_q")]:
            del ss[k]
        st.rerun()

    if st.button("Upload New Resume"):
        if ss.get("is_recording", False):
            try:
                stop_recording()
            except Exception:
                pass
            ss.is_recording = False
            ss.recording_path = None
        cleanup_temp_file(ss.get("temp_resume_path"))
        for key in list(ss.keys()):
            del ss[key]
        st.session_state.stage = "upload"
        st.rerun()
