# app.py
from __future__ import annotations

import io
import os
import time
import traceback
import hashlib
import uuid
from pathlib import Path

import streamlit as st

# --- Core / Agents ---
from core.resume_parser import parse_resume
from agent.round_manger import AVAILABLE_ROUNDS   # <-- fix: was round_manger
from agent.interview_agent import InterviewAgent
from utils.config import TEMP_AUDIO_FILENAME
from utils import config

# --- Audio I/O (cloud-safe: no PortAudio) ---
from core.audio_io import speak_text_bytes, transcribe_audio_bytes, transcribe_audio

from core.resume_parser import classify_document
# --- Mic widgets (optional) ---
HAVE_MIC_RECORDER = False
HAVE_AUDIO_RECORDER = False
MIC_IMPORT_ERROR = None
CONTACT_EMAIL = st.secrets.get("CONTACT_EMAIL", "jahnaviashok99@gmail.com")

try:
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    HAVE_MIC_RECORDER = True
except Exception as e:
    MIC_IMPORT_ERROR = str(e)
    try:
        from audio_recorder_streamlit import audio_recorder
        HAVE_AUDIO_RECORDER = True
    except Exception:
        pass


from core.demo_mode_llm import generate_llm_demo_qa


# --- Feedback ---
from core.feedback_generator import generate_feedback_and_scores


# ============================== Streamlit UI ==================================
st.set_page_config(page_title="Interview GenAIe", layout="wide")
st.title("Interview GenAIe")
st.markdown("Upload your resume, choose **an** interview round, and practice with an AI interviewer!")

# Quick state
first = (st.session_state.get("candidate_first") or "there").split()[0].title()
stage_map = {"upload": "Upload", "select_round": "Select round",
             "interviewing": "Interview", "feedback": "Feedback"}
stage_label = stage_map.get(st.session_state.get("stage"), "—")
round_label = AVAILABLE_ROUNDS.get(st.session_state.get("selected_round_key"), {}).get("name", "—")
duration_min = st.session_state.get("interview_duration_min", 10)

# Time left (shows during interview)
left_secs = 0
if st.session_state.get("interview_end_ts"):
    left_secs = max(0, int(st.session_state["interview_end_ts"] - time.time()))
def _mmss(n): 
    m, s = divmod(max(0, int(n)), 60)
    return f"{m:02d}:{s:02d}"

# Key presence (don’t re-call the APIs here to keep sidebar snappy)
openai_present = bool(config.OPENAI_API_KEY)
el_present = bool(config.ELEVENLABS_API_KEY)

# Sidebar CSS
st.sidebar.markdown(
    """
    <style>
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg,#0f172a 0%, #111827 100%);
        color:#e5e7eb; padding: 6px 10px 14px 10px;
      }
      [data-testid="stSidebar"] *{ color:#e5e7eb; }
      .sb-card{ background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:12px 14px; margin:10px 0 12px 0; }
      .toprow{ display:flex; gap:.75rem; align-items:center; margin-bottom:.35rem; }
      .avatar{ width:42px; height:42px; border-radius:50%; background:#0ea5e9;
               display:flex; align-items:center; justify-content:center; font-weight:700; }
      .title{ font-weight:700; line-height:1.05; }
      .muted{ opacity:.8; font-size:.85rem; }
      .kv{ display:flex; justify-content:space-between; gap:8px; font-size:.92rem; margin:.18rem 0; }
      .pill{ padding:.12rem .55rem; border-radius:999px; font-size:.8rem;
             background: rgba(59,130,246,.15); color:#93c5fd; border:1px solid rgba(59,130,246,.35); }
      .badge{ display:inline-flex; align-items:center; gap:.35rem; font-weight:600;
              padding:.22rem .55rem; border-radius:999px; font-size:.82rem; border:1px solid;}
      .ok{ background:rgba(16,185,129,.14); color:#34d399; border-color:rgba(16,185,129,.35); }
      .err{ background:rgba(239,68,68,.18); color:#f87171; border-color:rgba(239,68,68,.35); }
      .hr{ height:1px; margin:.6rem 0; background: linear-gradient(90deg,transparent,rgba(255,255,255,0.18),transparent); }
      .cta a{ color:#93c5fd; text-decoration:none; } .cta a:hover{ text-decoration:underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div class="sb-card">
      <div class="toprow">
        <div class="avatar">AI</div>
        <div>
          <div class="title">Interview GenAIe</div>
          <div class="muted">Mock interviews, real feedback</div>
        </div>
      </div>
      <div class="kv"><span>Candidate</span><span class="pill">{first}</span></div>
      <div class="kv"><span>Stage</span><span class="pill">{stage_label}</span></div>
      <div class="kv"><span>Round</span><span class="pill">{round_label}</span></div>
      <div class="kv"><span>Duration</span><span class="pill">{duration_min} min</span></div>
      {"<div class='kv'><span>Time left</span><span class='pill'>" + _mmss(left_secs) + "</span></div>" if stage_label=='Interview' else ""}
    </div>

    <div class="sb-card">
      <div style="font-weight:600">Auth</div>
      <div style="display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.4rem;">
        <span class="badge {('ok' if openai_present else 'err')}">🔑 OpenAI {('OK' if openai_present else 'Missing')}</span>
        <span class="badge {('ok' if el_present else 'err')}">🎙 ElevenLabs {('OK' if el_present else 'Missing')}</span>
      </div>
      <div class="hr"></div>
      <div class="muted">Keys are loaded from Streamlit <em>Secrets</em>.</div>
    </div>

    <div class="sb-card cta">
      <div class="kv">
        <span>Help</span>
        <span>📧 <a href="mailto:{CONTACT_EMAIL}?subject=Interview%20GenAIe%20support&body=Hi%20team%2C%0A%0A(Describe%20your%20issue%20here)%0A"
                 >Contact</a></span>
      </div>
      <div class="kv">
        <span>Source</span>
        <span>🌐 <a href="https://github.com/BharathKalaleRaviprakash/Interview_GenAIe" target="_blank">GitHub</a></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
try:
    if config.OPENAI_API_KEY:
        from openai import OpenAI
        OpenAI(api_key=config.OPENAI_API_KEY).models.list()
        st.sidebar.success("✅ OpenAI auth OK")
except Exception as e:
    st.sidebar.error(f"❌ OpenAI auth failed: {e}")

try:
    if config.ELEVENLABS_API_KEY:
        from elevenlabs.client import ElevenLabs
        ElevenLabs(api_key=config.ELEVENLABS_API_KEY).voices.get_all()
        st.sidebar.success("✅ ElevenLabs auth OK")
except Exception as e:
    st.sidebar.error(f"❌ ElevenLabs auth failed: {e}")

# ---------- Keys / Feature flags ----------
missing = []
if not getattr(config, "OPENAI_API_KEY", None):
    missing.append("OpenAI")
if not getattr(config, "ELEVENLABS_API_KEY", None):
    st.info("ElevenLabs key not found — interviewer voice will fall back to on-screen text.")
keys_blocking = "OpenAI" in missing
if keys_blocking:
    st.error("OpenAI API key not found. Add it in **Streamlit → App → Settings → Secrets** and rerun.")
    st.stop()

# ---------- Small helpers ----------
def _cleanup_file(path: str | None):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

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
    """Save a Streamlit UploadedFile to disk and return its path."""
    if not uploaded_file:
        return None
    try:
        # Limit to 10 MB
        max_mb = 10
        if uploaded_file.size and uploaded_file.size > max_mb * 1024 * 1024:
            st.error(f"File too large (> {max_mb} MB).")
            return None

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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


# ===== Demo helpers (no mic needed) =====
def _demo_generate_answers_offline(resume_text: str, questions: list[str]) -> list[str]:
    """
    No-API, deterministic STAR-ish answers so you can test UI quickly.
    """
    who = "I"
    resume_hint = (resume_text or "").split("\n")[0][:120]
    answers = []
    for q in questions:
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
    Optional OpenAI-powered answers. Falls back offline if unavailable.
    """
    try:
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


# ===== Suggested answers (ideal responses for the feedback tab) =====
def _suggest_answers_offline(resume_text: str, round_name: str, questions: list[str], user_answers: list[str]) -> list[str]:
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
    try:
        import json
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("no OPENAI_API_KEY")

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
        data = json.loads(raw)
        suggested = data.get("suggested", [])
        if not isinstance(suggested, list) or not suggested:
            raise ValueError("LLM returned empty/invalid suggestions")
        return (suggested + [""] * len(questions))[:len(questions)]
    except Exception:
        return _suggest_answers_offline(resume_text, round_name, questions, user_answers)


# ---------- Session ----------
ss = st.session_state
ss.setdefault("stage", "upload")
ss.setdefault("resume_text", None)
ss.setdefault("interview_agent", None)
ss.setdefault("selected_round_key", None)
ss.setdefault("questions", [])
ss.setdefault("current_question_index", 0)
ss.setdefault("interview_history", [])
ss.setdefault("feedback", None)
ss.setdefault("temp_resume_path", None)
ss.setdefault("rec_audio", {})       # {qid: bytes}
ss.setdefault("rec_transcript", {})  # {qid: str}
# TTS cache per question
ss.setdefault("tts_audio", {})
ss.setdefault("tts_done", {})

# NEW — timeboxed interview
ss.setdefault("interview_duration_min", 10)  # default duration (minutes)
ss.setdefault("interview_start_ts", None)    # epoch seconds
ss.setdefault("interview_end_ts", None)      # epoch seconds

# app.py (imports)
from core.name_extractor import extract_candidate_name


# === Timeboxing helpers ===
def _now() -> float:
    return time.time()

def _time_left_sec() -> int:
    """Remaining seconds (0 if finished or not started)."""
    if not ss.get("interview_end_ts"):
        return 0
    return max(0, int(ss["interview_end_ts"] - _now()))

def _fmt_mmss(secs: int) -> str:
    m, s = divmod(max(0, int(secs)), 60)
    return f"{m:02d}:{s:02d}"

def _ensure_more_questions(agent, round_name: str, batch: int = 5):
    """
    If we’re close to running out of questions while time remains, fetch more and append uniques.
    """
    extra = []
    try:
        if hasattr(agent, "generate_questions"):
            extra = agent.generate_questions(round_name, batch)
        else:
            extra = agent._generate_questions(round_name, batch)  # noqa
    except Exception:
        extra = []
    if extra:
        seen = set(ss.questions)
        ss.questions.extend([q for q in extra if q not in seen])




# ---------- Stage 1: Upload ----------
if ss.stage == "upload":
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        ss.temp_resume_path = save_uploaded_file(uploaded_file)
        if ss.temp_resume_path:
            file_hash = _hash_file(ss.temp_resume_path)
            with st.spinner("Parsing resume..."):
                ss.resume_text = _parse_resume_cached(ss.temp_resume_path, file_hash)

            if ss.resume_text:
                # Quick guard for scanned/image-only PDFs
                if len(ss.resume_text.split()) < 50:
                    st.error("This file has very little selectable text (possibly a scanned PDF). Please upload a text-based PDF or DOCX resume.")
                    _cleanup_file(ss.temp_resume_path)
                    ss.temp_resume_path = None
                    ss.resume_text = None
                    st.stop()

                # STRICT: Only allow resumes
                doc_type, info = classify_document(ss.resume_text)
                if doc_type != "resume":
                    pretty = doc_type.replace("_", " ")
                    st.error(f"This looks like a *{pretty or 'non-resume document'}*, not a resume. Please upload a PDF/DOCX resume.")
                    with st.expander("Why it was rejected"):
                        st.write({
                            "resume_score": info.get("resume_score"),
                            "cover_score": info.get("cover_score"),
                            "jd_score": info.get("jd_score"),
                            "resume_signals": info.get("resume_signals"),
                            "cover_signals": info.get("cover_signals"),
                            "jd_signals": info.get("jd_signals"),
                        })
                    st.stop()  # stay on the upload page
                
                # ... after successful parse & validation ...
                full_name, first_name = extract_candidate_name(ss.resume_text)
                ss.candidate_name = full_name or ss.get("candidate_name") or ""
                ss.candidate_first = first_name or (ss.candidate_name.split()[0] if ss.candidate_name else "")

                st.success("Resume parsed and validated successfully! ✅")
                ss.candidate_name = st.text_input("Confirm your name", value=ss.candidate_name).strip()
                if ss.candidate_name:
                    ss.candidate_first = ss.candidate_name.split()[0].title()
                try:
                    ss.interview_agent = InterviewAgent(ss.resume_text)
                    ss.stage = "select_round"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize interview agent: {e}")
                    ss.resume_text = None
                    ss.interview_agent = None

            else:
                st.error("Could not extract text from the resume. Try a different file.")
                _cleanup_file(ss.temp_resume_path)
                ss.temp_resume_path = None


# ---------- Stage 2: Select Round ----------
if ss.stage == "select_round":
    st.header("Select Interview Round")
    # Personalized greeting on the select-round page
    first = (ss.get("candidate_first") or "").strip() or "there"
    st.markdown(f"### 👋 Hi {first}, choose your interview round")
    st.caption("We’ll address you by name throughout the interview.")
    if not ss.interview_agent:
        st.error("Interview agent not initialized. Please upload the resume first.")
        ss.stage = "upload"
        _cleanup_file(ss.temp_resume_path)
        ss.temp_resume_path = None
        st.rerun()

    round_options = {key: info["name"] for key, info in AVAILABLE_ROUNDS.items()}
    ss.selected_round_key = st.selectbox(
        "Choose the type of interview round:",
        options=list(round_options.keys()),
        format_func=lambda key: round_options[key],
    )
    # NEW — choose interview duration
    ss.interview_duration_min = st.slider(
        "Interview duration (minutes)",
        min_value=5, max_value=60, step=5, value=ss.interview_duration_min,
        help="The interviewer will keep asking questions until this time is up."
    )


    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Interview Round", key="start_interview"):
            if ss.selected_round_key:
                info = AVAILABLE_ROUNDS[ss.selected_round_key]
                ss.current_question_index = 0
                ss.interview_history = []
                ss.feedback = None

                agent = ss.interview_agent
                round_name = info["name"]

                with st.spinner(f"Generating questions for the {round_name} round..."):
                    try:
                        if hasattr(agent, "generate_questions"):
                            ss.questions = agent.generate_questions(round_name, info["num_questions"])
                        else:
                            ss.questions = agent._generate_questions(round_name, info["num_questions"])  # noqa
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")
                        st.error(traceback.format_exc())
                        ss.questions = []

                if ss.questions:
                    # NEW — start the timer
                    ss.interview_start_ts = _now()
                    ss.interview_end_ts = ss.interview_start_ts + ss.interview_duration_min * 60

                    # NEW — optional: ensure enough questions for the allotted time
                    # Heuristic ~1.5 minutes per question (tweak to your style)
                    target_q = max(len(ss.questions), int((ss.interview_duration_min / 1.5) + 1))
                    if len(ss.questions) < target_q:
                        _ensure_more_questions(agent, round_name, batch=target_q - len(ss.questions))

                    ss.stage = "interviewing"
                    st.success(f"Questions ready. Timer started for {ss.interview_duration_min} min.")

                    # Personalized welcome w/ timeboxing
                    fn = ss.get("candidate_first") or "there"
                    welcome = (
                        f"Welcome, {fn}. This is the {round_name} round. "
                        f"I’ll keep asking questions until time is up ({ss.interview_duration_min} minutes)."
                    )
                    tts_welcome = speak_text_bytes(welcome)
                    if tts_welcome:
                        st.audio(io.BytesIO(tts_welcome), format="audio/mp3")
                    else:
                        st.info(welcome)
                    st.rerun()
                else:
                    st.error("Failed to generate questions. Please try again.")
            else:
                st.warning("Please select a round first.")


    with c2:
            # New: Q&A-only LLM demo (no interview flow, no feedback)
        if st.button("🧪 Generate Q&A Only (LLM, resume-based)", key="qa_only_demo"):
            if not ss.resume_text or len(ss.resume_text.strip()) < 50:
                st.warning("Please upload a résumé first (at least ~50 characters of text).")
            else:
                info = AVAILABLE_ROUNDS.get(ss.selected_round_key)
                round_name = (info["name"] if info else "auto")
                with st.spinner("Generating resume-based questions & answers..."):
                    try:
                        qa_items = generate_llm_demo_qa(
                            resume_text=ss.resume_text,
                            role="Software Engineer / Data Scientist",
                            round_name=round_name,
                            num_questions=8,
                            model="gpt-4o-mini",
                            temperature=0.35,
                            candidate_name=ss.get("candidate_name") or None,  # now accepted
                        )


                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
                        qa_items = []

                if qa_items:
                    st.markdown("---")
                    st.subheader("Resume-Based Questions & Suitable Answers")
                    md = ["# Questions & Answers\n"]
                    for i, item in enumerate(qa_items, 1):
                        st.markdown(f"**Q{i}. {item['question']}**")
                        st.markdown(item["answer"])
                        st.markdown("---")
                        md.append(f"**Q{i}. {item['question']}**\n\n{item['answer']}\n\n---\n")

                    st.download_button(
                        "Download as Markdown",
                        data=("".join(md)).encode("utf-8"),
                        file_name="resume_based_q_and_a.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                else:
                    st.info("No Q&A generated. Check OpenAI credentials and try again.")

# ---------- Stage 3: Interviewing ----------
if ss.stage == "interviewing":
    round_name = AVAILABLE_ROUNDS[ss.selected_round_key]["name"]

    # If time is over, go straight to feedback
    if _time_left_sec() <= 0:
        ss.stage = "feedback"
        st.rerun()

    st.header(f"Interviewing: {round_name}")

    # Timer UI
    left = _time_left_sec()
    total = ss.interview_duration_min * 60
    st.info(f"⏳ Time remaining: **{_fmt_mmss(left)}**")
    st.progress(max(0.0, min(1.0, 1 - (left / total))))

    # If we’re about to exhaust questions and still have time, top up
    if ss.current_question_index >= len(ss.questions) - 1 and left > 0:
        _ensure_more_questions(ss.interview_agent, round_name, batch=5)

    # Guard: no questions / index beyond list
    if not ss.questions:
        st.error("No questions available. Please start a round again.")
        st.stop()
    if ss.current_question_index >= len(ss.questions):
        # Try to top up once if time remains; else finish
        if left > 0:
            _ensure_more_questions(ss.interview_agent, round_name, batch=5)
            if ss.current_question_index >= len(ss.questions):
                ss.stage = "feedback"
                st.rerun()
        else:
            ss.stage = "feedback"
            st.rerun()

    # --- Current question ---
    question_text = ss.questions[ss.current_question_index]
    st.markdown(f"**Interviewer:** {question_text}")

    # --- TTS per question (cache once) ---
    qid = f"hr-q-{ss.current_question_index}"
    if not ss["tts_done"].get(qid):
        tts_bytes = speak_text_bytes(question_text)  # MP3 bytes
        if tts_bytes:
            ss["tts_audio"][qid] = tts_bytes
        ss["tts_done"][qid] = True

    tts_bytes = ss["tts_audio"].get(qid)
    if tts_bytes:
        st.audio(io.BytesIO(tts_bytes), format="audio/mp3")
    else:
        st.info("TTS not available (check API key/voice).")

    # --- Record via browser mic ---
    audio_bytes = None

    if HAVE_MIC_RECORDER:
        rec = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            key=f"mic_{qid}",
            # NOTE: no 'format' kwarg in this component version
        )
        if isinstance(rec, dict) and rec.get("bytes"):
            audio_bytes = rec["bytes"]

    elif HAVE_AUDIO_RECORDER:
        # Fallback widget
        audio_bytes = audio_recorder(
            text="Click to record / click again to stop",
            sample_rate=16_000
        )

    else:
        st.warning("No mic widget available. You can upload audio instead.")
        up = st.file_uploader("Upload an audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"], key=f"upload_{qid}")
        if up:
            audio_bytes = up.read()

    if audio_bytes:
        st.audio(io.BytesIO(audio_bytes))
        if qid not in ss.rec_transcript:
            with st.spinner("Transcribing..."):
                ss.rec_transcript[qid] = transcribe_audio_bytes(audio_bytes) or ""

    if ss.rec_transcript.get(qid):
        st.markdown("**Transcript:**")
        st.write(ss.rec_transcript[qid])

    # --- Actions ---
    cols = st.columns([1, 1, 6])
    with cols[0]:
        disabled = not bool(ss.rec_transcript.get(qid))
        if st.button("✅ Submit", disabled=disabled, key=f"submit_{qid}"):
            text = ss.rec_transcript.get(qid, "")
            if not text:
                st.warning("Please record (and let it auto-transcribe) before submitting.")
            else:
                ss.interview_history.append({"question": question_text, "answer": text})
                # clear per-Q buffers
                ss.rec_audio.pop(qid, None)
                ss.rec_transcript.pop(qid, None)
                ss["tts_done"].pop(qid, None)
                ss["tts_audio"].pop(qid, None)

                # If time has expired after this answer, end
                if _time_left_sec() <= 0:
                    ss.stage = "feedback"
                    st.rerun()

                # Otherwise advance; top up if near end and still have time
                ss.current_question_index += 1
                if ss.current_question_index >= len(ss.questions) - 1 and _time_left_sec() > 0:
                    _ensure_more_questions(ss.interview_agent, round_name, batch=5)
                st.rerun()

    with cols[1]:
        if st.button("⏹️ End now"):
            ss.stage = "feedback"
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
                    resume_text=ss.resume_text or "",
                    round_name=round_name,
                    qa_pairs=ss.interview_history,
                )
            except Exception as e:
                st.error(f"Failed to generate feedback: {e}")
                st.error(traceback.format_exc())

    # ---- Suggested answers (generate once and cache) ----
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
            st.info("Feedback is not available for this round.")

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
                    st.text_area("Copy/edit:", value=sug_a, height=200, key=f"copy_sug_{i}")
        else:
            st.info("Suggested answers will appear here once questions and feedback are available.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Another Round"):
            ss.stage = "select_round"
            ss.selected_round_key = None
            ss.questions = []
            ss.current_question_index = 0
            ss.interview_history = []
            ss.feedback = None
            ss.suggested_answers = None
            ss.tts_audio = {}
            ss.tts_done = {}
            # NEW — reset timer
            ss.interview_start_ts = None
            ss.interview_end_ts = None
            st.rerun()

    with c2:
        if st.button("Upload New Resume"):
            _cleanup_file(ss.get("temp_resume_path"))
            for key in list(ss.keys()):
                del ss[key]
            st.session_state.stage = "upload"
            st.rerun()
