# 🎙️ Interview GenAIe  
### *Practice. Feedback. Confidence. — All powered by AI.*

---

## 💡 Inspiration  
Preparing for interviews can be overwhelming — especially when you’re unsure what questions to expect, how to sound confident, or how to improve.  
Traditional mock interviews are difficult to schedule and often lack actionable feedback.

We wanted to build an **AI-powered interview coach** that feels like a **mentor on demand** — one that listens, asks contextually relevant questions, and helps you **practice smarter, not harder.**

That idea became **Interview GenAIe ✈️ — your AI copilot for mastering interviews.**

---

## 🧠 What it does  
**Interview GenAIe** is a **voice-first, resume-aware interview coach** that simulates realistic interviews using natural AI voices and delivers personalized feedback.

### Core Capabilities
- 🎤 **Voice-based AI interviewer** — holds mock interviews through realistic AI voices.  
- 📄 **Resume-tailored questions** — upload your resume and receive questions aligned with your background, skills, and role.  
- 💬 **Instant AI feedback** — on clarity, confidence, tone, and structure.  
- 🔁 **Iterative coaching** — refine your answers through continuous AI guidance.  
- 🧠 **Multimodal support** — combines speech recognition, voice synthesis, and LLM reasoning.  

In short, **Interview GenAIe** acts as your **24/7 personal interviewer**, helping you prepare for **technical, behavioral, and situational** interviews anytime, anywhere.

---

## ⚙️ How we built it  

### 🧩 Tech Architecture  
We built a complete **voice-to-AI loop** combining speech recognition, voice synthesis, and generative reasoning.

| Component | Technology |
|------------|-------------|
| **Frontend/UI** | Streamlit |
| **Voice Generation** | ElevenLabs API (`eleven_multilingual_v2`) |
| **Speech-to-Text** | Google Web Speech API via `SpeechRecognition` |
| **LLM Core** | OpenAI GPT-4 / GPT-3.5-Turbo |
| **Audio Processing** | `sounddevice`, `soundfile`, `simpleaudio`, `numpy` |
| **Resume Parsing** | PyPDF2 + SpaCy |
| **Feedback Engine** | GPT-based scoring module analyzing tone, clarity, and completeness |

### 🧠 Model Stack  
| Task | Model / Library |
|------|-----------------|
| Question Generation | GPT-4 |
| Feedback Evaluation | GPT-4 tuned with rubric-based prompt templates |
| Voice Output | ElevenLabs Multilingual v2 |
| Audio Capture | sounddevice (WAV, 44.1 kHz) |
| UI Framework | Streamlit |

---

## 🚧 Challenges we ran into  
- 🎙️ Handling **FFmpeg/audio driver dependencies** across OS environments.  
- ⚡ Balancing **latency vs. realism** in ElevenLabs streaming voices.  
- 📄 Improving **resume parsing accuracy** for varied PDF formats.  
- 🧠 Designing **LLM prompts** that maintain clarity and context.  
- 🔁 Managing **multi-turn conversations** across interview rounds.

---

## 🏆 Accomplishments we’re proud of  
- Built a **fully functional AI voice interviewer** within the hackathon timeline.  
- Integrated **speech-to-text, text-to-speech, and GPT feedback** into one seamless user experience.  
- Developed **resume-based question generation** for personalized practice.  
- Designed an **intuitive Streamlit interface** for natural interview flow.  
- Delivered a **working MVP** demonstrating real-time feedback and voice interaction.

---

## 📚 What we learned  
- How to integrate **multimodal AI (voice + text)** for interactive experiences.  
- The importance of **prompt engineering** for structured, consistent responses.  
- Managing multiple APIs (OpenAI, ElevenLabs, Google Speech) in real time.  
- That **UX and latency optimization** are as important as model accuracy.

---

## 🚀 What’s next for AceBot / Interview GenAIe  
- 🤝 **Panel interview simulation** with multiple AI voices.  
- 📈 **Progress analytics** to track improvement over time.  
- 🌐 **Web deployment** on Streamlit Cloud or Hugging Face Spaces.  
- 💼 **LinkedIn & job board integration** for contextual question generation.  
- 🎧 **Emotion analysis** — detect tone, confidence, and hesitation.  
- 📲 **Mobile version** for on-the-go interview prep.

---

## 🧰 Tech Stack Summary  
| Category | Tools |
|-----------|-------|
| Core Language | Python 3.10+ |
| Framework | Streamlit |
| LLM | OpenAI GPT-4 / GPT-3.5-Turbo |
| Voice | ElevenLabs API |
| Speech-to-Text | SpeechRecognition |
| Audio | sounddevice • soundfile • simpleaudio |
| Resume Parsing | PyPDF2 • SpaCy |
| Environment | dotenv • VS Code / Anaconda |

---

## 🧑‍💻 Quick Start  

```bash
# 1️⃣ Clone repository
git clone https://github.com/<your-username>/Interview-GenAIe.git
cd Interview-GenAIe

# 2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS / Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add environment variables (.env)
OPENAI_API_KEY=sk-xxxxx
ELEVENLABS_API_KEY=xxxxx
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL

# 5️⃣ Run the app
streamlit run app.py

