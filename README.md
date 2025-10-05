## 💡 Inspiration
Interview prep is stressful — juggling multiple rounds, guessing what questions might come up, and trying to assess your own performance without any real feedback. Traditional mock interviews are expensive, time-consuming, and rarely personalized.

Our goal was to create something that feels like having a **personal mentor available anytime** — one that knows your resume, understands the kind of role you’re targeting, and can coach you with empathy and intelligence.

That’s how **Interview GenAIe** — your **AI-powered interview coach** — was born.

---

## 🚀 What it does
**Interview GenAIe** is an intelligent, voice-first interview simulator that tailors every question to your **resume**, **chosen interview round**, and **career goals**.

It allows users to:
- 🎯 **Choose an interview type:** HR, Technical, Managerial, or General.
- 🧾 **Upload a resume:** the system analyzes it and generates context-aware, role-specific questions.
- 🎙️ **Answer in real time:** responses are recorded so users can review their performance.
- 💬 **Receive instant AI feedback:** detailed analysis on clarity, confidence, tone, and content quality.
- 🧠 **Get suggested model answers:** see how to improve and iterate smarter.

In short, **Interview GenAIe** helps users *practice, improve, and gain confidence* — anytime, anywhere — making interview prep **interactive, data-driven, and deeply personal.**

---

## 🧩 How we built it
**Interview GenAIe** combines modular AI services with a user-friendly Streamlit front-end for a seamless, voice-based experience.

### 💻 Frontend
- Built entirely in **Streamlit**, offering a responsive, voice-first interface.
- Manages user sessions, resume uploads, audio recording, and round selection through `app.py`.

### ⚙️ Core Logic & AI Services
- **LangChain + OpenAI APIs:** dynamic question generation, resume-based context retrieval, and AI feedback.
- **Custom Prompt Modules:** `question_prompts.py` and `feedback_prompts.py` generate round-specific, structured responses.
- **Feedback Engine:** `feedback_generator.py` evaluates user performance and produces detailed improvement suggestions.

### 🎧 Audio Pipeline
- **Speech-to-Text:** powered by **Whisper** for precise transcription.
- **Text-to-Speech:** integrated **ElevenLabs** to create realistic, conversational interviewer voices.
- **Audio Management:** `audio_io.py` handles recording, playback, and conversion.

### 📄 Resume Parsing
- Implemented in `resume_parser.py` using **PyMuPDF**, **pypdf**, and **docx2txt** to extract relevant skills and experience for contextualized question generation.

### 🔁 Session Management
- `round_manager.py` manages multi-round interview flow and stateful interactions.
- `interview_agent.py` orchestrates the overall logic across modules.

### ☁️ Deployment
- Deployed on **Streamlit Community Cloud**, ensuring zero-setup accessibility and fast iteration.

---

### 🧠 Model Stack
- 🧩 **LLM:** OpenAI GPT models (via LangChain)
- 🗣️ **Speech-to-Text:** Whisper
- 🎧 **Text-to-Speech:** ElevenLabs
- 🖥️ **Framework:** Streamlit
- 📄 **Resume Parsing:** PyMuPDF, docx2txt, pypdf
- 💬 **Feedback Engine:** Custom prompt templates + scikit-learn utilities
- ☁️ **Deployment:** Streamlit Community Cloud

---

## ⚙️ Challenges we ran into
- Implementing seamless **audio recording** within Streamlit required extensive exploration, as the framework doesn’t natively support advanced audio handling. We experimented with multiple community components and custom integrations before achieving a smooth, real-time recording experience.
- Ensuring **accurate and fast transcription** while keeping latency low during live sessions.
- Making the **feedback tone sound human and constructive** instead of purely algorithmic.
- Handling **resume-parsing edge cases** — ensuring the model accurately understands varied resume formats and skills.

---

## 🏆 Accomplishments that we’re proud of
- Created a **fully functional voice-based interview simulator** that dynamically adapts questions from a user’s resume.
- Successfully integrated **Whisper** and **ElevenLabs** into a single, low-latency audio loop for natural two-way conversation.
- Built a **modular architecture** with clear separation of logic — parsing, audio, feedback, and LLM orchestration.
- Designed and deployed an **end-to-end AI system** on Streamlit Cloud that anyone can access instantly.

---

## 🎓 What we learned
- Extending **Streamlit** for real-time voice interaction is possible with creativity — custom components and async workflows are key.
- Combining **Whisper + ElevenLabs** enables rich, lifelike conversational experiences but requires careful latency handling.
- **LangChain prompt engineering** greatly influences quality — small tweaks in context windows can make or break relevance.
- **Empathetic AI feedback** is critical — human-sounding voice + balanced phrasing encourages user confidence.
- Building **modularly** (separate parsing, audio, LLM layers) ensures scalability for future enterprise or educational use.

---

## 🌟 What’s next for Interview GenAIe
- 📊 **Progress Dashboard** — visualize user improvement across communication, technical accuracy, and confidence.
- 🎯 **Adaptive Follow-up Engine** — use performance metrics to generate progressive difficulty levels and deeper questions.
- 🧠 **Expanded Feedback Metrics** — measure tone, pacing, filler words, and emotional expression for richer analytics.
- 📱 **Mobile App Version** — enable quick practice sessions via voice on Android and iOS.
- 🎓 **University & Career Platform Partnerships** — integrate **Interview GenAIe** into placement and training programs.
