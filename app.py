import streamlit as st
import os
import time
import traceback

from core.resume_parser import parse_resume
from agent.round_manger import AVAILABLE_ROUNDS
from agent.interview_agent import InterviewAgent
from core.audio_io import speak_text, transcribe_audio

from core.feedback_generator import generate_feedback_and_scores
from utils.config import TEMP_AUDIO_FILENAME
from utils import config


st.set_page_config(page_title="AceBot Interviewer", layout="wide")

UPLOAD_DIR = "data/uploads"

def save_uploaded_file(uploaded_file):
    """"Saves uploaded file temporarily for parsing"""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error in saving uploaded file: {e}")
        return None

def cleanup_temp_file(file_path):
    """Removes a specific temporary file"""
    try: 
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.warning(f"Could not remove temporary file {file_path}: {e}")

if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'interview_agent' not in st.session_state:
    st.session_state.interview_agent = None
if 'selected_round_key' not in st.session_state:
    st.session_state.selected_round_key = None
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'interview_history' not in st. session_state:
    st.session_state.interview_history = []  # List of {'question': q, 'answer': a}
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'temp_resume_path' not in st.session_state:
    st.session_state.temp_resume_path = None   #Store path for cleanup 

keys_loaded = bool(config.OPENAI_API_KEY and config.ELEVENLABS_API_KEY)
if not keys_loaded:
    st.error("API keys for OpenAI or ElevenLabs not found. Please check your .env file")
    st.stop()


st.title("AceBot Interviewer")
st.markdown("Upload your resume, choose and interview round and practice with an AI interviewer!")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses AI (OpenAI & ElevenLabs) to conduct a mock interview based on your resume."
    "Upload your resume, select a round and answer the questions."
    "Receive feedback and scores to help you prepare."
)

# --- Stage 1: Upload Resume---
if st.session_state.stage == 'upload':
    st.header("1. Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a resume file (PDF or DOCX)", type=['pdf', 'docx'])

    if uploaded_file is not None:
        #Save and parse
        st.session_state.temp_resume_path = save_uploaded_file(uploaded_file)
        if st.session_state.temp_resume_path:
            with st.spinner("Parsing resume..."):
                st.session_state.resume_text = parse_resume(st.session_state.temp_resume_path)
        
            if st.session_state.resume_text: 
                st.success("Resume parsed successfully!")

                try:
                    st.session_state.resume_text = InterviewAgent(st.session_state.resume_text)
                    st.session_state.staeg = 'select_round'
                    st.rerun()  #  Rerun to move to the next staeg UI immediately
                
                except Exception as e:
                    st.error(f"Failed to initialize interview agent: {e}")
                    st.session_state.resume_text = None # Reset on failure
                    st.session_state.interview_agent = None
                    cleanup_temp_file(st.session_state.temp_resume_path) #cleanup failed attempt's file

            else:
                st.error("Could not extract text from the resume. Please try a different file")
                cleanup_temp_file(st.session_state.temp_resume_path)
                # clean up failed attempt's file
                st.session_state.temp_resume_path = None


# ---Stage 2: Select Round---
if st.session_state.stage == 'select_round':
    st.header("2. Select Interview Round")

    if not st.session_state.interview_agent:
        st.error("Interview agent not initialized. Please upload the resume first.")
        st.session_state.stage = 'Upload' # Go back to upload stage
        # Clean up just in case
        cleanup_temp_file(st.session_state.temp_resume_path)
        st.session_state.temp_resume_path = None
        st.rerun()

    round_options = {key: info['name'] for key, info in AVAILABLE_ROUNDS.items()}  
    st.session_state.selected_round_key = st.selectbox(
        "Choose the type of interview round:",
        options=list(round_options.keys()),
        format_func=lambda key: round_options[key]  # show names in dropdown
    )
    if st.button("Start Interview Round", key="start_interview"):
        if st.session_state.selected_round_key:
            selected_round_info = AVAILABLE_ROUNDS[st.session_state.selected_round_key]
            st.session_state.current_question_index = 0
            st.session_state.interview_history = []
            st.session_state.feedback = None

            # Generate questions for the round 
            agent = st.session_state.interview_agent
            if agent:
                with st.spinner(f"Generating questions for the 
                {selected_round_info['name']} round..."):
                    try:
                        # access internal method carefully (or refactor agent for this)
                        st.session_state.questions = agent._generate_questions(
                            selected_round_info['name'],
                            selected_round_info['num_questions']
                        )
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")
                        st.error(traceback.format_exc())  #Print detailed error
                        st.session_state.questions = []  # Ensure its empty on failure
   
                       
                if st.session_state.questions:
                    st.session_state.stage = 'interviewing'
                    st.success(f"Questions generated. Starting  the 
                    {selected_round_info['name']} round!")
                    time.sleep(1)  #Give user a moment to read the message
                    # Speak welcome message for the round
                    try:
                        speak_text(f"Welcome to the {selected_round_info['name']} round. I will 
                        ask you {len(st.session_state.questions)} questions. Let's begin with the first question.")
                    except Exception as e:
                        st.warning(f"Could not play welcome audio: {e}.Starting interview.")
                    st.rerun()
                else:
                    st.error("Failed to generate questions for the round. Please try selecting the round again or check the logs/API keys.")
            else:
                st.error("Interview agent not found. Please restart the process by uploading the resume again.")
                st.session_state.stage = 'upload'
                cleanup_temp_file(st.session_state.temp_resume_path)
                st.session_state.temp_resume_path = None
                st.rerun()
        else:
            st.warning("Please select a round first.")


# ---Stage 3: Interviewing ---
if st.session_state.stage == 'interviewing':
    st.header(f"Interviewing in Progress: {AVAILABLE_ROUNDS[st.session_state.selected_round_key]['name']} Round")

    # Check if questions are loaded
    if not st.session_state.questions:
        st.error("No questions loaded for this round. Please go back and select the round again.")
        if st.button("Go Back to Round Selection"):
            st.session_state.stage = 'select_round'
            st.rerun()
        st.stop()

    # Get current question index
    q_index = st.session_state.current_question_index
    if q_index < len(st.session_state.questions):
        current_question = st.session_state.questions[q_index]

        st.subheader(f"Question {q_index + 1}/{len(st.session_state.questions)}")
        st.markdown(f"**Interviewer:** {current_question}")

        # --- Placeholder for Audio Recording ---
        st.markdown("**Your ANswer (Type Below):**")
        # Use a unique key for the text_area based on the question index
        user_answer = st.text_area("Enter your answer here:", key=f"answer_q{q_index}", height=150)

        # Speak the question only once per question display
        if f"spoken_q{q_index}" not in st.session_state:
            try:
                # Use a spinner while speaking maybe?
                # with st.spinner("Intyerviewer is speaking..."): # This might be annoying if long
                speak_text(current_question)
                st.session_state[f"spoken_q{q_index}"] = True
            except Exception as e:
                st.warning(f"Could not play question audio: {e}")
                st.session_state[f"spoken_q{q_index}"] = True # Mark as 'spoken' anyway to avoid retry loop

        # --- Submit Answer Button ---
        if st.button("Submit Answer", key=f"submit_q{q_index}"):
            if user_answer and user_answer.strip():
                # Store the answer
                st.session_state.interview_history.append({
                    "question": current_question,
                    "answer": user_answer.strip()
                })

                # Move to the next question 
                st.session_state.current_question_index += 1

                # If there are more questions, speak the next one (or transition)
                if st.session_state.current_question_index < len(st.session_state_questions):
                    next_q_index = st.session_state.current_question_index
                    next_question = st.session_state.questions[next_q_index]
                    try:
                        # Maybe just say "Next question." or similar to keep it shorter
                        speak_text("Okay, thank you. Next question.")
                        # Let Streamlit rerun handle displaying the next question text
                    except Exception as e:
                        st.warning(f"Audio notification error: {e}")

                st.rerun() # Rerun to display the next question or move to feedback stage

            else:
                st.warning("Please enter your answer before submitting.")
    
    else:
        # All questions answered, move to feedback stage
        st.success("All questions for this round are completed")
        st.session_state.stage = 'feedback'
        try:
            speak_text("Thank you. That concludes the question for this round. I will now prepare your feedback.")
        except Exception as e:
            st.warning(f"Audio notification error: {e}")
        st.rerun() 
        

# ---Stage 4: Feedback ---

if st.session_state.stage == 'feedback':
    st.header("Interview Complete - Feedback")
    
    Agent = st.session_state.interview_agent
    round_name = AVAILABLE_ROUNDS[st.session_state.selected_round_key]['name']

    # Generate feedback only if it hasn't been generated yest for this round
    if not st.session_state.feedback and agent and st.session_state.interview_history:
        with st.spinner("Generating feedback... This may take a moment"):
            try:
                # Generate feedback using the agents's method (or directly call the function)
                # We need resume_text, round_name, and history from session_state
                # Inside the 'feedback stage in app.py
                st.session_state.feedback = generate_feedback_and_scores( 
                    resume_text = st.session_state.resume_text,
                    round_name = round_name,
                    qa_pairs = st.session_state.interview_history
                )
                # Store feedback in agent as well if needed by its internal logic
                # agent.feedback = st.session_state.feedback # if agent class uses self.feedback
            except Exception as e:
            st.error(f"Failed to generate feedback: {e}")
            st.error(traceback.format_exc())  # Print detailed error

    # Display Feedback if available
    if st.session_state.feedback:
        feedback_data = st.session_state.feedback

        st.subheader("Overall Feedback")
        st.markdown(feedback_data.get("Overall_feedback", "N/A"))

        st.subheader("Suggestions for Improvement")
        st.markdown(feedback_data.get("Suggestions", "N/A"))

        st.subheader("Scores per Question")
        score = feedback_data.get("Scores_per_question", [])
        total_score = feedback_data.get("total_score", 0)
        max_score = len(st.session_state.interview_history) * 10

        if scores and len(scores)  == len(st.session_state.interview_history):
            for i, score in enumerate(scores):
                st.markdown(f"- **Q{i+1}:** {score}/10")
        elif scores:
            st.warning(f"Note: Number of scores ({len(scores)})
            doesn't match number of questions ({len(st.session_state.interview_history)}). Display raw scores: {scores}")
        else:
            st.markdown("Scores could not be determined.")

        st.subheader("Total Score for Round")
        if max_score > 0:
            st.markdown(f"**{total_score} / {max_score}**")
        else:
            st.markdown("N/A (No questions answered)")

        # Optionally Show raw feedback for debugging
        with st.expander("Show Raw Feedback Data (for debugging)")
            st.json(feedback_data)

    else:
        st.warning("Feedback is not available for this round.")


    st.markdown("---")
    if st.button("Start Another Round"):
        # Reset state for a new round, keeping resume and agent
        st.session_state.stage = 'select_round'
        st.session_state.selected_round_key = None
        st.session_state.questions = []
        st.session_state.current_question_index = 0
        st.session_state.interview_history = []
        st.session_state.feedback = None
        # Clear spoken flags for questions
        keys_to_clear = [k for k in st.session_state if k.startswith('spoken_q')]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

    if st.button("Uploaded New Resume"):
        #  Reset everything including resume and agent
        cleanup_temp_file(st.session_state.temp_resume_path) # Cleanup the old resume file
        for key in list(st.session_state.keys()):
            del st.session_state[key]  # Clear all session state
        st.session_state.stage = 'upload' # Go back to start
        st.rerun()
