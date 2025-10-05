import json
from core.llm_service import generate_completion
from core.audio_io import speak_text, record_audio, transcribe_audio
from core.feedback_generator import generate_feedback_and_scores
from prompts.question_prompts import get_question_generation_prompt

class InterviewAgent:
    def __init__(self, resume_text:str):
        self.resume_text = resume_text
        self.interview_history = []
        self.current_round_info = None
        self.feedback = None
    
    def _generate_questions(self, round_name: str, num_of_questions: int) -> list[str]:
        print(f"\nGenerating {num_of_questions} questions for the {round_name} round based on your resume...")
        prompt = get_question_generation_prompt(self.resume_text, round_name, num_of_questions)
        raw_response = generate_completion(prompt, max_tokens=300 * num_of_questions, temperature=0.6)

        try:
            raw_response = raw_response.strip().strip('```python').strip('```').strip()
            questions = ast.literal_eval(raw_response)
            if isinstance(questions, list) and all(isinstance(q, str)
            for q in questions):
                if len(questions) > num_of_questions:
                    print(f"Warning: LLM generated {len(questions)} questions, expected {num_of_questions}. Using the first {num_of_questions}.")
                    questions = questions[:num_of_questions]
                elif len(questions) < num_of_questions:
                    print(f"Warning: LLM generated only {len(questions)} questions, expected {num_of_questions}.")

                print("Questions generated successfully.")
                return questions
            else:
                raise ValueError("Parsed result is not a list of strings.")
            
        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Error parsing questions from LLM response: {e}")
            print(f"Raw response was: {raw_response}")

            lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
            if lines and len(lines) >= num_of_questions //2:
                print("Falling back to line splitting for questions.")
                return lines[:num_of_questions]
            else:
                print("Could not generate questions properly. Using generic questions.")
                return [
                    f"Tell me about your experience relevant to the {round_name} role based on your resume.", "What is your biggesr strength related to this area?", "Can you describe a challenge you faced and how you overcame it?", "Where do you see yourself in 5 years?", "Do you have any questions for me?"
                ][:num_of_questions]
            


                                                   




    def conduct_round(self, round_info: dict):
        self.current_round_info = round_info
        round_name = round_info['name']
        num_questions = round_info['num_questions']
        self.interview_history = []

        print(f"\n--- Starting {round_name} Round ---")
        speak_text(f"Welcome to the {round_name} round. I will ask you {num_questions} questions based on your resume. Please answer clearly after I finish speaking. ")

        questions = self._generate_questions(
            round_name, num_questions
        )

        for i, question in enumerate(questions): 
            print(f"\nQuestion {i+1}/{len(questions)}:")
            speak_text(question)
            record_duration = 30
            audio_file = record_audio(duration=record_duration)
            answer = None
            if audio_file:
                answer = transcribe_audio(audio_file)
            if not answer:
                speak_text("I didn't catch that. Let's move to the next question.")
                answer = "[No response recorded]"
            
            self.interview_history.append({"question": question, "answer": answer})

        print(f"\n--- {round_name} Round Complete ---")
        speak_text("Thank you. That concludes the questions for this round.")

        self.feedback = generate_feedback_and_scores(
            self.resume_text, round_name, self.interview_history
        )

        return self.feedback

    def display_feedback(self):
        print("\n--- Interview Feedback ---")
        print(f"\nRound: {self.current_round_info['name']}")

        print("\n[ Overall Feedback ]")
        print(self.feedback.get("overall_feedback", "N/A"))

        print("\n[ Suggestions for Improvement ]")
        print(self.feedback.get("suggestions", "N/A"))

        print("\n [ Scored per Question ]")

        scores = self.feedback.get("scored_per_question", [])
        if scores and len(scores) == len(self.interview_history):
            for i, score in enumerate(scores):
                print(f" Q{i+1}: {score}/10")
        elif scores:
            print(f"  (Raw scores: {scores} - Mismatch in count check 'raw_output')")
        else:
            print("  Scores not available or parsing failed.")

        print(f"\n [ Total Score for Round ]")
        print(f"  {self.feedback.get('total_score', 'N/A')}/{len(self.interview_history) * 10 if self.interview_history else 'N/A'}")

    def get_total_score(self) -> int | None:
        if self.feedback:
            return self.feedback.get("total_score")
        return None




    


