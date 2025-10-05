# agent/interview_agent.py
from __future__ import annotations

import re
import ast
from typing import List, Dict, Any

from core.llm_service import generate_completion
from core.feedback_generator import generate_feedback_and_scores
from prompts.question_prompts import get_question_generation_prompt


class InterviewAgent:
    def __init__(self, resume_text: str):
        self.resume_text = resume_text
        self.interview_history: List[Dict[str, str]] = []
        self.current_round_info: Dict[str, Any] | None = None
        self.feedback: Dict[str, Any] | None = None

    # ------------------------- Question Generation -------------------------

    def generate_questions(self, round_name: str, num_of_questions: int) -> List[str]:
        """
        Public method used by the app. Generates interview questions from the resume.
        """
        return self._generate_questions(round_name, num_of_questions)

    def _generate_questions(self, round_name: str, num_of_questions: int) -> List[str]:
        """
        Uses your LLM service to create a list[str] of questions.
        Robust parsing handles code fences and non-JSON outputs.
        """
        prompt = get_question_generation_prompt(self.resume_text, round_name, num_of_questions)
        raw = generate_completion(prompt, max_tokens=300 * num_of_questions, temperature=0.6)

        # Strip ``` blocks if present
        raw = re.sub(r"^```(?:python)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)

        # Try to parse as Python list literal (e.g., ["Q1", "Q2", ...])
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                questions = parsed[:num_of_questions]
                if len(parsed) < num_of_questions:
                    # If fewer than requested, just return what we have
                    pass
                return questions
            raise ValueError("Parsed result is not a list[str].")
        except Exception:
            # Fallback: line-split
            lines = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
            if lines:
                return lines[:num_of_questions]

        # Final fallback: generic questions
        return [
            f"Tell me about your experience relevant to the {round_name} role based on your resume.",
            "What is your biggest strength related to this area?",
            "Can you describe a challenge you faced and how you overcame it?",
            "Where do you see yourself in 5 years?",
            "Do you have any questions for me?",
        ][:num_of_questions]

    # ------------------------- Feedback Generation -------------------------

    def build_feedback(self, round_name: str, qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Given a list of {'question': str, 'answer': str} pairs, call your feedback generator.
        Stores and returns the feedback dict.
        """
        self.current_round_info = {"name": round_name, "num_questions": len(qa_pairs)}
        self.interview_history = qa_pairs

        self.feedback = generate_feedback_and_scores(
            resume_text=self.resume_text,
            round_name=round_name,
            qa_pairs=qa_pairs,
        )
        return self.feedback

    # ------------------------- Accessors / Utils -------------------------

    def get_total_score(self) -> int | None:
        if self.feedback:
            return self.feedback.get("total_score")
        return None
