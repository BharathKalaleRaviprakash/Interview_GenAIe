from core.llm_service import get_llm
from prompts.feedback_prompts import get_feedback_prompt
import re

def generate_feedback_and_scores(resume_text: str, round_name: str, qa_pairs: list[dict]) -> dict:

    """
    Generate feedback and scores based on the interview Q&A pairs and resume text.
    
    Args:
        resume_text (str): The candidate's resume text.
        round_name (str): The name of the interview round.
        qa_pairs (list[dict]): List of question-answer pairs from the interview.
        
    Returns:
        dict: A dictionary containing feedback and scores.
    """
    print("\n Generating feedback and scores...")
    prompt = get_feedback_prompt(resume_text, round_name, qa_pairs)
    raw_feedback = get_llm(model="gpt-4o-mini", temperature=0.5, max_tokens=1000).invoke(prompt).content
    feedback_data = {
        "overall_feedback": "Could not parse feedback.",
        "suggestions": "Could not parse suggestions.",
        "scores_per_question": [],
        "total_score": 0,
        "raw_output": raw_feedback
    }
    
    try:
        overall_match = re.search(r"Overall\s*Feedback:\s*(.*?)(?:\s*Suggestions\s*for\s*Improvement\s*:|$)", raw_feedback, re.IGNORECASE | re.DOTALL)
        if overall_match:
            feedback_data["overall_feedback"] = overall_match.group(1).strip()
        suggestions_match = re.search(r"Suggestions\s*for\s*Improvement\s*:\s*(.*?)(?:\s*Scores\s*per\s*Question\s*:|\s*Total\s*Score\s*:|$)", raw_feedback, re.IGNORECASE | re.DOTALL)
        if suggestions_match:
            feedback_data["suggestions"] = suggestions_match.group(1).strip()
        score_matches = re.findall(r"Q\d+ Score:\s*(\d+)", raw_feedback, re.IGNORECASE)
        if score_matches:
            scores = [int(s) for s in score_matches]
            if len(scores) == len(qa_pairs):
                feedback_data["scores_per_question"] = scores
            else:
                print("[WARN] Mismatch in number of scores and questions.")
                feedback_data["scores_per_question"] = scores

        total_score_match = re.search(r"Total Score:\s*(\d+)", raw_feedback, re.IGNORECASE)
        
        if total_score_match:
            feedback_data["total_score"] = int(total_score_match.group(1))
        elif feedback_data["scores_per_question"] and len(feedback_data["scores_per_question"]) == len(qa_pairs):
            feedback_data["total_score"] = sum(feedback_data["scores_per_question"])
            print("Calculated total score from individual scores.")
        else:
            print("Warning: Could not determine total score.")
            feedback_data["total_score"] = sum(feedback_data["scores_per_question"])
    except Exception as e:
        print(f"[ERROR] Parsing feedback failed: {e}")
        print("Returning raw feedback only.")

    print("Feedback and scores generation complete.\n")
    return feedback_data    

            
    
