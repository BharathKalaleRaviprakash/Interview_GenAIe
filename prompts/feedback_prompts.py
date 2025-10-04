from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

_DEFAULT_SCORING = """Score each answer out of 10 based on the following criteria:
1. Relevance: How well does the answer address the question? (0–3 points)
2. Clarity: How clear and concise is the answer? (0–2 points)
3. Detail/Examples: Does the answer include sufficient detail or examples (e.g., STAR)? (0–3 points)
4. Resume Alignment: How well does the answer align with the candidate's resume? (0–2 points)"""

def _format_qa(qa_pairs: List[Dict[str, str]]) -> str:
    return "\n".join(f"Q: {i['question']}\nA: {i['answer']}" for i in qa_pairs)

def get_feedback_prompt(
    resume_text: str,
    round_name: str,
    qa_pairs: List[Dict[str, str]],
    scoring_criteria: Optional[str] = None
):
    """
    Returns LangChain chat messages for evaluating a mock interview.
    Use with:  ai_msg = llm.invoke(messages)
    """
    scoring = scoring_criteria or _DEFAULT_SCORING
    formatted_qa = _format_qa(qa_pairs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert interviewer evaluating a mock interview. "
         "Be constructive, specific, and follow the scoring criteria strictly."),
        ("human",
         "The interview was for the '{round_name}' round.\n\n"
         "Resume Context:\n---\n{resume_text}\n---\n\n"
         "Interview Questions and Answers:\n---\n{formatted_qa}\n---\n\n"
         "Instructions:\n"
         "1. Provide overall constructive feedback for the candidate's performance in this round (strengths and areas for improvement).\n"
         "2. Give specific suggestions for improvement based on their answers.\n"
         "3. Score each answer individually based on the provided criteria.\n"
         "4. Calculate a total score for the round (sum of individual scores).\n"
         "5. Format the output in this order: Overall Feedback, Suggestions, Per-Question Scores, Total Score.\n\n"
         "Scoring Criteria (per question, max 10 points): {scoring_criteria}\n\n"
         "Generate the feedback and scores now.")
    ])

    return prompt.format_messages(
        round_name=round_name,
        resume_text=resume_text,
        formatted_qa=formatted_qa,
        scoring_criteria=scoring
    )


'''def get_feedback_prompt(resume_text: str, round_name: str, qa_pairs:list[dict], scoring_criteria: str = None) -> str:
    "Creates prompts"

    if not scoring_criteria:
        scoring_criteria = """Score each answer out of 10 based on the following criteria:
        1. Relevance: How well does the answer address the question? (0-3 points)
        2. Clarity: How clear and concise is the answer?(0-2 points)
        3. Detail/Examples: Does the answer have sufficient detail or examples (like STAR method where applicable)? (0-3 points)
        4. Resume Alignment: How well does the answer align with the candidate's resume? (0-2 points)"""

    formatted_qa = "\n".join([f"Q: {item['question']} \nA: {item['answer']}" for item in qa_pairs])

    prompt = f"""You are an expert interviewer provide -- on a mock interview.
    The interview was for the '{round_name}' round.
    The candidate's resume is provided below for context.
    Analyze the following question and answer pairs from the interview.

    Resume Context:
    ---
    {resume_text}
    ---

    Interview Questions and Answers:
    ---
    {formatted_qa}
    ---

    Instructions:
    1. Provide overall constructive feedabck for the condidate's performance in this round. 
    Focus on strengths and areas for improvement.
    2. Give specific suggestions for improvement based on their answers.
    3. Score each answer individually based on the provided criteria
    4. Calculate a total score for the round (sum of individual scores).
    5. Format the output clearly, starting with Overall Feedback, then Suggestions, then a list of scores per question. and finally the total score.

    Scoring Criteria (per question, max 10 points): {scoring_criteria} 

    Generate the feedback and scores now:
    """
    return prompt'''