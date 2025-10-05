# core/feedback_generator.py
from __future__ import annotations
import os, json, time, math, random
from typing import Any, Dict, List

# ---------- OpenAI client (new + legacy compatible) ----------
_NEW_CLIENT = None
_LEGACY_OPENAI = None

def _get_openai_client():
    """
    Returns (mode, client)
    mode: "new" for openai>=1.0 Responses API, "legacy" for openai<1.0 Chat API
    """
    global _NEW_CLIENT, _LEGACY_OPENAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        # New SDK
        from openai import OpenAI
        if _NEW_CLIENT is None:
            _NEW_CLIENT = OpenAI(api_key=api_key)
        return "new", _NEW_CLIENT
    except Exception:
        # Legacy SDK
        try:
            import openai as legacy_openai
            legacy_openai.api_key = api_key
            _LEGACY_OPENAI = legacy_openai
            return "legacy", _LEGACY_OPENAI
        except Exception as e:
            raise RuntimeError("Failed to initialize OpenAI client") from e

# ---------- Public API expected by your app.py ----------
def generate_feedback_and_scores(
    *, resume_text: str, round_name: str, qa_pairs: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      - overall_feedback: str
      - suggestions: list[str]
      - scores_per_question: list[int]
      - total_score: int
    """
    # If no answers yet, short-circuit
    if not qa_pairs:
        return {
            "overall_feedback": "No answers were recorded, so there's nothing to evaluate yet.",
            "suggestions": ["Record an answer for at least one question and resubmit."],
            "scores_per_question": [],
            "total_score": 0,
        }

    try:
        mode, client = _get_openai_client()
    except Exception:
        # If OpenAI is unavailable, use heuristic fallback
        return _heuristic_feedback(resume_text=resume_text, round_name=round_name, qa_pairs=qa_pairs)

    qa_text_blocks = []
    for i, qa in enumerate(qa_pairs, start=1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        qa_text_blocks.append(f"Q{i}: {q}\nA{i}: {a}")
    qa_text = "\n\n".join(qa_text_blocks)

    rubric_hint = {
        "Behavioral": "Emphasize STAR, ownership, conflict resolution, impact.",
        "System Design": "Emphasize requirements, trade-offs, scalability, bottlenecks.",
        "Data Science": "Emphasize problem framing, metrics, data quality, modeling tradeoffs.",
        "DS-ML": "Emphasize problem framing, metrics, data quality, modeling tradeoffs.",
    }.get(round_name, "Emphasize clarity, relevance, structure, confidence, and technical depth when applicable.")

    prompt = f"""
You are an expert interview coach. Given the candidate resume excerpt and the interview Q&A for the round "{round_name}", produce STRICT JSON only with this schema:

{{
  "overall_feedback": "one concise paragraph (<= 140 words) synthesizing performance across all answers",
  "suggestions": ["short actionable improvement 1", "short actionable improvement 2", "… (<=4 total)"],
  "scores_per_question": [0-10 for each question, same length/order as Q&A],
  "total_score": integer 0..(10 * number_of_questions) (sum of scores_per_question)
}}

Guidance:
- {rubric_hint}
- Be specific and actionable in suggestions.
- Keep tone supportive but candid.
- Ensure the scores list length EXACTLY matches the number of Q&A pairs ({len(qa_pairs)}).
- JSON only. No markdown, no backticks.

RESUME (excerpt):
{(resume_text or "")[:6000]}

INTERVIEW Q&A:
{qa_text[:8000]}
"""

    # ---- Call OpenAI with robust JSON handling ----
    raw = ""
    last_err = None
    for attempt in range(3):
        try:
            if mode == "new":
                # Responses API (modern)
                resp = client.responses.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    input=prompt,
                    temperature=0.2,
                    max_output_tokens=800,
                    response_format={"type": "json_object"},
                )
                raw = (resp.output_text or "").strip()
            else:
                # Legacy chat.completions
                raw = client.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=800,
                )["choices"][0]["message"]["content"].strip()

            data = _safe_json(raw)
            return _normalize_to_ui(data, len(qa_pairs))

        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))

    # On repeated failure, fallback heuristic so UI keeps working
    if last_err:
        return _heuristic_feedback(resume_text=resume_text, round_name=round_name, qa_pairs=qa_pairs)

    # Shouldn't get here
    return _heuristic_feedback(resume_text=resume_text, round_name=round_name, qa_pairs=qa_pairs)

# ---------- Helpers ----------
def _safe_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        # rescue JSON substring if model added accidental noise
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        raise

def _normalize_to_ui(data: Dict[str, Any], n_q: int) -> Dict[str, Any]:
    overall = str(data.get("overall_feedback") or data.get("summary") or "").strip()
    sugg = data.get("suggestions") or data.get("improvements") or []
    if isinstance(sugg, str):
        sugg = [s.strip() for s in sugg.split("\n") if s.strip()]
    sugg = [s for s in sugg if s][:4]

    spq = data.get("scores_per_question") or data.get("scores") or []
    # coerce to ints in 0..10
    spq = [int(max(0, min(10, int(round(float(x)))))) for x in spq if _is_number(x)]
    # fix length
    spq = (spq + [0] * n_q)[:n_q]

    total = int(sum(spq))
    if not overall:
        overall = "We evaluated your responses and prepared targeted suggestions. Focus on being specific and structured."

    if not sugg:
        sugg = ["Use concrete examples and quantify impact.", "Structure answers with STAR and finish with results."]

    return {
        "overall_feedback": overall,
        "suggestions": sugg,
        "scores_per_question": spq,
        "total_score": total,
    }

def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

# ---------- Heuristic fallback (no API required) ----------
def _heuristic_feedback(*, resume_text: str, round_name: str, qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Very simple rule-based scorer so the app remains functional without API calls.
    Scores based on answer length & presence of basic structure keywords.
    """
    keywords = ["because", "therefore", "impact", "metric", "designed", "optimized", "measured", "trade-off", "STAR", "result", "scalable"]
    scores = []
    for qa in qa_pairs:
        a = (qa.get("answer") or "").strip()
        if not a:
            scores.append(0)
            continue
        length_score = min(10, max(2, len(a.split()) // 20))  # rough length heuristic
        kw_hit = sum(1 for k in keywords if k.lower() in a.lower())
        structure_bonus = min(4, kw_hit)
        s = int(max(0, min(10, length_score + structure_bonus)))
        scores.append(s)

    total = int(sum(scores))
    suggestions = [
        "Use the STAR method (Situation, Task, Action, Result) to structure answers.",
        "Quantify outcomes (e.g., % improvement, time saved, cost reduced).",
        "State trade-offs and why you chose one approach over another.",
        "Conclude each answer with the concrete result or learning.",
    ][:4]

    overall = (
        f"For the {round_name} round, your responses show potential. "
        "Aim for clear structure and measurable outcomes. "
        "Highlight specific actions you took and finish with the impact."
    )

    return {
        "overall_feedback": overall,
        "suggestions": suggestions,
        "scores_per_question": scores,
        "total_score": total,
    }
