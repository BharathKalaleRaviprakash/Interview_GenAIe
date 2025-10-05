# core/demo_mode_llm.py
from __future__ import annotations
import json, re
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, conint, constr

from core.llm_service import generate_completion  # your OpenAI SDK wrapper

class QAItem(BaseModel):
    question: constr(strip_whitespace=True, min_length=5)
    answer:   constr(strip_whitespace=True, min_length=5)

class QAPayload(BaseModel):
    role: constr(strip_whitespace=True, min_length=2)
    round: constr(strip_whitespace=True, min_length=2)
    num_questions: conint(ge=1, le=20)
    items: List[QAItem]

SYSTEM_PROMPT = (
    "You are a strict interview content generator. "
    "Given a candidate résumé, produce high-quality interview questions "
    "and concise, strong sample answers tailored ONLY to the résumé. "
    "Answers must be 3–6 sentences, concrete, and metric/outcome-oriented."
)

def _json_extract_first(text: str) -> str:
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    return m.group(1) if m else text

def generate_llm_demo_qa(
    *,
    resume_text: str,
    role: str = "Software Engineer / Data Scientist",
    round_name: str = "auto",
    num_questions: int = 8,
    model: Optional[str] = None,
    temperature: float = 0.4,
) -> List[Dict[str, str]]:
    num_questions = max(1, min(20, int(num_questions)))

    user_prompt = f"""
You are generating interview Q&A grounded ONLY on the following résumé:

<RÉSUMÉ>
{(resume_text or '').strip()[:15000]}
</RÉSUMÉ>

ROLE: {role}
ROUND: {round_name}   # behavioral | swe | ds-ml | system-design | auto
NUM_QUESTIONS: {num_questions}

Rules:
- Generate exactly NUM_QUESTIONS items.
- Questions must explicitly reference skills, projects, metrics, domains, or tools from the résumé.
- Answers must be strong, concise (3–6 sentences), and grounded in the résumé.
- No placeholders or invented employers/schools.

Return ONLY valid JSON with this schema:
{{
  "role": "{role}",
  "round": "{round_name}",
  "num_questions": {num_questions},
  "items": [
    {{"question": "…", "answer": "…"}}
  ]
}}
"""

    raw = generate_completion(
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        model=model,
        temperature=temperature,
        stream=False,
    )
    if not isinstance(raw, str):
        raw = "".join(list(raw))

    blob = _json_extract_first(raw)
    try:
        data = json.loads(blob)
    except Exception:
        blob2 = blob.replace("\n", " ").replace("\t", " ")
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob2)
        data = json.loads(blob2)

    try:
        payload = QAPayload(**data)
        return [it.model_dump() for it in payload.items]
    except ValidationError:
        # best-effort fallback
        items = []
        if isinstance(data, dict):
            for d in (data.get("items") or []):
                q = (d.get("question") or "").strip()
                a = (d.get("answer") or "").strip()
                if len(q) > 5 and len(a) > 5:
                    items.append({"question": q, "answer": a})
        if items:
            return items
        return [{"question": "Tell me about your most impactful project.",
                 "answer": "I led a project where … (fallback)."}]
