# core/demo_mode_llm.py
from __future__ import annotations
import json, re
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, conint, constr
from core.llm_service import generate_completion

class QAItem(BaseModel):
    question: constr(strip_whitespace=True, min_length=5)
    answer:   constr(strip_whitespace=True, min_length=5)

class QAPayload(BaseModel):
    role: constr(strip_whitespace=True, min_length=2)
    round: constr(strip_whitespace=True, min_length=2)
    num_questions: conint(ge=1, le=20)
    items: List[QAItem]

SYSTEM_PROMPT = (
    "You are a strict interview content generator.\n"
    "You MUST return ONLY valid JSON (UTF-8, no comments, no trailing commas, no code fences).\n"
    "Do not include any text before or after the JSON.\n"
)

# --- Robust JSON extraction helpers ---
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_JSON_RE  = re.compile(r"(\{.*\}|\[.*\])", re.S)

def _extract_json_any(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.lstrip("\ufeff").strip()  # strip BOM/newlines
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _JSON_RE.search(text)
    if m:
        return m.group(1).strip()
    return None

def _parse_json_loose(blob: str) -> dict:
    return json.loads(blob)

def _validate_payload(data: dict) -> List[Dict[str, str]]:
    payload = QAPayload(**data)
    return [it.model_dump() for it in payload.items]

def generate_llm_demo_qa(
    *,
    resume_text: str,
    role: str = "Software Engineer / Data Scientist",
    round_name: str = "auto",
    num_questions: int = 8,
    model: Optional[str] = None,
    temperature: float = 0.35,
    candidate_name: Optional[str] = None,   # <-- NEW
) -> List[Dict[str, str]]:
    num_questions = max(1, min(20, int(num_questions)))

    # Optional personalization
    name_instr = ""
    if candidate_name:
        first = candidate_name.split()[0].title()
        name_instr = f'Address the candidate by first name "{first}" when appropriate.'

    user_prompt = f"""
Read the résumé and produce interview Q&A grounded ONLY in it.

<RÉSUMÉ>
{(resume_text or '').strip()[:15000]}
</RÉSUMÉ>

ROLE: {role}
ROUND: {round_name}   # behavioral | swe | ds-ml | system-design | auto
NUM_QUESTIONS: {num_questions}

{name_instr}

Rules:
- Generate exactly NUM_QUESTIONS items.
- Questions must reference skills/projects/metrics/tools from the résumé.
- Answers must be strong and concise (3–6 sentences), with metrics/trade-offs when relevant.
- No placeholders, no invented employers/schools.

Return ONLY valid JSON matching this schema (no markdown fences, no extra text):
{{
  "role": "{role}",
  "round": "{round_name}",
  "num_questions": {num_questions},
  "items": [
    {{"question": "…", "answer": "…"}}
  ]
}}
"""

    # ---- First attempt (force JSON via Chat Completions) ----
    raw = generate_completion(
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        model=model or "gpt-4o-mini",
        temperature=temperature,
        stream=False,
        response_format={"type": "json_object"},  # JSON-only output
    )
    if not isinstance(raw, str):
        raw = "".join(list(raw))

    blob = _extract_json_any(raw)
    if blob:
        try:
            data = _parse_json_loose(blob)
            return _validate_payload(data)
        except Exception:
            pass  # fall through to repair

    # ---- Repair attempt: ask for JSON-only rewrite ----
    repair_prompt = f"""
Rewrite the following content as valid JSON ONLY (no prose, no fences), conforming to the schema.
If it is not JSON, synthesize valid JSON that matches the intent.

SCHEMA:
{{
  "role": "{role}",
  "round": "{round_name}",
  "num_questions": {num_questions},
  "items": [
    {{"question": "…", "answer": "…"}}
  ]
}}

CONTENT TO FIX:
{raw[:12000]}
"""
    repaired = generate_completion(
        prompt=repair_prompt,
        system="Return only valid JSON. No markdown or extra text.",
        model=model or "gpt-4o-mini",
        temperature=0.0,
        stream=False,
        response_format={"type": "json_object"},  # JSON-only output
    )
    if not isinstance(repaired, str):
        repaired = "".join(list(repaired))

    blob2 = _extract_json_any(repaired) or repaired.strip()
    try:
        data2 = _parse_json_loose(blob2)
        return _validate_payload(data2)
    except Exception:
        # Final minimal fallback so UI doesn't crash
        return [{
            "question": "Tell me about your most impactful project.",
            "answer": "I led a project where I clarified goals, chose a pragmatic approach, and delivered measurable impact (e.g., latency -35%, quality +8%)."
        }]
