# core/llm_service.py
from __future__ import annotations
from typing import Optional, Iterable, Generator
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from utils.config import OPENAI_API_KEY

# --- LangChain LLM factory ---
def get_llm(
    model: str = "gpt-4o-mini",     # use a model that actually exists
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your secrets/env.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens,
    )

def _lc_msg(msg: dict) -> BaseMessage:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    return SystemMessage(content=content) if role == "system" else HumanMessage(content=content)

# --- Generate via LangChain (compatible signature) ---
def generate_completion(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful AI assistant.",
    # NEW aliases / optional args so callers can pass these without breaking:
    system: Optional[str] = None,                  # alias for system_prompt
    stream: bool = False,                          # ignored/handled below
    extra_messages: Optional[Iterable[dict]] = None,
    response_format: Optional[dict] = None,        # ignored in LangChain path
) -> str | Generator[str, None, None]:
    """
    Generates a chat completion using LangChain's ChatOpenAI.
    Accepts `system=` alias and ignores `response_format` gracefully.
    Returns a string (default) or a generator of chunks if stream=True.
    """
    try:
        llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)

        sys_txt = (system or system_prompt or "").strip()
        msgs: list[BaseMessage] = []
        if sys_txt:
            msgs.append(SystemMessage(content=sys_txt))

        if extra_messages:
            msgs.extend(_lc_msg(m) for m in extra_messages)

        msgs.append(HumanMessage(content=prompt))

        if stream:
            # Stream tokens as they arrive
            def _gen() -> Generator[str, None, None]:
                try:
                    for chunk in llm.stream(msgs):
                        piece = getattr(chunk, "content", None)
                        if piece:
                            yield piece
                except Exception as e:
                    logging.exception("Streaming failed: %s", e)
            return _gen()

        ai_message = llm.invoke(msgs)
        content = getattr(ai_message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # Fallback if LC returns segmented content
        try:
            text = " ".join(seg.get("text", "") for seg in content).strip()
            return text or "Error: No content in response."
        except Exception:
            return str(content) if content else "Error: No content in response."

    except Exception as e:
        logging.exception("Error during LangChain LLM call")
        msg = str(e)
        if "401" in msg or "auth" in msg.lower():
            return "Error: OpenAI Authentication Failed. Check your OPENAI_API_KEY."
        if "rate" in msg.lower():
            return "Error: OpenAI Rate Limit Exceeded."
        return f"Error: Could not generate completion - {e}"
