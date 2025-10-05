# core/llm_service.py
from __future__ import annotations
from typing import Optional, Iterable, Generator
import logging, os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from utils.config import OPENAI_API_KEY

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: Optional[int] = None) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your secrets/env.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY, max_tokens=max_tokens)

def _lc_msg(msg: dict) -> BaseMessage:
    return SystemMessage(content=msg["content"]) if msg.get("role") == "system" else HumanMessage(content=msg.get("content",""))

def _get_openai_client():
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found.")
    from openai import OpenAI
    return OpenAI(api_key=key)

def generate_completion(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 800,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful AI assistant.",
    system: Optional[str] = None,                 # alias
    stream: bool = False,
    extra_messages: Optional[Iterable[dict]] = None,
    response_format: Optional[dict] = None,       # -> use Chat Completions JSON mode
) -> str | Generator[str, None, None]:

    sys_txt = (system or system_prompt or "").strip()

    # JSON mode path (Chat Completions)
    if response_format and not stream:
        client = _get_openai_client()
        messages = []
        if sys_txt:
            messages.append({"role": "system", "content": sys_txt})
        if extra_messages:
            messages.extend(list(extra_messages))
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,     # {"type":"json_object"}
            )
        except TypeError:
            # Older SDK without response_format support – try without it (we'll parse manually)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return (resp.choices[0].message.content or "").strip()

    # Default LangChain path
    try:
        llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        msgs: list[BaseMessage] = []
        if sys_txt:
            msgs.append(SystemMessage(content=sys_txt))
        if extra_messages:
            msgs.extend(_lc_msg(m) for m in extra_messages)
        msgs.append(HumanMessage(content=prompt))

        if stream:
            def _gen():
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
        try:
            text = " ".join(seg.get("text", "") for seg in content).strip()
            return text or "Error: No content in response."
        except Exception:
            return str(content) if content else "Error: No content in response."
    except Exception as e:
        logging.exception("Error during LLM call")
        msg = str(e)
        if "401" in msg or "auth" in msg.lower():
            return "Error: OpenAI Authentication Failed. Check your OPENAI_API_KEY."
        if "rate" in msg.lower():
            return "Error: OpenAI Rate Limit Exceeded."
        return f"Error: Could not generate completion - {e}"
