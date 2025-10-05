# llm_service.py
from __future__ import annotations  # safe; helps if you add Pydantic models later

from typing import Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import OPENAI_API_KEY  # <- use root config.py

# --- LangChain LLM factory ---
def get_llm(
    model: str = "gpt-4.1-nano",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI client."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env.")
    return ChatOpenAI(
        model=model,              # e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-0125"
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens,
    )

# --- Generate via LangChain ---
def generate_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """Generates a chat completion using LangChain's ChatOpenAI."""
    try:
        llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        ai_message = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        content = ai_message.content
        if isinstance(content, str):
            return content.strip() or "Error: No content in response."
        # Sometimes tools return segments; normalize to text
        try:
            text = " ".join(seg.get("text", "") for seg in content).strip()
            return text or "Error: No content in response."
        except Exception:
            return str(content)
    except Exception as e:
        logging.exception("Error during LangChain LLM call")
        msg = str(e)
        if "401" in msg or "auth" in msg.lower():
            return "Error: OpenAI Authentication Failed. Check your OPENAI_API_KEY."
        if "rate" in msg.lower():
            return "Error: OpenAI Rate Limit Exceeded."
        return f"Error: Could not generate completion - {e}"



'''from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os, json, re, logging
import openai
from utils.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

#task - from langchain access open ai - implement model --->> here opneai is used we need to use langchain

def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",   # or "gpt-4o-mini" if you prefer
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )

def generate_completion(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int=500, temperature: float = 0.7) -> str:
    """Generates text completion using OpenAI API."""
    try:
        response = openai.chat.completions.create(
            model=model, //
            messages=[
                {"role": "system", "content":"You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
        #check if response.choices exists and has items
        if response.choices and len(response.choices)>0:
            #check if message exixts and has content
            if response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print("Warning: LLM response message or content is empty.")
                return "Error: No content in response."
        else:
            print("Warning: LLM response choices list is empty.")
            return "Error: No choices"
        
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication Error: {e}")
        print("Please check your OPENAI_API_KEY in the .env file.")
        return "Error: OpenAI Authentication Failed."
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: OpenAI Rate Limit Exceeded."
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return f"Error: Could not generate completion - {e}"'''