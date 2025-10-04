from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import logging

from utils.config import OPENAI_API_KEY

# --- LangChain LLM factory ---
def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI client.
    """
    return ChatOpenAI(
        model=model,              # e.g., "gpt-3.5-turbo" or "gpt-4o-mini"
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens     # you can leave this None and pass per-call if preferred
    )

# --- Generate via LangChain (no direct openai.* calls) ---
def generate_completion(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 500,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """
    Generates a chat completion using LangChain's ChatOpenAI.
    """
    try:
        llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        # LangChain call
        ai_message = llm.invoke(messages)

        # ai_message.content is usually a string, but can be a list (if tool-calls etc).
        content = ai_message.content
        if isinstance(content, str):
            text = content.strip()
        else:
            # Fallback: join any non-string segments to text
            try:
                text = " ".join([seg.get("text", "") for seg in content]).strip()
            except Exception:
                text = str(content)

        if not text:
            logging.warning("Warning: LLM response content is empty.")
            return "Error: No content in response."

        return text

    except Exception as e:
        # LangChain wraps provider errors; keep a simple, user-friendly message.
        logging.exception("Error during LangChain LLM call")
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower():
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