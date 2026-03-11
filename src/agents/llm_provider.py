"""
LLM Provider with Automatic Fallback
--------------------------------------
Provides LLM instances that automatically fall back to Google Gemini
when Groq hits its daily rate limit.

This wraps the LLM call so the rest of the codebase doesn't
need to handle rate limit errors.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

load_dotenv()


class FallbackLLM(BaseChatModel):
    """
    An LLM wrapper that tries Groq first, falls back to Gemini on rate limits.

    This is transparent to the rest of the code — you use it exactly
    like any other LangChain chat model.
    """
    primary: BaseChatModel = None
    fallback: BaseChatModel = None
    _using_fallback: bool = False

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fallback_llm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Try primary (Groq), fall back to Gemini on rate limit."""
        try:
            if not self._using_fallback:
                return self.primary._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print("  ⚠️  Groq rate limited — switching to Gemini")
                self._using_fallback = True
            else:
                raise

        return self.fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> FallbackLLM:
    """
    Get an LLM with automatic Groq → Gemini fallback.

    Usage:
        llm = get_llm(temperature=0.1)
        response = llm.invoke("Hello")
        # Works exactly like ChatGroq or ChatGoogleGenerativeAI
    """
    primary = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    fallback = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    return FallbackLLM(primary=primary, fallback=fallback)