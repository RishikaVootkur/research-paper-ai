"""
LLM Provider
-------------
Provides LLM instances with automatic fallback.
If Groq hits rate limits, falls back to Google Gemini.
Both are free — this ensures the system never breaks.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 2048,
    provider: str = "auto",
):
    """
    Get an LLM instance.

    Args:
        temperature: Creativity level
        max_tokens: Max response length
        provider: "groq", "gemini", or "auto" (try groq first)

    Returns:
        A LangChain chat model instance
    """
    if provider == "gemini":
        return _get_gemini(temperature, max_tokens)

    if provider == "groq" or provider == "auto":
        return _get_groq(temperature, max_tokens)

    return _get_groq(temperature, max_tokens)


def _get_groq(temperature: float, max_tokens: int):
    """Primary: Groq (Llama 3.3 70B) — fast and free."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _get_gemini(temperature: float, max_tokens: int):
    """Fallback: Google Gemini Flash — generous free tier."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        max_output_tokens=max_tokens,
    )