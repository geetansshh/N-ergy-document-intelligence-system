"""
Central LLM factory.

Set the environment variable LLM_PROVIDER to switch providers:
  - "groq"   (default) — llama-3.3-70b-versatile via Groq API
  - "gemini"            — gemini-2.0-flash via Google AI API
"""

import os

PROVIDER_GROQ   = "groq"
PROVIDER_GEMINI = "gemini"


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", PROVIDER_GROQ).lower()


def get_llm(temperature: float = 0.0):
    """Return the LangChain chat model for the active provider."""
    provider = get_provider()

    if provider == PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # Default: Groq
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )
