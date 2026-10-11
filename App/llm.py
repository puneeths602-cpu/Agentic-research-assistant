"""LLM provider selection.

Uses OpenAI or Gemini if an API key is set, otherwise falls back to a
deterministic offline model so the pipeline, tests and evaluation run anywhere.
"""
import os
import re


class OfflineLLM:
    """Deterministic stand-in: plans by splitting the question and writes
    extractive answers from the supplied context. Not a real language model."""

    name = "offline-extractive"

    def invoke(self, prompt: str) -> str:
        if prompt.startswith("[PLAN]"):
            q = prompt.split("\n", 1)[1]
            parts = [p.strip() for p in re.split(r"\band\b|;|\?\s", q) if len(p.strip()) > 8]
            return "\n".join(parts[:3] or [q.strip()])
        if prompt.startswith("[WRITE]"):
            ctx = prompt.split("CONTEXT:", 1)[-1]
            lines = [l.strip() for l in ctx.splitlines() if l.strip().startswith("[")]
            return "\n".join(f"- {l}" for l in lines[:6]) or "No supporting context found."
        return ""


def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        m = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        return _Wrap(m, "openai")
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), temperature=0)
        return _Wrap(m, "gemini")
    return OfflineLLM()


class _Wrap:
    def __init__(self, model, name):
        self.model, self.name = model, name

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt.replace("[PLAN]", "").replace("[WRITE]", "")).content
