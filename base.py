# ebpm_agents/agents/base.py
from __future__ import annotations
from typing import Optional
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL, require_openai

class BaseAgent:
    def __init__(self, model: Optional[str] = None):
        require_openai()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or OPENAI_MODEL

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1600) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
