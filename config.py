# ebpm_agents/config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
]
for p in _ENV_CANDIDATES:
    try:
        if p.is_file():
            load_dotenv(p)
            break
    except Exception:
        pass
# fallback
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def require_openai() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
