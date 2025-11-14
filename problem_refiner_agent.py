# ebpm_agents/agents/problem_refiner_agent.py
from __future__ import annotations
from typing import Dict, Any
import json
from .base import BaseAgent

SYSTEM = "あなたは行政政策課題の要件定義の専門家。日本語で簡潔に、JSONのみ返す。"

PROMPT = """ユーザ入力の政策課題を、意思決定に足るレベルに詳細化してください。
出力は厳格なJSONのみで、以下のスキーマに従うこと。

{
  "refined_problem": {
    "title": "",
    "background": "",
    "stakeholders": ["", "..."],
    "constraints": ["", "..."],         // 法規制・財政・人材・時間等
    "assumptions": ["", "..."],         // 暫定前提
    "kpi_candidates": ["", "..."],      // 成功指標の候補（定義不要）
    "scope": {"geo": "", "population": "", "period": ""},
    "risk_points": ["", "..."],         // 主要リスク
    "questions_for_confirmation": ["", "..."] // 重要な確認質問
  },
  "confirm_prompt": "この内容で進めてよいですか？ (y/n)"
}

ユーザ入力:
{user_query}
"""

class ProblemRefinerAgent(BaseAgent):
    def run(self, user_query: str) -> Dict[str, Any]:
        # JSONスキーマを含むテンプレは .format に不向き。ユーザ入力を後置で連結して安全化。
        out = self.chat(
            SYSTEM,
            f"{PROMPT}\n\nユーザ要件:\n{user_query}",
            temperature=0.2,
            max_tokens=1600,
        )
        try:
            return json.loads(out)
        except Exception:
            return {"refined_problem": {"title": user_query}, "confirm_prompt": "この内容で進めてよいですか？ (y/n)"}
