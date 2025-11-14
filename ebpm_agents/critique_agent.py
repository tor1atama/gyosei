# ebpm_agents/agents/critique_agent.py
from __future__ import annotations
from typing import Dict, Any, List
import json
from .base import BaseAgent

SYSTEM = "あなたは政策案の批判的レビュー担当。日本語で端的に、JSONのみ返す。"

PROMPT = """以下の政策案群を、批判的に評価し、何が犠牲になるかも含めて整理してください。
出力はJSONのみ。スキーマ:
{{
  "reviews": [
    {{
      "strategy_name": "",
      "strengths": ["","..."],
      "weaknesses": ["","..."],
      "trade_offs": ["","..."],         // 何が犠牲になるか（例: 地域間公平性、医療者負荷、他事業機会費用）
      "risks": ["","..."],
      "mitigations": ["","..."],
      "unknowns": ["","..."]
    }}
  ],
  "cross_cutting_observations": ["","..."]
}}

政策案群:
{strategies}
"""

class CritiqueAgent(BaseAgent):
    def run(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        j = json.dumps(strategies, ensure_ascii=False)
        out = self.chat(SYSTEM, PROMPT.format(strategies=j), temperature=0.2, max_tokens=2400)
        try:
            return json.loads(out)
        except Exception:
            return {"reviews": [], "cross_cutting_observations": []}
