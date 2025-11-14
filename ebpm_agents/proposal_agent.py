# ebpm_agents/agents/proposal_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseAgent
import json

class KPIProposalAgent(BaseAgent):
    def run(self, user_need: str, current_kpis: List[Dict[str, Any]]) -> Dict[str, Any]:
        kjson = json.dumps(current_kpis, ensure_ascii=False)
        prompt = (
            f"要件:{user_need}\n既存KPI:{kjson}\n"
            "不足があれば追加提案。不要なら空。JSON配列のみ: {name,horizon,justification,definition,unit,direction}"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.3)
        try:
            data = json.loads(out)
            if isinstance(data, list):
                return {"added_kpis": data}
        except Exception:
            pass
        return {"added_kpis": []}
