# ebpm_agents/agents/kpi_extractor_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseAgent

class KPIExtractorAgent(BaseAgent):
    def run(self, user_need: str, policy_hits: List[Dict[str, Any]], paper_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        pol = "\n".join([f"- {h.get('title','')} :: {h.get('url','')}" for h in (policy_hits or [])[:10]])
        pap = "\n".join([f"- {h.get('title','')} :: {h.get('url','')}" for h in (paper_hits or [])[:10]])
        prompt = (
            f"要件: {user_need}\n政策候補:\n{pol}\n\n論文候補:\n{pap}\n\n"
            "KPI候補を短期/中期/長期で分類し、{name,horizon,definition,unit,data_source_hint,direction,baseline_expected,risk_note} の配列(JSONのみ)で返せ。"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.2)
        import json
        try:
            arr = json.loads(out)
            if isinstance(arr, list): return {"kpis": arr}
        except Exception:
            pass
        return {"kpis": []}
