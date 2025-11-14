# ebpm_agents/agents/graph_builder_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseAgent
import json

class GraphBuilderAgent(BaseAgent):
    """
    KPI群から因果パス案（軽量版）。高度版は pathway_builder_agent を使用。
    """
    def run(self, kpis: List[Dict[str, Any]], max_scenarios: int = 3) -> Dict[str, Any]:
        kpi_json = json.dumps(kpis, ensure_ascii=False)
        prompt = (
            "以下のKPIノードを用いて因果パス案を複数提示。"
            "各案: {name,nodes:[...],edges:[{source,target,sign,lag_years,confidence}],comments}。JSONのみ。\n"
            f"KPI: {kpi_json}\nシナリオ数:{max_scenarios}"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.3, max_tokens=2000)
        try:
            arr = json.loads(out)
            if isinstance(arr, list): return {"graphs": arr}
        except Exception:
            pass
        return {"graphs": []}
