# ebpm_agents/pathway_builder_agent.py
from __future__ import annotations
from typing import Any, Dict, List
import json
from string import Template
from .base import BaseAgent

PATHWAY_PROMPT_TPL = Template(r"""あなたは政策因果ネットワーク設計者。
指定KPIに対し、予算(BUDGET)→…→KPI_DELTA のDAGを構築せよ。説明文は禁止、厳格JSONのみ。

スキーマ:
{
  "kpi":"${kpi_name}",
  "graph":{"nodes":[...],"edges":[...]},
  "path_decomposition":[...],
  "calculation_stub":{"assumptions":[...],"pseudocode":[...]},
  "sources_used":{"pdf_pages_cited":[], "web_urls":[]},
  "assumptions":[...],
  "flags":{"acyclic":true, "needs_more_evidence":false}
}

材料（PDF要約断片/検索結果）:
【PDF】${pdf_context}
【WEB】${web_context}
""")

class PathwayBuilderAgent(BaseAgent):
    def run(self, kpi_name: str, pdf_context: str, web_context: str) -> Dict[str, Any]:
        prompt = PATHWAY_PROMPT_TPL.substitute(kpi_name=kpi_name, pdf_context=pdf_context, web_context=web_context)
        out = self.chat("JSONのみ返す。", prompt, temperature=0.0, max_tokens=4000)
        try:
            return json.loads(out)
        except Exception:
            from .utils import safe_json_loads
            return safe_json_loads(out)
