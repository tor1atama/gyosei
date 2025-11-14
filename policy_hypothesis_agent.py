# ebpm_agents/policy_hypothesis_agent.py
from __future__ import annotations
from typing import Any, Dict, List
import json

from .base import BaseAgent
from .utils import safe_json_loads

SYSTEM = (
    "あなたは公共政策の仮説形成を支援する専門家です。"
    "入力された論点と検索ヒットを基に、複数の政策仮説をJSONで返してください。"
    "説明文や余計な文字は不要で、厳格なJSONのみ。"
)

PROMPT = """ユーザ課題: {user_query}
サブプロブレム: {subproblem}
階層論点: {layer_label} ({tier})

検索ヒット（タイトルとURL、最大6件）:
{hits}

要件:
- 2件以上の仮説を出す（ヒットが少ない場合でも工夫する）
- 各仮説は {{"name","summary","expected_effect","kpi","evidence"}} を持つ
- evidence には参照したヒットのタイトルを列挙
- JSONのみで出力:
{{
  "hypotheses": [
    {{
      "name": "",
      "summary": "",
      "expected_effect": "",
      "kpi": ["", "..."],
      "evidence": ["", "..."]
    }}
  ]
}}
"""


class PolicyHypothesisAgent(BaseAgent):
    def run(
        self,
        user_query: str,
        subproblem: str,
        layer_label: str,
        tier: str,
        hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lines = [
            f"- {h.get('title','no-title')} :: {h.get('url','')}" for h in hits[:6]
        ]
        prompt = PROMPT.format(
            user_query=user_query,
            subproblem=subproblem,
            layer_label=layer_label,
            tier=tier,
            hits="\n".join(lines) if lines else "(該当ヒットなし)",
        )
        out = self.chat(SYSTEM, prompt, temperature=0.4, max_tokens=1200)
        try:
            return safe_json_loads(out)
        except Exception:
            fallback = []
            for h in hits[:2]:
                fallback.append(
                    {
                        "name": f"{layer_label}対策案",
                        "summary": f"{h.get('title','参考事例')}を参考にした施策案。",
                        "expected_effect": "関連KPIの改善を想定。",
                        "kpi": ["KPI候補"],
                        "evidence": [h.get("title","参考事例")],
                    }
                )
            if not fallback:
                fallback = [
                    {
                        "name": f"{layer_label}仮説A",
                        "summary": "検索ヒット不足のため、一般的な改善策を仮定。",
                        "expected_effect": "対象領域の効率化。",
                        "kpi": ["暫定KPI"],
                        "evidence": [],
                    }
                ]
            return {"hypotheses": fallback}
