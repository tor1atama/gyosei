# ebpm_agents/topic_map_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from .base import BaseAgent
from .utils import safe_json_loads

SYSTEM = (
    "あなたは政策課題の論点マッピングを行う専門アナリストです。"
    "日本語で簡潔なテキストのみを含む厳格なJSONを出力し、余計な説明は不要です。"
)

PROMPT = """ユーザの課題とサブプロブレムに基づき、検索前に論点を階層化してください。

要件:
- 3層（macro/meso/micro）を意識して論点を洗い出す
- 各層ごとに、論点の狙い・想定するデータ源・具体的な検索クエリ案（政策/エビデンス）を含める
- 県や市などの地理的フォーカスが想定される場合は child_nodes に記述する
- JSONのみ出力。スキーマ:
{{
  "topic_layers": [
    {{
      "subproblem": "",
      "overview": "",
      "layers": [
        {{
          "tier": "macro|meso|micro",
          "label": "",
          "policy_focus": "",
          "keywords": ["", "..."],
          "angles": ["", "..."],
          "sample_queries": {{
            "policy": ["", "..."],
            "evidence": ["", "..."]
          }},
          "child_nodes": [
            {{
              "label": "",
              "scope": "",
              "signals": ["", "..."],
              "sample_sources": ["", "..."]
            }}
          ]
        }}
      ]
    }}
  ],
  "global_queries": {{
    "broad": ["", "..."],
    "focused": ["", "..."]
  }}
}}

ユーザ課題: {user_query}
詳細化要約: {refined_summary}
サブプロブレム一覧(JSON): {subproblems}
"""


class TopicExplorerAgent(BaseAgent):
    def run(
        self,
        user_query: str,
        workplan: Dict[str, Any],
        refined_problem: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        refined_summary = refined_problem or {}
        if "refined_problem" in refined_summary:
            refined_summary = refined_summary.get("refined_problem") or {}
        subproblems = workplan.get("subproblems") or []
        prompt = PROMPT.format(
            user_query=user_query,
            refined_summary=json.dumps(refined_summary, ensure_ascii=False),
            subproblems=json.dumps(subproblems, ensure_ascii=False),
        )
        out = self.chat(
            SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=2000,
        )
        try:
            return safe_json_loads(out)
        except Exception:
            return {
                "topic_layers": [
                    {
                        "subproblem": (subproblems[0]["name"] if subproblems else user_query),
                        "overview": "論点マップ生成に失敗したため、手動で補完してください。",
                        "layers": [],
                    }
                ],
                "global_queries": {"broad": [user_query], "focused": []},
            }
