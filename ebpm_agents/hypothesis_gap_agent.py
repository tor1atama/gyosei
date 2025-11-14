# ebpm_agents/hypothesis_gap_agent.py
from __future__ import annotations
from typing import Any, Dict, List
import json

from .base import BaseAgent
from .utils import safe_json_loads

SYSTEM = (
    "あなたは政策仮説のピアレビュー担当。仮説の妥当性を疑い、補強に必要なデータやエビデンスを列挙する。"
    "回答は厳格なJSONのみ。説明は不要。"
)

PROMPT = """以下は政策課題に関する仮説一覧です。各仮説の弱点・疑問点と、それを補強するために必要なデータソースを列挙してください。
入力（JSON）:
{hypotheses}

出力スキーマ:
{{
  "gap_analysis": [
    {{
      "hypothesis": "",
      "concern": "",
      "needed_data": ["", ""],
      "priority": "high|medium|low"
    }}
  ]
}}

ルール:
- 各仮説を少なくとも1回は疑うこと。
- needed_data には具体的な統計/調査/質的データの例を挙げる。
- priority はエビデンス不足の大きさで決める。
"""

FALLBACK_GUIDE = {
    "医師偏在": ["医師届出票の地域別データ", "救急搬送拒否件数"],
    "非臨床業務過多": ["勤務表の時間配分調査", "電子カルテ操作ログ"],
    "救急搬送遅延": ["搬送時間と距離のタイムスタンプ", "ヘリ出動記録"],
    "ICT不足": ["システム稼働率ログ", "職員ITリテラシー調査"]
}


class HypothesisGapAgent(BaseAgent):
    def run(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not hypotheses:
            return {"gap_analysis": []}
        prompt = PROMPT.format(hypotheses=json.dumps(hypotheses, ensure_ascii=False))
        try:
            out = self.chat(SYSTEM, prompt, temperature=0.1, max_tokens=1200)
            data = safe_json_loads(out)
            if isinstance(data, dict) and isinstance(data.get("gap_analysis"), list):
                return data
        except Exception:
            pass
        fallback = []
        for row in hypotheses:
            label = row.get("cluster_lv1") or row.get("cause") or "仮説"
            needs = FALLBACK_GUIDE.get(label, ["統計データ", "住民ヒアリング"])
            fallback.append({
                "hypothesis": label,
                "concern": "裏付けとなる定量データが不足",
                "needed_data": needs,
                "priority": "medium"
            })
        return {"gap_analysis": fallback}


__all__ = ["HypothesisGapAgent"]
