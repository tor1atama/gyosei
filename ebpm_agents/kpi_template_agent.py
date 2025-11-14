# ebpm_agents/kpi_template_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from .base import BaseAgent
from .utils import safe_json_loads

__all__ = ["KPITemplateAgent", "DEFAULT_KPI_TEMPLATES"]

SYSTEM = (
    "あなたは政策KPI設計の専門家。日本語で厳格なJSONのみを返し、余計な文は入れない。"
)

PROMPT = """ユーザ課題: {user_query}
既知ドメインヒント: {domain_hint}
既存KPIカタログ例: {catalog}

KPI候補を最大5件挙げ、各候補に以下のキーを含める:
- name: KPI名称
- definition: 測定対象の定義
- unit: 単位
- direction: up/down (大きいほど良い or 小さいほど良い)
- threshold_type: min/max (最低確保 or 上限)
- threshold_hint: 参考となる閾値の数値（float）
- data_source: 主なデータソース
- legal_floor: 関連する法制度や基準名（無ければ""）
- rationale: 先行事例や根拠の短い説明

JSONのみで以下形式で返答:
{{
  "kpi_candidates": [ {{...}} ]
}}
"""

DEFAULT_KPI_TEMPLATES = [
    {
        "name": "救急搬送受入率",
        "definition": "搬送要請のうち受入できた割合",
        "unit": "%",
        "direction": "up",
        "threshold_type": "min",
        "threshold_hint": 80.0,
        "data_source": "消防庁救急搬送統計",
        "legal_floor": "医療法第25条",
        "rationale": "過去の地域医療構想の評価指標"
    },
    {
        "name": "医師1人あたり患者数",
        "definition": "常勤換算医師1人が担当する年間患者数",
        "unit": "人/年",
        "direction": "down",
        "threshold_type": "max",
        "threshold_hint": 1800.0,
        "data_source": "地域医療計画・医師届出票",
        "legal_floor": "",
        "rationale": "OECDベンチマークを参照"
    }
]


class KPITemplateAgent(BaseAgent):
    def run(
        self,
        user_query: str,
        domain_hint: Optional[str] = None,
        catalog_examples: Optional[List[str]] = None,
        max_items: int = 5,
    ) -> Dict[str, Any]:
        catalog_json = json.dumps(catalog_examples or [], ensure_ascii=False)
        prompt = PROMPT.format(
            user_query=user_query,
            domain_hint=domain_hint or "",
            catalog=catalog_json,
        )
        out = self.chat(
            SYSTEM,
            prompt,
            temperature=0.2,
            max_tokens=1200,
        )
        try:
            data = safe_json_loads(out)
            items = data.get("kpi_candidates")
            if isinstance(items, list) and items:
                return {"kpi_candidates": items[:max_items]}
        except Exception:
            pass
        return {"kpi_candidates": DEFAULT_KPI_TEMPLATES[:max_items]}
