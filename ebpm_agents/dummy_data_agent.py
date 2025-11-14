# ebpm_agents/dummy_data_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import random

from .base import BaseAgent

SYSTEM = (
    "あなたは政策調査補助エージェント。実際の検索結果が得られない場合に、それっぽい既存資料風データを生成する。"
    "返答は JSON のみで、信ぴょう性が低い旨を snippet に含める。"
)

PROMPT = """ユーザ課題: {user_query}
サブプロブレム: {subproblem}
要求種別: {data_type}
生成件数: {count}

下記フォーマットで JSON のみ出力せよ:
{{
  "items": [
    {{
      "title": "",
      "url": "https://example.com/...",
      "snippet": "",
      "source": "existing"
    }}
  ]
}}

備考: snippet に「既存資料参照」である旨と仮想エビデンスの種類（ガイドライン、県報告等）を含めること。
"""

USE_LLM_FOR_DUMMY = False

FALLBACK_TITLES = [
    "架空県 医療計画 2024 既存資料",
    "既存市 救急受入れレビュー",
    "フィクション救命センター 既存評価記録",
]


class DummyDataAgent(BaseAgent):
    def run(
        self,
        user_query: str,
        subproblem: Optional[str],
        data_type: str,
        count: int = 3,
    ) -> Dict[str, Any]:
        if USE_LLM_FOR_DUMMY:
            prompt = PROMPT.format(
                user_query=user_query or "",
                subproblem=subproblem or "",
                data_type=data_type,
                count=count,
            )
            try:
                out = self.chat(SYSTEM, prompt, temperature=0.1, max_tokens=600)
                data = self._safe_parse(out)
                if data:
                    return data
            except Exception:
                pass
        return {"items": self._fallback_items(count)}

    def _safe_parse(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return data
        except Exception:
            return None
        return None

    def _fallback_items(self, count: int) -> List[Dict[str, str]]:
        items = []
        for _ in range(max(1, count)):
            title = random.choice(FALLBACK_TITLES)
            items.append({
                "title": title,
                "url": f"https://example.com/existing/{random.randint(1000, 9999)}",
                "snippet": f"既存資料参照: {title} (参考用)",
                "source": "existing",
            })
        return items


__all__ = ["DummyDataAgent"]
