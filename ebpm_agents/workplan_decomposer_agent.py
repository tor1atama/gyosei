# ebpm_agents/agents/workplan_decomposer_agent.py
from __future__ import annotations
from typing import Dict, Any
import json
from .base import BaseAgent

SYSTEM = "あなたは政策調査の検索計画を設計する専門家。日本語でJSONのみ返す。説明や前置きは禁止。"

PROMPT = """以下の詳細化された政策課題を、検索に適した作業単位へ分解し、言語（日本語・英語）別のクエリを設計してください。
厳格にJSONのみで出力し、最低でも3件の subproblems を含めてください。各クエリは具体的な名詞を含め、曖昧語を避けてください。スキーマ:
{
  "subproblems": [
    {
      "name": "",
      "objective": "",
      "queries": { "ja": ["...", "..."], "en": ["...", "..."] },
      "policy_keywords": ["", "..."],
      "paper_evidence_needed": ["systematic review","RCT","guideline","observational","quasi-experimental"],
      "kpi_hints": ["", "..."]
    }
  ]
}

詳細化課題:
{refined}
"""

class WorkplanDecomposerAgent(BaseAgent):
    def run(self, refined_problem: Dict[str, Any]) -> Dict[str, Any]:
        j = json.dumps(refined_problem, ensure_ascii=False)
        # .formatを避けて安全に連結
        out = self.chat(
            SYSTEM,
            f"{PROMPT}\n\n詳細化課題(再掲):\n{j}",
            temperature=0.0,
            max_tokens=2000,
        )
        # JSON救済パーサ
        try:
            from .utils import safe_json_loads
            data = safe_json_loads(out)
        except Exception:
            try:
                data = json.loads(out)
            except Exception:
                data = None
        if isinstance(data, dict) and isinstance(data.get("subproblems"), list) and data["subproblems"]:
            return data
        # フォールバック: refined_problem から素朴に3件生成
        rp = refined_problem or {}
        title = (rp.get("refined_problem", {}) or {}).get("title") or rp.get("title") or "政策課題"
        kpis = (rp.get("refined_problem", {}) or {}).get("kpi_candidates") or rp.get("kpi_candidates") or []
        subs = [
            {
                "name": f"現状把握と課題定量化: {title}",
                "objective": "統計/行政データから現状と格差を定量化する",
                "queries": {"ja": [f"{title} 統計 データ", f"{title} 地域差 指標"], "en": [f"{title} statistics by region", f"{title} inequality indicators"]},
                "policy_keywords": ["白書", "統計", "行政データ"],
                "paper_evidence_needed": ["systematic review","observational"],
                "kpi_hints": kpis[:3] if isinstance(kpis, list) else []
            },
            {
                "name": f"対策候補の収集とレビュー: {title}",
                "objective": "国内外の政策事例・評価を収集し有望策を洗い出す",
                "queries": {"ja": [f"{title} 政策 事例 評価", f"{title} レビュー PDF"], "en": [f"{title} policy case evaluation", f"{title} review report PDF"]},
                "policy_keywords": ["レビュー", "評価", "施策"],
                "paper_evidence_needed": ["systematic review","RCT","quasi-experimental"],
                "kpi_hints": kpis[:3] if isinstance(kpis, list) else []
            },
            {
                "name": f"KPI/データ源の確定: {title}",
                "objective": "KPIの定義・単位・データ取得方法を確定する",
                "queries": {"ja": [f"{title} KPI 定義", f"{title} 指標 単位 データソース"], "en": [f"{title} KPI definition", f"{title} indicator unit data source"]},
                "policy_keywords": ["KPI", "指標", "データソース"],
                "paper_evidence_needed": ["guideline","observational"],
                "kpi_hints": kpis[:5] if isinstance(kpis, list) else []
            },
        ]
        return {"subproblems": subs}
