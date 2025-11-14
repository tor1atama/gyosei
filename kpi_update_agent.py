# ebpm_agents/kpi_update_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json, re, unicodedata, time
from pathlib import Path
from .base import BaseAgent
from .utils import extract_pdf_text, join_with_markers, chunk_by_chars, safe_json_loads

def norm_name(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip().casefold()

def merge_kpi(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    out = {"quantitative_kpi": [], "hard_to_quantify_kpi": []}
    seen_q = {norm_name(x["name"]) for x in base.get("quantitative_kpi", []) if x.get("name")}
    seen_h = {norm_name(x["name"]) for x in base.get("hard_to_quantify_kpi", []) if x.get("name")}
    out["quantitative_kpi"].extend(base.get("quantitative_kpi", []))
    out["hard_to_quantify_kpi"].extend(base.get("hard_to_quantify_kpi", []))
    for x in add.get("quantitative_kpi", []):
        if norm_name(x.get("name")) not in seen_q:
            out["quantitative_kpi"].append(x); seen_q.add(norm_name(x.get("name")))
    for x in add.get("hard_to_quantify_kpi", []):
        if norm_name(x.get("name")) not in seen_h:
            out["hard_to_quantify_kpi"].append(x); seen_h.add(norm_name(x.get("name")))
    return out

EXTRACT_TPL = """あなたはKPI設計の専門家。以下の複数断片(intermediate)に含まれるKPI候補を、
A:quantitative / B:hard_to_quantify に分類して厳格なJSONのみで返せ。
{
  "quantitative_kpi":[{"name":"","note":""}],
  "hard_to_quantify_kpi":[{"name":"","note":""}]
}
入力:
{payload}
"""

UPDATE_TPL = """既存KPI(重複除外基準: 名称/意味が同じなら除外):
{seed}

以下のPDFチャンクから、新規・改良KPIのみをJSONで返せ（同スキーマ）。
PDFテキスト(チャンク):
{chunk}
"""

class KPIUpdateAgent(BaseAgent):
    def seed_from_intermediate(self, fragments: List[str]) -> Dict[str, Any]:
        payload = "\n\n---\n".join(fragments)
        out = self.chat("JSONのみ返す。", EXTRACT_TPL.format(payload=payload), temperature=0.0, max_tokens=1600)
        try:
            data = safe_json_loads(out)
            if isinstance(data, dict):
                data.setdefault("quantitative_kpi", []); data.setdefault("hard_to_quantify_kpi", [])
                return data
        except Exception:
            pass
        return {"quantitative_kpi": [], "hard_to_quantify_kpi": []}

    def update_from_pdfs(self, seed: Dict[str, Any], pdf_paths: List[str], max_chars: int = 8000, sleep: float = 0.05) -> Dict[str, Any]:
        updated = seed
        for p in pdf_paths:
            pages = extract_pdf_text(p)
            full = join_with_markers(pages)
            chunks = chunk_by_chars(full, max_chars=max_chars)
            for _, c in chunks:
                out = self.chat("JSONのみ返す。", UPDATE_TPL.format(seed=json.dumps(updated, ensure_ascii=False), chunk=c), temperature=0.0, max_tokens=1600)
                try:
                    add = safe_json_loads(out)
                    updated = merge_kpi(updated, add)
                except Exception:
                    pass
                time.sleep(sleep)
        return updated
