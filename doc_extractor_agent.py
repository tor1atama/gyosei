# ebpm_agents/doc_extractor_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json, time
from .base import BaseAgent
from .utils import extract_pdf_text, join_with_markers, chunk_by_chars, safe_json_loads

class ReviewSheetPrimer(BaseAgent):
    def build(self, rs_pdf_text: str) -> Dict[str, Any]:
        prompt = (
            "あなたは行政レビューシートの専門家。"
            "以下の全文を読み、JSONのみで返せ: "
            "{\"what_is_review_sheet\":{\"definition\":\"\",\"must_include\":[],\"synonyms\":[]},"
            "\"what_is_effect_model\":{\"definition\":\"\",\"must_include\":[],\"synonyms\":[]}}"
            f"\n\n【RS_format全文】\n{rs_pdf_text}"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.0, max_tokens=2000)
        return safe_json_loads(out)

MAP_TMPL = """あなたはレビューシート抽出器。以下の断片から
1) 概要・目的（原文抜粋,要約禁止）
2) 効果発現経路（Activity→Output→Outcome short/mid/long）部分
のみを抽出し、厳格JSONのみで返す。

JSON:
{
  "overview_candidates":[{"page":"pX","text":"..."}],
  "graph_candidates":{
    "activities":[{"label":"...","page":"pX"}],
    "outputs":[{"label":"...","page":"pX"}],
    "outcomes":{"short":[...],"mid":[...],"long":[...]},
    "edges":[{"from":"<label>","to":"<label>","type":"activity->output|output->short|short->mid|mid->long","evidence":{"page":"pX","quote":"..."}}]
  },
  "pages_seen":["pX","pY"]
}

【テキスト断片】:
{chunk}
"""

REDUCE_TMPL = """あなたは統合器。配列で渡す候補JSONを重複・同義統合し、
以下の最終スキーマ(JSONのみ)で出力せよ。
{
  "skip": false,
  "file_path": "{file_path}",
  "title": "",
  "matched_RS": true,
  "overview_purpose": "",
  "effect_pathway": {
    "activities":[{"id":"A1","label":""}],
    "outputs":[{"id":"O1","label":""}],
    "outcomes":{"short":[{"id":"S1","label":""}],"mid":[{"id":"M1","label":""}],"long":[{"id":"L1","label":""}]},
    "edges":[
      {"from":"A1","to":"O1","type":"activity->output","evidence":{"page":"pX","quote":""}},
      {"from":"O1","to":"S1","type":"output->short","evidence":{"page":"pY","quote":""}},
      {"from":"S1","to":"M1","type":"short->mid","evidence":{"page":"pZ","quote":""}},
      {"from":"M1","to":"L1","type":"mid->long","evidence":{"page":"pW","quote":""}}
    ]
  },
  "pages_cited": ["pX","pY"],
  "confidence": 0.0,
  "flags": {"missing_overview_purpose": false, "incomplete_graph": false, "needs_ocr": false}
}

【候補配列】:
{maps}
"""

class ReviewSheetExtractor(BaseAgent):
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        pages = extract_pdf_text(file_path)
        total_chars = sum(len(t) for _, t in pages)
        needs_ocr = total_chars < 50
        full = join_with_markers(pages)
        chunks = chunk_by_chars(full, max_chars=8000)

        maps = []
        for _, chunk in chunks:
            out = self.chat("JSONのみ返す。", MAP_TMPL.format(chunk=chunk), temperature=0.0, max_tokens=2000)
            try: maps.append(safe_json_loads(out))
            except Exception: maps.append({"overview_candidates": [], "graph_candidates":{"activities":[],"outputs":[],"outcomes":{"short":[],"mid":[],"long":[]},"edges": []}, "pages_seen":[]})
            time.sleep(0.05)

        maps_json = json.dumps(maps, ensure_ascii=False)
        reduced = self.chat("JSONのみ返す。", REDUCE_TMPL.format(file_path=file_path, maps=maps_json), temperature=0.0, max_tokens=3000)
        final = safe_json_loads(reduced)
        if needs_ocr:
            final.setdefault("flags",{}).update({"needs_ocr": True})
        return final
