# ebpm_agents/web_timeseries_agent.py
from __future__ import annotations
from typing import Any, Dict, List
import time, json, re
from urllib.parse import quote_plus, urlparse
import requests
from bs4 import BeautifulSoup
import trafilatura
from .base import BaseAgent
from .utils import safe_json_loads

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
DDG_HTML = "https://duckduckgo.com/html/?q="

PREFERRED = [
    "go.jp","e-stat.go.jp","mhlw.go.jp","cao.go.jp","meti.go.jp","mext.go.jp","soumu.go.jp","stat.go.jp",
    "data.go.jp","who.int","oecd.org","worldbank.org","ilo.org","imf.org","un.org","gov.uk","europa.eu"
]

class TimeSeriesWebAgent(BaseAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def propose_queries(self, label: str) -> Dict[str, Any]:
        prompt = (
            "与えられた指標候補（名詞句）に対し、過去時系列を探す検索クエリ案を日英で設計。"
            "JSONのみ: {\"queries\":[\"...\"],\"hints\":{\"unit_candidates\":[],\"source_hints\":[],\"notes\":\"\"}}\n"
            f"ノード: {label}"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.0)
        try:
            return safe_json_loads(out)
        except Exception:
            return {"queries":[label],"hints":{"unit_candidates":[],"source_hints":[],"notes":""}}

    def ddg_search(self, q: str, max_results: int = 6) -> List[Dict[str, str]]:
        url = DDG_HTML + quote_plus(q)
        r = self.session.get(url, timeout=15); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html5lib")
        out = []
        for a in soup.select("a.result__a"):
            href = a.get("href"); title = a.get_text(" ", strip=True)
            if href: out.append({"title": title, "url": href})
            if len(out) >= max_results: break
        # 公的サイトを優先
        def is_pref(u: str) -> bool: return any(dom in urlparse(u).netloc.lower() for dom in PREFERRED)
        uniq, seen = [], set()
        for it in out:
            u = it["url"]
            if u in seen: continue
            seen.add(u); uniq.append(it)
        uniq.sort(key=lambda x: (not is_pref(x["url"])))
        return uniq

    def fetch_text(self, url: str) -> Dict[str, str]:
        try:
            r = self.session.get(url, timeout=20); r.raise_for_status()
            text = trafilatura.extract(r.text, include_tables=True, no_fallback=False, url=url) or ""
            try:
                soup = BeautifulSoup(r.text, "html5lib")
                title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
            except Exception:
                title = ""
            return {"url": url, "title": title, "text": text[:200000]}
        except Exception:
            return {"url": url, "title": "", "text": ""}

    def corpus_from_hits(self, hits: List[Dict[str, str]], max_pages: int = 12) -> str:
        pages = []
        for h in hits[:max_pages]:
            page = self.fetch_text(h["url"])
            if len(page.get("text","")) >= 400: pages.append(page)
            time.sleep(0.8)
        merged = "\n\n".join([f"### {p['title']}\n{p['text']}" for p in pages])
        srcs = "\n".join([f"- {p['title']} [SRC] {p['url']}" for p in pages])
        return (merged + "\n\n[SOURCES]\n" + srcs)[:120000]

    def extract_timeseries(self, label: str, corpus: str) -> Dict[str, Any]:
        prompt = (
            "複数ページ本文から、与えたノードに関係する時系列データを新しい順で最大10点、"
            "厳格JSONのみで返せ。スキーマ:{\"label\":\"\",\"timeseries\":[{\"date\":\"YYYY-MM-DD\",\"value\":0.0,"
            "\"unit\":\"\",\"measure_name\":\"\",\"source_url\":\"\",\"source_title\":\"\",\"unit_note\":\"\",\"confidence\":0.0}]}\n"
            f"ノード:{label}\n本文:\n{corpus}"
        )
        out = self.chat("JSONのみ返す。", prompt, temperature=0.0, max_tokens=2200)
        try:
            js = safe_json_loads(out)
        except Exception:
            js = {"label": label, "timeseries": []}
        # 整形：上位10点
        try:
            ts = js.get("timeseries", [])
            ts = sorted(ts, key=lambda d: d.get("date",""), reverse=True)[:10]
            js["timeseries"] = ts
        except Exception:
            js = {"label": label, "timeseries": []}
        return js

    def run_label(self, label: str) -> Dict[str, Any]:
        qobj = self.propose_queries(label)
        queries = (qobj.get("queries") or [label])[:6]
        all_hits = []
        for q in queries:
            all_hits.extend(self.ddg_search(q, max_results=6))
        corpus = self.corpus_from_hits(all_hits, max_pages=12)
        return self.extract_timeseries(label, corpus)
