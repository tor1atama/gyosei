# ebpm_agents/agents/budget_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re, json, time
from urllib.parse import quote_plus, urlparse
import requests
from bs4 import BeautifulSoup, FeatureNotFound
import trafilatura
from .base import BaseAgent

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
DDG_HTML = "https://duckduckgo.com/html/?q="

def _is_rs(u: str) -> bool:
    try:
        return "rssystem.go.jp" in (urlparse(u).netloc or "").lower()
    except Exception:
        return False

def _make_soup(html: str) -> BeautifulSoup:
    for parser in ("html5lib", "lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
    return BeautifulSoup(html, "html.parser")


def _ddg_rs_search(q: str, max_results: int = 6) -> List[Dict[str, str]]:
    query = f'site:rssystem.go.jp "{q}" 予算 OR 事業費 OR 総事業費 project'
    url = DDG_HTML + quote_plus(query)
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": UA})
        r.raise_for_status()
    except Exception:
        return []
    soup = _make_soup(r.text)
    out: List[Dict[str, str]] = []
    for a in soup.select("a.result__a"):
        href = a.get("href"); title = a.get_text(" ", strip=True)
        if href and _is_rs(href):
            out.append({"title": title, "url": href})
        if len(out) >= max_results: break
    # 重複除去
    seen, uniq = set(), []
    for it in out:
        if it["url"] in seen: continue
        seen.add(it["url"]); uniq.append(it)
    return uniq

_money_rx = re.compile(
    r"(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*(?P<unit>兆円|億円|千万円|百万円|万円|円)"
)

def _yen_from_text(num: str, unit: str) -> float:
    x = float(num.replace(",", ""))
    if unit == "兆円": return x * 1e12
    if unit == "億円": return x * 1e8
    if unit == "千万円": return x * 1e7
    if unit == "百万円": return x * 1e6
    if unit == "万円": return x * 1e4
    return x

def _fetch_text(u: str) -> Dict[str, str]:
    try:
        r = requests.get(u, timeout=25, headers={"User-Agent": UA})
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_tables=True, no_fallback=False, url=u) or ""
        soup = _make_soup(r.text)
        title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
        return {"title": title, "url": u, "text": text[:200000]}
    except Exception:
        return {"title": "", "url": u, "text": ""}

def _extract_budgets(text: str) -> List[Tuple[float, str]]:
    if not text: return []
    out = []
    for m in _money_rx.finditer(text):
        num = m.group("num"); unit = m.group("unit")
        yen = _yen_from_text(num, unit)
        # 近傍に「予算」「事業費」等があると加点
        start = max(0, m.start()-20); end = min(len(text), m.end()+20)
        ctx = text[start:end]
        weight = 1.0 + (1.0 if any(k in ctx for k in ["予算","事業費","総事業費","補助"]) else 0.0)
        out.append((yen * weight, f"{num}{unit}"))
    # 降順
    out.sort(key=lambda x: x[0], reverse=True)
    return out

BUDGET_ALLOC_TMPL = """以下の政策ロジックツリー(activities/outputs/outcomes)において、各「活動(activity)」へ推定予算(円)を割り当て、さらに
段階別(活動→アウトプット→アウトカム短/中/長期)へ比率配分を提案してください。JSONのみ。スキーマ:
{{
  "per_activity_budget": [{{"activity_id":"A1","activity_label":"","estimated_yen":0}}],
  "allocation_profile": {{"activity":0.6,"output":0.25,"outcome_short":0.1,"outcome_mid":0.04,"outcome_long":0.01}}
}}

候補予算(単位:円, 上から有力順・重複含む):
{budget_list}

ロジックツリー:
{logic_tree}
"""

class BudgetAgent(BaseAgent):
    def search_rs_budgets(self, policy_terms: List[str]) -> List[Dict[str, Any]]:
        pages = []
        for term in policy_terms:
            hits = _ddg_rs_search(term, max_results=6)
            for h in hits:
                pages.append(_fetch_text(h["url"]))
                time.sleep(0.5)
        return pages

    def estimate_budget_for_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        # 1) 政策名でRS検索
        policy_terms = [p.get("name","") for p in strategy.get("policies", []) if p.get("name")]
        pages = self.search_rs_budgets(policy_terms)

        # 2) テキストから金額っぽい箇所を拾う
        candidates = []
        for p in pages:
            for yen, raw in _extract_budgets(p.get("text","")):
                candidates.append({"source_url": p["url"], "source_title": p["title"], "yen": float(yen), "raw": raw})
        # 上位のみ
        candidates = sorted(candidates, key=lambda x: x["yen"], reverse=True)[:30]

        # 3) LLMで活動ごとの割当と段階別配分を決定
        logic_tree = strategy.get("logic_tree", {})
        j_tree = json.dumps(logic_tree, ensure_ascii=False)
        budget_list = "\n".join([f"- {c['yen']:.0f} ({c['raw']}) {c['source_url']}" for c in candidates]) or "- 0"
        out = self.chat("JSONのみ返す。", BUDGET_ALLOC_TMPL.format(budget_list=budget_list, logic_tree=j_tree), temperature=0.0, max_tokens=1600)
        try:
            alloc = json.loads(out)
        except Exception:
            # Fallback: 単一ポリシーに最大値をアサイン
            total = candidates[0]["yen"] if candidates else 0.0
            acts = logic_tree.get("activities", [])
            per = []
            if acts:
                share = total / len(acts) if total > 0 else 0.0
                for a in acts:
                    per.append({"activity_id": a.get("id",""), "activity_label": a.get("label",""), "estimated_yen": share})
            alloc = {"per_activity_budget": per, "allocation_profile": {"activity":0.6,"output":0.25,"outcome_short":0.1,"outcome_mid":0.04,"outcome_long":0.01}}

        return {"pages": pages, "candidates": candidates, "allocation": alloc}
