# ebpm_agents/agents/search_paper_agent.py
from __future__ import annotations
"""
Bilingual Paper Search Agent
- 日本語 / 英語の両言語で学術的な情報源を優先して検索
- DDG (duckduckgo_search) のテキスト検索を用い、サイト制約でノイズを削減
- エビデンス強度（総説/メタ解析 > RCT > 準実験 > 観察 > 解説/ガイドライン）をヒューリスティクでスコア
- 日本語/英語のクエリを自動生成して両方実行 → マージ・重複排除・再ランク
- 出力は orchestrator 互換: {"query": str, "results": [{"rank", "title", "url", "evidence", "note"}]}

必要依存:
  pip install duckduckgo_search beautifulsoup4 html5lib
"""
from typing import Any, Dict, List, Tuple, Optional
import re, time
from urllib.parse import urlparse, urlunparse, urlsplit, urlunsplit, quote_plus

import requests
from bs4 import BeautifulSoup
from .base import BaseAgent

# --------- 設定（必要に応じて調整可） ---------
# 日本語サイドで優先したいドメイン
JP_DOMAINS = [
    "go.jp", "ac.jp", "jstage.jst.go.jp", "ci.nii.ac.jp", "ndl.go.jp",
    "mhlw.go.jp", "e-stat.go.jp", "stat.go.jp", "cao.go.jp", "mext.go.jp",
]
# 英語サイドで優先したいドメイン
EN_DOMAINS = [
    "nih.gov", "ncbi.nlm.nih.gov", "who.int", "oecd.org", "worldbank.org",
    "gov.uk", "europa.eu", "cdc.gov", "education", "harvard.edu", "stanford.edu",
]

# DDG 1クエリあたりの取得件数
PER_QUERY_RESULTS = 3
# 日本語・英語クエリそれぞれの生成上限（短縮）
MAX_QUERIES_EACH_LANG = 3
MAX_QUERIES_MULTI_EXTRA = 1
MAX_UNIQUE_HITS = 16
MAX_TOTAL_SECONDS = 3.0
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# 文字種で日本語を判定（簡易）
_rx_jp = re.compile(r"[ぁ-んァ-ン一-龥]")

def _normalize_url(u: str) -> str:
    """フラグメント/クエリのノイズを落として正規化（重複排除用）"""
    try:
        us = urlsplit(u)
        # fragment やトラッキングクエリを除去（必要なら拡張）
        clean = (us.scheme, us.netloc.lower(), us.path, "", "")
        return urlunsplit(clean)
    except Exception:
        return (u or "").strip()

def _lang_of_title_snippet(title: str, snippet: str) -> str:
    text = f"{title} {snippet}"
    return "ja" if _rx_jp.search(text) else "en"

def _is_preferred_domain(u: str) -> Tuple[bool, int]:
    """学術・公的を優先（ドメイン重みを返す）"""
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        host = ""
    score = 0
    pref = False
    for d in JP_DOMAINS:
        if d in host:
            score += 3
            pref = True
    for d in EN_DOMAINS:
        if d in host:
            score += 3
            pref = True
    # edu は広めに +1（上で特定大学に+3）
    if host.endswith(".edu"):
        score += 1
        pref = True
    return pref, score

def _evidence_tag_and_score(title: str, snippet: str) -> Tuple[str, int]:
    """
    簡易ヒューリスティックでエビデンス種別とスコアを付与。
    上位ほどスコア高 → ランクに効かせる。
    """
    text = f"{title} {snippet}".lower()

    # 英語
    if any(k in text for k in ["systematic review", "systematic literature review", "umbrella review", "meta-analysis", "meta analysis"]):
        return "systematic_review/meta_analysis", 6
    if any(k in text for k in ["randomized controlled trial", "randomised controlled trial", "rct"]):
        return "randomized_controlled_trial", 5
    if any(k in text for k in ["quasi-experimental", "difference-in-differences", "regression discontinuity", "instrumental variable"]):
        return "quasi_experimental", 4
    if any(k in text for k in ["cohort", "case-control", "cross-sectional", "observational study"]):
        return "observational", 3
    if any(k in text for k in ["guideline", "recommendation", "position paper"]):
        return "guideline", 3
    if any(k in text for k in ["preprint", "working paper"]):
        return "working/preprint", 1

    # 日本語
    jtext = f"{title} {snippet}"
    if any(k in jtext for k in ["システマティックレビュー", "系統的レビュー", "メタ解析"]):
        return "systematic_review/meta_analysis", 6
    if any(k in jtext for k in ["無作為化比較試験", "ランダム化比較試験", "RCT", "ランダム化"]):
        return "randomized_controlled_trial", 5
    if any(k in jtext for k in ["準実験", "回帰不連続", "操作変数", "差の差"]):
        return "quasi_experimental", 4
    if any(k in jtext for k in ["観察研究", "コホート", "症例対照", "横断研究"]):
        return "observational", 3
    if any(k in jtext for k in ["ガイドライン", "推奨", "提言"]):
        return "guideline", 3
    if any(k in jtext for k in ["プレプリント", "ワーキングペーパー"]):
        return "working/preprint", 1

    return "other", 2

def _build_queries_ja(user_query: str) -> List[str]:
    kws = [
        "システマティックレビュー", "系統的レビュー", "メタ解析",
        "無作為化比較試験", "ランダム化比較試験",
        "準実験", "観察研究", "ガイドライン",
    ]
    sites = " OR ".join([f"site:{d}" for d in JP_DOMAINS])
    queries = []
    for kw in kws:
        q = f'{user_query} {kw} {sites}'
        queries.append(q)
        if len(queries) >= MAX_QUERIES_EACH_LANG:
            break
    return queries

def _build_queries_en(user_query: str) -> List[str]:
    kws = [
        "systematic review", "meta-analysis",
        "randomized controlled trial",
        "quasi-experimental", "observational study",
        "guideline",
    ]
    sites = " OR ".join([f"site:{d}" for d in EN_DOMAINS] + ["site:edu"])
    queries = []
    for kw in kws:
        q = f'{user_query} {kw} {sites}'
        queries.append(q)
        if len(queries) >= MAX_QUERIES_EACH_LANG:
            break
    return queries

def _build_queries_multi(user_query: str) -> List[str]:
    """日本語/英語以外の代表的な言語でケーススタディ/社会学系の語を混ぜたクエリを少数生成。"""
    terms = {
        "fr": ["étude de cas", "sociologie"],
        "de": ["Fallstudie", "Soziologie"],
        "es": ["estudio de caso", "sociología"],
        "zh": ["案例研究", "社会学"],
        "ko": ["사례 연구", "사회학"],
    }
    site_terms = ["go.jp", "ac.jp", "edu", "org", "who.int", "oecd.org", "worldbank.org"]
    sites = " OR ".join([f"site:{d}" for d in site_terms])
    out: List[str] = []
    for lang, kws in terms.items():
        for kw in kws[:MAX_QUERIES_MULTI_EXTRA]:
            out.append(f"{user_query} {kw} {sites}")
    return out

def _ddg_text(q: str, n: int = PER_QUERY_RESULTS, timeout: int = 2) -> List[Dict[str, str]]:
    """DuckDuckGo HTML (軽量版) でタイトル/URLを取得。"""
    url = "https://duckduckgo.com/html/?q=" + quote_plus(q)
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html5lib")
        out: List[Dict[str, str]] = []
        for a in soup.select("a.result__a"):
            href = a.get("href", "")
            title = a.get_text(" ", strip=True)
            if not href:
                continue
            out.append({"title": title, "url": href, "snippet": ""})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []

def _collect_bilingual_hits(user_query: str, deadline: float) -> List[Dict[str, str]]:
    all_hits: List[Dict[str, str]] = []
    # 日本語クエリ
    for q in _build_queries_ja(user_query):
        if time.monotonic() > deadline:
            break
        all_hits.extend(_ddg_text(q, n=PER_QUERY_RESULTS))
        if len(all_hits) >= MAX_UNIQUE_HITS:
            break
    # 英語クエリ
    for q in _build_queries_en(user_query):
        if time.monotonic() > deadline or len(all_hits) >= MAX_UNIQUE_HITS:
            break
        all_hits.extend(_ddg_text(q, n=PER_QUERY_RESULTS))
    # 多言語（少数）
    for q in _build_queries_multi(user_query):
        if time.monotonic() > deadline or len(all_hits) >= MAX_UNIQUE_HITS:
            break
        all_hits.extend(_ddg_text(q, n=PER_QUERY_RESULTS))
    # 正規化 & 重複除去
    uniq, seen = [], set()
    for h in all_hits:
        nu = _normalize_url(h.get("url",""))
        if not nu or nu in seen:
            continue
        seen.add(nu)
        h["url"] = nu
        uniq.append(h)
    return uniq

def _score_item(it: Dict[str, str]) -> Tuple[int, int, int]:
    """(evidence_score, domain_score, lang_bonus) を返す。ソートは降順。"""
    ev_tag, ev_score = _evidence_tag_and_score(it.get("title",""), it.get("snippet",""))
    it["evidence"] = ev_tag
    pref, dom_score = _is_preferred_domain(it.get("url",""))
    # 日本語/英語どちらも許容。質問言語に合わせたバイアスを掛けない（混在で拾う）
    lang = _lang_of_title_snippet(it.get("title",""), it.get("snippet",""))
    it["lang"] = lang
    # わずかに日本語ソースに +1（国内政策利用を想定）。必要なければ 0 に。
    lang_bonus = 1 if lang == "ja" else 0
    return ev_score, dom_score, lang_bonus

class PaperSearchAgent:
    """
    日本語＋英語の両言語で、政策決定に有用な学術情報を優先的に検索・ランク。
    run() の戻り値は orchestrator 既定の構造に合わせる。
    """
    def __init__(self):
        pass

    def run(self, query: str, *, max_results: int = 16, max_seconds: Optional[float] = None) -> Dict[str, Any]:
        budget = max_seconds if max_seconds is not None else MAX_TOTAL_SECONDS
        budget = float(max(1.5, budget))
        deadline = time.monotonic() + budget
        hits = _collect_bilingual_hits(query, deadline)

        # スコア付与
        for it in hits:
            ev, dom, bonus = _score_item(it)
            it["_score_tuple"] = (ev, dom, bonus)

        # ソート（エビデンス > ドメイン > 言語ボーナス）
        hits.sort(key=lambda x: x["_score_tuple"], reverse=True)

        # 整形：rank付与 + メモ
        results: List[Dict[str, Any]] = []
        for i, it in enumerate(hits[:max_results], start=1):
            note_parts = []
            if it.get("lang") == "ja": note_parts.append("ja")
            if it.get("lang") == "en": note_parts.append("en")
            # ドメイン優先かどうか（便宜上表示）
            pref, _ = _is_preferred_domain(it["url"])
            if pref: note_parts.append("preferred-domain")
            note = ",".join(note_parts) if note_parts else ""
            results.append({
                "rank": i,
                "title": it.get("title",""),
                "url": it.get("url",""),
                "evidence": it.get("evidence","other"),
                "note": note
            })

        return {"query": query, "results": results}
