# ebpm_agents/agents/search_policy_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import io
import re
import time
from copy import deepcopy
from urllib.parse import urlparse, quote_plus
import requests
from bs4 import BeautifulSoup
# LLMを使わず高速にヒューリスティクでスコアリングする

RS_BASE = "https://rssystem.go.jp"
RS_PROJECT = f"{RS_BASE}/project"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

_CACHE_TTL_RS = 900.0
_CACHE_TTL_DDG_SITE = 600.0
_CACHE_TTL_DDG_HTML = 300.0
_rs_index_cache: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, str]]]] = {}
_ddg_site_cache: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, str]]]] = {}
_ddg_html_cache: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, str]]]] = {}
PDF_SNIPPET_LIMIT = 3
PDF_SNIPPET_MAX_BYTES = 250_000
PDF_SNIPPET_MAX_CHARS = 400
PDF_SNIPPET_TIMEOUT = 2
ENABLE_PDF_SNIPPET = False
_MIN_HTTP_TIMEOUT = 0.5


def _cache_get(store: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, str]]]], key: Tuple[Any, ...], ttl: float) -> Optional[List[Dict[str, str]]]:
    entry = store.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.monotonic() - ts > ttl:
        store.pop(key, None)
        return None
    return deepcopy(value)


def _cache_set(store: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, str]]]], key: Tuple[Any, ...], value: List[Dict[str, str]]) -> None:
    store[key] = (time.monotonic(), deepcopy(value))


def _remaining_time(deadline: Optional[float], default: float) -> float:
    if deadline is None:
        return default
    remain = deadline - time.monotonic()
    if remain <= 0:
        return 0.0
    return max(_MIN_HTTP_TIMEOUT, min(default, remain))


def _looks_like_pdf(url: str) -> bool:
    return bool(re.search(r"\.pdf($|[?#])", (url or ""), re.IGNORECASE))


def _fetch_pdf_snippet(url: str) -> str:
    try:
        r = requests.get(url, timeout=PDF_SNIPPET_TIMEOUT, headers={"User-Agent": UA})
        r.raise_for_status()
        data = r.content
        if len(data) > PDF_SNIPPET_MAX_BYTES:
            data = data[:PDF_SNIPPET_MAX_BYTES]
        try:
            from pdfminer.high_level import extract_text
        except Exception:
            return "PDFリンク (プレビュー不可)"
        bio = io.BytesIO(data)
        text = extract_text(bio)
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[:PDF_SNIPPET_MAX_CHARS] or "PDFリンク"
    except Exception:
        return ""


def _maybe_enrich_pdf_snippets(items: List[Dict[str, Any]], limit: int = PDF_SNIPPET_LIMIT) -> None:
    fetched = 0
    for it in items:
        if fetched >= limit:
            break
        url = it.get("url", "")
        if not url or not _looks_like_pdf(url):
            continue
        if it.get("snippet"):
            continue
        snippet = _fetch_pdf_snippet(url)
        if snippet:
            it["snippet"] = snippet
        fetched += 1

def _is_rs_project_url(url: str) -> bool:
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        return "rssystem.go.jp" in host and "/project" in (u.path or "/")
    except Exception:
        return False

def _fetch(url: str, timeout: int = 2) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _rs_index_candidates(max_pages: int = 1) -> List[Dict[str, str]]:
    """
    RSシステムの /project インデックスを直接パースして候補URLを収集。
    具体的な一覧/検索UIの構造に依存するため、汎用に anchor 抽出で実装。
    """
    cache_key = ("rs-index", max_pages)
    cached = _cache_get(_rs_index_cache, cache_key, _CACHE_TTL_RS)
    if cached is not None:
        return cached
    html = _fetch(RS_PROJECT)
    items: List[Dict[str, str]] = []
    if not html:
        return items
    try:
        soup = BeautifulSoup(html, "html5lib")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            title = a.get_text(" ", strip=True)
            if not href:
                continue
            if href.startswith("/"):
                url = RS_BASE + href
            elif href.startswith("http"):
                url = href
            else:
                url = RS_BASE + "/" + href.lstrip("./")
            if _is_rs_project_url(url):
                items.append({"title": title[:300], "url": url, "snippet": ""})
    except Exception:
        pass
    # 重複除去
    seen, uniq = set(), []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"]); uniq.append(it)
    _cache_set(_rs_index_cache, cache_key, uniq)
    return uniq

def _ddg_site_search(query: str, force_keyword: str = "政策", max_results: int = 12, timeout: float = 2.0) -> List[Dict[str, str]]:
    """
    DuckDuckGoのHTML簡易検索で、site:rssystem.go.jp と /project を強く縛る。
    """
    q = f'site:rssystem.go.jp "{force_keyword}" "{query}" project'
    url = "https://duckduckgo.com/html/?q=" + quote_plus(q)
    cache_key = (q, max_results, timeout)
    cached = _cache_get(_ddg_site_cache, cache_key, _CACHE_TTL_DDG_SITE)
    if cached is not None:
        return cached
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
            if _is_rs_project_url(href):
                out.append({"title": title[:300], "url": href, "snippet": ""})
            if len(out) >= max_results:
                break
        # 重複除去
        seen, uniq = set(), []
        for it in out:
            if it["url"] in seen:
                continue
            seen.add(it["url"]); uniq.append(it)
        _cache_set(_ddg_site_cache, cache_key, uniq)
        return uniq
    except Exception:
        return []

def _score_policy_like(title: str, snippet: str, page_text: str) -> float:
    """
    タイトル/本文に「政策」「事業」「施策」「レビュー」「評価」などがあるほどスコアを高く。
    """
    keys = ["政策", "施策", "事業", "レビュー", "評価", "効果", "KPI"]
    s = 0.0
    text = f"{title}\n{snippet}\n{page_text or ''}"
    for k in keys:
        if k in text:
            s += 1.0
    # タイトルに「政策」があると加点
    if "政策" in (title or ""):
        s += 2.0
    return s

CORE_GOV_SITES_JA = [
    "go.jp", "cao.go.jp", "mhlw.go.jp", "mext.go.jp", "soumu.go.jp", "meti.go.jp", "stat.go.jp", "e-stat.go.jp"
]

EXTENDED_SITE_FILTERS = CORE_GOV_SITES_JA + [
    "lg.jp", "metro.tokyo.jp", "pref.osaka.lg.jp", "pref.aichi.lg.jp", "pref.hokkaido.lg.jp",
    "pref.fukuoka.lg.jp", "city.yokohama.lg.jp", "city.nagoya.lg.jp"
]

LOOSE_SITE_FILTERS = [
    "go.jp", "lg.jp", "ac.jp", "or.jp", "gr.jp", "co.jp"
]

def _ddg_html_search(query: str, site_filters: List[str], max_results: int = 3, timeout: float = 3.0) -> List[Dict[str, str]]:
    """DuckDuckGo HTMLを使った軽量検索（サイトフィルタを OR で結合）。"""
    filters_key = tuple(site_filters or [])
    cache_key = (query, filters_key, max_results, timeout)
    cached = _cache_get(_ddg_html_cache, cache_key, _CACHE_TTL_DDG_HTML)
    if cached is not None:
        return cached
    site_q = " OR ".join([f"site:{s}" for s in filters_key]) if filters_key else ""
    q = f'"{query}" {site_q}'.strip()
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
            out.append({"title": title[:300], "url": href, "snippet": ""})
            if len(out) >= max_results:
                break
        # 重複除去
        uniq, seen = [], set()
        for it in out:
            u = it["url"]
            if u in seen:
                continue
            seen.add(u); uniq.append(it)
        _cache_set(_ddg_html_cache, cache_key, uniq)
        return uniq
    except Exception:
        return []

def _build_policy_queries(user_query: str) -> List[Dict[str, str]]:
    """政策階層ごとに日本語/英語のクエリを生成。"""
    queries: List[Dict[str, str]] = []
    # マクロ（骨太・基本方針・ガイドライン）
    macro_ja = [
        f"{user_query} 骨太の方針", f"{user_query} 基本方針", f"{user_query} ガイドライン", f"{user_query} 行動計画"
    ]
    macro_en = [
        f"{user_query} basic policy site:go.jp", f"{user_query} guideline site:go.jp", f"{user_query} action plan site:go.jp"
    ]
    # 近縁（関連政策）
    related_ja = [
        f"{user_query} 政策 事例", f"{user_query} 施策 レビュー", f"{user_query} 施策 評価 PDF"
    ]
    # 過去政策・評価
    hist_ja = [
        f"{user_query} 過去 施策 評価", f"{user_query} 事業 評価 結果", f"{user_query} 検証 報告"
    ]
    for q in macro_ja[:1]: queries.append({"layer":"macro","q": q})
    for q in macro_en[:1]: queries.append({"layer":"macro","q": q})
    for q in related_ja[:1]: queries.append({"layer":"related","q": q})
    for q in hist_ja[:1]: queries.append({"layer":"historical","q": q})
    return queries


def _build_loosened_queries(user_query: str) -> List[Dict[str, Any]]:
    base = [
        {"layer": "pdf", "q": f"{user_query} 政策 PDF 評価", "filters": ["go.jp"], "max_results": 6},
        {"layer": "pdf", "q": f"{user_query} filetype:pdf policy evaluation", "filters": ["go.jp", "lg.jp"], "max_results": 6},
        {"layer": "loose", "q": f"{user_query} 政策 事例 レビュー", "filters": EXTENDED_SITE_FILTERS, "max_results": 6},
        {"layer": "loose", "q": f"{user_query} 施策 効果 報告", "filters": LOOSE_SITE_FILTERS, "max_results": 6},
        {"layer": "global", "q": f"{user_query} policy evaluation report", "filters": [], "max_results": 6},
    ]
    return base


def _layer_weight(layer: str) -> int:
    return {"macro": 3, "related": 2, "historical": 1, "rs-system": 4, "pdf": 2, "loose": 1}.get(layer, 0)


def _loosened_policy_hits(user_query: str, need: int, deadline: Optional[float] = None) -> List[Dict[str, str]]:
    if need <= 0:
        return []
    hits: List[Dict[str, str]] = []
    for obj in _build_loosened_queries(user_query):
        if deadline is not None and time.monotonic() >= deadline:
            break
        q = obj.get("q", "")
        filters = obj.get("filters") or []
        max_each = obj.get("max_results", 5)
        take = min(max_each, max(3, need * 2))
        timeout = _remaining_time(deadline, 5.0)
        if timeout <= 0:
            break
        loose_hits = _ddg_html_search(q, filters, max_results=take, timeout=timeout)
        for h in loose_hits:
            h["layer"] = obj.get("layer", "loose")
            hits.append(h)
            if len(hits) >= need:
                return hits
    return hits

class PolicySearchAgent:
    """
    ポリシー検索を RSシステム由来に固定。
    - まず /project インデックスをパースし anchor を候補化
    - 併せて DDG で site:rssystem.go.jp を限定検索（force_keyword=政策）
    - 収集候補の本文を軽くフェッチ→ポリシーらしさをスコア→rank付け
    """
    def run(
        self,
        query: str,
        *,
        prefer_rs_system: bool = True,
        force_keyword: str = "政策",
        max_results: int = 12,
        max_runtime: Optional[float] = None,
    ) -> Dict[str, Any]:
        target_results = max(10, max_results)
        max_candidates = max(30, target_results * 2)
        deadline = time.monotonic() + max_runtime if max_runtime else None

        def _timed_out() -> bool:
            return bool(deadline and time.monotonic() >= deadline)

        def _site_hits() -> List[Dict[str, str]]:
            timeout = _remaining_time(deadline, 6.0)
            if timeout <= 0:
                return []
            return _ddg_site_search(query, force_keyword=force_keyword, max_results=max_results, timeout=timeout)

        # 1) RSインデックス候補 + RS限定DDG
        rs_items = _rs_index_candidates()
        rs_ddg = _site_hits()

        # 2) 政策階層クエリ（日本語/英語）を go.jp 系で検索
        layer_queries = _build_policy_queries(query)
        gov_hits: List[Dict[str, str]] = []
        for obj in layer_queries:
            if _timed_out():
                break
            layer = obj["layer"]
            q = obj["q"]
            timeout = _remaining_time(deadline, 3.0)
            if timeout <= 0:
                break
            hits = _ddg_html_search(q, CORE_GOV_SITES_JA, max_results=3, timeout=timeout)
            for h in hits:
                h["layer"] = layer
                gov_hits.append(h)

        # 3) マージ + ヒューリスティクスコアリング（高速化のため本文フェッチは行わない）
        merged: List[Dict[str, Any]] = []
        seen = set()
        def _append_hits(items: List[Dict[str, Any]]) -> None:
            for it in items:
                if _timed_out():
                    return
                if len(merged) >= max_candidates:
                    return
                u = it.get("url", "")
                if not u or u in seen:
                    continue
                seen.add(u)
                merged.append(it)

        _append_hits(rs_items)
        _append_hits(rs_ddg)
        _append_hits(gov_hits)

        if len(merged) < target_results and not _timed_out():
            loose_hits = _loosened_policy_hits(query, target_results - len(merged), deadline)
            _append_hits(loose_hits)

        if len(merged) < target_results and not _timed_out():
            timeout = _remaining_time(deadline, 4.0)
            if timeout > 0:
                general_hits = _ddg_html_search(
                    query,
                    [],
                    max_results=max(5, (target_results - len(merged)) * 2),
                    timeout=timeout,
                )
                for h in general_hits:
                    h["layer"] = "loose"
                _append_hits(general_hits)

        if ENABLE_PDF_SNIPPET:
            _maybe_enrich_pdf_snippets(merged)

        scored: List[Dict[str, Any]] = []
        for it in merged:
            title = it.get("title", "")
            snippet = it.get("snippet", "")
            page_text = ""  # 本文フェッチ省略で高速化
            score = _score_policy_like(title, snippet, page_text)
            layer = "rs-system" if _is_rs_project_url(it.get("url", "")) else it.get("layer", "related")
            host = urlparse(it.get("url", "")).netloc.lower()
            gov_hosts = CORE_GOV_SITES_JA + ["lg.jp"]
            gov_boost = 1.5 if any(s in host for s in gov_hosts) else 1.0
            total = score * gov_boost + _layer_weight(layer)
            reason_note = f"score={round(float(total),3)} layer={layer}"
            if _looks_like_pdf(it.get("url", "")):
                reason_note += " pdf"
            scored.append({
                "title": title,
                "url": it.get("url", ""),
                "snippet": snippet,
                "source": "rs-system" if layer == "rs-system" else "gov",
                "layer": layer,
                "policy_likeliness": round(float(total), 3),
                "_reason": reason_note
            })

        # 4) ソート: RS最優先 → マクロ → 関連 → 過去→PDF/loose
        scored.sort(key=lambda x: (
            x["source"] != "rs-system",
            -_layer_weight(x["layer"]),
            -x["policy_likeliness"],
            len(x["title"] or "")
        ))

        # 5) rank付与して返却
        results = []
        for i, r in enumerate(scored[:target_results], start=1):
            results.append({
                "rank": i,
                "title": r["title"],
                "url": r["url"],
                "layer": r["layer"],
                "source": r["source"],
                "reason": r.get("_reason") or f"score={r['policy_likeliness']} layer={r['layer']}"
            })
        return {"query": query, "results": results}
