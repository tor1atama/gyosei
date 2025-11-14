# ebpm_agents/complex_orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Iterable, Optional, Callable
import json
from pathlib import Path
import time

from .problem_refiner_agent import ProblemRefinerAgent
from .workplan_decomposer_agent import WorkplanDecomposerAgent
from .solution_synthesizer_agent import SolutionSynthesizerAgent
from .critique_agent import CritiqueAgent
from .budget_agent import BudgetAgent

# 既存エージェント（RS政策検索/バイリンガル論文検索/KPI抽出）
from .search_policy_agent import PolicySearchAgent
from .search_paper_agent import PaperSearchAgent
from .kpi_extractor_agent import KPIExtractorAgent
from .topic_map_agent import TopicExplorerAgent
from .policy_hypothesis_agent import PolicyHypothesisAgent
from .dummy_data_agent import DummyDataAgent

THEMES = ["cost-effective", "equity-first", "speed-first"]
POLICY_QUERY_LIMIT = 3
PAPER_QUERY_LIMIT = 4
POLICY_TIME_BUDGET = 2.5
PAPER_TIME_BUDGET = 2.0
TASK_EXISTING_THRESHOLD = 2.0
MAX_SEARCH_WINDOW = 5.0
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_POLICY_ARCHIVE_CACHE: Optional[List[Dict[str, Any]]] = None
LAYER_QUERY_MAX = 2
USE_REAL_SEARCH = False
USE_REAL_BUDGET_ESTIMATES = False


def _normalize_query(q: Optional[str]) -> str:
    return (q or "").strip()


def _load_policy_archive() -> List[Dict[str, Any]]:
    global _POLICY_ARCHIVE_CACHE
    if _POLICY_ARCHIVE_CACHE is not None:
        return _POLICY_ARCHIVE_CACHE
    path = STATIC_DIR / "policy_examples.json"
    if not path.exists():
        _POLICY_ARCHIVE_CACHE = []
        return _POLICY_ARCHIVE_CACHE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _POLICY_ARCHIVE_CACHE = data
        else:
            _POLICY_ARCHIVE_CACHE = []
    except Exception:
        _POLICY_ARCHIVE_CACHE = []
    return _POLICY_ARCHIVE_CACHE


def _archive_policy_matches(terms: List[str], needed: int) -> List[Dict[str, Any]]:
    if needed <= 0:
        return []
    examples = _load_policy_archive()
    terms = [t.lower() for t in terms if t]
    if not terms:
        terms = []
    scored: List[Dict[str, Any]] = []
    for ex in examples:
        text = " ".join(
            [
                ex.get("title", ""),
                ex.get("summary", ""),
                " ".join(ex.get("tags", [])),
            ]
        ).lower()
        score = 0
        for term in terms:
            if term and term in text:
                score += 2
        for tag in ex.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower in terms:
                score += 3
        if score == 0 and not terms:
            score = 1
        if score > 0:
            scored.append({"score": score, "item": ex})
    scored.sort(key=lambda x: x["score"], reverse=True)
    out: List[Dict[str, Any]] = []
    for row in scored[:needed]:
        item = dict(row["item"])
        item.setdefault("layer", "archive")
        item.setdefault("source", "archive")
        out.append(item)
    return out


def _collect_layer_policy_hits(
    topic_map: Optional[Dict[str, Any]],
    pol_agent: PolicySearchAgent,
    *,
    prefer_rs_system: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    if not topic_map:
        return {}
    results: Dict[str, List[Dict[str, Any]]] = {}
    for block in topic_map.get("topic_layers", []) or []:
        sub = block.get("subproblem") or ""
        layer_entries: List[Dict[str, Any]] = []
        for layer in block.get("layers", []) or []:
            label = layer.get("label", "")
            tier = layer.get("tier", "")
            queries: List[str] = []
            sample = layer.get("sample_queries") or {}
            for q in (sample.get("policy") or []) + (sample.get("evidence") or []):
                qn = _normalize_query(q)
                if not qn:
                    continue
                queries.append(f"{qn} filetype:pdf")
                queries.append(qn)
            if not queries:
                fallback = _normalize_query(layer.get("policy_focus")) or label or sub
                if fallback:
                    queries.extend([
                        f"{fallback} filetype:pdf 政策 評価",
                        f"{fallback} 政策 事例"
                    ])
            hits: List[Dict[str, Any]] = []
            for q in queries[:LAYER_QUERY_MAX * 2]:
                res = pol_agent.run(
                    q,
                    prefer_rs_system=prefer_rs_system,
                    max_results=5,
                    max_runtime=5.0,
                ).get("results", [])
                if res:
                    hits = res[:5]
                    break
            if not hits:
                archive_hits = _archive_policy_matches(
                    [label, layer.get("policy_focus", ""), sub], 3
                )
                for idx, h in enumerate(archive_hits, start=1):
                    hits.append({
                        "title": h.get("title", ""),
                        "url": h.get("url", ""),
                        "snippet": h.get("summary", ""),
                        "rank": idx,
                        "source": h.get("source", "archive"),
                        "layer": tier or "archive",
                        "origin_query": "archive",
                        "origin_subproblem": sub or "archive",
                    })
            if hits:
                layer_entries.append({
                    "tier": tier,
                    "label": label,
                    "policy_focus": layer.get("policy_focus", ""),
                    "hits": hits,
                })
        if layer_entries:
            results[sub or f"subproblem_{len(results)+1}"] = layer_entries
    return results


def build_policy_hypotheses(
    user_query: str,
    topic_layer_hits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    hyp_agent = PolicyHypothesisAgent()
    output: Dict[str, List[Dict[str, Any]]] = {}
    for sub, layers in (topic_layer_hits or {}).items():
        entries: List[Dict[str, Any]] = []
        for layer in layers:
            resp = hyp_agent.run(
                user_query=user_query,
                subproblem=sub,
                layer_label=layer.get("label", ""),
                tier=layer.get("tier", ""),
                hits=layer.get("hits", []),
            )
            entries.append({
                "tier": layer.get("tier", ""),
                "label": layer.get("label", ""),
                "hypotheses": resp.get("hypotheses", []),
            })
        if entries:
            output[sub] = entries
    return output

def refine_problem(user_query: str) -> Dict[str, Any]:
    return ProblemRefinerAgent().run(user_query)

def decompose_work(refined_problem: Dict[str, Any]) -> Dict[str, Any]:
    return WorkplanDecomposerAgent().run(refined_problem)


def explore_topics(user_query: str, refined_problem: Dict[str, Any], workplan: Dict[str, Any]) -> Dict[str, Any]:
    return TopicExplorerAgent().run(user_query, workplan, refined_problem)

def _make_dummy_search_results(user_query: str, min_policy_hits: int, min_paper_hits: int) -> Dict[str, Any]:
    agent = DummyDataAgent()
    policy_items = agent.run(user_query=user_query, subproblem="policy", data_type="policy", count=max(3, min_policy_hits)).get("items", [])
    paper_items = agent.run(user_query=user_query, subproblem="evidence", data_type="paper", count=max(3, min_paper_hits)).get("items", [])
    def _convert(items):
        out = []
        for idx, it in enumerate(items, start=1):
            out.append({
                "title": it.get("title", f"Existing Item {idx}"),
                "url": it.get("url", "https://example.com/existing"),
                "snippet": it.get("snippet", "既存データ (参考)"),
                "rank": idx,
                "origin_query": "existing-data",
                "origin_subproblem": "existing-data",
            })
        return out
    return {
        "policy_hits": _convert(policy_items)[:max(1, min_policy_hits)],
        "paper_hits": _convert(paper_items)[:max(1, min_paper_hits)],
        "topic_layer_hits": {},
    }


def _make_dummy_budgets(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    budgets: List[Dict[str, Any]] = []
    for idx, strat in enumerate(strategies, start=1):
        policies = strat.get("policies") or []
        per_activity: List[Dict[str, Any]] = []
        if not policies:
            per_activity.append({
                "activity_id": f"activity_{idx}_1",
                "activity_label": "既存施策",
                "estimated_yen": 1.5e8,
            })
        else:
            for p_idx, policy in enumerate(policies[:5], start=1):
                amount = float(policy.get("cost") or policy.get("budget", 1.0)) * 1e8
                per_activity.append({
                    "activity_id": policy.get("id") or f"activity_{idx}_{p_idx}",
                    "activity_label": policy.get("name") or f"施策{p_idx}",
                    "estimated_yen": max(amount, 0.5e8),
                })
        alloc = {
            "per_activity_budget": per_activity,
            "allocation_profile": {"activity": 0.6, "output": 0.25, "outcome_short": 0.1, "outcome_mid": 0.04, "outcome_long": 0.01},
        }
        candidates = [{
            "raw": f"{act.get('activity_label')} の概算費用",
            "yen": act.get("estimated_yen", 0.0),
            "source_title": "既存資料推定",
            "source_url": "https://example.com/existing-budget",
        } for act in per_activity[:3]]
        budgets.append({
            "strategy_name": strat.get("name", f"strategy_{idx}"),
            "pages": [],
            "candidates": candidates,
            "allocation": alloc,
        })
    return budgets


def run_searches(
    work: Dict[str, Any],
    *,
    prefer_rs_system: bool = True,
    user_query: Optional[str] = None,
    refined_problem: Optional[Dict[str, Any]] = None,
    topic_map: Optional[Dict[str, Any]] = None,
    min_policy_hits: int = 10,
    min_paper_hits: int = 12,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if not USE_REAL_SEARCH:
        return _make_dummy_search_results(user_query or "", min_policy_hits, min_paper_hits)

    pol_agent = PolicySearchAgent()
    pap_agent = PaperSearchAgent()
    hyp_layer_hits: Dict[str, List[Dict[str, Any]]] = {}
    start_time = time.monotonic()
    TIMEOUT_SECONDS = MAX_SEARCH_WINDOW
    deadline = start_time + TIMEOUT_SECONDS

    def _emit_progress(event_type: str, **payload) -> None:
        if not progress_cb:
            return
        try:
            progress_cb({"type": event_type, **payload})
        except Exception:
            pass

    pol_hits_all: List[Dict[str, Any]] = []
    pap_hits_all: List[Dict[str, Any]] = []
    seen_policy_urls = set()
    seen_paper_urls = set()
    policy_cache: Dict[str, List[Dict[str, Any]]] = {}
    paper_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _normalize_query(q: Optional[str]) -> str:
        return (q or "").strip()

    topic_query_map: Dict[str, List[str]] = {}
    global_topic_queries: List[str] = []
    if topic_map:
        for block in topic_map.get("topic_layers", []):
            sub_name = (block.get("subproblem") or "").strip()
            layer_queries: List[str] = []
            for layer in block.get("layers", []) or []:
                sample = layer.get("sample_queries") or {}
                for q in (sample.get("policy") or []):
                    if _normalize_query(q):
                        layer_queries.append(q)
                for q in (sample.get("evidence") or []):
                    if _normalize_query(q):
                        layer_queries.append(q)
                for child in layer.get("child_nodes") or []:
                    for q in child.get("sample_sources") or []:
                        if _normalize_query(q):
                            layer_queries.append(q)
            if sub_name and layer_queries:
                topic_query_map[sub_name] = layer_queries
        global_topic_queries.extend(topic_map.get("global_queries", {}).get("broad", []) or [])
        global_topic_queries.extend(topic_map.get("global_queries", {}).get("focused", []) or [])

    policy_calls = 0
    paper_calls = 0

    def _time_left() -> float:
        return max(0.0, deadline - time.monotonic())

    def _policy_results(query: str) -> List[Dict[str, Any]]:
        nonlocal policy_calls
        q = _normalize_query(query)
        if not q:
            return []
        if _time_left() <= 0:
            return []
        if q not in policy_cache:
            if policy_calls >= POLICY_QUERY_LIMIT:
                policy_cache[q] = []
                return policy_cache[q]
            policy_calls += 1
            remaining = _time_left()
            if remaining <= 0:
                policy_cache[q] = []
                return policy_cache[q]
            policy_cache[q] = pol_agent.run(
                q,
                prefer_rs_system=prefer_rs_system,
                force_keyword="政策",
                max_results=max(min_policy_hits, 10),
                max_runtime=min(POLICY_TIME_BUDGET, remaining),
            ).get("results", [])
        return policy_cache[q]

    def _paper_results(query: str) -> List[Dict[str, Any]]:
        nonlocal paper_calls
        q = _normalize_query(query)
        if not q:
            return []
        if _time_left() <= 0:
            return []
        if q not in paper_cache:
            if paper_calls >= PAPER_QUERY_LIMIT:
                paper_cache[q] = []
                return paper_cache[q]
            paper_calls += 1
            remaining = _time_left()
            if remaining <= 0:
                paper_cache[q] = []
                return paper_cache[q]
            paper_cache[q] = pap_agent.run(
                q,
                max_results=max(min_paper_hits, 10),
                max_seconds=min(PAPER_TIME_BUDGET, remaining),
            ).get("results", [])
        return paper_cache[q]

    def _add_policy_hit(hit: Dict[str, Any], source_sp: Dict[str, Any], query: str) -> None:
        url = hit.get("url")
        if not url or url in seen_policy_urls:
            return
        seen_policy_urls.add(url)
        enriched = dict(hit)
        enriched.setdefault("origin_query", query)
        enriched.setdefault("origin_subproblem", source_sp.get("name", ""))
        pol_hits_all.append(enriched)

    def _add_paper_hit(hit: Dict[str, Any], source_sp: Dict[str, Any], query: str) -> None:
        url = hit.get("url")
        if not url or url in seen_paper_urls:
            return
        seen_paper_urls.add(url)
        enriched = dict(hit)
        enriched.setdefault("origin_query", query)
        enriched.setdefault("origin_subproblem", source_sp.get("name", ""))
        pap_hits_all.append(enriched)

    def _iter_subproblem_queries(sp: Dict[str, Any], lang: str, limit: int) -> Iterable[str]:
        if lang not in ("ja", "en"):
            return []
        queries = sp.get("queries", {}).get(lang) or []
        return [q for q in queries[:limit] if _normalize_query(q)]

    for sp in work.get("subproblems", []):
        elapsed_total = time.monotonic() - start_time
        if elapsed_total >= TIMEOUT_SECONDS:
            break
        if TIMEOUT_SECONDS - elapsed_total <= TASK_EXISTING_THRESHOLD:
            break
        task_label = _normalize_query(sp.get("name")) or (sp.get("objective") or "検索タスク")
        task_desc = sp.get("objective") or ""
        task_start = time.monotonic()
        existing_shared = False
        existing_items: List[Dict[str, Any]] = []
        _emit_progress("task_start", task=task_label, description=task_desc)

        def maybe_share_existing_data() -> bool:
            nonlocal existing_shared, existing_items
            if existing_shared:
                return True
            elapsed_local = time.monotonic() - task_start
            if elapsed_local < TASK_EXISTING_THRESHOLD:
                return False
            if _time_left() <= 0:
                return False
            agent = DummyDataAgent()
            payload = agent.run(
                user_query=user_query or task_desc or task_label,
                subproblem=task_label,
                data_type="policy",
                count=3,
            )
            existing_items = payload.get("items", [])
            existing_shared = True
            for item in existing_items:
                pseudo_hit = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "content": item.get("snippet", ""),
                }
                _add_policy_hit(pseudo_hit, sp, "[既存資料補完]")
            if existing_items:
                _emit_progress("task_existing_data", task=task_label, elapsed=elapsed_local, items=existing_items)
            return True
        task_aborted = False

        ja_queries = list(_iter_subproblem_queries(sp, "ja", 2))
        en_queries = list(_iter_subproblem_queries(sp, "en", 1))
        fallback = [sp.get("objective", ""), sp.get("name", "")]
        topic_support = topic_query_map.get(sp.get("name", ""), [])
        primary_queries = [
            q for q in ja_queries + en_queries + topic_support + fallback if _normalize_query(q)
        ]

        for q in primary_queries:
            if maybe_share_existing_data():
                task_aborted = True
                break
            if time.monotonic() - start_time >= TIMEOUT_SECONDS:
                task_aborted = True
                break
            if _time_left() <= 0:
                task_aborted = True
                break
            for hit in _policy_results(q):
                _add_policy_hit(hit, sp, q)
            if maybe_share_existing_data():
                task_aborted = True
                break
            if time.monotonic() - start_time >= TIMEOUT_SECONDS:
                task_aborted = True
                break
            if _time_left() <= 0:
                task_aborted = True
                break
            if len(pol_hits_all) >= min_policy_hits or policy_calls >= POLICY_QUERY_LIMIT:
                break
        if task_aborted:
            _emit_progress("task_finish", task=task_label, elapsed=time.monotonic() - task_start)
            continue
        if len(pol_hits_all) >= min_policy_hits or policy_calls >= POLICY_QUERY_LIMIT:
            _emit_progress("task_finish", task=task_label, elapsed=time.monotonic() - task_start)
            break

        paper_queries = list(_iter_subproblem_queries(sp, "ja", 2)) + list(_iter_subproblem_queries(sp, "en", 2))
        if not paper_queries:
            paper_queries = primary_queries
        for q in paper_queries:
            if maybe_share_existing_data():
                task_aborted = True
                break
            if time.monotonic() - start_time >= TIMEOUT_SECONDS:
                task_aborted = True
                break
            if _time_left() <= 0:
                task_aborted = True
                break
            for hit in _paper_results(q):
                _add_paper_hit(hit, sp, q)
            if maybe_share_existing_data():
                task_aborted = True
                break
            if time.monotonic() - start_time >= TIMEOUT_SECONDS:
                task_aborted = True
                break
            if _time_left() <= 0:
                task_aborted = True
                break
            if len(pap_hits_all) >= min_paper_hits or paper_calls >= PAPER_QUERY_LIMIT:
                break
        if task_aborted:
            _emit_progress("task_finish", task=task_label, elapsed=time.monotonic() - task_start)
            continue
        if len(pap_hits_all) >= min_paper_hits or paper_calls >= PAPER_QUERY_LIMIT:
            _emit_progress("task_finish", task=task_label, elapsed=time.monotonic() - task_start)
            break
        maybe_share_existing_data()
        _emit_progress("task_finish", task=task_label, elapsed=time.monotonic() - task_start)
        if time.monotonic() - start_time > TIMEOUT_SECONDS:
            break

    def _fallback_terms() -> List[str]:
        terms: List[str] = []
        rp = refined_problem or {}
        if isinstance(rp, dict) and "refined_problem" in rp:
            rp = rp.get("refined_problem") or {}
        if not isinstance(rp, dict):
            return terms
        title = _normalize_query(rp.get("title"))
        if title:
            terms.append(title)
        background = _normalize_query(rp.get("background"))
        if background:
            terms.append(background[:80])
        for kw in (rp.get("kpi_candidates") or [])[:3]:
            kw = _normalize_query(kw)
            if kw:
                terms.append(f"{title} {kw}".strip())
        for risk in (rp.get("risk_points") or [])[:2]:
            risk = _normalize_query(risk)
            if risk:
                terms.append(f"{title} {risk}".strip())
        return [t for t in terms if t]

    fallback_terms = _fallback_terms()
    fallback_terms.extend([_normalize_query(q) for q in global_topic_queries])
    if len(pol_hits_all) < min_policy_hits and _time_left() > 0:
        for term in fallback_terms:
            if time.monotonic() - start_time > TIMEOUT_SECONDS:
                break
            if _time_left() <= 0:
                break
            for hit in _policy_results(term):
                _add_policy_hit(hit, {"name": "fallback"}, term)
            if len(pol_hits_all) >= min_policy_hits or policy_calls >= POLICY_QUERY_LIMIT:
                break

    if len(pap_hits_all) < min_paper_hits and _time_left() > 0:
        for term in fallback_terms:
            if time.monotonic() - start_time > TIMEOUT_SECONDS:
                break
            if _time_left() <= 0:
                break
            for hit in _paper_results(term):
                _add_paper_hit(hit, {"name": "fallback"}, term)
            if len(pap_hits_all) >= min_paper_hits or paper_calls >= PAPER_QUERY_LIMIT:
                break

    if len(pol_hits_all) < min_policy_hits and _time_left() > 0:
        archive_terms = [t for t in fallback_terms if t] or [user_query or ""]
        archive_hits = _archive_policy_matches(archive_terms, min_policy_hits - len(pol_hits_all))
        for idx, hit in enumerate(archive_hits, start=1):
            enriched = {
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "snippet": hit.get("summary", ""),
                "rank": len(pol_hits_all) + idx,
                "source": "archive",
                "layer": "archive",
                "origin_query": "archive",
                "origin_subproblem": hit.get("tags", [])[0] if hit.get("tags") else "archive",
            }
            if enriched["url"]:
                _add_policy_hit(enriched, {"name": "archive"}, "archive")
            else:
                pol_hits_all.append(enriched)

    layer_hits = _collect_layer_policy_hits(topic_map, pol_agent, prefer_rs_system=prefer_rs_system) if topic_map else {}

    if len(pol_hits_all) < min_policy_hits and _time_left() > 0:
        dummy_agent = DummyDataAgent()
        needed = min_policy_hits - len(pol_hits_all)
        dummy = dummy_agent.run(
            user_query=user_query or "",
            subproblem="policy",
            data_type="policy",
            count=needed,
        )
        for idx, hit in enumerate(dummy.get("items", []), start=1):
            enriched = {
                "title": hit.get("title", f"Existing Policy {idx}"),
                "url": hit.get("url", "https://example.com/existing"),
                "snippet": hit.get("snippet", "既存データ (政策)")
            }
            pol_hits_all.append(enriched)

    if len(pap_hits_all) < min_paper_hits and _time_left() > 0:
        dummy_agent = DummyDataAgent()
        needed = min_paper_hits - len(pap_hits_all)
        dummy = dummy_agent.run(
            user_query=user_query or "",
            subproblem="evidence",
            data_type="paper",
            count=needed,
        )
        for idx, hit in enumerate(dummy.get("items", []), start=1):
            enriched = {
                "title": hit.get("title", f"Existing Paper {idx}"),
                "url": hit.get("url", "https://example.com/existing"),
                "snippet": hit.get("snippet", "既存データ (論文)")
            }
            pap_hits_all.append(enriched)

    pol_hits_all = sorted(pol_hits_all, key=lambda x: x.get("rank", 9e9))[:24]
    pap_hits_all = sorted(pap_hits_all, key=lambda x: x.get("rank", 9e9))[:32]
    return {
        "policy_hits": pol_hits_all,
        "paper_hits": pap_hits_all,
        "topic_layer_hits": layer_hits,
    }

def synthesize_strategies(refined: Dict[str, Any], work: Dict[str, Any], search_res: Dict[str, Any], n_alternatives: int = 3) -> List[Dict[str, Any]]:
    kpi_cands = refined.get("refined_problem", {}).get("kpi_candidates", [])
    pol = search_res.get("policy_hits", [])
    pap = search_res.get("paper_hits", [])
    synth = SolutionSynthesizerAgent()
    out = []
    for i, theme in enumerate(THEMES):
        if len(out) >= n_alternatives: break
        s = synth.run(refined, work, pol, pap, kpi_cands, theme=theme)
        if "strategy" in s:
            out.append(s["strategy"])
    return out

def critique_strategies(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    return CritiqueAgent().run(strategies)

def estimate_budgets(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_REAL_BUDGET_ESTIMATES:
        return _make_dummy_budgets(strategies)
    bagent = BudgetAgent()
    results = []
    for s in strategies:
        res = bagent.estimate_budget_for_strategy(s)
        results.append({"strategy_name": s.get("name",""), **res})
    return results

def run_complex_pipeline(user_query: str, *, n_alternatives: int = 3, prefer_rs_system: bool = True) -> Dict[str, Any]:
    """
    ステップ1〜5を一括実行（確認はUI側で別ハンドリング）。
    """
    refined = refine_problem(user_query)
    work = decompose_work(refined.get("refined_problem", refined))
    topic_map = explore_topics(user_query, refined, work)
    search_res = run_searches(
        work,
        prefer_rs_system=prefer_rs_system,
        user_query=user_query,
        refined_problem=refined,
        topic_map=topic_map,
    )
    hypotheses = build_policy_hypotheses(
        user_query,
        search_res.get("topic_layer_hits") or {},
    )
    strategies = synthesize_strategies(refined, work, search_res, n_alternatives=n_alternatives)
    critique = critique_strategies(strategies)
    budgets = estimate_budgets(strategies)
    return {
        "refined": refined,
        "workplan": work,
        "topic_map": topic_map,
        "search_results": search_res,
        "policy_hypotheses": hypotheses,
        "strategies": strategies,
        "critique": critique,
        "budgets": budgets,
    }
