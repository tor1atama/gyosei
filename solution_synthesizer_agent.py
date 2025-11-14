# ebpm_agents/solution_synthesizer_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from .base import BaseAgent

BASE_SYSTEM = "あなたは政策デザイナー。日本語で厳密なJSONのみ返す。"

BASE_PROMPT = """以下の材料を統合し、複数KPI間の対立構造を説明できる政策案を生成してください。
出力スキーマ:
{{
  "strategy": {{
    "name": "",
    "theme": "{theme}",
    "summary": "課題→因果→施策→KPI",
    "policies": [
      {{"name":"","rationale":"","expected_kpis":[""],"evidence_refs":[{{"title":"","url":"","level":"high|medium|low"}}]}}
    ],
    "logic_tree": {{
      "goal": "最終アウトカム",
      "nodes": [{{"id":"N1","label":"","detail":""}}],
      "edges": [{{"from":"","to":"","evidence":"","evidence_level":"high|medium|low","conflict":"対象KPI/利害"}}]
    }},
    "risks": [""],
    "assumptions": [""],
    "timeline": [{{"milestone":"","due":"YYYY-Qn"}}]
  }}
}}
ルール: policies>=3, risks>=2, timeline>=2, logic_treeノード>=5で複数枝分かれ、エッジごとに evidence_level/ conflict を記載。

[材料]
- 詳細化課題:{refined}
- 作業分解:{workplan}
- 政策ヒット:{policy_hits}
- 論文ヒット:{paper_hits}
- KPI候補:{kpi_candidates}
"""

CRITIC_PROMPT = """あなたは批評エージェント。以下の戦略のロジックツリーを査読し、欠落・弱点・対立を列挙しJSONで返す。
スキーマ: {"feedback":"","conflicts":[""],"missing_nodes":[""],"weak_links":[""],"suggested_nodes":[{"id":"","label":"","detail":"","type":"Input|Activity|Output|Outcome|KPI|Impact"}]}。
政策案:{strategy}
"""

REFINE_PROMPT = """あなたは提案エージェント。批評の指摘を反映し、ロジックツリー・policies・risksを改良せよ。
要件: ノード>=6, エッジの evidence_level で色分け可能に、conflict は具体的なKPI利害を書き、weak_linksは補強案(detail)を含む。
出力: {"strategy": {...}} 同スキーマ。
旧戦略:{strategy}
批評:{feedback}
"""

ITERATIONS = 2

class SolutionSynthesizerAgent(BaseAgent):
    def run(
        self,
        refined_problem: Dict[str, Any],
        workplan: Dict[str, Any],
        policy_hits: List[Dict[str, Any]],
        paper_hits: List[Dict[str, Any]],
        kpi_candidates: List[Dict[str, Any]],
        *,
        theme: str = "cost-effective",
    ) -> Dict[str, Any]:
        ctx = {
            "refined": refined_problem,
            "workplan": workplan,
            "policy_hits": policy_hits[:12],
            "paper_hits": paper_hits[:12],
            "kpi_candidates": kpi_candidates[:12],
        }
        payload = BASE_PROMPT.format(
            refined=json.dumps(ctx["refined"], ensure_ascii=False),
            workplan=json.dumps(ctx["workplan"], ensure_ascii=False),
            policy_hits=json.dumps(ctx["policy_hits"], ensure_ascii=False),
            paper_hits=json.dumps(ctx["paper_hits"], ensure_ascii=False),
            kpi_candidates=json.dumps(ctx["kpi_candidates"], ensure_ascii=False),
            theme=theme,
        )
        out = self.chat(BASE_SYSTEM, payload, temperature=0.2, max_tokens=2400)
        strategy = self._safe_get_strategy(out, theme) or {
            "name": f"{theme}-fallback",
            "theme": theme,
            "summary": "",
            "policies": [],
            "logic_tree": {"goal": "", "nodes": [], "edges": []},
            "risks": [],
            "assumptions": [],
            "timeline": [],
        }
        strategy = self._normalize_strategy(strategy)
        strategy = self._evolve_logic_tree(strategy)
        return {"strategy": strategy}

    def _safe_get_strategy(self, raw: str, theme: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("strategy"), dict):
                return data["strategy"]
        except Exception:
            return None
        return None

    def _normalize_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        strat = dict(strategy)
        strat.setdefault("summary", strat.get("rationale", ""))
        policies = strat.get("policies") or []
        normalized_policies: List[Dict[str, Any]] = []
        for idx, p in enumerate(policies, start=1):
            normalized_policies.append({
                "name": p.get("name") or f"施策{idx}",
                "rationale": p.get("rationale", ""),
                "expected_kpis": p.get("expected_kpis", []),
                "evidence_refs": p.get("evidence_refs", []),
            })
        if len(normalized_policies) < 3:
            normalized_policies += [
                {"name": f"補完施策{len(normalized_policies)+i+1}", "rationale": "補完", "expected_kpis": [], "evidence_refs": []}
                for i in range(3 - len(normalized_policies))
            ]
        strat["policies"] = normalized_policies
        tree = strat.get("logic_tree") or {}
        nodes = tree.get("nodes") or []
        edges = tree.get("edges") or []
        if len(nodes) < 3:
            nodes = [
                {"id": "N1", "label": "Input", "detail": "予算"},
                {"id": "N2", "label": "Activity", "detail": normalized_policies[0]["name"]},
                {"id": "N3", "label": "Outcome", "detail": "主要KPI"},
            ]
        node_ids = {n["id"] for n in nodes}
        normalized_edges: List[Dict[str, Any]] = []
        for edge in edges:
            src = str(edge.get("from"))
            dst = str(edge.get("to"))
            if src in node_ids and dst in node_ids:
                normalized_edges.append({
                    "from": src,
                    "to": dst,
                    "evidence": edge.get("evidence", ""),
                    "evidence_level": edge.get("evidence_level", "medium"),
                    "conflict": edge.get("conflict", ""),
                })
        if not normalized_edges:
            normalized_edges = [
                {"from": nodes[i]["id"], "to": nodes[i+1]["id"], "evidence": "", "evidence_level": "medium", "conflict": ""}
                for i in range(len(nodes) - 1)
            ]
        tree["nodes"] = nodes
        tree["edges"] = normalized_edges
        strat["logic_tree"] = tree
        if len(strat.get("risks", [])) < 2:
            strat["risks"] = (strat.get("risks") or []) + ["不確実性", "リソース制約"]
        strat.setdefault("assumptions", ["データ入手可能", "関係者協力"])
        timeline = strat.get("timeline") or []
        if len(timeline) < 2:
            timeline += [{"milestone": "計画策定", "due": "2025-Q2"}, {"milestone": "中間レビュー", "due": "2026-Q4"}]
        strat["timeline"] = timeline[:3]
        return strat

    def _evolve_logic_tree(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        updated = self._normalize_strategy(strategy)
        best = dict(updated)
        best_score = self._score_strategy(best)
        critic_log: List[Dict[str, Any]] = []
        strat_json = json.dumps(updated, ensure_ascii=False)
        for idx in range(ITERATIONS):
            try:
                critic_raw = self.chat("あなたは批評者エージェント。JSONのみ返す。", CRITIC_PROMPT.format(strategy=strat_json), temperature=0.1, max_tokens=900)
                critic = json.loads(critic_raw)
            except Exception:
                critic = {"feedback": "", "conflicts": [], "missing_nodes": [], "weak_links": [], "suggested_nodes": []}
            critic_log.append({"iteration": idx + 1, "critic": critic})
            critic_json = json.dumps(critic, ensure_ascii=False)
            try:
                refine_raw = self.chat("あなたは提案エージェント。JSONのみ返す。", REFINE_PROMPT.format(strategy=strat_json, feedback=critic_json), temperature=0.2, max_tokens=2000)
                refine = json.loads(refine_raw)
                if isinstance(refine, dict) and isinstance(refine.get("strategy"), dict):
                    updated = self._normalize_strategy(refine["strategy"])
                    score = self._score_strategy(updated)
                    if score >= best_score:
                        best_score = score
                        best = dict(updated)
                    strat_json = json.dumps(updated, ensure_ascii=False)
            except Exception:
                continue
        best.setdefault("rlhf_meta", {})
        best["rlhf_meta"]["reward"] = round(best_score, 2)
        best["rlhf_meta"]["critic_log"] = critic_log
        return best

    def _score_strategy(self, strategy: Dict[str, Any]) -> float:
        tree = strategy.get("logic_tree") or {}
        nodes = tree.get("nodes") or []
        edges = tree.get("edges") or []
        node_count = len(nodes)
        adjacency: Dict[str, int] = {}
        evidence_weight = {"high": 3.0, "medium": 1.5, "low": 0.5}
        evidence_score = 0.0
        conflict_count = 0
        for edge in edges:
            src = str(edge.get("from"))
            adjacency[src] = adjacency.get(src, 0) + 1
            level = (edge.get("evidence_level") or "medium").lower()
            evidence_score += evidence_weight.get(level, 1.0)
            if edge.get("conflict"):
                conflict_count += 1
        multi_branch = sum(1 for deg in adjacency.values() if deg >= 2)
        policies = strategy.get("policies") or []
        kpi_coverage = len({k for p in policies for k in (p.get("expected_kpis") or [])})
        risk_diversity = len(strategy.get("risks") or [])
        timeline_span = len(strategy.get("timeline") or [])
        penalties = 0.0
        if node_count < 5:
            penalties += 4.0
        if multi_branch == 0:
            penalties += 3.0
        reward = (
            node_count * 0.6
            + multi_branch * 2.5
            + kpi_coverage * 1.2
            + evidence_score * 0.4
            + conflict_count * 1.0
            + risk_diversity * 0.2
            + timeline_span * 0.3
        )
        return reward - penalties
