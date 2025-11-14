"""Minimal script to run decomposition → topic map → search → KPI extraction."""
from __future__ import annotations
import argparse
from pprint import pprint

from ebpm_agents_old.complex_orchestrator import (
    refine_problem,
    decompose_work,
    run_searches,
    explore_topics,
    build_policy_hypotheses,
)
from ebpm_agents_old.kpi_extractor_agent import KPIExtractorAgent


def run_policy_flow(user_query: str, prefer_rs_system: bool = True) -> dict:
    refined = refine_problem(user_query)
    workplan = decompose_work(refined.get("refined_problem", refined))
    topic_map = explore_topics(user_query, refined, workplan)
    search_results = run_searches(
        workplan,
        prefer_rs_system=prefer_rs_system,
        user_query=user_query,
        refined_problem=refined,
        topic_map=topic_map,
    )
    layer_hits = search_results.get("topic_layer_hits") or {}
    hypotheses = build_policy_hypotheses(user_query, layer_hits)
    kpi_agent = KPIExtractorAgent()
    kpis = kpi_agent.run(
        user_query,
        search_results.get("policy_hits", []),
        search_results.get("paper_hits", []),
    ).get("kpis", [])
    return {
        "refined_problem": refined,
        "workplan": workplan,
        "topic_map": topic_map,
        "search_results": search_results,
        "policy_hypotheses": hypotheses,
        "kpis": kpis,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the EBPM policy flow end-to-end.")
    parser.add_argument("query", help="政策課題の概要（日本語推奨）")
    parser.add_argument("--no-rs", action="store_true", help="RSシステム限定検索を外す")
    args = parser.parse_args()
    payload = run_policy_flow(args.query, prefer_rs_system=not args.no_rs)
    pprint(payload)


if __name__ == "__main__":
    main()
