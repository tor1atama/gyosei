# ebpm_agents/orchestrator.py
from __future__ import annotations
from typing import Dict, Any
from .config import require_openai
from .search_policy_agent import PolicySearchAgent
from .search_paper_agent import PaperSearchAgent
from .kpi_extractor_agent import KPIExtractorAgent
from .proposal_agent import KPIProposalAgent
from .graph_builder_agent import GraphBuilderAgent

def run_pipeline(
    user_query: str,
    *,
    prefer_rs_system: bool = True,        # ★ 追加：RSシステム限定を既定でON
    force_policy_keyword: str = "政策",   # ★ 追加：必ず「政策」を入れる
) -> Dict[str, Any]:
    require_openai()

    policy_agent = PolicySearchAgent()
    paper_agent  = PaperSearchAgent()
    kpi_agent    = KPIExtractorAgent()
    prop_agent   = KPIProposalAgent()
    graph_agent  = GraphBuilderAgent()

    pol = policy_agent.run(
        user_query,
        prefer_rs_system=prefer_rs_system,
        force_keyword=force_policy_keyword,
        max_results=12,
    )
    pap = paper_agent.run(user_query)

    kpis = kpi_agent.run(user_query, pol.get("results",[]), pap.get("results",[])).get("kpis", [])
    added = prop_agent.run(user_query, kpis).get("added_kpis", [])
    all_kpis = kpis + added
    graphs = graph_agent.run(all_kpis, max_scenarios=3).get("graphs", [])

    return {
        "input": user_query,
        "policy_search": pol,
        "paper_search": pap,
        "kpi_extracted": kpis,
        "kpi_added": added,
        "kpis_all": all_kpis,
        "graphs": graphs,
    }
