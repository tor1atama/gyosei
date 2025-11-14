# ebpm_agents/agent_competition_manager.py
from __future__ import annotations
from typing import Dict, Any, List

from .solution_synthesizer_agent import SolutionSynthesizerAgent
from .critique_agent import CritiqueAgent


class AgentCompetitionManager:
    """Runs multiple synthesizers and lets CritiqueAgent pick highlights."""

    def __init__(self):
        self.synth = SolutionSynthesizerAgent()
        self.critique = CritiqueAgent()

    def run(
        self,
        refined: Dict[str, Any],
        work: Dict[str, Any],
        search_res: Dict[str, Any],
        kpi_cands: List[Dict[str, Any]],
        contestants: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for c in contestants:
            theme = c.get("theme") or c.get("name") or "default"
            label = c.get("name") or theme
            try:
                res = self.synth.run(refined, work, search_res.get("policy_hits", []), search_res.get("paper_hits", []), kpi_cands, theme=theme)
            except Exception:
                res = {}
            if "strategy" in res:
                entries.append({
                    "contestant": label,
                    "theme": theme,
                    "strategy": res["strategy"],
                })
        critique = self.critique.run([e["strategy"] for e in entries]) if entries else {}
        return {"entries": entries, "critique": critique}


DEFAULT_CONTESTANTS = [
    {"name": "Agent Alpha", "theme": "cost-effective"},
    {"name": "Agent Beta", "theme": "equity-first"},
    {"name": "Agent Gamma", "theme": "speed-first"},
]


__all__ = ["AgentCompetitionManager", "DEFAULT_CONTESTANTS"]
