# ebpm_agents/research_orchestrator.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path
from .utils import extract_pdf_text, join_with_markers, chunk_by_chars, ensure_dir
from .doc_extractor_agent import ReviewSheetPrimer, ReviewSheetExtractor
from .kpi_update_agent import KPIUpdateAgent
from .web_timeseries_agent import TimeSeriesWebAgent
from .causality_agent import run_te_and_granger

def build_rs_primer(rs_pdf_path: str) -> Dict[str, Any]:
    pages = extract_pdf_text(rs_pdf_path)
    full = join_with_markers(pages)
    primer = ReviewSheetPrimer().build(full)
    return primer

def extract_effect_pathway(rs_like_pdf_path: str) -> Dict[str, Any]:
    return ReviewSheetExtractor().process_pdf(rs_like_pdf_path)

def seed_kpis_from_fragments(fragments: List[str]) -> Dict[str, Any]:
    return KPIUpdateAgent().seed_from_intermediate(fragments)

def update_kpis_with_pdfs(seed: Dict[str, Any], pdf_paths: List[str]) -> Dict[str, Any]:
    return KPIUpdateAgent().update_from_pdfs(seed, pdf_paths)

def build_kpi_timeseries(labels: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    agent = TimeSeriesWebAgent()
    out = {}
    for lab in labels:
        out[lab] = agent.run_label(lab).get("timeseries", [])
    return out

def collect_edges(effect_payloads: List[Dict[str, Any]]) -> List[Tuple[str,str]]:
    edges = []
    for item in effect_payloads:
        ep = item.get("effect_pathway") or {}
        for e in ep.get("edges", []) or []:
            src, dst = e.get("from"), e.get("to")
            if src and dst:
                edges.append((src, dst))
    # 重複除去
    seen, uniq = set(), []
    for e in edges:
        if e in seen: continue
        seen.add(e); uniq.append(e)
    return uniq

def run_causality(per_node_ts: Dict[str, List[Dict[str, Any]]], edges: List[Tuple[str,str]]) -> Dict[str, Any]:
    return run_te_and_granger(per_node_ts, edges)
