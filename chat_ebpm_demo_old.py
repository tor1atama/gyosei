# chat_ebpm_demo.py
from __future__ import annotations
import json, re, time, uuid, os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import graphviz
import numpy as np
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False
from dotenv import load_dotenv
# ==== 追加 ====
from ebpm_agents_old.complex_orchestrator import (
    run_complex_pipeline, refine_problem, decompose_work, run_searches,
    synthesize_strategies, critique_strategies, estimate_budgets, explore_topics,
    build_policy_hypotheses
)
from ebpm_agents_old.dummy_data_agent import DummyDataAgent
from ebpm_agents_old.kpi_template_agent import KPITemplateAgent, DEFAULT_KPI_TEMPLATES
from ebpm_agents_old.hypothesis_gap_agent import HypothesisGapAgent
from ebpm_agents_old.agent_competition_manager import AgentCompetitionManager, DEFAULT_CONTESTANTS


# ==== Agents/Research orchestrators ====
_HAS_AGENTS = True
_AGENTS_IMPORT_ERR = ""
try:
    from ebpm_agents_old.orchestrator import run_pipeline as run_ebpm_agents
    from ebpm_agents_old.research_orchestrator import (
        build_rs_primer, extract_effect_pathway, seed_kpis_from_fragments,
        update_kpis_with_pdfs, build_kpi_timeseries, collect_edges, run_causality
    )
    from ebpm_agents_old.utils import extract_pdf_text, join_with_markers
except Exception as _e:
    _HAS_AGENTS = False
    _AGENTS_IMPORT_ERR = str(_e)

# .env
_ENV_PATHS = [
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / "Better-EBPM" / ".env",
    Path.cwd() / ".env",
]
_ENV_LOADED_FROM = None
for p in _ENV_PATHS:
    try:
        if p.is_file() and load_dotenv(dotenv_path=p):
            _ENV_LOADED_FROM = str(p); break
    except Exception:
        pass
if _ENV_LOADED_FROM is None:
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ===== OpenAI SDK 両対応（v0.28系 / v1系） =====
_HAS_OPENAI = False
_CLIENT_KIND = "none"
try:
    from openai import OpenAI  # v1.x
    _HAS_OPENAI = True; _CLIENT_KIND = "v1"
except Exception:
    try:
        import openai  # legacy
        _HAS_OPENAI = True; _CLIENT_KIND = "legacy"
    except Exception:
        _HAS_OPENAI = False

st.set_page_config(page_title="EBPM Chat Demo (Agents+Research)", layout="wide")

# --------------------
# ダミー知識ベース
# --------------------
KPI_CATALOG = {
    "地域医療": ["救急受入率(%)", "医師1人あたり患者数(人)", "初診待機日数(日)"],
    "子育て": ["保育所待機児童数(人)", "出生率(%)", "母子保健受診率(%)"],
    "雇用": ["有効求人倍率(倍)", "離職率(%)", "平均通勤時間(分)"],
}
POLICY_LIBRARY = {
    "奨学金返還免除":   {"cost": 3.0, "effect": 12.0, "lag": 2, "risk": "中"},
    "ICT効率化":       {"cost": 2.0, "effect": 9.0,  "lag": 1, "risk": "低"},
    "医療クラーク増員": {"cost": 1.5, "effect": 7.0,  "lag": 0, "risk": "中"},
    "給与引上げ":       {"cost": 4.0, "effect": 14.0, "lag": 1, "risk": "高"},
    "ドクターバンク":   {"cost": 2.5, "effect": 8.0,  "lag": 1, "risk": "中"},
}
RISK_NOTE = {"低": "実装容易", "中": "運用調整必要", "高": "政治・人材ハードル"}

POLICY_STAGE_FLOW = [
    {"name": "問題意識", "desc": "課題の背景と痛点を整理"},
    {"name": "仮説形成", "desc": "質的データから原因を洗い出し"},
    {"name": "施策設計", "desc": "施策候補を列挙し比較"},
    {"name": "検証・シミュ", "desc": "シナリオやKPIレンジを確認"},
    {"name": "意思決定", "desc": "コメント記録・関係者合意"},
]

STAGE_ACTION_HINTS = {
    "問題意識": ["課題をチャットに入力", "KPIテンプレートを生成"],
    "仮説形成": ["質的データをアップロード", "仮説の補強データを確認"],
    "施策設計": ["施策オプションを生成", "Agentコンペで案を比較"],
    "検証・シミュ": ["シナリオ最適化を実行", "KPIレンジで閾値違反をチェック"],
    "意思決定": ["政策担当者メモに意思を記録", "関係者向けテンプレを作成"],
}

SPECIAL_ACTION_DEFS = [
    {
        "name": "simulate",
        "keywords": ["シミュレーション", "効果を検証"],
        "message": "シミュレーションは Step 5 でシナリオ設定→最適化→KPIレンジ確認の順に実行できます。上部メニューの『5) 制約下での資源配分シミュレーション』をご利用ください。"
    },
    {
        "name": "stakeholder",
        "keywords": ["関係者", "アプローチ", "共有テンプレ"],
        "message": None
    }
]

QUAL_SAMPLE_TEXTS = [
    {"source": "厚労省_地域医療構想ヒアリング2023-07", "text": "都市部に医師が集中し、地方の夜間救急で受入拒否が相次いでいる。"},
    {"source": "県議会_医療提供体制委員会_議事録2024-03", "text": "医師が非臨床業務に追われ初診待機が長期化。ICT活用が進んでいない。"},
    {"source": "地域救急搬送実績レポート2024", "text": "高齢者が多い地域なのにドクターヘリが遠く、搬送時間が1時間以上かかる。"},
    {"source": "医療現場アンケート自由記述", "text": "医療クラークがいないためカルテ入力で残業が常態化。"},
]

CAUSE_KEYWORDS = {
    "医師偏在": ["医師", "偏在", "都市", "地方", "偏り"],
    "非臨床業務過多": ["非臨床", "事務", "クラーク", "入力", "残業"],
    "救急搬送遅延": ["救急", "搬送", "ヘリ", "受入", "待機"],
    "ICT不足": ["ICT", "デジタル", "電子カルテ", "DX"],
}
CAUSE_DATA_NEEDS = {
    "医師偏在": ["地域別医師届出票", "救急拒否件数の推移"],
    "非臨床業務過多": ["勤務表の時間配分", "カルテ入力ログ"],
    "救急搬送遅延": ["搬送時間と距離の対照", "ヘリ/救急車の到着記録"],
    "ICT不足": ["システム稼働率ログ", "職員ITリテラシー調査"],
    "その他": ["追加ヒアリング記録", "統計局オープンデータ"]
}

POLICY_OPTION_DB = {
    "医師偏在": [
        {"name": "奨学金返還免除", "cost": 3.0, "effect": 12.0, "lag": 2, "risk": "中", "kpi_target": "救急受入率(%)", "staff_need": 5, "evidence": "総務省 地域枠報告 2022"},
        {"name": "ドクターバンク制度", "cost": 2.5, "effect": 8.0, "lag": 1, "risk": "中", "kpi_target": "医師1人あたり患者数(人)", "staff_need": 3, "evidence": "北海道ドクターバンク"}
    ],
    "非臨床業務過多": [
        {"name": "医療クラーク増員", "cost": 1.5, "effect": 7.0, "lag": 0, "risk": "低", "kpi_target": "初診待機日数(日)", "staff_need": 8, "evidence": "厚労科研 ICT活用報告"},
        {"name": "ICT効率化", "cost": 2.0, "effect": 9.0, "lag": 1, "risk": "低", "kpi_target": "初診待機日数(日)", "staff_need": 4, "evidence": "成長戦略 2023"}
    ],
    "救急搬送遅延": [
        {"name": "ドクターヘリ共同運航", "cost": 3.5, "effect": 11.0, "lag": 1, "risk": "中", "kpi_target": "救急受入率(%)", "staff_need": 6, "evidence": "新潟県ヘリ整備"}
    ],
    "ICT不足": [
        {"name": "遠隔画像診断ネットワーク", "cost": 2.2, "effect": 8.5, "lag": 1, "risk": "中", "kpi_target": "救急受入率(%)", "staff_need": 4, "evidence": "北海道遠隔医療"}
    ],
}

SCENARIO_TEMPLATES = [
    {"name": "A: 成長", "budget": 8.0, "staff_limit": 30, "start_year": 2025, "duration": 6, "growth": 0.02, "lag_multiplier": 1.0, "discount_rate": 0.02},
    {"name": "B: 税収減", "budget": 5.0, "staff_limit": 20, "start_year": 2025, "duration": 6, "growth": -0.01, "lag_multiplier": 1.2, "discount_rate": 0.01},
]

RISK_SAMPLE = [
    {"risk": "財政状況悪化", "probability": 0.3, "impact": -4.0, "mitigation": "交付税の弾力運用"},
    {"risk": "人材確保失敗", "probability": 0.4, "impact": -6.0, "mitigation": "民間委託と研修充実"},
]

# --------------------
# 汎用ユーティリティ
# --------------------
def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def shorten(txt: str, n: int=160) -> str:
    if txt is None: return ""
    s = str(txt).replace("\n"," ")
    return s if len(s)<=n else s[:n]+"…"
def normalize_message(m: dict) -> dict:
    return {"id": m.get("id", str(uuid.uuid4())), "t": m.get("t", now_str()),
            "role": m.get("role", "assistant"), "content": m.get("content",""),
            **{k:v for k,v in m.items() if k not in {"id","t","role","content"}}}

# --------------------
# 可視化
# --------------------
def bubble_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty: return go.Figure()
    fig = px.scatter(df, x="コスト(億円)", y="効果(中位)", size="効果(中位)", color="リスク",
                     text="施策", hover_data=df.columns, size_max=60)
    fig.update_traces(textposition="top center")
    fig.update_layout(title="費用対効果マップ", xaxis_title="コスト(億円)", yaxis_title="効果(中位)")
    return fig

def band_chart(
    years: List[int],
    base: List[float],
    low: List[float],
    high: List[float],
    thr: Optional[float],
    y_title: str,
    extra_thresholds: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=high, name="Best", mode="lines"))
    fig.add_trace(go.Scatter(x=years, y=low,  name="Worst", mode="lines", fill="tonexty"))
    fig.add_trace(go.Scatter(x=years, y=base, name="Base", mode="lines+markers"))
    if extra_thresholds:
        for th in extra_thresholds:
            val = th.get("value")
            label = th.get("label") or "Threshold"
            if val is not None:
                fig.add_hline(y=val, line_dash="dash", annotation_text=label)
    elif thr is not None:
        fig.add_hline(y=thr, line_dash="dash", annotation_text=f"Threshold={thr}")
    fig.update_layout(title="KPI予測レンジ", xaxis_title="Year", yaxis_title=y_title)
    return fig

def logic_model_figure() -> go.Figure:
    nodes = {"課題": (0.05,0.5), "Input(予算/人員)": (0.25,0.5), "Activity(施策実行)": (0.45,0.5),
             "Output(短期)": (0.65,0.5), "Outcome(KPI)": (0.85,0.5), "Impact(社会効果)": (0.95,0.5)}
    fig = go.Figure()
    for name,(x,y) in nodes.items():
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[name],textposition="bottom center",
                                 marker=dict(size=14),name=name,hoverinfo="text"))
    edges=[("課題","Input(予算/人員)",""),("Input(予算/人員)","Activity(施策実行)","実装"),
           ("Activity(施策実行)","Output(短期)","達成"),("Output(短期)","Outcome(KPI)","波及"),
           ("Outcome(KPI)","Impact(社会効果)","長期")]
    for s,d,l in edges:
        x0,y0=nodes[s]; x1,y1=nodes[d]
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,showarrow=True,arrowhead=3,arrowsize=1,arrowwidth=1,text=l)
    fig.update_layout(title="Logic Model", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False, height=420)
    return fig

# --------------------
# EBPM 計算
# --------------------
def greedy_allocation(candidates: pd.DataFrame, budget: float) -> Tuple[pd.DataFrame, float, float]:
    if candidates is None or candidates.empty: return pd.DataFrame(), 0.0, 0.0
    df=candidates.copy(); df["eff_ratio"]=df["効果(中位)"]/df["コスト(億円)"].replace(0,np.nan)
    df=df.sort_values("eff_ratio",ascending=False)
    picked=[]; cost_sum=0.0; eff_sum=0.0
    for _,r in df.iterrows():
        if cost_sum + r["コスト(億円)"] <= budget:
            picked.append(r); cost_sum+=r["コスト(億円)"]; eff_sum+=r["効果(中位)"]
    return pd.DataFrame(picked), float(cost_sum), float(eff_sum)

def simulate_kpi(years: List[int], base_start: float, drift: float, policies: List[dict], lag_profile: Dict[int,float], noise=0.0) -> List[float]:
    vals=[]; states=[{"e":p["effect"],"lag":p["lag"],"age":0} for p in policies]
    for i,_ in enumerate(years):
        base=base_start+drift*i; yearly=0.0
        for s in states:
            idx = max(0, min(max(lag_profile), s["age"]-s["lag"]))
            w = lag_profile.get(idx, 0.0)
            yearly += s["e"]*max(0.0,w); s["age"]+=1
        vals.append(base+yearly+(np.random.normal(0,noise) if noise>0 else 0.0))
    return vals


def _to_float(val: Any) -> Optional[float]:
    try:
        if val is None: return None
        return float(val)
    except (TypeError, ValueError):
        return None


def threshold_breaches(years: List[int], low: List[float], high: List[float], constraint: Dict[str, Any]) -> List[str]:
    breaches: List[str] = []
    thr_value = _to_float(constraint.get("threshold_hint"))
    if thr_value is None:
        return breaches
    th_type = (constraint.get("threshold_type") or "min").lower()
    unit = constraint.get("unit") or ""
    name = constraint.get("name") or "KPI"
    if th_type == "min":
        for y, val in zip(years, low):
            if val < thr_value:
                breaches.append(f"{y}年に {name} が最低値 {thr_value}{unit} を下回る想定 ({val:.1f})")
                break
    elif th_type == "max":
        for y, val in zip(years, high):
            if val > thr_value:
                breaches.append(f"{y}年に {name} が上限 {thr_value}{unit} を超過する想定 ({val:.1f})")
                break
    return breaches


def load_qualitative_entries(files: List[Any]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for f in files or []:
        try:
            text = f.read().decode("utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append({"source": f.name, "text": line})
    return entries


def detect_cause_keyword(text: str) -> str:
    if not text:
        return "その他"
    t = text.lower()
    for cause, keywords in CAUSE_KEYWORDS.items():
        for k in keywords:
            if k.lower() in t:
                return cause
    return "その他"


def analyze_qualitative(entries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counter = Counter()
    for i, entry in enumerate(entries):
        cause = detect_cause_keyword(entry.get("text", ""))
        counter[cause] += 1
        rows.append({
            "quote_id": f"Q{i+1}",
            "source": entry.get("source", ""),
            "quote": entry.get("text", ""),
            "cause": cause,
            "cluster_lv1": cause,
            "cluster_lv2": cause,
            "importance": 1,
            "evidence_link": entry.get("source", "")
        })
    for row in rows:
        row["importance"] = counter[row["cause"]]
    return rows


def summarize_hypotheses(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    pivot = df.groupby("cluster_lv1").agg({"importance": "max", "quote_id": "count"}).rename(columns={"quote_id": "frequency"}).reset_index()
    pivot = pivot.sort_values("frequency", ascending=False)
    evidence_gaps = [
        {"cause": r["cluster_lv1"], "issue": "エビデンスリンク未設定"}
        for r in rows if not r.get("evidence_link")
    ]
    return pivot, evidence_gaps


def generate_policy_options_from_hypotheses(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen = set()
    for h in hypotheses or []:
        cause = h.get("cluster_lv1") or h.get("cause")
        for opt in POLICY_OPTION_DB.get(cause, []):
            key = opt["name"]
            if key in seen:
                continue
            seen.add(key)
            options.append({
                "施策": opt.get("name"),
                "原因カテゴリ": cause,
                "コスト(億円)": opt.get("cost", 0.0),
                "効果(中位)": opt.get("effect", 0.0),
                "効果(悲観)": round(opt.get("effect", 0.0) * 0.7, 1),
                "効果(楽観)": round(opt.get("effect", 0.0) * 1.2, 1),
                "ラグ(年)": opt.get("lag", 0),
                "スタッフ需要": opt.get("staff_need", 0),
                "KPI紐付け": opt.get("kpi_target", ""),
                "リスク": opt.get("risk", "中"),
                "根拠": opt.get("evidence", ""),
            })
    return options


def optimize_scenario_allocation(options: pd.DataFrame, budget: float, staff_limit: float) -> pd.DataFrame:
    if options is None or options.empty:
        return pd.DataFrame()
    df = options.copy()
    if "スタッフ需要" not in df.columns:
        df["スタッフ需要"] = 0
    df["スタッフ需要"] = df["スタッフ需要"].fillna(0)
    df["eff_ratio"] = df["効果(中位)"] / df["コスト(億円)"].replace(0, np.nan)
    df = df.sort_values("eff_ratio", ascending=False)
    picked = []
    cost_sum = 0.0
    staff_sum = 0.0
    for _, row in df.iterrows():
        cost = row["コスト(億円)"]
        staff = row.get("スタッフ需要", 0)
        if cost_sum + cost <= budget and staff_sum + staff <= staff_limit:
            picked.append(row)
            cost_sum += cost
            staff_sum += staff
    return pd.DataFrame(picked)


def simulate_scenario(years: List[int], scenario: Dict[str, Any], selected: pd.DataFrame) -> Dict[str, Any]:
    lag_profile = {0:0.0,1:0.4,2:0.7,3:1.0}
    start = scenario.get("start_year", years[0])
    base_start = 70.0 + scenario.get("growth", 0.0) * 100
    drift = scenario.get("growth", 0.0) * 50
    sel_df = selected if isinstance(selected, pd.DataFrame) else pd.DataFrame()
    policies = []
    for _, r in sel_df.iterrows():
        policies.append({"effect": r.get("効果(中位)",0.0), "lag": int(r.get("ラグ(年)",0))})
    mid = simulate_kpi(years, base_start, drift, policies, lag_profile)
    low = [v * 0.9 for v in mid]
    high = [v * 1.1 for v in mid]
    return {"years": years, "mid": mid, "low": low, "high": high}


def calc_risk_exposure(risks: List[Dict[str, Any]], picked: pd.DataFrame) -> pd.DataFrame:
    if not risks:
        return pd.DataFrame()
    df = pd.DataFrame(risks)
    df["expected_impact"] = df.get("probability", 0).astype(float) * df.get("impact", 0).astype(float)
    return df


def find_contention_points(hypotheses: List[Dict[str, Any]]) -> List[str]:
    counter = Counter()
    for h in hypotheses or []:
        counter[h.get("cluster_lv1", "")] += 1
    if len(counter) <= 1:
        return []
    most_common = counter.most_common(3)
    return [f"論点: {name}（言及 {freq} 件）" for name, freq in most_common]


def find_evidence_gaps(hypotheses: List[Dict[str, Any]]) -> List[str]:
    gaps = []
    for h in hypotheses or []:
        if not h.get("evidence_link"):
            gaps.append(f"{h.get('quote_id')} {h.get('cluster_lv1')} : 出典不明")
    return gaps


def fallback_gap_analysis(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   _entries = []
   for r in rows or []:
       cause = r.get("cluster_lv1") or r.get("cause") or "その他"
       needs = CAUSE_DATA_NEEDS.get(cause, CAUSE_DATA_NEEDS["その他"])
       _entries.append({
           "hypothesis": cause,
           "concern": "定量データや比較対象が不足しています",
           "needed_data": needs,
           "priority": "medium"
       })
   return _entries

# --------------------
# セッション
# --------------------
def _init_state():
    if "messages" not in st.session_state: st.session_state.messages=[]
    else: st.session_state.messages=[normalize_message(m) for m in st.session_state.messages]
    if "sessions" not in st.session_state: st.session_state.sessions=[]
    if "current_session_id" not in st.session_state: st.session_state.current_session_id=str(uuid.uuid4())
    if "context" not in st.session_state:
        st.session_state.context = {
            "domain": None, "kpi": None, "thr": None, "budget": None,
            "candidates": None, "picked": None, "base_start": 70.0, "drift": 0.0,
            "years": list(range(2025, 2031)),
            # Agents/Research
            "agents": None,
            "primer": None,
            "rs_result": None,
            "kpi_seed_fragments": [],
            "kpi_catalog": None,
            "effect_models": [],
            "per_node_timeseries": {},
            "topic_map": None,
            "topic_layer_hits": None,
            "policy_hypotheses": None,
            "kpi_templates": [],
            "kpi_constraints": [],
            "kpi_targets": [],
            "kpi_threshold_type": None,
            "qual_entries": [],
            "hypothesis_clusters": [],
            "policy_options": [],
            "scenario_configs": [],
            "scenario_results": {},
            "risk_register": [],
            "evidence_gaps": [],
            "contention_points": [],
            "hypothesis_gap_analysis": [],
            "policy_stage": "問題意識",
            "stage_notes": {},
            "decision_notes": "",
            "feedback_log": [],
            "stakeholder_templates": [],
            "agent_competition": {},
            "causality": None
        }
        # サイドバー内の設定に追記
        st.session_state.restrict_rs = st.checkbox("RSシステム限定（政策検索）", value=True)
        # 代替案の数はセッションに保持（後段の参照で NameError を避ける）
        if "n_alts" not in st.session_state:
            st.session_state.n_alts = 3
        st.number_input(
            "代替案の数（オーケストレーション）", min_value=1, max_value=5, value=st.session_state.n_alts, step=1, key="n_alts"
        )
    if "openai_api_key" not in st.session_state: st.session_state.openai_api_key=""
    if "tavily_api_key" not in st.session_state: st.session_state.tavily_api_key=""
    if "use_web_search" not in st.session_state: st.session_state.use_web_search=True
    if "show_agent_blocks" not in st.session_state: st.session_state.show_agent_blocks=True
_init_state()

def log_message(role: str, content: str, extra: dict | None = None):
    msg = {"id": str(uuid.uuid4()), "t": now_str(), "role": role, "content": content}
    if extra: msg.update(extra)
    st.session_state.messages.append(normalize_message(msg))

def start_new_conversation():
    started_at = now_str()
    if st.session_state.messages:
        first = normalize_message(st.session_state.messages[0]); started_at = first.get("t", now_str())
    st.session_state.sessions.append({
        "id": st.session_state.current_session_id, "started_at": started_at, "ended_at": now_str(),
        "messages": [normalize_message(m) for m in st.session_state.messages],
        "context": st.session_state.context.copy(),
    })
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.context.update({
        "domain": None, "kpi": None, "thr": None, "budget": None, "candidates": None, "picked": None,
        "base_start": 70.0, "drift": 0.0, "years": list(range(2025, 2031)),
        "agents": None, "primer": None, "rs_result": None, "kpi_seed_fragments": [],
        "kpi_catalog": None, "effect_models": [], "per_node_timeseries": {},
        "topic_map": None, "topic_layer_hits": None, "policy_hypotheses": None,
        "kpi_templates": [], "kpi_constraints": [], "kpi_threshold_type": None,
        "kpi_targets": [],
        "qual_entries": [], "hypothesis_clusters": [], "policy_options": [],
        "scenario_configs": [], "scenario_results": {}, "risk_register": [],
        "evidence_gaps": [], "contention_points": [], "hypothesis_gap_analysis": [],
        "policy_stage": "問題意識", "stage_notes": {}, "decision_notes": "", "feedback_log": [],
        "stakeholder_templates": [], "agent_competition": {},
        "causality": None
    })


def render_topic_map(topic_map: Dict[str, Any]):
    layers = topic_map.get("topic_layers") or []
    if not layers:
        st.info("論点マップがまだ生成されていません。")
        return

    def bullet(text: str, level: int = 0):
        indent = "&nbsp;" * (level * 4)
        st.markdown(f"{indent}• {text}", unsafe_allow_html=True)

    for node in layers:
        sub_name = node.get("subproblem") or "サブプロブレム"
        overview = node.get("overview") or ""
        with st.expander(f"📂 {sub_name}", expanded=False):
            if overview:
                st.markdown(f"> {overview}")
            for layer in node.get("layers", []) or []:
                tier = layer.get("tier", "")
                label = layer.get("label", "")
                focus = layer.get("policy_focus", "")
                bullet(f"[{tier.upper()}] {label} ― {focus}", level=0)
                if layer.get("keywords"):
                    bullet("キーワード: " + ", ".join(layer["keywords"]), level=1)
                if layer.get("angles"):
                    bullet("検討角度: " + ", ".join(layer["angles"]), level=1)
                sample = layer.get("sample_queries") or {}
                policy_q = sample.get("policy") or []
                evidence_q = sample.get("evidence") or []
                if policy_q:
                    bullet("政策検索: " + " / ".join(policy_q), level=1)
                if evidence_q:
                    bullet("エビデンス検索: " + " / ".join(evidence_q), level=1)
                for child in layer.get("child_nodes") or []:
                    child_label = child.get("label", "")
                    scope = child.get("scope", "")
                    bullet(f"{child_label} ({scope})", level=1)
                    if child.get("signals"):
                        bullet("把握したい指標: " + ", ".join(child["signals"]), level=2)
                    if child.get("sample_sources"):
                        bullet("参考ソース: " + ", ".join(child["sample_sources"]), level=2)
    gq = topic_map.get("global_queries") or {}
    if gq:
        st.markdown("**グローバル検索キーワード**")
        if gq.get("broad"):
            st.write("広義:", ", ".join(gq["broad"]))
        if gq.get("focused"):
            st.write("重点:", ", ".join(gq["focused"]))


def render_policy_hypotheses(hypotheses: Dict[str, List[Dict[str, Any]]]):
    if not hypotheses:
        st.info("政策仮説がまだ生成されていません。")
        return
    for sub, layers in hypotheses.items():
        with st.expander(f"🧠 {sub}", expanded=False):
            for layer in layers:
                st.markdown(f"**[{layer.get('tier','')}] {layer.get('label','')}**")
                for i, hyp in enumerate(layer.get("hypotheses", []), start=1):
                    st.markdown(f"- ({i}) {hyp.get('name','仮説')}")
                    st.caption(hyp.get("summary", ""))
                    if hyp.get("expected_effect"):
                        st.write("　効果想定:", hyp["expected_effect"])
                    if hyp.get("kpi"):
                        st.write("　KPI:", ", ".join(hyp["kpi"]))
                    if hyp.get("evidence"):
                        st.write("　参考:", ", ".join(hyp["evidence"]))


def render_policy_actions(policies: List[Dict[str, Any]]):
    if not policies:
        st.info("採用政策候補がまだありません。")
        return
    df = pd.DataFrame(policies)
    if not df.empty:
        st.dataframe(
            df.rename(columns={"name": "施策", "description": "概要", "kpi_links": "紐付KPI"}),
            use_container_width=True,
        )
    else:
        st.json(policies)


def render_logic_tree_graph(tree: Dict[str, Any]):
    nodes = (tree or {}).get("nodes") or []
    edges = (tree or {}).get("edges") or []
    if not nodes:
        st.info("ロジックツリー情報が不足しています。")
        return
    dot = graphviz.Digraph()
    ids = []
    for node in nodes:
        node_id = str(node.get("id") or node.get("label") or f"N{len(ids)+1}")
        ids.append(node_id)
        label = node.get("label") or node_id
        detail = node.get("detail") or ""
        dot.node(node_id, f"{label}\n{detail}")
    for edge in edges:
        src = str(edge.get("from"))
        dst = str(edge.get("to"))
        if not src or not dst:
            continue
        ev = edge.get("evidence", "")
        level = (edge.get("evidence_level") or "medium").lower()
        conflict = edge.get("conflict") or ""
        color = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}.get(level, "#7f7f7f")
        edge_label = ev
        if conflict:
            edge_label = f"{ev}\n⚠ {conflict}" if ev else f"⚠ {conflict}"
        dot.edge(src, dst, label=edge_label, color=color)
    st.graphviz_chart(dot, use_container_width=True)


def _normalize_kpi_links(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        items = [s.strip() for s in re.split(r"[,/、;|]", value) if s.strip()]
        if not items and value.strip():
            items = [value.strip()]
        return items
    return []


def get_kpi_targets(ctx: Dict[str, Any]) -> List[str]:
    if not ctx:
        return []
    targets = ctx.get("kpi_targets") or []
    if targets:
        return [t for t in targets if t]
    fallback = ctx.get("kpi")
    return [fallback] if fallback else []


def get_primary_kpi_name(ctx: Dict[str, Any]) -> Optional[str]:
    targets = get_kpi_targets(ctx)
    return targets[0] if targets else None


def set_kpi_targets(ctx: Dict[str, Any], targets: List[str]):
    clean = [t for t in targets if t]
    ctx["kpi_targets"] = clean
    ctx["kpi"] = clean[0] if clean else None


def epsilon_constraint_allocation(effect_matrix: np.ndarray, y_base: np.ndarray, budget: float, epsilons: np.ndarray):
    m, n = effect_matrix.shape
    solutions = {}
    for target in range(m):
        model = pulp.LpProblem(f"Maximize_KPI_{target}", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
        y = [y_base[k] + pulp.lpSum(effect_matrix[k][i] * x[i] for i in range(n)) for k in range(m)]
        model += pulp.lpSum(x) == budget
        for k in range(m):
            if k != target:
                model += y[k] >= epsilons[k]
        model += y[target]
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        x_sol = [round(pulp.value(var) or 0.0, 4) for var in x]
        y_sol = [round(pulp.value(val) or 0.0, 4) for val in y]
        solutions[target] = {"allocation": x_sol, "KPI_pred": y_sol}
    return solutions


def build_effect_inputs(state: Dict[str, Any]):
    app_ctx: Dict[str, Any]
    if hasattr(state, "context"):
        app_ctx = state.context
    elif isinstance(state, dict):
        app_ctx = state
    else:
        app_ctx = {}
    complex_ctx = app_ctx.setdefault("complex", {})

    constraints = complex_ctx.get("kpi_constraints") or app_ctx.get("kpi_constraints") or []
    if not constraints:
        domain = app_ctx.get("domain") or next(iter(KPI_CATALOG.keys()))
        kpis = KPI_CATALOG.get(domain) or next(iter(KPI_CATALOG.values()))
        defaults = []
        base = 70.0
        for idx, name in enumerate(kpis):
            defaults.append({
                "name": name,
                "definition": f"{name} の達成度",
                "unit": "%",
                "direction": "up",
                "threshold_type": "min",
                "threshold_hint": round(base + idx * 5, 1),
                "data_source": "自動提案サンプル",
                "legal_floor": "",
                "rationale": "ステップ未入力のため自動補完",
                "baseline": round(base - 5, 1),
            })
        complex_ctx["kpi_constraints"] = defaults
        app_ctx["kpi_constraints"] = defaults
        if defaults:
            set_kpi_targets(app_ctx, [defaults[0].get("name")] if defaults[0].get("name") else [])
        constraints = defaults

    policies = complex_ctx.get("policy_options") or []
    if not policies:
        auto_policies: List[Dict[str, Any]] = []
        for cause, opts in POLICY_OPTION_DB.items():
            for opt in opts:
                auto_policies.append({
                    "施策": opt.get("name"),
                    "原因カテゴリ": cause,
                    "コスト(億円)": opt.get("cost", 0.0),
                    "効果(中位)": opt.get("effect", 0.0),
                    "効果(悲観)": round(opt.get("effect", 0.0) * 0.7, 1),
                    "効果(楽観)": round(opt.get("effect", 0.0) * 1.2, 1),
                    "ラグ(年)": opt.get("lag", 0),
                    "スタッフ需要": opt.get("staff_need", 0),
                    "KPI紐付け": opt.get("kpi_target", ""),
                    "リスク": opt.get("risk", "中"),
                    "根拠": opt.get("evidence", ""),
                })
        complex_ctx["policy_options"] = auto_policies
        policies = auto_policies

    kpi_names = [c.get("name") for c in constraints if c.get("name")]
    if not kpi_names:
        return None
    n = len(policies)
    effect = np.zeros((len(kpi_names), n))
    kpi_index = {name: idx for idx, name in enumerate(kpi_names)}
    for j, opt in enumerate(policies):
        linked = opt.get("KPI紐付け") or kpi_names
        value = float(opt.get("効果(中位)") or 0.0)
        if not linked:
            linked = kpi_names
        for name in linked:
            idx = kpi_index.get(name)
            if idx is not None:
                effect[idx, j] = value / max(1, len(linked))
    y_base = np.array([
        float(c.get("baseline") or c.get("threshold_hint") or 0.0)
        for c in constraints if c.get("name") in kpi_index
    ])
    if len(y_base) != len(kpi_names):
        y_base = np.array([float(c.get("threshold_hint") or 0.0) for c in constraints if c.get("name")])
    eps = np.array([
        float(c.get("threshold_hint") or y_base[i] if i < len(y_base) else 0.0)
        for i, c in enumerate(constraints) if c.get("name")
    ])
    budget = float(ctx.get("budget") or sum(float(opt.get("コスト(億円)") or 1.0) for opt in policies))
    policy_names = [opt.get("施策") or opt.get("name") or f"施策{idx+1}" for idx, opt in enumerate(policies)]
    return effect, y_base, eps, budget, kpi_names, policy_names
def local_critique(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    reviews = []
    for strat in strategies or []:
        name = strat.get("name", "strategy")
        policies = strat.get("policies", [])
        kpis = [k for p in policies for k in p.get("expected_kpis", [])]
        reviews.append({
            "strategy_name": name,
            "strengths": [f"政策数 {len(policies)} 件を束ねている"],
            "weaknesses": ["ローカル批評: エビデンス強度を確認する必要あり"],
            "trade_offs": ["公平性と即効性のバランス"],
            "risks": ["実行体制の調整"],
            "mitigations": ["関係機関との連絡会を設定"],
            "unknowns": ["KPIデータの最新性"],
            "kpi_focus": list(dict.fromkeys(kpis))[:3],
        })
    return {
        "reviews": reviews,
        "cross_cutting_observations": ["ローカル批評: エビデンス詳細はLLM実行時に補完してください"]
    }


def render_stage_guide(ctx: Dict[str, Any]):
    current = ctx.get("policy_stage", POLICY_STAGE_FLOW[0]["name"])
    st.markdown("### 🧭 政策ステージガイド")
    cols = st.columns(len(POLICY_STAGE_FLOW))
    for col, stage in zip(cols, POLICY_STAGE_FLOW):
        with col:
            is_active = stage["name"] == current
            st.button(
                f"{stage['name']}\n{stage['desc']}",
                type="primary" if is_active else "secondary",
                key=f"stage_btn_{stage['name']}",
                on_click=lambda s=stage['name']: ctx.update({"policy_stage": s})
            )
    hints = STAGE_ACTION_HINTS.get(current, [])
    if hints:
        st.info("現在のステージ推奨アクション: " + " / ".join(hints))


def build_stakeholder_template(ctx: Dict[str, Any]) -> str:
    domain = ctx.get("domain") or "政策課題"
    stage = ctx.get("policy_stage", "問題意識")
    targets = get_kpi_targets(ctx)
    primary_kpi = targets[0] if targets else (ctx.get("kpi") or "主要KPI")
    kpi_label = primary_kpi if len(targets) <= 1 else f"{primary_kpi} ほか{len(targets)-1}指標"
    thr = ctx.get("thr")
    thr_text = f"目標 {kpi_label} >= {thr}" if thr else f"{kpi_label} を改善"
    template = (
        f"件名: {domain} の進め方について\n"
        f"現状: {stage} 段階での気づきと論点を共有します。\n"
        f"KPI目線: {thr_text}\n"
        "求めるアクション: ①最新データ共有 ②合意形成のためのコメント ③追加懸念点\n"
        "返信期限: ○月○日までにコメントいただけると助かります。"
    )
    return template


def get_polaris_flow(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.setdefault(
        "polaris_flow",
        {
            "stage": "ask_problem",
            "messages": [],
            "problem": "",
            "qual_entries": [],
            "policy_choice": None,
            "last_search_duration": None,
        },
    )


def polaris_log(ctx: Dict[str, Any], role: str, text: str, payload: Optional[Dict[str, Any]] = None):
    pf = get_polaris_flow(ctx)
    entry = {"role": role, "content": text, "ts": now_str()}
    if payload:
        entry["payload"] = payload
    pf["messages"].append(entry)
    pf["messages"] = pf["messages"][-60:]


def render_polaris_chat(ctx: Dict[str, Any]):
    pf = get_polaris_flow(ctx)
    for msg in pf["messages"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])
            payload = msg.get("payload")
            if payload:
                kind = payload.get("kind")
                label = payload.get("label")
                if kind == "table":
                    data = payload.get("data") or []
                    if data:
                        df = pd.DataFrame(data)
                        if label:
                            st.caption(label)
                        st.dataframe(df, use_container_width=True)
                elif kind == "strategies":
                    data = payload.get("data") or []
                    if label:
                        st.caption(label)
                    for strat in data:
                        st.markdown(f"**{strat.get('name','strateg y')}** ({strat.get('theme','')})  \n{shorten(strat.get('summary',''), 220)}")
                elif kind == "markdown":
                    if label:
                        st.caption(label)
                    st.markdown(payload.get("text", ""))


def _autofill_kpi_constraints(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    refined = ctx["complex"].get("refined") or {}
    candidates = refined.get("refined_problem", {}).get("kpi_candidates") or []
    if not candidates:
        domain = ctx.get("domain") or next(iter(KPI_CATALOG.keys()))
        base = KPI_CATALOG.get(domain) or next(iter(KPI_CATALOG.values()))
        candidates = [{"name": name, "threshold_hint": 75 + idx * 3} for idx, name in enumerate(base)]
    constraints = []
    for idx, cand in enumerate(candidates):
        if isinstance(cand, str):
            name = cand
            threshold = 75 + idx * 2
            unit = "%"
            direction = "up"
        else:
            name = cand.get("name") or f"KPI{idx+1}"
            threshold = cand.get("threshold_hint") or (75 + idx * 2)
            unit = cand.get("unit") or "%"
            direction = cand.get("direction") or "up"
        constraints.append({
            "name": name,
            "definition": cand.get("definition") if isinstance(cand, dict) else "",
            "unit": unit,
            "direction": direction,
            "threshold_type": cand.get("threshold_type", "min") if isinstance(cand, dict) else "min",
            "threshold_hint": threshold,
            "data_source": cand.get("source", "POLARIS推定") if isinstance(cand, dict) else "POLARIS推定",
            "legal_floor": cand.get("legal_floor", "") if isinstance(cand, dict) else "",
            "rationale": cand.get("rationale", "") if isinstance(cand, dict) else "",
            "baseline": cand.get("baseline", threshold - 5) if isinstance(cand, dict) else threshold - 5,
        })
    return constraints[:5]


def advance_polaris_flow(ctx: Dict[str, Any]):
    pf = get_polaris_flow(ctx)
    while True:
        stage = pf.get("stage", "ask_problem")
        if stage == "refine":
            problem = pf.get("problem") or ctx["complex"].get("user_query")
            if not problem:
                pf["stage"] = "ask_problem"
                break
            with st.spinner("詳細化を実行しています…"):
                refined = refine_problem(problem)
            ctx["complex"]["refined"] = refined
            ctx["complex"]["user_query"] = problem
            ctx["complex"].update({
                "workplan": None,
                "topic_map": None,
                "topic_layer_hits": None,
                "policy_hypotheses": None,
                "search_results": None,
                "strategies": [],
                "critique": None,
                "budgets": [],
                "qual_entries": [],
                "hypothesis_clusters": [],
                "policy_options": [],
                "scenario_configs": [],
                "scenario_results": {},
            })
            pf["stage"] = "kpi_proposal"
            summary = refined.get("refined_problem", {}).get("title") or "詳細化済みの政策課題"
            polaris_log(ctx, "assistant", f"政策課題を詳細化しました。\n\n- 主題: **{summary}**\n- 重点論点を整理しました。次に KPI の候補を提示します。")
            continue
        if stage == "kpi_proposal":
            constraints = _autofill_kpi_constraints(ctx)
            ctx["complex"]["kpi_constraints"] = constraints
            ctx["complex"]["kpi_templates"] = constraints
            ctx["kpi_constraints"] = constraints
            set_kpi_targets(ctx, [c["name"] for c in constraints if c.get("name")])
            table_data = [
                {
                    "KPI": c.get("name"),
                    "目標": f"{c.get('threshold_hint')} {c.get('unit','')}".strip(),
                    "方向": "向上" if (c.get("direction") or "up") == "up" else "抑制",
                }
                for c in constraints
            ]
            polaris_log(
                ctx,
                "assistant",
                "「KPI候補の提案」を行いました。\n\n2) KPI設定と制約条件の明確化を行いましょう。テーブルで閾値を調整したら『KPI設定完了』を押してください。",
                payload={"kind": "table", "label": "KPI候補一覧", "data": table_data},
            )
            pf["stage"] = "kpi_confirm"
            continue
        if stage == "qual_analyze":
            entries = pf.get("qual_entries")
            if not entries:
                pf["stage"] = "qual_prompt"
                break
            with st.spinner("質的データをKJ法的に整理しています…"):
                rows = analyze_qualitative(entries)
            pivot, gaps = summarize_hypotheses(rows)
            ctx["complex"]["qual_entries"] = entries
            ctx["complex"]["hypothesis_clusters"] = rows
            ctx["complex"]["evidence_gaps"] = [g["issue"] if isinstance(g, dict) else str(g) for g in gaps]
            ctx["complex"]["contention_points"] = find_contention_points(rows)
            table = pivot.rename(columns={"cluster_lv1": "原因クラスタ", "frequency": "件数", "importance": "重要度"}).to_dict("records")
            polaris_log(
                ctx,
                "assistant",
                f"質的データを解析しました。仮説クラスタ数: {len(pivot)}。原因別の施策オプションを列挙します。",
                payload={"kind": "table", "label": "仮説クラスタ概要", "data": table},
            )
            pf["stage"] = "policy_auto"
            continue
        if stage == "policy_auto":
            hypotheses = ctx["complex"].get("hypothesis_clusters") or []
            options = generate_policy_options_from_hypotheses(hypotheses)
            if not options:
                options = [
                    {"施策": n, "原因カテゴリ": "参考", "コスト(億円)": info["cost"], "効果(中位)": info["effect"],
                     "効果(悲観)": round(info["effect"]*0.7,1), "効果(楽観)": round(info["effect"]*1.2,1),
                     "ラグ(年)": info["lag"], "スタッフ需要": 5, "KPI紐付け": get_primary_kpi_name(ctx),
                     "リスク": info["risk"], "根拠": "POLARISライブラリ"}
                    for n, info in POLICY_LIBRARY.items()
                ]
            ctx["complex"]["policy_options"] = options
            table = [
                {
                    "施策": opt.get("施策"),
                    "原因": opt.get("原因カテゴリ"),
                    "効果(中位)": opt.get("効果(中位)"),
                    "コスト(億円)": opt.get("コスト(億円)"),
                }
                for opt in options[:6]
            ]
            polaris_log(
                ctx,
                "assistant",
                f"原因別の施策オプションを {len(options)} 件抽出しました。この方針で進めてよろしいですか？（はい/いいえを選択）",
                payload={"kind": "table", "label": "施策オプション（抜粋）", "data": table},
            )
            st.session_state.pop("polaris_policy_choice", None)
            pf["stage"] = "policy_confirm"
            continue
        if stage == "decompose_run":
            refined = ctx["complex"].get("refined")
            if not refined:
                pf["stage"] = "ask_problem"
                break
            with st.spinner("タスクを分解し、論点を階層化しています…"):
                workplan = decompose_work(refined.get("refined_problem", refined))
                try:
                    topic_map = explore_topics(ctx["complex"].get("user_query") or "", refined, workplan)
                except Exception:
                    topic_map = {}
            ctx["complex"]["workplan"] = workplan
            ctx["complex"]["topic_map"] = topic_map
            polaris_log(ctx, "assistant", "タスク分解と論点マップを生成しました。分解に基づき検索を実行します。")
            pf["stage"] = "search_run"
            continue
        if stage == "search_run":
            workplan = ctx["complex"].get("workplan")
            if not workplan:
                pf["stage"] = "decompose_run"
                continue
            start = time.time()
            status_placeholder = st.empty()
            with st.spinner("分解に基づき検索を実行しています…"):
                search_res = run_searches(
                    workplan,
                    prefer_rs_system=st.session_state.restrict_rs,
                    user_query=ctx["complex"].get("user_query"),
                    refined_problem=ctx["complex"].get("refined"),
                    topic_map=ctx["complex"].get("topic_map"),
                )
            policy_hits = search_res.get("policy_hits", [])
            paper_hits = search_res.get("paper_hits", [])
            polaris_log(
                ctx,
                "assistant",
                "分解に基づき検索を実行しています…",
                payload={
                    "kind": "table",
                    "label": "検索途中のステータス",
                    "data": [
                        {"種別": "政策ヒット件数", "値": len(policy_hits)},
                        {"種別": "論文ヒット件数", "値": len(paper_hits)},
                    ],
                },
            )
            duration = time.time() - start
            ctx["complex"]["search_results"] = search_res
            if duration > 10 or not policy_hits:
                dummy_agent = DummyDataAgent()
                dummy = dummy_agent.run(
                    ctx["complex"].get("user_query") or "",
                    subproblem="policy",
                    data_type="policy",
                    count=3,
                )
                for hit in dummy.get("items", []):
                    policy_hits.append({
                        "title": hit.get("title", "Dummy Policy"),
                        "url": hit.get("url", "https://example.com/dummy"),
                        "snippet": hit.get("snippet", "ダミーデータ (政策)"),
                        "rank": len(policy_hits) + 1,
                    })
                search_res["policy_hits"] = policy_hits
                polaris_log(ctx, "assistant", "検索結果が10秒以内に揃わなかったため、ダミーデータで補完しました。")
            else:
                polaris_log(ctx, "assistant", f"{duration:.1f}秒で政策候補 {len(policy_hits)} 件、論文 {len(search_res.get('paper_hits', []))} 件を取得しました。複数案を生成します。")
            pf["stage"] = "strategy_run"
            continue
        if stage == "strategy_run":
            refined = ctx["complex"].get("refined")
            workplan = ctx["complex"].get("workplan")
            search_res = ctx["complex"].get("search_results") or {}
            if not (refined and workplan and search_res):
                pf["stage"] = "search_run"
                continue
            with st.spinner("複数の政策案を合成しています…"):
                n_alts = int(st.session_state.get("n_alts", 3))
                strategies = synthesize_strategies(refined, workplan, search_res, n_alternatives=n_alts)
            ctx["complex"]["strategies"] = strategies
            summaries = [
                {
                    "name": s.get("name", f"strategy {idx+1}"),
                    "theme": s.get("theme"),
                    "summary": s.get("summary") or s.get("rationale", ""),
                }
                for idx, s in enumerate(strategies)
            ]
            polaris_log(
                ctx,
                "assistant",
                f"最適と思われる複数案が生成されました（{len(strategies)}件）。横並びで比較しつつ批判的検討に進みます。",
                payload={"kind": "strategies", "label": "生成された政策案サマリ", "data": summaries},
            )
            pf["stage"] = "critique_run"
            continue
        if stage == "critique_run":
            strategies = ctx["complex"].get("strategies") or []
            if not strategies:
                pf["stage"] = "strategy_run"
                continue
            with st.spinner("批判的検討（何が犠牲になるか）を実行中…"):
                if _HAS_OPENAI and (OPENAI_API_KEY or "").strip():
                    critique = critique_strategies(strategies)
                else:
                    critique = local_critique(strategies)
            ctx["complex"]["critique"] = critique
            polaris_log(ctx, "assistant", "批判的検討を完了しました。次に制約下での資源配分シミュレーションを行います。")
            pf["stage"] = "simulation_run"
            continue
        if stage == "simulation_run":
            options = ctx["complex"].get("policy_options") or []
            if not options:
                pf["stage"] = "risk_run"
                continue
            scenario_cfgs = ctx["complex"].get("scenario_configs") or SCENARIO_TEMPLATES
            ctx["complex"]["scenario_configs"] = scenario_cfgs
            scenario_results = ctx["complex"].setdefault("scenario_results", {})
            years = ctx.get("years", list(range(2025, 2031)))
            options_df = pd.DataFrame(options)
            if options_df.empty:
                pf["stage"] = "risk_run"
                continue
            last_allocation = pd.DataFrame()
            with st.spinner("制約下での資源配分シミュレーションを実行しています…"):
                for scenario in scenario_cfgs[:2]:
                    selected = optimize_scenario_allocation(
                        options_df,
                        scenario.get("budget", 0.0),
                        scenario.get("staff_limit", 0.0),
                    )
                    last_allocation = selected
                    result = simulate_scenario(years, scenario, selected)
                    scenario_results[scenario["name"]] = result
            ctx["complex"]["scenario_results"] = scenario_results
            if isinstance(last_allocation, pd.DataFrame) and not last_allocation.empty:
                ctx["picked"] = last_allocation
            polaris_log(ctx, "assistant", f"{len(scenario_results)} シナリオで資源配分シミュレーションを実施しました。続いて効果とリスクを可視化します。")
            pf["stage"] = "risk_run"
            continue
        if stage == "risk_run":
            if not ctx["complex"].get("risk_register"):
                ctx["complex"]["risk_register"] = RISK_SAMPLE
            exposure = calc_risk_exposure(ctx["complex"]["risk_register"], ctx.get("picked"))
            ctx["complex"]["risk_exposure"] = exposure.to_dict("records") if not exposure.empty else []
            payload = None
            if not exposure.empty:
                payload = {"kind": "table", "label": "リスク感度分析", "data": exposure.to_dict("records")}
            polaris_log(ctx, "assistant", "その案の効果・リスクを整理しました。RSシステムから過去予算を探索し、段階別に可視化します…", payload=payload)
            pf["stage"] = "budget_run"
            continue
        if stage == "budget_run":
            strategies = ctx["complex"].get("strategies") or []
            if not strategies:
                pf["stage"] = "done"
                continue
            with st.spinner("RSシステムからデータを読み込み中…予算情報を推定しています"):
                budgets = estimate_budgets(strategies)
            ctx["complex"]["budgets"] = budgets
            polaris_log(ctx, "assistant", "RSシステムから過去予算を探索し、段階別に可視化しました。POLARIS フローは完了です。")
            pf["stage"] = "done"
            continue
        break


def handle_special_actions(prompt: str, ctx: Dict[str, Any]) -> List[str]:
    outputs: List[str] = []
    for conf in SPECIAL_ACTION_DEFS:
        if any(keyword in prompt for keyword in conf["keywords"]):
            if conf["name"] == "stakeholder":
                template = build_stakeholder_template(ctx)
                ctx.setdefault("stakeholder_templates", []).append({"time": now_str(), "template": template})
                outputs.append("📨 関係者アプローチ案:\n\n" + template)
            else:
                outputs.append(conf.get("message") or "追加機能の準備中です。")
    return outputs
# --------------------
# OpenAI ラッパ
# --------------------
def get_openai_client():
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not (_HAS_OPENAI and key): return None
    if _CLIENT_KIND == "v1":
        return OpenAI(api_key=key)
    else:
        import openai as _legacy
        _legacy.api_key = key
        return "legacy"

def llm_chat(messages: List[dict], model: str = "gpt-4o") -> str:
    client = get_openai_client()
    if client is None:
        return "⚠️ OpenAI API Key を環境変数から取得できません。.env に OPENAI_API_KEY=... を設定してください。"
    try:
        if _CLIENT_KIND == "v1":
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=600)
            return resp.choices[0].message.content.strip()
        else:
            import openai as _legacy
            resp = _legacy.ChatCompletion.create(model=model, messages=messages, temperature=0.2, max_tokens=600)
            return resp.choices[0].message["content"].strip()
    except Exception as e:
        err = str(e)
        if any(x in err for x in ["401","403","Unauthorized","Authentication","invalid_api_key","api_key"]):
            return "❌ 認証エラー: APIキーを確認してください。"
        if any(x in err for x in ["404","model_not_found","No such model","is not permitted"]):
            return "❌ モデル未対応/存在しません。モデル名を 'gpt-4o-mini' 等に変更してください。"
        return f"❌ LLM呼び出しエラー: {err}"

# --------------------
# Tavily 簡易検索（既存）
# --------------------
def web_search_tavily(query: str, max_results: int = 5) -> List[dict]:
    key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")
    if not key: return []
    try:
        r = requests.post("https://api.tavily.com/search", json={
            "api_key": key, "query": query, "max_results": max_results,
            "include_answer": False, "include_raw_content": False, "search_depth": "basic",
        }, timeout=30)
        data = r.json()
        return data.get("results", [])
    except Exception as e:
        return [{"title":"検索エラー","url":"","content":str(e)}]

def summarize_with_citations(question: str, results: List[dict]) -> str:
    if not results: return "（オンライン検索が無効、または結果が見つかりませんでした）"
    snippets, links = [], []
    for i, r in enumerate(results[:5], start=1):
        title = shorten(r.get("title",""), 80); content = shorten(r.get("content",""), 400); url = r.get("url","")
        snippets.append(f"[{i}] {title}\n{content}"); links.append(f"[{i}] {title} — {url}")
    system = {"role":"system","content":"事実と推測を分け、[1]形式で根拠番号を示す。"}
    user = {"role":"user","content":f"質問: {question}\n\n抜粋:\n" + "\n\n".join(snippets) + "\n\n3-6項目で要点を書き、最後に参考リンクを列挙。"}
    answer = llm_chat([system, user]); answer += "\n\n参考:\n" + "\n".join(links)
    return answer

# --------------------
# サイドバー
# --------------------
with st.sidebar:
    st.header("Settings & Controls")
    st.caption("APIキーは .env から自動読み込み")
    st.write(f".env: {_ENV_LOADED_FROM or '不明'}")
    st.write(f"OpenAI: {'設定済み' if (OPENAI_API_KEY or '').strip() else '未設定'}")
    st.write(f"Tavily: {'設定済み' if (TAVILY_API_KEY or '').strip() else '未設定'}")
    st.session_state.use_web_search = st.checkbox("オンライン検索を有効化", value=st.session_state.use_web_search)
    st.session_state.restrict_rs = st.checkbox("RSシステム限定（政策検索）", value=True)

    with st.expander("診断 (Debug)"):
        st.write(f"HAS_OPENAI: {_HAS_OPENAI}")
        st.write(f"CLIENT_KIND: {_CLIENT_KIND}")
        st.write(f"HAS_AGENTS: {_HAS_AGENTS}")
        if not _HAS_AGENTS: st.write(f"AGENTS_IMPORT_ERR: {_AGENTS_IMPORT_ERR}")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🆕 新しい会話"):
            start_new_conversation(); st.rerun()
    with colB:
        if st.button("🧹 クリア"):
            st.session_state.messages=[]; _init_state(); st.rerun()

    st.markdown("---")
    st.session_state.show_agent_blocks = st.checkbox("🔎 エージェント結果表示", value=st.session_state.show_agent_blocks)

    if st.button("▶️ 直近のユーザ入力でエージェント実行"):
        last_user = next((m.get("content") for m in reversed(st.session_state.messages) if m.get("role")=="user"), None)
        if last_user:
            with st.spinner("エージェントを実行中…"):
                agents_out = {"error":"エージェント未ロード"}
                if _HAS_AGENTS:
                    agents_out = run_ebpm_agents(
                        last_user,
                        prefer_rs_system=st.session_state.restrict_rs,       # ★ 追加
                        force_policy_keyword="政策",                          # ★ 追加
                    )
            st.session_state.context["agents"] = agents_out
            log_message("assistant", "（エージェント実行完了）サイドの『エージェント結果表示』をONで確認できます。")
            st.rerun()
        else:
            st.warning("ユーザ入力がまだありません。Chatタブで課題を入力してください。")


    st.markdown("---")
    export_payload = {
        "current_session_id": st.session_state.current_session_id,
        "messages": st.session_state.messages,
        "context": st.session_state.context,
        "sessions": st.session_state.sessions,
        "exported_at": now_str(),
        "app_version": "chat-ebpm-demo-logs-2.0-research",
    }
    st.download_button("⬇️ 全ログJSONをエクスポート", data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                       file_name=f"ebpm_chat_logs_{int(time.time())}.json", mime="application/json")
    st.write("インポート（JSON）")
    up = st.file_uploader("選択", type=["json"], key="import_json")
    if up:
        try:
            data = json.load(up)
            st.session_state.current_session_id = data.get("current_session_id", str(uuid.uuid4()))
            st.session_state.messages = [normalize_message(m) for m in data.get("messages", [])]
            st.session_state.context  = data.get("context", st.session_state.context)
            st.session_state.sessions = data.get("sessions", [])
            st.success("読み込みました")
        except Exception as e:
            st.error(f"読み込み失敗: {e}")

    st.markdown("---")
    feedback_text = st.text_area("即時フィードバック", key="sidebar_feedback")
    if st.button("フィードバック送信", key="btn_sidebar_feedback"):
        if feedback_text.strip():
            st.session_state.context.setdefault("feedback_log", []).append({"time": now_str(), "text": feedback_text.strip()})
            st.success("フィードバックを保存しました。")
        else:
            st.warning("内容を入力してください。")
    if st.session_state.context.get("feedback_log"):
        latest_fb = st.session_state.context["feedback_log"][-1]
        st.caption(f"最新フィードバック ({latest_fb['time']}): {latest_fb['text']}")

# --------------------
# タブ
# --------------------
tabs = st.tabs(["💬 Chat", "🌟 POLARIS", "📄 文書抽出", "📈 時系列/因果", "🧾 現在の会話ログ", "🗂 セッション履歴", "ℹ️ 使い方"])
tab_chat, tab_orch, tab_docs, tab_ts, tab_current, tab_sessions, tab_help = tabs

# ===== Chat =====
with tab_chat:
    st.title("行政EBPM支援ツール（対話型 + 研究機能）")
    st.caption("KPI→施策比較→配分→効果レンジ→ロジックモデル→モニタ + LLM会話 + オンライン検索 + エージェント群 + PDF抽出/因果分析")
    render_stage_guide(st.session_state.context)
    with st.expander("✍️ 政策担当者メモ", expanded=False):
        stage = st.session_state.context.get("policy_stage", "問題意識")
        current_note = st.session_state.context.get("stage_notes", {}).get(stage, "")
        memo = st.text_area(f"{stage} ステージのメモ", value=current_note, key=f"stage_note_box_{stage}")
        if st.button("このステージのメモを保存", key=f"save_stage_note_{stage}"):
            st.session_state.context.setdefault("stage_notes", {})[stage] = memo
            st.success("メモを保存しました。")

    if len(st.session_state.messages)==0:
        log_message("assistant", "政策課題を教えてください（例: 地域医療の救急受入率を改善したい）。目標や予算があれば一緒に。")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    prompt = st.chat_input("入力…（/web 〜 で検索, /agent 〜 でエージェント実行）")
    if prompt:
        log_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        ctx=st.session_state.context; responded=False

        # エージェントトリガ
        agent_triggered=False
        if prompt.strip().startswith("/agent") or "エージェント" in prompt:
            agent_query = prompt.replace("/agent","").strip() or prompt; agent_triggered=True
        elif st.session_state.show_agent_blocks and any(k in prompt for k in ["政策","論文","KPI","レビュー","因果","グラフ"]):
            agent_query = prompt; agent_triggered=True
        else:
            agent_query = prompt

        if agent_triggered and _HAS_AGENTS:
            with st.chat_message("assistant"):
                st.markdown(f"🧠 エージェント実行: **{agent_query}**")
                with st.spinner("エージェント群を走らせています…"):
                    agents_out = run_ebpm_agents(
                        agent_query,
                        prefer_rs_system=st.session_state.restrict_rs,   # ★ 追加
                        force_policy_keyword="政策",                      # ★ 追加
                    )
            st.session_state.context["agents"] = agents_out
            k_n=len(agents_out.get("kpis_all", [])); p_n=len((agents_out.get("policy_search",{}) or {}).get("results",[])); a_n=len((agents_out.get("paper_search",{}) or {}).get("results",[]))
            msg_ag=f"✅ エージェント結果: 政策 {p_n} / 論文 {a_n} / KPI {k_n}（『エージェント結果表示』で詳細）。"
            log_message("assistant", msg_ag)
            with st.chat_message("assistant"):
                st.success(msg_ag)

        # 既存意図検出
        if any(k in prompt for k in ["医療","救急","医師","地域医療"]):
            ctx["domain"]="地域医療"; kpis=KPI_CATALOG["地域医療"]
            msg=f"**KPI候補**: {', '.join(kpis)}\n\n目標しきい値（例: *救急受入率 80%*）を入力してください。"
            log_message("assistant", msg)
            with st.chat_message("assistant"):
                st.markdown(msg)
            responded=True
        m_thr = re.search(r"(?:目標|しきい値|threshold).{0,6}?(\d{2,3})\s*[%％]?", prompt)
        if m_thr:
            ctx["thr"]=float(m_thr.group(1)); msg=f"目標しきい値を **{m_thr.group(1)}** に設定。KPI名も指定してください。"
            log_message("assistant", msg)
            with st.chat_message("assistant"):
                st.markdown(msg)
            responded=True
        m_kpi = re.search(r"(救急受入率|医師1人あたり患者数|初診待機日数)", prompt)
        if m_kpi:
            kpi_name = m_kpi.group(1)
            set_kpi_targets(ctx, [kpi_name])
            msg=f"KPIを **{kpi_name}** に設定。予算額（億円）を入力してください。"
            log_message("assistant", msg)
            with st.chat_message("assistant"):
                st.markdown(msg)
            responded=True
        m_budget = re.search(r"(\d+(?:\.\d+)?)\s*億", prompt)
        if m_budget:
            ctx["budget"]=float(m_budget.group(1)); pessim, optim = 0.2, 0.2
            df = pd.DataFrame([{"施策": n, "コスト(億円)": info["cost"], "効果(中位)": info["effect"],
                                "効果(悲観)": round(info["effect"]*(1-pessim),1), "効果(楽観)": round(info["effect"]*(1+optim),1),
                                "ラグ(年)": info["lag"], "リスク": info["risk"]} for n,info in POLICY_LIBRARY.items()])
            ctx["candidates"]=df
            msg=f"予算 **{m_budget.group(1)}億円** で候補施策を提示。"; log_message("assistant", msg);
            with st.chat_message("assistant"):
                st.markdown(msg)
                st.plotly_chart(bubble_chart(df), use_container_width=True)
                st.caption("前提: コスト/効果は仮のヒューリスティック値です。編集セクションで調整できます。")
            responded=True
        if any(k in prompt for k in ["採択","配分","最適","選んで","決めて"]):
            picked, cost_sum, eff_sum = greedy_allocation(ctx.get("candidates"), ctx.get("budget") or 0.0)
            ctx["picked"]=picked; msg=f"**採択結果**（貪欲法）\n- コスト合計: **{cost_sum:.1f}億円**\n- 効果(中位)合計: **{eff_sum:.1f}**"
            log_message("assistant", msg, {"picked_count": int(len(picked))})
            with st.chat_message("assistant"): st.markdown(msg); st.dataframe(picked, use_container_width=True); responded=True
        if any(k in prompt for k in ["将来","推移","レンジ","効果","シナリオ","グラフ","可視化"]):
            years=ctx["years"]; base_start=ctx["base_start"]; drift=ctx["drift"]; thr=ctx["thr"]; kpi=get_primary_kpi_name(ctx) or "KPI"
            picked=ctx["picked"] or pd.DataFrame(columns=["効果(中位)","効果(悲観)","効果(楽観)","ラグ(年)"])
            lag_profile={0:0.0,1:0.5,2:0.8,3:1.0}
            pol_mid =[{"effect":r["効果(中位)"],"lag":int(r["ラグ(年)"])} for _,r in picked.iterrows()]
            pol_low =[{"effect":r["効果(悲観)"],"lag":int(r["ラグ(年)"])} for _,r in picked.iterrows()]
            pol_high=[{"effect":r["効果(楽観)"],"lag":int(r["ラグ(年)"])} for _,r in picked.iterrows()]
            base = simulate_kpi(years, base_start, drift, [], lag_profile)
            mid  = simulate_kpi(years, base_start, drift, pol_mid, lag_profile)
            low  = simulate_kpi(years, base_start, drift, pol_low, lag_profile)
            high = simulate_kpi(years, base_start, drift, pol_high, lag_profile)
            primary_constraint = None
            for c in ctx.get("kpi_constraints", []) or []:
                if c.get("name") == kpi:
                    primary_constraint = c
                    break
            threshold_lines = None
            if primary_constraint:
                thr_value = _to_float(primary_constraint.get("threshold_hint"))
                if thr_value is not None:
                    label_prefix = "≧" if (primary_constraint.get("threshold_type") or "min") == "min" else "≦"
                    unit = primary_constraint.get("unit") or ""
                    threshold_lines = [{"value": thr_value, "label": f"{kpi} {label_prefix} {thr_value}{unit}"}]
                    thr_display = None
                else:
                    thr_display = thr
            else:
                thr_display = thr
            if not primary_constraint:
                primary_constraint = None
            alerts = threshold_breaches(years, low, high, primary_constraint) if primary_constraint else []
            msg = "**KPI予測レンジ**と**ロジックモデル**を表示。"; log_message("assistant", msg)
            with st.chat_message("assistant"):
                st.markdown(msg)
                st.plotly_chart(band_chart(years, mid, low, high, thr_display, kpi, threshold_lines), use_container_width=True)
                st.caption("前提: 効果値は選択施策の仮定値を合算。閾値線を下回ると警告が表示されます。")
                if alerts:
                    for al in alerts:
                        st.warning(al)
                st.plotly_chart(logic_model_figure(), use_container_width=True)
                responded=True
        if any(k in prompt for k in ["実績","CSV","重ね","レビュー","モニタ"]):
            msg = "CSV（列: `Year,Value`）をアップロードしてください。"; log_message("assistant", msg)
            with st.chat_message("assistant"): st.markdown(msg)
            uploaded=st.file_uploader("実績CSV", type=["csv"], key=f"real_csv_{uuid.uuid4()}")
            if uploaded:
                df_real=pd.read_csv(uploaded)
                if {"Year","Value"}.issubset(df_real.columns):
                    years=ctx["years"]; kpi=get_primary_kpi_name(ctx) or "KPI"; thr=ctx["thr"]; picked=ctx["picked"] or pd.DataFrame(columns=["効果(中位)","ラグ(年)"])
                    lag_profile={0:0.0,1:0.5,2:0.8,3:1.0}
                    pol_mid=[{"effect":r["効果(中位)"],"lag":int(r["ラグ(年)"])} for _,r in picked.iterrows()]
                    mid=simulate_kpi(years, ctx["base_start"], ctx["drift"], pol_mid, lag_profile)
                    merged_years=sorted(set(years).union(set(df_real["Year"].tolist())))
                    mid_series=pd.Series(mid, index=years).reindex(merged_years).interpolate()
                    fig=go.Figure(); fig.add_trace(go.Scatter(x=merged_years, y=mid_series, name="予測(中位)", mode="lines"))
                    fig.add_trace(go.Scatter(x=df_real["Year"], y=df_real["Value"], name="実績", mode="lines+markers"))
                    threshold_lines = None
                    primary_constraint = None
                    for c in ctx.get("kpi_constraints", []) or []:
                        if c.get("name") == kpi:
                            primary_constraint = c
                            break
                    if primary_constraint:
                        thr_value = _to_float(primary_constraint.get("threshold_hint"))
                        if thr_value is not None:
                            label_prefix = "≧" if (primary_constraint.get("threshold_type") or "min") == "min" else "≦"
                            unit = primary_constraint.get("unit") or ""
                            threshold_lines = [{"value": thr_value, "label": f"{kpi} {label_prefix} {thr_value}{unit}"}]
                    if threshold_lines:
                        for th in threshold_lines:
                            fig.add_hline(y=th["value"], line_dash="dash", annotation_text=th["label"])
                    elif thr is not None:
                        fig.add_hline(y=thr, line_dash="dash", annotation_text=f"Threshold={thr}")
                    with st.chat_message("assistant"): st.plotly_chart(fig, use_container_width=True)
                    latest=int(df_real["Year"].max()); pred=float(mid_series.loc[latest]); real=float(df_real.loc[df_real["Year"]==latest,"Value"].iloc[0]); diff=real-pred
                    msg2=f"最新年の乖離（実績-予測）: **{diff:+.2f}**"
                    log_message("assistant", msg2)
                    with st.chat_message("assistant"):
                        st.markdown(msg2)
                else:
                    with st.chat_message("assistant"): st.error("列名が不正です。'Year','Value' を含むCSVを指定してください。")
        special_msgs = handle_special_actions(prompt, ctx)
        for smsg in special_msgs:
            with st.chat_message("assistant"):
                st.markdown(smsg)
            responded = True
        web_cmd=None
        if prompt.strip().startswith("検索:"): web_cmd = prompt.strip().split("検索:",1)[1].strip()
        elif prompt.strip().startswith("/web"): web_cmd = prompt.strip().split("/web",1)[1].strip()
        elif st.session_state.use_web_search and any(k in prompt for k in ["調べて","検索","最新","事例","論文","データ"]):
            web_cmd = prompt
        if web_cmd:
            with st.chat_message("assistant"): st.markdown(f"オンライン検索: **{web_cmd}** を実行中…")
            results = web_search_tavily(web_cmd, max_results=5); summary = summarize_with_citations(web_cmd, results)
            log_message("assistant", summary)
            with st.chat_message("assistant"):
                st.markdown(summary)

    # エージェント結果表示
    agents = st.session_state.context.get("agents")
    if st.session_state.show_agent_blocks and agents:
        st.markdown("---"); st.subheader("🔎 エージェント結果")
        if "error" in agents: st.error(agents["error"])
        else:
            c1,c2,c3=st.columns(3)
            with c1:
                st.markdown("**政策検索**")
                pol=(agents.get("policy_search",{}) or {}).get("results",[])
                st.dataframe(pd.DataFrame(pol)[["rank","title","url"]].head(10) if pol else pd.DataFrame(), use_container_width=True)
            with c2:
                st.markdown("**論文検索**")
                pap=(agents.get("paper_search",{}) or {}).get("results",[])
                df=pd.DataFrame(pap); cols=[c for c in ["rank","title","url","evidence","note"] if c in df.columns]
                st.dataframe(df[cols].head(10) if not df.empty else pd.DataFrame(), use_container_width=True)
            with c3:
                all_kpis=agents.get("kpis_all",[]); st.metric("KPI数", len(all_kpis))
                if all_kpis:
                    st.write(", ".join(sorted({str(x.get("name","")) for x in all_kpis})[:30]))

# ===== 文書抽出 =====
with tab_docs:
    st.subheader("📄 レビューシート抽出（PDF → 概要・因果パス）")
    ctx = st.session_state.context
    c1, c2 = st.columns(2)
    with c1:
        rs_file = st.file_uploader("RSフォーマット（レビューシート解説などの基準PDF）", type=["pdf"], key="rs_pdf")
    with c2:
        target_pdf = st.file_uploader("対象レビューシート（または政策資料）PDF", type=["pdf"], key="rs_target_pdf")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("① RSプリマ生成"):
            if rs_file:
                tmp = Path(st.secrets.get("tmp_dir",".")) / f"rs_{uuid.uuid4().hex}.pdf"
                tmp.write_bytes(rs_file.read())
                with st.spinner("プリマ生成…"):
                    primer = build_rs_primer(str(tmp)) if _HAS_AGENTS else None
                ctx["primer"] = primer; st.success("生成しました"); st.json(primer or {})
            else:
                st.warning("RSフォーマットPDFを指定してください。")
    with colB:
        if st.button("② レビューシート抽出（Map/Reduce）"):
            if target_pdf:
                tmp = Path(st.secrets.get("tmp_dir",".")) / f"doc_{uuid.uuid4().hex}.pdf"
                tmp.write_bytes(target_pdf.read())
                with st.spinner("抽出中…"):
                    rs = extract_effect_pathway(str(tmp)) if _HAS_AGENTS else None
                ctx["rs_result"] = rs; st.success("抽出しました"); st.json(rs or {})
            else:
                st.warning("対象PDFを指定してください。")

    st.markdown("---")
    st.subheader("🧪 KPIシード（断片→A/B分類）とコーパスPDFでのKPI増補")
    frag = st.text_area("KPI断片（複数を改行で）", height=120, key="kpi_frag")
    add_pdf = st.file_uploader("KPI増補用の追加PDF（複数可）", type=["pdf"], accept_multiple_files=True, key="kpi_add_pdfs")
    cA, cB = st.columns(2)
    with cA:
        if st.button("③ 断片からKPIシード生成"):
            if frag.strip():
                fragments = [s for s in frag.split("\n") if s.strip()]
                with st.spinner("KPIシード抽出…"):
                    seed = seed_kpis_from_fragments(fragments) if _HAS_AGENTS else None
                ctx["kpi_seed_fragments"]=fragments; ctx["kpi_catalog"]=seed
                st.success("抽出しました"); st.json(seed or {})
            else:
                st.warning("断片を入力してください。")
    with cB:
        if st.button("④ 追加PDFでKPI増補/マージ"):
            if ctx.get("kpi_catalog") and add_pdf:
                paths=[]
                for f in add_pdf:
                    p = Path(st.secrets.get("tmp_dir",".")) / f"add_{uuid.uuid4().hex}.pdf"
                    p.write_bytes(f.read()); paths.append(str(p))
                with st.spinner("増補中…"):
                    updated = update_kpis_with_pdfs(ctx["kpi_catalog"], paths) if _HAS_AGENTS else None
                ctx["kpi_catalog"]=updated; st.success("更新しました"); st.json(updated or {})
            else:
                st.warning("KPIシード生成と追加PDFの両方が必要です。")

# ===== 時系列/因果 =====
with tab_ts:
    st.subheader("📈 時系列収集（Web）→ 🔁 因果推定（TE / VAR–Granger）")
    ctx = st.session_state.context
    if st.button("⑤ 効果発現経路（エッジ）を収集"):
        rs = ctx.get("rs_result") or {}
        models = [rs] if rs else []
        ctx["effect_models"]=models
        st.json({"edges": collect_edges(models)})
    kpi_names = []
    catalog_obj = ctx.get("kpi_catalog")
    if isinstance(catalog_obj, dict):
        kpi_names += [x.get("name", "") for x in catalog_obj.get("quantitative_kpi", [])]
        kpi_names += [x.get("name", "") for x in catalog_obj.get("hard_to_quantify_kpi", [])]
    elif isinstance(catalog_obj, list):
        kpi_names += [x.get("name", "") for x in catalog_obj if isinstance(x, dict)]
    kpi_names = sorted({n for n in kpi_names if n})
    st.write(f"KPI候補数: {len(kpi_names)}")
    if st.button("⑥ KPI名リストを表示"):
        st.write(", ".join(kpi_names[:50]))

    labels_input = st.text_area("ノード（指標）名（改行区切り、空なら自動推定不可）", height=120, key="ts_labels")
    if st.button("⑦ 指標ごとにWebから時系列を収集（最大10点）"):
        labels = [s.strip() for s in labels_input.split("\n") if s.strip()]
        if not labels:
            st.warning("ノード名を入力してください。")
        else:
            with st.spinner("収集中…やや時間がかかります"):
                per_node_ts = build_kpi_timeseries(labels) if _HAS_AGENTS else {}
            ctx["per_node_timeseries"]=per_node_ts
            for lab, ts in per_node_ts.items():
                st.write(f"**{lab}**"); st.dataframe(pd.DataFrame(ts), use_container_width=True)

    if st.button("⑧ Transfer Entropy / VAR–Granger を実行"):
        models = ctx.get("effect_models") or []
        edges = collect_edges(models)
        ts = ctx.get("per_node_timeseries") or {}
        if not edges:
            st.warning("エッジがありません（文書抽出→⑤を実行してください）。")
        elif not ts:
            st.warning("時系列がありません（⑦を実行してください）。")
        else:
            with st.spinner("因果推定…（数分かかることがあります）"):
                res = run_causality(ts, edges) if _HAS_AGENTS else {}
            ctx["causality"]=res
            st.success("完了")
            st.subheader("Transfer Entropy")
            st.dataframe(pd.json_normalize(res.get("transfer_entropy", [])), use_container_width=True)
            st.subheader("VAR–Granger")
            st.dataframe(pd.DataFrame(res.get("granger", [])), use_container_width=True)

# ===== 現在の会話ログ =====
with tab_current:
    st.subheader("現在の会話ログ")
    if st.session_state.messages:
        safe = [normalize_message(m) for m in st.session_state.messages]
        df_log = pd.DataFrame(safe)[["t","role","content"]]
        df_log["content_preview"] = df_log["content"].apply(lambda s: (s.replace("\n"," ")[:200]+"…") if len(s)>200 else s)
        st.dataframe(df_log, use_container_width=True); st.caption("全文は Chat タブで確認。Export でJSON保存可。")
    else:
        st.info("まだメッセージがありません。")

# ===== セッション履歴 =====
with tab_sessions:
    st.subheader("セッション履歴")
    if st.session_state.sessions:
        for i, ses in enumerate(reversed(st.session_state.sessions), start=1):
            with st.expander(f"[{i}] {ses['started_at']} - {ses['ended_at']}  / msgs={len(ses['messages'])}"):
                ctx = ses.get("context", {})
                kpi_labels = ", ".join(get_kpi_targets(ctx) or ([ctx.get('kpi')] if ctx.get('kpi') else [])) or "未設定"
                st.write(f"- domain: {ctx.get('domain')}, kpi: {kpi_labels}, thr: {ctx.get('thr')}, budget: {ctx.get('budget')}")
                st.write(f"- agents: {'あり' if ctx.get('agents') else 'なし'} / RS抽出: {'あり' if ctx.get('rs_result') else 'なし'} / KPIカタログ: {'あり' if ctx.get('kpi_catalog') else 'なし'}")
                for m in ses["messages"]:
                    mm = normalize_message(m); st.markdown(f"**{mm['t']} [{mm['role']}]**  \n{mm['content']}")
                if st.button("このセッションを復元", key=f"restore_{ses['id']}"):
                    st.session_state.current_session_id = ses["id"]
                    st.session_state.messages = [normalize_message(m) for m in ses["messages"]]
                    st.session_state.context  = ses.get("context", st.session_state.context)
                    st.rerun()
    else:
        st.info("保存済みのセッションはありません。")

# ===== 使い方 =====
with tab_help:
    st.subheader("使い方（要点）")
    st.markdown("""
- **普通に話す** → OpenAIで応答（API Key要）。
- **オンライン検索** → `検索: ...` or `/web ...`。サイドバーで「オンライン検索を有効化」。
- **エージェント** → `/agent ...` またはサイドバーの実行ボタン。
- **📄 文書抽出** → RS基準PDF→プリマ生成→対象PDFから 概要/因果パス（Map/Reduce）。
- **KPI** → 断片からA/B分類でシード→追加PDFで増補/マージ。
- **📈 時系列/因果** → ノード名を入力→Webから最大10点抽出→TE/VAR–Granger。
""")

with tab_orch:
    col_logo, col_title = st.columns([1, 2])
    with col_logo:
        logo_path = Path("static/polaris_logo.png")
        if logo_path.exists():
            st.image(str(logo_path), use_column_width=True)
        else:
            st.info("static/polaris_logo.png を配置すると POLARIS ロゴが表示されます。")
    with col_title:
        st.title("🌟 POLARIS")
        st.markdown("**Policy Optimization through Linked Analysis of Reliable Indicators & Statistics**")
        st.caption("詳細化 → 検索 → 複数案 → 批判 → 予算可視化")
    ctx = st.session_state.context
    if "complex" not in ctx:
        ctx["complex"] = {
            "refined": None,
            "confirmed": False,
            "workplan": None,
            "topic_map": None,
            "topic_layer_hits": None,
            "policy_hypotheses": None,
            "kpi_templates": [],
            "kpi_constraints": [],
            "search_results": None,
            "strategies": [],
            "critique": None,
            "budgets": [],
            "user_query": None,
            "qual_entries": [],
            "hypothesis_clusters": [],
            "policy_options": [],
            "scenario_configs": [],
            "scenario_results": {},
            "risk_register": [],
            "evidence_gaps": [],
            "contention_points": [],
            "hypothesis_gap_analysis": [],
            "agent_competition": {},
            "decision_notes": "",
        }

    # --- Step 1: POLARISチャットフロー ---
    st.subheader("1) 政策課題の詳細化（POLARISチャット）")
    if st.button("POLARISフローをリセット", key="btn_reset_polaris_flow"):
        ctx.pop("polaris_flow", None)
        ctx["complex"].update({
            "refined": None,
            "confirmed": False,
            "workplan": None,
            "topic_map": None,
            "topic_layer_hits": None,
            "policy_hypotheses": None,
            "kpi_templates": [],
            "kpi_constraints": [],
            "search_results": None,
            "strategies": [],
            "critique": None,
            "budgets": [],
            "user_query": None,
            "qual_entries": [],
            "hypothesis_clusters": [],
            "policy_options": [],
            "scenario_configs": [],
            "scenario_results": {},
            "risk_register": [],
            "evidence_gaps": [],
            "contention_points": [],
            "hypothesis_gap_analysis": [],
            "agent_competition": {},
            "decision_notes": "",
        })
        st.rerun()
    pf = get_polaris_flow(ctx)
    if not pf["messages"]:
        polaris_log(ctx, "assistant", "POLARISです。政策課題について論じてください。背景・現状・気になっている指標などを自由にどうぞ。")
    advance_polaris_flow(ctx)
    render_polaris_chat(ctx)
    stage = pf.get("stage", "ask_problem")
    if stage == "ask_problem":
        user_reply = st.chat_input("政策課題について論じてください", key="polaris_problem_input")
        if user_reply:
            pf["problem"] = user_reply
            ctx["complex"]["user_query"] = user_reply
            polaris_log(ctx, "user", user_reply)
            polaris_log(ctx, "assistant", "ありがとうございます。詳細化（Step1）に取り掛かります。")
            pf["stage"] = "refine"
            st.rerun()
    elif stage == "kpi_confirm":
        st.info("テーブルでKPIと制約条件を調整後、『KPI設定完了』を押してください。")
        if st.button("KPI設定完了", key="btn_polaris_kpi_done"):
            polaris_log(ctx, "assistant", "質的データから原因仮説を生成・整理します。ファイルを入力してください。")
            pf["stage"] = "qual_prompt"
            st.rerun()
    elif stage == "qual_prompt":
        st.info("質的データから原因仮説を生成・整理します。txt/csv/mdファイルをアップロードするか、ダミーデータを使用してください。")
        qual_files = st.file_uploader("質的データファイル（複数可）", type=["txt", "csv", "md"], accept_multiple_files=True, key="polaris_qual_files")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("アップロードしたファイルを解析", key="btn_polaris_qual_upload"):
                entries = load_qualitative_entries(qual_files)
                if not entries:
                    st.warning("ファイルが指定されていません。")
                else:
                    pf["qual_entries"] = entries
                    polaris_log(ctx, "assistant", "入力しました。質的データを解析します。")
                    pf["stage"] = "qual_analyze"
                    st.rerun()
        with col_q2:
            if st.button("ダミーデータを解析", key="btn_polaris_qual_dummy"):
                pf["qual_entries"] = [dict(sample) for sample in QUAL_SAMPLE_TEXTS]
                polaris_log(ctx, "assistant", "入力しました（今回はダミーデータ）。質的データを解析します。")
                pf["stage"] = "qual_analyze"
                st.rerun()
    elif stage == "policy_confirm":
        st.info("この方針で進めてよろしいですか？")
        choice = st.radio("POLARISの提案方針に同意しますか？", ["選択してください", "はい", "いいえ"], index=0, key="polaris_policy_choice")
        if st.button("回答する", key="btn_polaris_policy_confirm"):
            if choice == "はい":
                polaris_log(ctx, "assistant", "了解しました。タスクを分解させ、自動で検索→複数案の比較に進みます。")
                pf["stage"] = "decompose_run"
                st.rerun()
            elif choice == "いいえ":
                polaris_log(ctx, "assistant", "承知しました。質的データから再度整理しますので、必要なファイルを入力してください。")
                pf["stage"] = "qual_prompt"
                st.rerun()
            else:
                st.warning("「はい」または「いいえ」を選択してください。")
    elif stage in {"decompose_run", "search_run", "strategy_run", "critique_run", "simulation_run", "risk_run", "budget_run"}:
        st.info("各ステップを自動実行中です。進行状況はチャットログをご確認ください。")
    elif stage == "done":
        st.success("POLARIS フローが完了しました。必要に応じて下部の各セクションで詳細を編集してください。")

    if ctx["complex"].get("refined"):
        with st.expander("詳細化した内容（JSON）", expanded=False):
            st.json(ctx["complex"]["refined"])

    # --- Step 1-b: KPI設定と制約 ---
    st.markdown("### 1-b) KPI設定と制約条件の明確化")
    kpi_domain_hint = st.text_input("主となる政策ドメイン (例: 地域医療)", value=ctx.get("domain") or "", key="kpi_domain_hint")
    col_kpi1, col_kpi2 = st.columns([2,1])
    with col_kpi1:
        st.caption("課題の文脈から KPI 候補と想定閾値を自動提案。閾値種別: min=最低保証, max=上限。")
    with col_kpi2:
        auto_fill = st.checkbox("ドメインに応じて自動入力", value=True, key="kpi_auto_fill")

    if st.button("KPI候補を提案", key="btn_kpi_templates"):
        base_query = ctx["complex"].get("refined") or {}
        refined_problem = base_query.get("refined_problem") if isinstance(base_query, dict) else {}
        user_query_text = (
            (refined_problem or {}).get("title")
            or ctx["complex"].get("user_query")
            or user_seed
        )
        if not user_query_text:
            st.warning("課題を入力してからKPIを生成してください。")
        else:
            with st.spinner("KPIテンプレートを生成しています…"):
                if _HAS_OPENAI and OPENAI_API_KEY:
                    agent = KPITemplateAgent()
                    resp = agent.run(
                        user_query=user_query_text,
                        domain_hint=kpi_domain_hint or ctx.get("domain"),
                        catalog_examples=KPI_CATALOG.get(kpi_domain_hint, []) if auto_fill else [],
                    )
                    templates = resp.get("kpi_candidates", [])
                else:
                    templates = [dict(item) for item in DEFAULT_KPI_TEMPLATES]
            ctx["complex"]["kpi_templates"] = templates
            ctx["complex"]["kpi_constraints"] = templates
            ctx["kpi_constraints"] = templates
            ctx["kpi_catalog"] = templates
            if templates:
                first_name = templates[0].get("name")
                set_kpi_targets(ctx, [first_name] if first_name else [])
                ctx["thr"] = templates[0].get("threshold_hint")
                ctx["kpi_threshold_type"] = templates[0].get("threshold_type", "min")
            st.success("KPI候補を更新しました。下表で閾値や定義を調整してください。")

    current_constraints = ctx["complex"].get("kpi_constraints") or ctx.get("kpi_constraints") or []
    if current_constraints:
        df_constraints = pd.DataFrame(current_constraints)
        column_config = {
            "name": st.column_config.TextColumn("KPI名", required=True),
            "definition": st.column_config.TextColumn("定義"),
            "unit": st.column_config.TextColumn("単位"),
            "direction": st.column_config.SelectboxColumn("指標方向", options=["up", "down"], help="up=値が大きいほど良い"),
            "threshold_type": st.column_config.SelectboxColumn("閾値種別", options=["min", "max"], help="min=下限を守る, max=上限を超えない"),
            "threshold_hint": st.column_config.NumberColumn("閾値", format="%.2f"),
            "data_source": st.column_config.TextColumn("データソース"),
            "legal_floor": st.column_config.TextColumn("法令・基準"),
            "rationale": st.column_config.TextColumn("根拠/出典")
        }
        edited_df = st.data_editor(
            df_constraints,
            num_rows="dynamic",
            use_container_width=True,
            column_config=column_config,
            key="kpi_constraints_editor"
        )
        primary_options = [str(r.get("name", "")) for _, r in edited_df.iterrows() if str(r.get("name", ""))]
        default_targets = [t for t in get_kpi_targets(ctx) if t in primary_options]
        if not default_targets and primary_options:
            default_targets = [primary_options[0]]
        selected_targets = st.multiselect(
            "シミュレーション対象の主要KPI（複数選択可）",
            primary_options,
            default=default_targets,
            key="primary_kpi_select"
        )
        if st.button("KPI設定を保存", key="btn_save_kpi_constraints"):
            records = edited_df.fillna("").to_dict("records")
            ctx["complex"]["kpi_constraints"] = records
            ctx["kpi_constraints"] = records
            ctx["complex"]["kpi_templates"] = ctx["complex"].get("kpi_templates") or records
            if records:
                targets_to_set = selected_targets or [records[0].get("name")]
            else:
                targets_to_set = selected_targets
            set_kpi_targets(ctx, targets_to_set or [])
            primary_name = get_primary_kpi_name(ctx)
            primary_constraint = next((r for r in records if r.get("name") == primary_name), records[0] if records else None)
            if primary_constraint:
                try:
                    ctx["thr"] = float(primary_constraint.get("threshold_hint"))
                except (TypeError, ValueError):
                    ctx["thr"] = None
                ctx["kpi_threshold_type"] = primary_constraint.get("threshold_type", "min")
            st.success("KPI制約を保存しました。")
        with st.expander("KPIコメント/合意状況", expanded=False):
            kpi_comment = st.text_area("コメント", value=ctx["complex"].get("stage_notes", {}).get("KPI_COMMENT", ""), key="kpi_comment_box")
            if st.button("コメント保存", key="btn_save_kpi_comment"):
                ctx["complex"].setdefault("stage_notes", {})["KPI_COMMENT"] = kpi_comment
                st.success("KPIコメントを保存しました。")
    else:
        st.info("KPI候補がまだありません。『KPI候補を提案』を押すか、手動で表に追加してください。")

    # --- Step 2: 質的データから仮説を抽出 ---
    st.markdown("---")
    st.subheader("2) 質的データから原因仮説を生成・整理")
    st.caption("ヒアリング・議事録・SNSなどのテキストを解析し、KJ法的にグルーピング")
    qual_files = st.file_uploader("質的データファイル（txt/csv, 複数可）", type=["txt","csv","md"], accept_multiple_files=True, key="qual_files")

    if st.button("質的データを解析", key="btn_analyze_qual"):
        entries = load_qualitative_entries(qual_files)
        used_sample = False
        if not entries:
            entries = [dict(sample) for sample in QUAL_SAMPLE_TEXTS]
            used_sample = True
        if not entries:
            st.warning("質的データを1件以上指定してください。")
        else:
            rows = analyze_qualitative(entries)
            pivot, gaps = summarize_hypotheses(rows)
            ctx["complex"]["qual_entries"] = entries
            ctx["complex"]["hypothesis_clusters"] = rows
            ctx["complex"]["evidence_gaps"] = [g["issue"] if isinstance(g, dict) else str(g) for g in gaps]
            ctx["complex"]["contention_points"] = find_contention_points(rows)
            st.success(f"{len(rows)}件の断片から仮説を抽出しました。グルーピングを編集してください。")
            if used_sample:
                st.info("アップロードがなかったため、公表資料タイトル風のサンプルデータで解析しました。")
            if not pivot.empty:
                st.dataframe(pivot, use_container_width=True)

    hypothesis_rows = ctx["complex"].get("hypothesis_clusters", [])
    if hypothesis_rows:
        df_hyp = pd.DataFrame(hypothesis_rows)
        edited_hyp = st.data_editor(
            df_hyp,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "quote_id": st.column_config.TextColumn("ID", disabled=True),
                "source": st.column_config.TextColumn("ソース"),
                "quote": st.column_config.TextColumn("引用"),
                "cause": st.column_config.TextColumn("自動推定原因", disabled=True),
                "cluster_lv1": st.column_config.TextColumn("中項目"),
                "cluster_lv2": st.column_config.TextColumn("小項目"),
                "importance": st.column_config.NumberColumn("重要度", format="%d"),
                "evidence_link": st.column_config.TextColumn("エビデンスリンク")
            },
            key="kj_editor"
        )
        if st.button("仮説クラスタを保存", key="btn_save_hypothesis"):
            records = edited_hyp.to_dict("records")
            ctx["complex"]["hypothesis_clusters"] = records
            ctx["complex"]["contention_points"] = find_contention_points(records)
            ctx["complex"]["evidence_gaps"] = find_evidence_gaps(records)
            st.success("仮説クラスタを保存しました。")
        if st.button("仮説の補強データを提案", key="btn_gap_agent"):
            rows = ctx["complex"].get("hypothesis_clusters") or edited_hyp.to_dict("records")
            if not rows:
                st.warning("仮説データがありません。")
            else:
                try:
                    if _HAS_OPENAI and OPENAI_API_KEY:
                        gap_agent = HypothesisGapAgent()
                        resp = gap_agent.run(rows)
                    else:
                        raise RuntimeError("no-openai")
                except Exception:
                    resp = {"gap_analysis": fallback_gap_analysis(rows)}
                ctx["complex"]["hypothesis_gap_analysis"] = resp.get("gap_analysis", [])
                st.success("仮説の補強に必要なデータを整理しました。")
        st.markdown("**対立点 / 重要論点**")
        for item in ctx["complex"].get("contention_points", []):
            st.info(item)
        if ctx["complex"].get("evidence_gaps"):
            with st.expander("エビデンス不足の指摘"):
                for gap in ctx["complex"]["evidence_gaps"]:
                    st.warning(gap)
        if ctx["complex"].get("hypothesis_gap_analysis"):
            st.markdown("**補強に必要なデータ提案**")
            for entry in ctx["complex"]["hypothesis_gap_analysis"]:
                hypothesis = entry.get("hypothesis", "仮説")
                concern = entry.get("concern", "")
                needed = ", ".join(entry.get("needed_data", []))
                priority = entry.get("priority", "medium")
                st.warning(f"[{priority}] {hypothesis}: {concern} → 推奨データ: {needed}")
        with st.expander("仮説に関するコメント", expanded=False):
            hyp_comment = st.text_area("合意状況/懸念", value=ctx["complex"].get("stage_notes", {}).get("HYP_COMMENT", ""), key="hyp_comment_box")
            if st.button("仮説コメントを保存", key="btn_save_hyp_comment"):
                ctx["complex"].setdefault("stage_notes", {})["HYP_COMMENT"] = hyp_comment
                st.success("仮説コメントを保存しました。")
    else:
        st.info("質的データの解析結果がまだありません。")

    # --- Step 3: 施策オプション列挙・比較 ---
    st.markdown("---")
    st.subheader("3) 原因別の施策オプションを列挙・比較")
    if st.button("原因仮説から施策案を生成", key="btn_generate_policies"):
        options = generate_policy_options_from_hypotheses(ctx["complex"].get("hypothesis_clusters") or [])
        if not options:
            options = [
                {"施策": n, "原因カテゴリ": "参考", "コスト(億円)": info["cost"], "効果(中位)": info["effect"],
                 "効果(悲観)": round(info["effect"]*0.7,1), "効果(楽観)": round(info["effect"]*1.2,1),
                 "ラグ(年)": info["lag"], "スタッフ需要": 5, "KPI紐付け": get_primary_kpi_name(ctx),
                 "リスク": info["risk"], "根拠": "ライブラリ"}
                for n, info in POLICY_LIBRARY.items()
            ]
        ctx["complex"]["policy_options"] = options
        st.success(f"{len(options)}件の施策案を取得しました。")

    policy_options = ctx["complex"].get("policy_options") or []
    if policy_options:
        df_options = pd.DataFrame(policy_options)
        edited_options = st.data_editor(
            df_options,
            num_rows="dynamic",
            use_container_width=True,
            key="policy_options_editor"
        )
        ctx["complex"]["policy_options"] = edited_options.to_dict("records")
        st.plotly_chart(bubble_chart(edited_options.rename(columns={"効果(中位)": "効果(中位)", "コスト(億円)": "コスト(億円)"})), use_container_width=True)
        st.caption("前提: 効果=KPI改善スコア、コスト=年間予算。データエディタで各値を更新してください。")
        selected_names = st.multiselect("比較する施策", edited_options["施策"].tolist(), default=edited_options["施策"].tolist()[:2], key="policy_select")
        if selected_names:
            subset = edited_options[edited_options["施策"].isin(selected_names)]
            st.dataframe(subset, use_container_width=True)
            cost_sum = subset["コスト(億円)"].sum()
            eff_sum = subset["効果(中位)"].sum()
            st.info(f"選択施策の合計: コスト {cost_sum:.1f} 億円 / 想定効果 {eff_sum:.1f}")
        with st.expander("施策比較メモ", expanded=False):
            plan_comment = st.text_area("メモ", value=ctx["complex"].get("stage_notes", {}).get("PLAN_COMMENT", ""), key="plan_comment_box")
            if st.button("施策メモを保存", key="btn_save_plan_comment"):
                ctx["complex"].setdefault("stage_notes", {})["PLAN_COMMENT"] = plan_comment
                st.success("施策メモを保存しました。")
    else:
        st.info("施策オプションはまだ生成されていません。")

    st.markdown("---")
    st.subheader("Agentコンペ: 複数シンセサイザの提案を比較")
    comp_ctx = ctx["complex"].get("agent_competition") or {}
    if st.button("コンペを実行", key="btn_agent_competition"):
        if not (ctx["complex"].get("refined") and ctx["complex"].get("workplan") and ctx["complex"].get("search_results")):
            st.warning("詳細化・分解・検索を完了させてください。")
        else:
            manager = AgentCompetitionManager()
            refined_payload = ctx["complex"].get("refined") or {}
            work_payload = ctx["complex"].get("workplan") or {}
            search_payload = ctx["complex"].get("search_results") or {}
            kpi_cands = refined_payload.get("refined_problem", {}).get("kpi_candidates", [])
            comp_ctx = manager.run(refined_payload, work_payload, search_payload, kpi_cands, DEFAULT_CONTESTANTS)
            ctx["complex"]["agent_competition"] = comp_ctx
            st.success("コンペ結果を更新しました。")
    if comp_ctx.get("entries"):
        for entry in comp_ctx["entries"]:
            st.markdown(f"**{entry['contestant']} ({entry['theme']})**")
            st.json(entry["strategy"], expanded=False)
        if comp_ctx.get("critique"):
            st.markdown("**Critique Agent コメント**")
            st.json(comp_ctx["critique"], expanded=False)
    else:
        st.info("まだコンペ結果がありません。")

    # --- Step 4 & 5: 分解→検索→複数案合成（既存） ---
    st.markdown("---")
    st.subheader("4) 作業分解→検索→解決案 (単一)")
    st.caption("5) テーマを変えつつ、上記を繰り返して複数案を生成します。")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("分解を生成"):
            if not ctx["complex"]["refined"] or not ctx["complex"]["confirmed"]:
                st.warning("まず詳細化し、yで確認してください。")
            else:
                with st.spinner("分解中…"):
                    ctx["complex"]["workplan"] = decompose_work(ctx["complex"]["refined"].get("refined_problem", ctx["complex"]["refined"]))
                with st.spinner("論点洗い出し中…"):
                    ctx["complex"]["topic_map"] = explore_topics(
                        ctx["complex"].get("user_query") or "",
                        ctx["complex"].get("refined") or {},
                        ctx["complex"]["workplan"],
                    )
                st.success("分解と論点マップ生成が完了しました。")
    with colB:
        if st.button("分解に基づき検索を実行"):
            if not ctx["complex"]["workplan"]:
                st.warning("先に『分解を生成』してください。")
            else:
                with st.spinner("検索中…"):
                    ctx["complex"]["search_results"] = run_searches(
                        ctx["complex"]["workplan"],
                        prefer_rs_system=st.session_state.restrict_rs,
                        user_query=ctx["complex"].get("user_query"),
                        refined_problem=ctx["complex"].get("refined"),
                        topic_map=ctx["complex"].get("topic_map"),
                    )
                    ctx["complex"]["topic_layer_hits"] = ctx["complex"]["search_results"].get("topic_layer_hits")
                    ctx["complex"]["policy_hypotheses"] = build_policy_hypotheses(
                        ctx["complex"].get("user_query") or "",
                        ctx["complex"]["topic_layer_hits"] or {},
                    )
                st.success("検索完了")
    with colC:
        if st.button("複数案を生成（テーマ: 費用対効果/公平性/スピード）"):
            if not (ctx["complex"]["refined"] and ctx["complex"]["workplan"] and ctx["complex"]["search_results"]):
                st.warning("詳細化→分解→検索を済ませてください。")
            else:
                with st.spinner("政策案合成中…"):
                    n_alts_val = int(st.session_state.get("n_alts", 3))
                    ctx["complex"]["strategies"] = synthesize_strategies(
                        ctx["complex"]["refined"], ctx["complex"]["workplan"], ctx["complex"]["search_results"], n_alternatives=n_alts_val
                    )
                st.success(f"政策案を {len(ctx['complex']['strategies'])} 件生成しました。")

    if ctx["complex"]["workplan"]:
        st.markdown("**分解結果**")
        st.json(ctx["complex"]["workplan"])
    if ctx["complex"].get("topic_map"):
        st.markdown("**論点マップ（階層表示）**")
        render_topic_map(ctx["complex"]["topic_map"])
    if ctx["complex"].get("policy_hypotheses"):
        st.markdown("**階層別 政策仮説**")
        render_policy_hypotheses(ctx["complex"]["policy_hypotheses"])

    # --- Step 5: シナリオと資源配分シミュレーション ---
    st.markdown("---")
    st.subheader("5) 制約下での資源配分シミュレーション")
    scenario_records = ctx["complex"].get("scenario_configs") or SCENARIO_TEMPLATES
    scenario_df = st.data_editor(
        pd.DataFrame(scenario_records),
        num_rows="dynamic",
        use_container_width=True,
        key="scenario_editor"
    )
    if st.button("シナリオ設定を保存", key="btn_save_scenarios"):
        ctx["complex"]["scenario_configs"] = scenario_df.fillna(0).to_dict("records")
        st.success("シナリオ設定を保存しました。")
    scenario_names = scenario_df.get("name", pd.Series(dtype=str)).tolist()
    if scenario_names:
        scenario_choice = st.selectbox("シミュレーション対象シナリオ", scenario_names, key="scenario_choice")
        if st.button("このシナリオで最適化", key="btn_run_scenario"):
            options_df = pd.DataFrame(ctx["complex"].get("policy_options") or [])
            scenario_row = scenario_df[scenario_df["name"] == scenario_choice].iloc[0].to_dict()
            selected = optimize_scenario_allocation(
                options_df,
                scenario_row.get("budget", 0.0),
                scenario_row.get("staff_limit", 0.0),
            )
            ctx["complex"]["picked"] = selected
            years = ctx["years"]
            result = simulate_scenario(years, scenario_row, selected)
            ctx["complex"].setdefault("scenario_results", {})[scenario_choice] = result
            st.success(f"{scenario_choice} の最適化を実行しました。")
            st.dataframe(selected, use_container_width=True)
            st.caption(f"前提: シナリオ {scenario_choice} の预算={scenario_row.get('budget')}億円 / 人員={scenario_row.get('staff_limit')}人。")
            targets = get_kpi_targets(ctx) or [ctx.get("kpi") or "KPI"]
            constraints = ctx.get("kpi_constraints", []) or []
            for target in targets:
                constraint = next((c for c in constraints if c.get("name") == target), None)
                thr_value = _to_float((constraint or {}).get("threshold_hint"))
                threshold_lines = None
                if thr_value is not None and constraint:
                    unit = constraint.get("unit") or ""
                    label_text = f"{target or 'KPI'} ≧ {thr_value}{unit}" if (constraint.get("threshold_type") or "min") == "min" else f"{target or 'KPI'} ≦ {thr_value}{unit}"
                    threshold_lines = [{"value": thr_value, "label": label_text}]
                st.markdown(f"**{target or 'KPI'} のレンジ**")
                st.plotly_chart(
                    band_chart(years, result["mid"], result["low"], result["high"], None, target or 'KPI', threshold_lines),
                    use_container_width=True,
                    key=f"scenario_band_{scenario_choice}_{target}"
                )
                alerts = threshold_breaches(years, result["low"], result["high"], constraint) if constraint else []
                for al in alerts:
                    st.error(al)

    # --- Step 6: リスクとエビデンス透明性 ---
    st.markdown("---")
    st.subheader("6) 効果・リスクの可視化と透明性")
    risk_df = st.data_editor(
        pd.DataFrame(ctx["complex"].get("risk_register") or RISK_SAMPLE),
        num_rows="dynamic",
        use_container_width=True,
        key="risk_editor"
    )
    ctx["complex"]["risk_register"] = risk_df.to_dict("records")
    exposure = calc_risk_exposure(ctx["complex"].get("risk_register"), ctx.get("picked"))
    if not exposure.empty:
        st.markdown("**リスク感度分析**")
        st.dataframe(exposure, use_container_width=True)
        st.bar_chart(exposure.set_index("risk")["expected_impact"], use_container_width=True)
    else:
        st.info("リスク登録がありません。")
    st.markdown("**対立点 / 未決論点**")
    if ctx["complex"].get("contention_points"):
        for item in ctx["complex"]["contention_points"]:
            st.info(item)
    else:
        st.write("現在登録されている対立点はありません。")
    st.markdown("**エビデンス不足箇所**")
    if ctx["complex"].get("evidence_gaps"):
        for gap in ctx["complex"]["evidence_gaps"]:
            st.warning(gap)
    else:
        st.write("エビデンスギャップは検出されていません。")
    st.markdown("**関係者向けテンプレ**")
    templates = st.session_state.context.get("stakeholder_templates", [])
    if templates:
        for tpl in reversed(templates[-3:]):
            st.code(tpl.get("template") if isinstance(tpl, dict) else str(tpl), language="markdown")
    else:
        st.write("まだテンプレは作成されていません。チャットで『関係者にアプローチ』等と入力して生成できます。")
    st.markdown("**意思決定メモ**")
    decision_input = st.text_area("最終判断メモ", value=ctx["complex"].get("decision_notes", ""), key="decision_notes_box")
    if st.button("意思決定メモを保存", key="btn_save_decision_notes"):
        ctx["complex"]["decision_notes"] = decision_input
        st.success("意思決定メモを保存しました。")
    if ctx["complex"]["search_results"]:
        st.markdown("**検索結果ダイジェスト**")
        pr = ctx["complex"]["search_results"].get("policy_hits", [])
        pa = ctx["complex"]["search_results"].get("paper_hits", [])
        st.write(f"政策候補: {len(pr)} 件 / 論文候補: {len(pa)} 件")
    if ctx["complex"]["strategies"]:
        st.markdown("**生成された政策案（複数）**")
        for i, s in enumerate(ctx["complex"]["strategies"], start=1):
            with st.expander(f"[{i}] {s.get('name','strategy')}", expanded=(i==1)):
                st.write(f"テーマ: {s.get('theme')}")
                st.write(s.get("summary", s.get("rationale", "")))
                st.write("**採用政策候補**")
                render_policy_actions(s.get("policies") or s.get("actions") or [])
                st.write("**ロジックツリー（グラフ）**")
                render_logic_tree_graph(s.get("logic_tree"))
                st.write("**ロジックツリー（JSON）**")
                st.json(s.get("logic_tree", {}))
                meta = s.get("rlhf_meta")
                if meta:
                    st.caption(f"RLHF風マルチエージェント評価スコア: {meta.get('reward', 0)}")
                    if meta.get("critic_log"):
                        with st.expander("批評ログ（提案者×批評者の対話）", expanded=False):
                            st.json(meta.get("critic_log"))

    # --- Step 4: 批判的検討 ---
    st.markdown("---")
    st.subheader("4) 批判的検討（何が犠牲になるか）")
    if st.button("批判レビューを実行"):
        if not ctx["complex"]["strategies"]:
            st.warning("先に複数案を生成してください。")
        else:
            with st.spinner("レビュー中…"):
                has_key = bool((OPENAI_API_KEY or "").strip())
                if _HAS_OPENAI and has_key:
                    remote = critique_strategies(ctx["complex"]["strategies"])
                else:
                    remote = {"reviews": [], "cross_cutting_observations": []}
                if not remote.get("reviews"):
                    remote = local_critique(ctx["complex"]["strategies"])
                ctx["complex"]["critique"] = remote
            st.success("レビュー完了")
    if ctx["complex"].get("critique"):
        st.json(ctx["complex"]["critique"])
    else:
        st.info("まだ批判レビューが実行されていません。")

    # --- Step 5: 予算推定と段階別可視化 ---
    st.markdown("---")
    st.subheader("5) RSシステムから過去予算を探索し、段階別に可視化")
    if st.button("予算探索と可視化（戦略ごと）"):
        if not ctx["complex"]["strategies"]:
            st.warning("先に複数案を生成してください。")
        else:
            with st.spinner("予算探索・配分…"):
                ctx["complex"]["budgets"] = estimate_budgets(ctx["complex"]["strategies"])
            st.success("予算推定完了")

    st.markdown("**KPI ε制約最適化**")
    effect_inputs = build_effect_inputs(st.session_state)
    if not HAS_PULP:
        st.warning("PuLP がインストールされていません。`pip install pulp` を実行してください。")
    elif not effect_inputs:
        st.info("KPI制約または施策オプションが不足しています。Step2/3でKPIと施策を登録すると最適化できます。")
    else:
        effect, y_base, eps_default, budget_default, kpi_names, policy_names = effect_inputs
        st.caption("各KPIの最低許容水準(ε)と総予算を調整すると、ε制約法でターゲットKPIを順番に最大化します。")
        epsilon_df = pd.DataFrame({
            "KPI": kpi_names,
            "現状値": np.round(y_base, 2),
            "ε制約(下限)": np.round(eps_default, 2),
        })
        column_config = {
            "KPI": st.column_config.TextColumn("KPI", disabled=True),
            "現状値": st.column_config.NumberColumn("現状値", format="%.2f", disabled=True),
            "ε制約(下限)": st.column_config.NumberColumn("ε制約(下限)", format="%.2f"),
        }
        edited_eps = st.data_editor(
            epsilon_df,
            use_container_width=True,
            column_config=column_config,
            key="epsilon_editor"
        )
        eps_arr = edited_eps["ε制約(下限)"].astype(float).to_numpy()
        budget_val = st.number_input(
            "総予算制約（億円）",
            min_value=0.0,
            value=float(budget_default),
            step=0.5,
            key="epsilon_budget_input"
        )
        if st.button("ε制約でKPI最適化", key="btn_epsilon_opt"):
            with st.spinner("ε制約に基づく最適配分を探索中…"):
                sols = epsilon_constraint_allocation(effect, y_base, budget_val, eps_arr)
                st.session_state.context["complex"].setdefault("epsilon_alloc", {})
                st.session_state.context["complex"]["epsilon_alloc"] = {
                    "solutions": sols,
                    "kpis": kpi_names,
                    "policies": policy_names,
                    "budget": budget_val,
                }
                st.success("ε制約最適化を実行しました。")
    eps_ctx = st.session_state.context.get("complex", {}).get("epsilon_alloc")
    if eps_ctx:
        kpi_names = eps_ctx.get("kpis", [])
        policy_names = eps_ctx.get("policies", [])
        sols = eps_ctx.get("solutions", {}) or {}
        targets = []
        for target, sol in sorted(sols.items()):
            target_name = kpi_names[target] if target < len(kpi_names) else f"KPI{target}"
            targets.append((target_name, sol))
        if targets:
            st.markdown("**最適化結果（ターゲット別）**")
            tabs = st.tabs([name for name, _ in targets])
            for tab, (name, sol) in zip(tabs, targets):
                with tab:
                    st.caption(f"ターゲットKPI: {name} ｜ 総予算 {eps_ctx.get('budget', 0):.2f} 億円")
                    df_alloc = pd.DataFrame({"施策": policy_names, "配分(億円)": sol.get("allocation", [])}).round(3)
                    st.dataframe(df_alloc, use_container_width=True)
                    df_pred = pd.DataFrame({"KPI": kpi_names, "予測値": sol.get("KPI_pred", [])}).round(3)
                    if not df_pred.empty:
                        st.bar_chart(df_pred.set_index("KPI"), use_container_width=True)
                    st.dataframe(df_pred, use_container_width=True)

    # Sankey 可視化
    if ctx["complex"]["budgets"]:
        import plotly.graph_objects as go
        st.markdown("### 予算の段階別Sankey（各戦略）")
        for i, (s, b) in enumerate(zip(ctx["complex"]["strategies"], ctx["complex"]["budgets"]), start=1):
            st.markdown(f"**[{i}] {s.get('name','strategy')}**")
            # ノード構成: Input(予算) → Activities(各政策) → Outputs/Outcomes(集約)
            acts = s.get("logic_tree", {}).get("activities", [])
            alloc = b.get("allocation", {})
            per = alloc.get("per_activity_budget", [])
            prof = alloc.get("allocation_profile", {"activity":0.6,"output":0.25,"outcome_short":0.1,"outcome_mid":0.04,"outcome_long":0.01})
            # ノード
            labels = ["予算"]
            idx = {"予算": 0}
            # 活動
            for a in acts:
                lab = a.get("label", a.get("id",""))
                idx[lab] = len(labels); labels.append(lab)
            # 段階ノード
            for L in ["Outputs","Outcomes(S)","Outcomes(M)","Outcomes(L)"]:
                idx[L] = len(labels); labels.append(L)

            # リンク
            links = {"source": [], "target": [], "value": []}
            total_activity = 0.0
            # 予算→活動
            for a in per:
                lab = a.get("activity_label") or a.get("activity_id","")
                v = float(a.get("estimated_yen", 0.0))
                total_activity += v
                if lab in idx and v > 0:
                    links["source"].append(idx["予算"]); links["target"].append(idx[lab]); links["value"].append(v)
            # 活動→出力/アウトカム（配分プロファイル）
            for a in per:
                lab = a.get("activity_label") or a.get("activity_id","")
                v = float(a.get("estimated_yen", 0.0))
                if lab not in idx or v <= 0: continue
                links["source"].append(idx[lab]); links["target"].append(idx["Outputs"]);      links["value"].append(v*float(prof.get("output",0.25)))
                links["source"].append(idx[lab]); links["target"].append(idx["Outcomes(S)"]); links["value"].append(v*float(prof.get("outcome_short",0.1)))
                links["source"].append(idx[lab]); links["target"].append(idx["Outcomes(M)"]); links["value"].append(v*float(prof.get("outcome_mid",0.04)))
                links["source"].append(idx[lab]); links["target"].append(idx["Outcomes(L)"]); links["value"].append(v*float(prof.get("outcome_long",0.01)))

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=10, thickness=16, line=dict(color="black", width=0.3), label=labels),
                link=links
            )])
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), title=f"予算推定合計: {total_activity:,.0f} 円（配分比率に基づく段階別フロー）")
            st.plotly_chart(fig, use_container_width=True, key=f"sankey_{i}")

        st.markdown("**抽出元ページ（上位）**")
        for i, b in enumerate(ctx["complex"]["budgets"], start=1):
            st.markdown(f"- 戦略[{i}] {b.get('strategy_name','')}")
            cands = b.get("candidates", [])[:8]
            for c in cands:
                st.write(f"  - {c['raw']} / {c['yen']:,.0f} 円  —  {c['source_title']}  ({c['source_url']})")
