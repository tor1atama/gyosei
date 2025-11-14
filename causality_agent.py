# ebpm_agents/causality_agent.py (minimal stub)
from __future__ import annotations
from typing import Any, Dict, List, Tuple

def run_te_and_granger(per_node_ts: Dict[str, List[Dict[str, Any]]], edges: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Minimal stub used to keep imports working when optional scientific
    dependencies are unavailable. Returns empty results.
    """
    return {"transfer_entropy": [], "granger": []}
    x2, y2 = make_stationary(x)
    idx = x2.index.intersection(y2.index)
    x2, y2 = x2.loc[idx], y2.loc[idx]
    if len(idx) < 8:
        return {"status":"skip_after_diff_too_short","n":int(len(idx))}
    import warnings
    warnings.filterwarnings("ignore")
    df = pd.DataFrame({"x": x2, "y": y2}).dropna()
    from statsmodels.tsa.api import VAR
    model = VAR(df)
    sel = model.select_order(maxlags=min(2, max(1, len(df)//3)))
    p = int(sel.selected_orders.get("bic", 1) or 1)
    p = max(1, min(p, 2))
    res = model.fit(p)
    gx = res.test_causality("y", ["x"], kind="f")
    gy = res.test_causality("x", ["y"], kind="f")
    return {
        "status":"ok","lag_selected":p,"n":int(len(df)),
        "p_x_to_y": float(gx.pvalue), "p_y_to_x": float(gy.pvalue),
        "aic": float(res.aic), "bic": float(res.bic)
    }

def run_te_and_granger(per_node_ts: Dict[str, List[Dict[str, Any]]], edges: List[Tuple[str,str]]) -> Dict[str, Any]:
    label2series = {lab: series_from_items(ts) for lab, ts in per_node_ts.items()}
    te_rows, gr_rows = [], []
    for src, dst in edges:
        sx, sy = label2series.get(src), label2series.get(dst)
        if sx is None or sy is None:
            te_rows.append({"from":src,"to":dst,"status":"missing_series"})
            gr_rows.append({"from":src,"to":dst,"status":"missing_series"})
            continue
        ax, ay = align_common(sx, sy, k=10)
        if len(ax)==0 or len(ay)==0:
            te_rows.append({"from":src,"to":dst,"status":"skip_not_enough_points"})
            gr_rows.append({"from":src,"to":dst,"status":"skip_not_enough_points"})
            continue
        te_xy = transfer_entropy_xy(ax.values, ay.values, n_bins=3, n_perm=2000, alpha=1.0, seed=0)
        te_yx = transfer_entropy_xy(ay.values, ax.values, n_bins=3, n_perm=2000, alpha=1.0, seed=1)
        te_rows.append({"from":src,"to":dst,"status":"ok","te_x_to_y":te_xy,"te_y_to_x":te_yx})
        gr = var_granger_xy(ax, ay)
        gr_rows.append({"from":src,"to":dst,**gr})
    return {"transfer_entropy": te_rows, "granger": gr_rows}
