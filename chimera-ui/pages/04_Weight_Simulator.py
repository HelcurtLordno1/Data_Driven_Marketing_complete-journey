"""
pages/04_Weight_Simulator.py
Tuning playground – weight sliders, live re-ranking, bump chart, side-by-side comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import load_top5, load_archetype_assignments, get_data_freshness
from utils.state_manager import (init_session_state, get_weights, normalise_weights,
                                  save_scenario, list_scenarios, load_scenario)
from utils.ui_components import (inject_css, render_header, make_bump_chart,
                                  PLOTLY_LAYOUT, COLOURS, COMPONENT_KEYS, COMPONENT_LABELS)
from utils.recompute     import rerank_households, compute_stability
from utils.scenario_io   import (save_scenario_to_disk, load_scenario_from_disk,
                                  list_scenario_names)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Weight Simulator & Scenario Comparison", freshness)

st.markdown("""
> **Purpose:** Explore how changing utility weights affects recommendation rankings.
> Compare two scenarios side-by-side.
""")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    top5       = load_top5()
    archetypes = load_archetype_assignments()

if top5.empty:
    st.error("Top-5 data not available.")
    st.stop()

# ── Select 5 sample households ────────────────────────────────────────────────
all_hh = top5["household_key"].unique().tolist()
sample_hh = all_hh[:5]  # deterministic first 5

# ── Two-column scenario layout ────────────────────────────────────────────────
st.markdown("### ⚖ Scenario A vs Scenario B")
col_a, col_b = st.columns(2, gap="large")

def _scenario_weights(label: str, default: dict) -> dict:
    st.markdown(f"**{label} Weights**")
    raw = {}
    for k, l in zip(COMPONENT_KEYS, COMPONENT_LABELS):
        raw[k] = st.slider(l, 0.0, 1.0, float(default.get(k, 0.25)), 0.05,
                           key=f"{label}_{k}")
    norm = normalise_weights(raw)
    st.caption("Normalised: " + " | ".join([f"{l}: {norm[k]:.2f}" for k, l in zip(COMPONENT_KEYS, COMPONENT_LABELS)]))
    st.latex(r"w_i^{norm} = \frac{w_i}{\sum_j w_j}")
    return norm

current_w = get_weights()
with col_a:
    weights_a = _scenario_weights("Scenario A", current_w)
with col_b:
    # Offset scenario B slightly
    alt_default = {k: max(0.05, v * 0.6 + (0.25 * 0.4)) for k, v in current_w.items()}
    weights_b = _scenario_weights("Scenario B", alt_default)

with st.spinner("Re-ranking…"):
    req_cols = {"Relevance", "Uplift", "Normalized_Margin", "Context"}
    if req_cols.issubset(set(top5.columns)):
        # Calculate for top 20 to allow rank shift tracking even if item drops out of top 5
        ranked_a = rerank_households(top5, weights_a, household_keys=sample_hh, top_k=20)
        ranked_b = rerank_households(top5, weights_b, household_keys=sample_hh, top_k=20)
    else:
        ranked_a = top5[top5["household_key"].isin(sample_hh)].copy()
        ranked_b = ranked_a.copy()

# Filter for display (Top 5)
display_a = ranked_a[ranked_a["new_rank"] <= 5]
display_b = ranked_b[ranked_b["new_rank"] <= 5]

# ── Stability score ───────────────────────────────────────────────────────────
if not ranked_a.empty and not ranked_b.empty:
    top1_a = ranked_a[ranked_a["new_rank"] == 1].set_index("household_key")["COMMODITY_DESC"]
    top1_b = ranked_b[ranked_b["new_rank"] == 1].set_index("household_key")["COMMODITY_DESC"]
    stability = compute_stability(top1_a, top1_b)
else:
    stability = 0.0

st.markdown("---")
stab_colour = "#50F08F" if stability >= 0.8 else ("#F0C850" if stability >= 0.5 else "#F05050")
st.markdown(f"""
<div style="text-align:center;background:#1B1F28;border:1px solid #2B3040;
            border-radius:10px;padding:14px;margin:10px 0;">
  <div style="font-size:0.78rem;color:#D1D5DB;text-transform:uppercase;letter-spacing:0.08em;">
    Stability Score (Rank-1 unchanged)
  </div>
  <div style="font-size:2.2rem;font-weight:900;color:{stab_colour};">
    {stability:.0%}
  </div>
  <div style="font-size:0.82rem;color:#D1D5DB;">
    of sample households keep same #1 recommendation across scenarios
  </div>
</div>
""", unsafe_allow_html=True)

# ── Bump chart ────────────────────────────────────────────────────────────────
st.markdown("### 📉 Rank Shift Across Scenarios")
if not ranked_a.empty and not ranked_b.empty and "COMMODITY_DESC" in ranked_a.columns:
    rank_col_a = "new_rank" if "new_rank" in ranked_a.columns else "rank"
    rank_col_b = "new_rank" if "new_rank" in ranked_b.columns else "rank"

    bump_data: dict = {}
    for hh in sample_hh:
        # Get Scenario A's top-1 item
        ra_top = ranked_a[(ranked_a["household_key"] == hh) & (ranked_a["new_rank"] == 1)]
        if ra_top.empty: continue
        
        target_item = ra_top["COMMODITY_DESC"].iloc[0]
        
        # Find its rank in Scenario B
        rb_item = ranked_b[(ranked_b["household_key"] == hh) & (ranked_b["COMMODITY_DESC"] == target_item)]
        
        rank_a_val = 1
        rank_b_val = int(rb_item["new_rank"].iloc[0]) if not rb_item.empty else 20 # Fallback to 20 if outside top-20
        
        bump_data[hh] = [rank_a_val, rank_b_val]
    
    if bump_data:
        st.plotly_chart(make_bump_chart(bump_data), use_container_width=True)
else:
    st.info("Bump chart requires Relevance, Uplift, Normalized_Margin, Context columns in top5.")

# ── Side-by-side KPI comparison ───────────────────────────────────────────────
st.markdown("### 📊 KPI Comparison")
comp_col_a, comp_col_b = st.columns(2)

def _kpi_block(label: str, ranked: pd.DataFrame, weights: dict) -> None:
    st.markdown(f"**{label}**")
    if ranked.empty:
        st.info("No data")
        return
    avg_m = float(ranked["Normalized_Margin"].mean()) if "Normalized_Margin" in ranked.columns else 0.0
    avg_u = float(ranked.get("Utility_new", ranked.get("Utility", pd.Series([0]))).mean())
    uniq_hh = int(ranked["household_key"].nunique())
    st.metric("Avg Margin",  f"{avg_m:.4f}")
    st.metric("Avg Utility", f"{avg_u:.4f}")
    st.metric("Households",  str(uniq_hh))
    w_str = " | ".join([f"{l}: {weights[k]:.2f}" for k, l in zip(COMPONENT_KEYS, COMPONENT_LABELS)])
    st.caption(f"Weights: {w_str}")

with comp_col_a:
    _kpi_block("Scenario A", display_a, weights_a)
with comp_col_b:
    _kpi_block("Scenario B", display_b, weights_b)

# ── Save scenario ─────────────────────────────────────────────────────────────
st.markdown("---")
save_col_a, save_col_b = st.columns(2)
with save_col_a:
    sc_a_name = st.text_input("Save Scenario A as…", key="sc_a_name", placeholder="e.g. Margin-Heavy")
    if st.button("💾 Save Scenario A") and sc_a_name:
        save_scenario_to_disk(sc_a_name, weights_a)
        st.success(f"Saved '{sc_a_name}'")

with save_col_b:
    sc_b_name = st.text_input("Save Scenario B as…", key="sc_b_name", placeholder="e.g. Discovery-Heavy")
    if st.button("💾 Save Scenario B") and sc_b_name:
        save_scenario_to_disk(sc_b_name, weights_b)
        st.success(f"Saved '{sc_b_name}'")

# ── Load scenario ──────────────────────────────────────────────────────────────
saved = list_scenario_names()
if saved:
    st.markdown("---")
    st.markdown("**📂 Load Saved Scenario into Global Weights**")
    load_sel = st.selectbox("Choose scenario", ["— select —"] + saved, key="load_sc_sim_sel")
    if load_sel != "— select —" and st.button("Load into Sidebar Weights"):
        sc = load_scenario_from_disk(load_sel)
        if sc:
            st.session_state["current_weights"] = sc["weights"]
            st.success(f"Loaded '{load_sel}'")
            st.rerun()
