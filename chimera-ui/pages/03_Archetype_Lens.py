"""
pages/03_Archetype_Lens.py
Segment-driven strategic view for category managers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import (load_top5, load_archetype_assignments,
                                  load_archetype_summary, get_data_freshness)
from utils.state_manager import init_session_state, get_weights
from utils.ui_components import (inject_css, render_header, make_radar,
                                  PLOTLY_LAYOUT, COLOURS, ARCHETYPE_COLOURS,
                                  COMPONENT_LABELS, COMPONENT_COLS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Archetype Lens", freshness)

st.markdown("""
> **Purpose:** Segment-driven strategic view — understand how each customer archetype behaves
> and how Chimera serves them differently.
""")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    top5       = load_top5()
    archetypes = load_archetype_assignments()
    arch_sum   = load_archetype_summary()
    weights    = get_weights()

ALL_ARCHETYPES = ["Routine Replenisher", "Deal-Driven Explorer",
                  "Premium Discoverer", "Frugal Loyalist"]

if archetypes.empty:
    st.error("Archetype assignments not found. Ensure `module8_archetype_assignments.csv` exists.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_col, _ = st.columns([1, 2])
with ctrl_col:
    selected_arch = st.selectbox("Select Archetype", ALL_ARCHETYPES, key="arch_lens_sel")
    stock_aware   = st.session_state.get("stock_aware", False)
    if stock_aware:
        st.info("📦 Stock-Aware Ranking is ON – inventory penalty applied.")

arch_colour = ARCHETYPE_COLOURS.get(selected_arch, "#D1D5DB")

# ── Filter households for selected archetype ──────────────────────────────────
arch_hh = archetypes[archetypes["archetype"] == selected_arch]["household_key"].tolist()
arch_top5 = top5[top5["household_key"].isin(arch_hh)] if not top5.empty else pd.DataFrame()

n_hh_arch = len(arch_hh)
avg_deal   = float(archetypes.loc[archetypes["archetype"] == selected_arch, "deal_sensitivity"].mean()) if not archetypes.empty else 0.0
avg_div    = float(archetypes.loc[archetypes["archetype"] == selected_arch, "basket_diversity"].mean()) if not archetypes.empty else 0.0
avg_margin = float(arch_top5["Normalized_Margin"].mean()) if not arch_top5.empty and "Normalized_Margin" in arch_top5 else 0.0
avg_util   = float(arch_top5["Utility"].mean()) if not arch_top5.empty and "Utility" in arch_top5 else 0.0

# Source mix
src_mix_txt = "N/A"
if not arch_top5.empty and "source_detail" in arch_top5.columns:
    src = arch_top5["source_detail"].str.upper().fillna("UNK").value_counts(normalize=True)
    src_mix_txt = " | ".join([f"{k}:{v:.0%}" for k, v in src.items()])

# ── Archetype header card ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#1B1F28;border:2px solid {arch_colour};border-radius:12px;
            padding:18px 24px;margin:12px 0;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:14px;height:14px;border-radius:50%;background:{arch_colour};
                box-shadow:0 0 10px {arch_colour};"></div>
    <div style="font-size:1.4rem;font-weight:900;color:#FFFFFF;">{selected_arch}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px;">
    <div><div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;">Households</div>
         <div style="font-size:1.4rem;font-weight:700;color:#FFFFFF;">{n_hh_arch:,}</div></div>
    <div><div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;">Avg Deal Sensitivity</div>
         <div style="font-size:1.4rem;font-weight:700;color:#F08F50;">{avg_deal:.3f}</div></div>
    <div><div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;">Avg Margin</div>
         <div style="font-size:1.4rem;font-weight:700;color:#50F08F;">{avg_margin:.3f}</div></div>
    <div><div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;">Source Mix</div>
         <div style="font-size:0.88rem;font-weight:600;color:#B050F0;">{src_mix_txt}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: Radar | Metrics table ─────────────────────────────────────────────
radar_col, tbl_col = st.columns([1.2, 1.8])

with radar_col:
    st.markdown("#### Mean Component Scores")
    if not arch_top5.empty:
        comp_avgs = [
            float(arch_top5.get("Relevance", pd.Series([0])).mean()),
            float(arch_top5.get("Uplift",    pd.Series([0])).mean()),
            float(arch_top5.get("Normalized_Margin", pd.Series([0])).mean()),
            float(arch_top5.get("Context",   pd.Series([0])).mean()),
        ]
        fig_radar = make_radar(
            categories=COMPONENT_LABELS,
            values=comp_avgs,
            title=f"{selected_arch} Profile",
            colour=arch_colour,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No recommendation data for this archetype.")

with tbl_col:
    st.markdown("#### Per-Archetype Metrics Summary")
    if not arch_sum.empty:
        st.dataframe(arch_sum.style.format(precision=3), use_container_width=True, height=220)
    else:
        # Compute on the fly from top5 + archetypes
        if not arch_top5.empty:
            comp_cols = [c for c in ["Relevance","Uplift","Normalized_Margin","Context","Utility"]
                         if c in arch_top5.columns]
            summary = arch_top5[comp_cols].agg(["mean", "std"]).round(4)
            st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ── Top recommended items for segment ────────────────────────────────────────
st.markdown("#### 🏆 Top Items Recommended Across Segment")
if not arch_top5.empty and "COMMODITY_DESC" in arch_top5.columns:
    item_freq = (
        arch_top5.groupby("COMMODITY_DESC")
        .agg(frequency=("household_key", "nunique"),
             avg_utility=("Utility", "mean"),
             avg_margin=("Normalized_Margin", "mean"))
        .sort_values("frequency", ascending=False)
        .head(15)
        .reset_index()
    )
    fig_top = px.bar(
        item_freq, x="frequency", y="COMMODITY_DESC",
        orientation="h",
        color="avg_utility",
        color_continuous_scale=[[0, "#1B1F28"], [0.5, arch_colour], [1, "#FFFFFF"]],
        title=f"Top Recommended Items for {selected_arch}",
    )
    fig_top.update_layout(
        title=dict(text=f"Top Recommended Items for {selected_arch}", font=dict(color="#FFFFFF", size=16)),
        **PLOTLY_LAYOUT
    )
    fig_top.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")

# ── Cohort Performance Tracking (simulated) ───────────────────────────────────
st.markdown("#### 📈 Cohort Performance Tracking")
with st.expander("Show cohort margin shift over snapshots", expanded=True):
    rng = np.random.default_rng(99)
    snapshots = list(range(1, 9))
    fig_cohort = go.Figure()
    for arch in ALL_ARCHETYPES:
        base = rng.uniform(0.02, 0.10)
        vals = [base + rng.normal(0, 0.005) * i * 0.3 for i in snapshots]
        fig_cohort.add_trace(go.Scatter(
            x=snapshots, y=vals, name=arch,
            mode="lines+markers",
            line=dict(color=ARCHETYPE_COLOURS.get(arch, "#D1D5DB"), width=2),
            marker=dict(size=6),
        ))
    fig_cohort.update_layout(
        title=dict(text="Avg Observed Margin Shift by Archetype (Simulated)", font=dict(color="#FFFFFF", size=16)),
        xaxis_title="Snapshot #", yaxis_title="Margin Shift",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

# ── Example Household button ──────────────────────────────────────────────────
st.markdown("---")
if arch_hh:
    if st.button(f"🔍 Show Example Household from {selected_arch}", use_container_width=True):
        rand_hh = random.choice(arch_hh)
        st.session_state["selected_household"] = rand_hh
        st.switch_page("pages/02_Household_Explorer.py")
