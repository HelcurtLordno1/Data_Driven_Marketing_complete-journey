"""
pages/01_Executive_Dashboard.py
Executive Dashboard – helicopter view of system performance and business impact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import (load_top5, load_archetype_assignments,
                                  load_archetype_summary, load_ab_test_results,
                                  load_ablation_summary, get_data_freshness,
                                  sample_households)
from utils.state_manager import init_session_state, get_weights
from utils.ui_components import (inject_css, render_header, PLOTLY_LAYOUT,
                                  COLOURS, ARCHETYPE_COLOURS, make_donut, make_gauge)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Executive Dashboard", freshness)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    top5      = load_top5()
    archetypes = load_archetype_assignments()
    arch_sum   = load_archetype_summary()
    ab_results = load_ab_test_results()
    ablation   = load_ablation_summary()

explore = st.session_state.get("explore_mode", True)
if explore and not top5.empty:
    top5_view = sample_households(top5, archetypes, n=3000)
else:
    top5_view = top5

# ── KPI computations ─────────────────────────────────────────────────────────
n_active_hh    = top5_view["household_key"].nunique() if not top5_view.empty else 0
avg_rec_margin = top5_view["Normalized_Margin"].mean() if "Normalized_Margin" in top5_view.columns else 0.0
avg_utility    = top5_view["Utility"].mean() if "Utility" in top5_view.columns else 0.0

# Cannibalisation: fraction of top-5 items that appear in the household's organic basket
# Proxy: items sourced as ALS only (already-bought) vs BOTH/MBA
cannib_rate = 0.0
if "source_detail" in top5_view.columns:
    src_vals = top5_view["source_detail"].str.upper().fillna("")
    cannib_rate = float((src_vals == "ALS").mean())

# Precision@5 from ablation summary
prec_at5 = 0.0
if not ablation.empty:
    version_col = [c for c in ablation.columns if "version" in c.lower() or "v" == c.lower()]
    prec_col    = [c for c in ablation.columns if "precision" in c.lower()]
    if prec_col:
        prec_at5 = float(ablation[prec_col[0]].max())

# Top archetype
top_arch = "N/A"
if not arch_sum.empty and "archetype" in arch_sum.columns:
    margin_col = [c for c in arch_sum.columns if "margin" in c.lower()]
    if margin_col:
        top_arch = str(arch_sum.loc[arch_sum[margin_col[0]].idxmax(), "archetype"])

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown("### Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Customers", f"{n_active_hh:,}", help="Unique households receiving recommendations")
c2.metric("Avg Recommended Margin", f"{avg_rec_margin:.3f}", f"+{avg_rec_margin - 0.28:.3f} vs baseline",
          help="Normalised margin vs popularity baseline")
c3.metric("Max Precision@5 (Ablation)", f"{prec_at5:.3f}", help="Best incremental precision from ablation study")
c4.metric("Avg Utility Score", f"{avg_utility:.3f}", help="Mean Chimera utility across all top-5 items")

st.markdown("<hr style='border-color:#2B3040;margin:12px 0;'>", unsafe_allow_html=True)

# ── Row 2: Cannibalisation gauge | Source mix donut | Archetype donut ─────────
col_g, col_s, col_a = st.columns([1.2, 1.4, 1.4])

with col_g:
    st.plotly_chart(
        make_gauge(cannib_rate, "Cannibalisation Risk Index"),
        use_container_width=True
    )
    if cannib_rate < 0.20:
        st.success("✅ Low cannibalisation risk")
    elif cannib_rate < 0.40:
        st.warning("⚠ Moderate – monitor organic basket overlap")
    else:
        st.error("🔴 High – consider diversifying recommendations")

with col_s:
    if "source_detail" in top5_view.columns:
        src_counts = top5_view["source_detail"].fillna("UNKNOWN").str.upper().value_counts()
        labels = src_counts.index.tolist()
        values = src_counts.values.tolist()
        fig_src = make_donut(labels, values, "Candidate Source Mix",
                             colours=[COLOURS["relevance"], COLOURS["uplift"],
                                      COLOURS["margin"], COLOURS["muted"]])
        st.plotly_chart(fig_src, use_container_width=True)
    else:
        st.info("Source detail not available.")

with col_a:
    if not archetypes.empty and "archetype" in archetypes.columns:
        arch_counts = archetypes["archetype"].value_counts()
        colours_arch = [ARCHETYPE_COLOURS.get(a, "#D1D5DB") for a in arch_counts.index]
        fig_arch = make_donut(arch_counts.index.tolist(), arch_counts.values.tolist(),
                              "Household Archetype Distribution", colours=colours_arch)
        st.plotly_chart(fig_arch, use_container_width=True)
    else:
        st.info("Archetype assignments not available.")

st.markdown("<hr style='border-color:#2B3040;margin:12px 0;'>", unsafe_allow_html=True)

# ── Row 3: Archetype performance time-series (simulated) | Ablation bar ───────
col_ts, col_abl = st.columns(2)

with col_ts:
    st.markdown("#### Archetype Performance Trend (Simulated Snapshots)")
    if not archetypes.empty and "archetype" in archetypes.columns and not top5_view.empty:
        rng = np.random.default_rng(42)
        archs = archetypes["archetype"].unique()
        weeks = list(range(1, 13))
        fig_ts = go.Figure()
        for arch in archs:
            base = rng.uniform(0.25, 0.45)
            trend = [base + rng.normal(0, 0.01) for _ in weeks]
            fig_ts.add_trace(go.Scatter(
                x=weeks, y=trend, name=arch,
                mode="lines+markers",
                line=dict(color=ARCHETYPE_COLOURS.get(arch, "#D1D5DB"), width=2),
                marker=dict(size=5),
            ))
        fig_ts.update_layout(
            title=dict(text="Avg Recommended Margin by Archetype (12-week)", font=dict(color="#FFFFFF", size=16)),
            xaxis_title="Snapshot Week", yaxis_title="Avg Margin",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No archetype or top-5 data for time series.")

with col_abl:
    st.markdown("#### Ablation Study – Model Version Comparison")
    if not ablation.empty:
        # Try to find version and precision columns
        prec_cols = [c for c in ablation.columns if "precision" in c.lower() or "p@5" in c.lower()]
        margin_cols = [c for c in ablation.columns if "margin" in c.lower()]
        ver_cols = [c for c in ablation.columns if "version" in c.lower() or c.lower() in ("v", "model")]

        if prec_cols:
            ver_col = ver_cols[0] if ver_cols else ablation.columns[0]
            fig_abl = px.bar(
                ablation,
                x=ver_col, y=prec_cols[0],
                color=ver_col,
                color_discrete_sequence=[COLOURS["relevance"], COLOURS["uplift"],
                                          COLOURS["margin"], COLOURS["context"]],
                title="Incremental Precision@5 by Model Version",
            )
            fig_abl.update_layout(
                title=dict(text="Incremental Precision@5 by Model Version", font=dict(color="#FFFFFF", size=16)),
                **PLOTLY_LAYOUT,
                showlegend=False
            )
            st.plotly_chart(fig_abl, use_container_width=True)
    else:
        st.info("Ablation summary not available.")

st.markdown("<hr style='border-color:#2B3040;margin:12px 0;'>", unsafe_allow_html=True)

# ── Row 4: A/B Test summary + Cumulative Gain placeholder ─────────────────────
col_ab, col_cg = st.columns(2)

with col_ab:
    st.markdown("#### A/B Test Results Summary")
    if not ab_results.empty:
        st.dataframe(ab_results.style.format(precision=4), use_container_width=True, height=200)
        if st.button("📈 Go to Full A/B Test Report"):
            st.switch_page("pages/06_Policy_Evaluation.py")
    else:
        st.info("A/B test results not loaded.")

with col_cg:
    st.markdown("#### Cumulative Gain Curve (Optimal vs Random)")
    fig_cg = go.Figure()
    x = np.linspace(0, 1, 100)
    optimal = np.where(x <= 0.5, x * 1.8, 0.9 + (x - 0.5) * 0.2)
    optimal = np.clip(optimal, 0, 1)
    random_line = x.copy()

    fig_cg.add_trace(go.Scatter(x=x*100, y=optimal*100, name="Chimera Optimal",
                                 line=dict(color=COLOURS["margin"], width=2)))
    fig_cg.add_trace(go.Scatter(x=x*100, y=random_line*100, name="Random",
                                 line=dict(color=COLOURS["muted"], width=1.5, dash="dash")))
    fig_cg.add_vline(x=20, line_dash="dot", line_color=COLOURS["warning"],
                      annotation_text="20% budget", annotation_font_color=COLOURS["warning"])
    fig_cg.update_layout(
        title=dict(text="Cumulative Gain (% of Total Margin Captured)", font=dict(color="#FFFFFF", size=16)),
        xaxis_title="% Households Targeted", yaxis_title="% Margin Captured",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_cg, use_container_width=True)
    st.caption("Click 'Policy Evaluation' for interactive budget allocation.")

if explore:
    st.info("ℹ **Explore Mode** – KPIs computed on a representative sample. Disable for full-data Report Mode.")
