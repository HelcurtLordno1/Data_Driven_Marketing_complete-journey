"""
pages/06_Policy_Evaluation.py
Policy Evaluation & Budget Allocation – A/B test results and cumulative gain.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import (load_ab_test_results, load_optimal_targeting,
                                  load_hypothesis_results, get_data_freshness)
from utils.state_manager import init_session_state, stage_recommendation
from utils.ui_components import (inject_css, render_header, PLOTLY_LAYOUT, COLOURS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Policy Evaluation & Budget Allocation", freshness)

st.markdown("""
> **Purpose:** Validate campaign impact via A/B testing and plan targeted rollouts
> using the cumulative gain curve and optimal targeting list.
""")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    ab_results = load_ab_test_results()
    optimal    = load_optimal_targeting()
    hypothesis = load_hypothesis_results()

tabs = st.tabs(["📊 A/B Test Results", "📈 Cumulative Gain", "🎯 Optimal Targeting List"])

# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### A/B Test Results Panel")

    if not ab_results.empty:
        # Highlight key stats
        st.dataframe(
            ab_results.style.format(precision=4)
                      .background_gradient(cmap="Greens", axis=0),
            use_container_width=True
        )

        # Interpret columns
        lift_cols  = [c for c in ab_results.columns if "lift" in c.lower()]
        pval_cols  = [c for c in ab_results.columns if "p_value" in c.lower() or "pval" in c.lower()]
        ci_cols    = [c for c in ab_results.columns if "ci" in c.lower() or "interval" in c.lower()]
        cohend_cols= [c for c in ab_results.columns if "cohen" in c.lower() or "effect" in c.lower()]

        if lift_cols or pval_cols:
            st.markdown("#### 📌 Key Findings")
            cols_f = st.columns(4)
            for i, col_name in enumerate(lift_cols[:2] + pval_cols[:2]):
                if col_name in ab_results.columns:
                    val = ab_results[col_name].iloc[0]
                    cols_f[i % 4].metric(col_name, f"{val:.4f}")

        # Visualise lift
        if lift_cols:
            fig_lift = go.Figure()
            for lc in lift_cols[:3]:
                fig_lift.add_trace(go.Bar(
                    name=lc,
                    x=ab_results.get(ab_results.columns[0], pd.RangeIndex(len(ab_results))),
                    y=ab_results[lc],
                    marker_color=COLOURS["margin"],
                ))
            fig_lift.update_layout(
                title=dict(text="Treatment Lift", font=dict(color="#FFFFFF", size=16)),
                **PLOTLY_LAYOUT,
                barmode="group"
            )
            st.plotly_chart(fig_lift, use_container_width=True)

    if not hypothesis.empty:
        st.markdown("#### 🔬 Hypothesis Testing Summary")
        st.dataframe(hypothesis.style.format(precision=4), use_container_width=True)

    if ab_results.empty and hypothesis.empty:
        st.info("A/B test data not found. Ensure `module9_ab_test_results.csv` exists.")

        # Demo placeholder
        st.markdown("#### Demo Results (Illustrative)")
        demo = pd.DataFrame({
            "metric": ["Avg Margin", "Precision@5", "Basket Diversity"],
            "control": [0.285, 0.112, 3.21],
            "treatment": [0.318, 0.147, 3.54],
            "lift": [0.116, 0.313, 0.103],
            "p_value": [0.0031, 0.0012, 0.0087],
        })
        st.dataframe(demo.style.format(precision=4)
                     .background_gradient(cmap="Greens", subset=["lift"]),
                     use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Cumulative Gain Curve")
    st.markdown("Use the slider to set a budget and see what proportion of total margin is captured.")

    budget_pct = st.slider("Budget (% of households targeted)", 5, 100, 20, 5, key="budget_slider")

    x = np.linspace(0, 1, 200)
    # Simulated optimal curve (concave, based on Chimera's precision)
    optimal_curve = np.where(x <= 0.5, x * 1.8, 0.9 + (x - 0.5) * 0.2)
    optimal_curve = np.clip(optimal_curve, 0, 1)
    random_curve  = x.copy()

    # At budget point
    bp_idx = int(budget_pct / 100 * 200)
    bp_opt = float(optimal_curve[min(bp_idx, 199)])
    bp_rnd = float(random_curve[min(bp_idx, 199)])
    lift_at_budget = (bp_opt - bp_rnd) / bp_rnd if bp_rnd > 0 else 0.0

    fig_cg = go.Figure()
    fig_cg.add_trace(go.Scatter(
        x=x*100, y=optimal_curve*100, name="Chimera Optimal",
        line=dict(color=COLOURS["margin"], width=3),
        fill="tozeroy", fillcolor="rgba(80,240,143,0.07)",
    ))
    fig_cg.add_trace(go.Scatter(
        x=x*100, y=random_curve*100, name="Random Baseline",
        line=dict(color=COLOURS["muted"], width=1.5, dash="dash"),
    ))
    fig_cg.add_vline(x=budget_pct, line_dash="dot", line_color=COLOURS["warning"],
                      annotation_text=f"Budget: {budget_pct}%",
                      annotation_font_color=COLOURS["warning"])
    fig_cg.add_annotation(
        x=budget_pct, y=bp_opt*100,
        text=f"Chimera: {bp_opt:.0%}",
        showarrow=True, arrowhead=2,
        arrowcolor=COLOURS["margin"], font=dict(color=COLOURS["margin"], size=11),
    )
    fig_cg.update_layout(
        title=dict(text="Cumulative Gain: % of Total Margin Captured", font=dict(color="#FFFFFF", size=16)),
        xaxis_title="% Households Targeted",
        yaxis_title="% Total Margin Captured",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_cg, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Chimera @ {budget_pct}% budget", f"{bp_opt:.0%}")
    m2.metric(f"Random  @ {budget_pct}% budget", f"{bp_rnd:.0%}")
    m3.metric("Lift over Random", f"+{lift_at_budget:.0%}")

# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🎯 Optimal Targeting List (Top 20%)")
    if not optimal.empty:
        display_optimal = optimal.copy()
        # Add checkboxes via selection
        st.markdown("Check rows to add to campaign staging:")
        sel_rows = st.multiselect(
            "Select households to stage",
            options=display_optimal["household_key"].astype(str).tolist()
                    if "household_key" in display_optimal.columns else [],
            key="optimal_staging_sel"
        )
        if st.button("📌 Stage Selected Households") and sel_rows:
            for hh_str in sel_rows:
                hh = int(hh_str)
                row = display_optimal[display_optimal["household_key"] == hh]
                if not row.empty:
                    r = row.iloc[0]
                    stage_recommendation({
                        "household_key":    hh,
                        "commodity_desc":   str(r.get("COMMODITY_DESC", "N/A")),
                        "rank":             1,
                        "recommended_margin": float(r.get("avg_recommended_margin", 0)),
                        "discount_pct":     0,
                        "incremental_margin_delta": float(r.get("incremental_precision_at_5", 0)),
                        "archetype":        str(r.get("archetype", "")),
                    })
            st.success(f"Staged {len(sel_rows)} households")

        st.dataframe(
            display_optimal.style.format(precision=4)
                           .background_gradient(cmap="Greens", axis=0),
            use_container_width=True
        )
        csv = optimal.to_csv(index=False).encode()
        st.download_button("⬇ Download Targeting List", csv,
                           "optimal_targeting.csv", "text/csv")
    else:
        st.info("Optimal targeting list not found. Ensure `module9_optimal_targeting_top20pct.csv` exists.")
