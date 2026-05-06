"""
pages/07_Model_Health.py
Model Health & Ablation – deep dive for ML engineers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import (load_ablation_summary, load_interpretability_summary,
                                  load_top5, load_archetype_assignments,
                                  get_data_freshness)
from utils.state_manager import init_session_state
from utils.ui_components import (inject_css, render_header, PLOTLY_LAYOUT, COLOURS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Model Health & Ablation", freshness)

st.markdown("""
> **Purpose:** Validate model quality, inspect ablation results, and trace data lineage.
> Designed for ML engineers and data scientists.
""")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    ablation  = load_ablation_summary()
    interp    = load_interpretability_summary()
    top5      = load_top5()
    archetypes= load_archetype_assignments()

tabs = st.tabs(["🔬 Ablation Study", "📊 Feature Importance", "🗂 Model Lineage", "🌡 Precision Heatmap"])

# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Ablation Study – Model Version Comparison")
    if not ablation.empty:
        st.dataframe(ablation.style.format(precision=4), use_container_width=True)

        # Auto-detect columns
        ver_cols  = [c for c in ablation.columns if "version" in c.lower() or c.lower() in ("v","model")]
        prec_cols = [c for c in ablation.columns if "precision" in c.lower() or "p@5" in c.lower()]
        mrg_cols  = [c for c in ablation.columns if "margin" in c.lower()]

        ver_col = ver_cols[0] if ver_cols else ablation.columns[0]
        colour_seq = [COLOURS["muted"], COLOURS["relevance"], COLOURS["margin"], COLOURS["context"]]

        if prec_cols:
            fig_p = px.bar(ablation, x=ver_col, y=prec_cols[0],
                           color=ver_col, color_discrete_sequence=colour_seq,
                           title=f"Incremental Precision@5 ({prec_cols[0]})")
            fig_p.update_layout(
                title=dict(text=f"Incremental Precision@5 ({prec_cols[0]})", font=dict(color="#FFFFFF", size=16)),
                **PLOTLY_LAYOUT,
                showlegend=False
            )
            st.plotly_chart(fig_p, use_container_width=True)

        if mrg_cols:
            fig_m = px.bar(ablation, x=ver_col, y=mrg_cols[0],
                           color=ver_col, color_discrete_sequence=colour_seq,
                           title=f"Avg Recommended Margin ({mrg_cols[0]})")
            fig_m.update_layout(
                title=dict(text=f"Avg Recommended Margin ({mrg_cols[0]})", font=dict(color="#FFFFFF", size=16)),
                **PLOTLY_LAYOUT,
                showlegend=False
            )
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Ablation summary not found. Ensure `module4_ablation_summary.csv` exists.")

        # Illustrative demo table
        st.markdown("**Illustrative Ablation Table (Demo)**")
        demo_abl = pd.DataFrame({
            "Version": ["V0 (Popularity)", "V1 (ALS only)", "V2 (ALS+MBA)", "V3 (Chimera Full)"],
            "Incremental_Precision_at_5": [0.000, 0.081, 0.103, 0.147],
            "Avg_Recommended_Margin":     [0.285, 0.291, 0.303, 0.318],
        })
        colour_seq = [COLOURS["muted"], COLOURS["relevance"], COLOURS["margin"], COLOURS["context"]]
        fig_demo = px.bar(demo_abl, x="Version", y="Incremental_Precision_at_5",
                          color="Version", color_discrete_sequence=colour_seq,
                          title="Demo: Precision@5 by Model Version")
        fig_demo.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_demo, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Global Feature / Component Importance")
    if not interp.empty:
        st.dataframe(interp.style.format(precision=4), use_container_width=True)
        imp_cols = [c for c in interp.columns if "importance" in c.lower() or "mean" in c.lower()]
        feat_cols = [c for c in interp.columns if "feature" in c.lower() or "component" in c.lower()]
        if imp_cols and feat_cols:
            fig_imp = px.bar(
                interp.sort_values(imp_cols[0], ascending=False),
                x=imp_cols[0], y=feat_cols[0], orientation="h",
                color=imp_cols[0],
                color_continuous_scale=[[0,"#1B1F28"],[0.5,COLOURS["relevance"]],[1,COLOURS["margin"]]],
                title="Permutation Importance by Component"
            )
            fig_imp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            fig_imp.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Interpretability summary not found.")
        # Demo bar chart
        demo_imp = pd.DataFrame({
            "Component":  ["Relevance", "Uplift", "Margin", "Context"],
            "Importance": [0.038, 0.024, 0.011, 0.007],
            "Std":        [0.004, 0.003, 0.002, 0.001],
        })
        fig_di = px.bar(demo_imp, x="Importance", y="Component",
                        orientation="h", error_x="Std",
                        color="Importance",
                        color_continuous_scale=[[0,"#1B1F28"],[0.5,COLOURS["relevance"]],[1,COLOURS["margin"]]],
                        title="Demo: Permutation Importance (RF Classifier)")
        fig_di.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig_di.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_di, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Model Lineage Panel")
    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "02_processed"
    files_of_interest = [
        "master_transactions.parquet",
        "top5_recommendations_module3.csv",
        "module8_archetype_assignments.csv",
        "module4_ablation_summary.csv",
        "module9_ab_test_results.csv",
        "candidate_set_module3_scored.csv",
        "module9_optimal_targeting_top20pct.csv",
    ]
    import os, datetime
    rows = []
    for fname in files_of_interest:
        fpath = _DATA_DIR / fname
        if fpath.exists():
            stat = os.stat(fpath)
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = stat.st_size / 1e6
            rows.append({"File": fname, "Size (MB)": round(size_mb, 2),
                         "Last Modified": mtime, "Status": "✅ Present"})
        else:
            rows.append({"File": fname, "Size (MB)": "-",
                         "Last Modified": "-", "Status": "❌ Missing"})
    lineage_df = pd.DataFrame(rows)
    st.dataframe(
        lineage_df.style.map(
            lambda v: "color:#50F08F" if v == "✅ Present" else "color:#F05050",
            subset=["Status"]
        ),
        use_container_width=True, hide_index=True
    )

    # Data freshness
    max_day = freshness.get("max_day", "N/A")
    days_old = freshness.get("days_since", "N/A")
    drift_status = "🟢 OK" if (isinstance(days_old, int) and days_old <= 7) else "🟡 Stale"
    st.markdown(f"""
    <div style="background:#1B1F28;border:1px solid #2B3040;border-radius:10px;
                padding:14px 18px;margin-top:12px;">
      <div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;margin-bottom:8px;">
        System Pulse Summary
      </div>
      <div>📅 Max Transaction Day: <b style="color:#4F8FF0;">{max_day}</b></div>
      <div>⏱ Data Age: <b>{days_old} days</b></div>
      <div>📡 Drift Status: <b>{drift_status}</b></div>
      <div>🔁 Model Retrain: <b style="color:#D1D5DB;">N/A (manual)</b></div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Household-Level Incremental Precision Heatmap")
    if not top5.empty and "Utility" in top5.columns and not archetypes.empty:
        with st.spinner("Building heatmap sample…"):
            sample_size = min(500, len(top5["household_key"].unique()))
            sample_hh = top5["household_key"].unique()[:sample_size]
            hmap_df = (
                top5[top5["household_key"].isin(sample_hh)]
                .groupby("household_key")[["Utility", "Normalized_Margin"]]
                .mean()
                .reset_index()
                .merge(archetypes[["household_key","archetype"]], on="household_key", how="left")
            )
            hmap_df["archetype"] = hmap_df["archetype"].fillna("Unknown")

        fig_heat = px.scatter(
            hmap_df, x="Utility", y="Normalized_Margin",
            color="archetype",
            color_discrete_map={
                "Routine Replenisher": COLOURS["relevance"],
                "Deal-Driven Explorer": COLOURS["uplift"],
                "Premium Discoverer": COLOURS["margin"],
                "Frugal Loyalist": COLOURS["context"],
                "Unknown": COLOURS["muted"],
            },
            hover_name="household_key",
            title=f"Utility vs Margin – Sampled {sample_size} Households",
            opacity=0.7,
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Top-5 data or archetype assignments not available for heatmap.")
