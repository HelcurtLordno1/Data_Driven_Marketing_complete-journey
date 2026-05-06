"""
pages/02_Household_Explorer.py
Deep dive into an individual shopper – fully explainable and actionable.
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
                                  load_candidate_set_scored, get_data_freshness,
                                  sample_households)
from utils.state_manager import (init_session_state, get_weights, stage_recommendation,
                                  log_feedback, get_staged)
from utils.ui_components import (inject_css, render_header, render_rec_card,
                                  PLOTLY_LAYOUT, COLOURS, ARCHETYPE_COLOURS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Household Explorer", freshness)

st.markdown("""
> **Purpose:** Deep dive into an individual shopper — fully explainable, interactive, and actionable.
""")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading recommendations…"):
    top5      = load_top5()
    archetypes = load_archetype_assignments()
    weights    = get_weights()

if top5.empty:
    st.error("Top-5 recommendations data not found. Please ensure `top5_recommendations_module3.csv` exists in `data/02_processed/`.")
    st.stop()

# ── Layout: left control panel | right card stack ─────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("### 🔎 Household Search")

    all_households = sorted(top5["household_key"].unique().tolist())

    # Autocomplete via selectbox
    hh_search = st.selectbox(
        "Select or search Household ID",
        options=all_households,
        index=0 if not st.session_state.get("selected_household") else
              (all_households.index(st.session_state["selected_household"])
               if st.session_state["selected_household"] in all_households else 0),
        key="hh_search_select",
        help="All households with recommendations",
    )
    st.session_state["selected_household"] = hh_search

    # Household profile card
    arch_row = archetypes[archetypes["household_key"] == hh_search] if not archetypes.empty else pd.DataFrame()

    archetype  = str(arch_row["archetype"].iloc[0]) if not arch_row.empty and "archetype" in arch_row else "Unknown"
    deal_sens  = float(arch_row["deal_sensitivity"].iloc[0]) if not arch_row.empty and "deal_sensitivity" in arch_row else 0.0
    basket_div = float(arch_row["basket_diversity"].iloc[0]) if not arch_row.empty and "basket_diversity" in arch_row else 0.0

    arch_colour = ARCHETYPE_COLOURS.get(archetype, "#D1D5DB")

    st.markdown(f"""
    <div style="background:#1B1F28;border:1px solid #2B3040;border-radius:10px;padding:16px 18px;margin:10px 0;">
      <div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;letter-spacing:0.08em;">Household Profile</div>
      <div style="font-size:1.5rem;font-weight:900;color:#FFFFFF;margin-top:4px;">HH #{hh_search}</div>
      <div style="margin-top:8px;">
        <span style="background:rgba(0,0,0,0.4);border:1px solid {arch_colour};
                     color:{arch_colour};font-size:0.78rem;font-weight:700;
                     padding:3px 12px;border-radius:999px;">{archetype}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;">
        <div>
          <div style="font-size:0.72rem;color:#D1D5DB;">Deal Sensitivity</div>
          <div style="font-size:1.1rem;font-weight:700;color:#F08F50;">{deal_sens:.3f}</div>
        </div>
        <div>
          <div style="font-size:0.72rem;color:#D1D5DB;">Basket Diversity</div>
          <div style="font-size:1.1rem;font-weight:700;color:#4F8FF0;">{basket_div:.3f}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Copy all 5 to staging
    if st.button("📌 Stage All 5 Recommendations", use_container_width=True):
        hh_recs = top5[top5["household_key"] == hh_search].head(5)
        for idx, row in hh_recs.iterrows():
            stage_recommendation({
                "household_key":    hh_search,
                "commodity_desc":   str(row.get("COMMODITY_DESC", "")),
                "rank":             int(row.get("rank", idx)),
                "recommended_margin": float(row.get("Normalized_Margin", 0)),
                "discount_pct":     0,
                "incremental_margin_delta": 0,
                "archetype":        archetype,
            })
        st.success("All 5 staged ✓")

    # Baseline overlay toggle
    st.session_state["show_baseline_overlay"] = st.toggle(
        "Compare vs Popularity Baseline", value=False, key="baseline_toggle"
    )

    # Weight normalisation formula
    st.markdown("---")
    st.markdown("**📐 Current Weight Formula**")
    st.latex(r"w_i^{norm} = \frac{w_i}{\sum_j w_j}")
    w_df = pd.DataFrame({"Component": list(weights.keys()),
                         "Weight": [f"{v:.3f}" for v in weights.values()]})
    st.dataframe(w_df, use_container_width=True, hide_index=True)


with right_col:
    hh_recs = top5[top5["household_key"] == hh_search].head(5).reset_index(drop=True)

    if hh_recs.empty:
        st.warning(f"No recommendations found for Household {hh_search}.")
    else:
        st.markdown(f"### Top-5 Recommendations for HH #{hh_search}")

        # Render each recommendation card
        for rank_idx, row in hh_recs.iterrows():
            rank = int(row.get("rank", rank_idx + 1))
            render_rec_card(
                row=row,
                rank=rank,
                weights=weights,
                household_key=hh_search,
                archetype=archetype,
                show_stage_btn=True,
            )

    # Baseline overlay
    if st.session_state.get("show_baseline_overlay") and not hh_recs.empty:
        st.markdown("---")
        st.markdown("##### 📊 Popularity Baseline (Top-5 by frequency)")
        if "source_detail" in top5.columns:
            pop_items = (
                top5.groupby("COMMODITY_DESC")["household_key"]
                .nunique()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            pop_items.columns = ["COMMODITY_DESC", "Households"]
            st.dataframe(
                pop_items.style.background_gradient(cmap="Blues", subset=["Households"]),
                use_container_width=True, hide_index=True
            )

# ── Utility Landscape scatter ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌐 Utility Landscape – All Candidates for this Household")

with st.expander("Show Utility Landscape (loads scored candidates)", expanded=False):
    with st.spinner("Loading candidate set…"):
        cand = load_candidate_set_scored()

    if cand is not None and not cand.empty:
        hh_cand = cand[cand["household_key"] == hh_search]
        if not hh_cand.empty and all(c in hh_cand.columns for c in ["Relevance", "Uplift", "Utility", "Normalized_Margin"]):
            # Sample for speed
            if len(hh_cand) > 500:
                hh_cand = hh_cand.sample(500, random_state=42)

            # Mark top-5
            top5_comms = set(hh_recs["COMMODITY_DESC"].values)
            hh_cand = hh_cand.copy()
            hh_cand["In Top-5"] = hh_cand["COMMODITY_DESC"].isin(top5_comms)

            fig_scatter = px.scatter(
                hh_cand,
                x="Relevance", y="Uplift",
                color="Utility",
                size="Normalized_Margin",
                hover_name="COMMODITY_DESC",
                color_continuous_scale=[[0, "#1B1F28"], [0.5, "#4F8FF0"], [1, "#50F08F"]],
                symbol="In Top-5",
                symbol_map={True: "star", False: "circle"},
                title=f"Candidate Landscape – HH #{hh_search}",
            )
            fig_scatter.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Required columns not in candidate set for this household.")
    else:
        st.info("Candidate set not available (file may be too large or not present).")

# ── Export staged ─────────────────────────────────────────────────────────────
staged = get_staged()
if staged:
    st.markdown("---")
    st.markdown(f"### 📦 Export Campaign Pack ({len(staged)} staged items)")
    from utils.state_manager import staged_as_dataframe
    df_export = staged_as_dataframe()
    st.dataframe(df_export, use_container_width=True, hide_index=True)
    csv_bytes = df_export.to_csv(index=False).encode()
    st.download_button(
        "⬇ Export Campaign Pack (CSV)",
        data=csv_bytes,
        file_name="chimera_campaign_pack.csv",
        mime="text/csv",
        use_container_width=True,
    )
