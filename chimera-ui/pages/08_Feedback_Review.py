"""
pages/08_Feedback_Review.py
HITL Feedback Review & Fine-Tuning – review collected thumbs up/down flags.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader   import get_data_freshness
from utils.state_manager import init_session_state, get_feedback_log
from utils.ui_components import (inject_css, render_header, make_donut,
                                  PLOTLY_LAYOUT, COLOURS, ARCHETYPE_COLOURS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("HITL Feedback Review", freshness)

st.markdown("""
> **Purpose:** Review and analyse human-in-the-loop feedback collected from recommendation cards.
> Export filtered logs for model retraining.
""")

# ── Load feedback ─────────────────────────────────────────────────────────────
fb_df = get_feedback_log()

# Running count card
n_total    = len(fb_df)
n_positive = int((fb_df["feedback"] == "👍").sum()) if not fb_df.empty else 0
n_negative = int((fb_df["feedback"] == "👎").sum()) if not fb_df.empty else 0
pct_pos    = (n_positive / n_total * 100) if n_total > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Feedback", n_total)
c2.metric("👍 Positive",    n_positive, f"{pct_pos:.0f}%")
c3.metric("👎 Negative",    n_negative, f"{100-pct_pos:.0f}%")
c4.metric("Archetypes Seen",
          fb_df["archetype"].nunique() if not fb_df.empty else 0)

st.markdown("---")

if fb_df.empty:
    st.info("No feedback collected yet. Use the 👍/👎 buttons on the Household Explorer page to log feedback.")
    st.markdown("""
    **How HITL Feedback Works:**
    1. Navigate to **Household Explorer** and search for a household.
    2. Click 👍 if a recommendation is relevant, or 👎 if it's not.
    3. Optionally add a reason when clicking 👎.
    4. All feedback is logged here and saved to `data/feedback_log.csv`.
    5. Use **Export for Training** below to extract a clean CSV for model retraining.
    """)
else:
    # ── Filters ────────────────────────────────────────────────────────────────
    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        arch_opts = ["All"] + sorted(fb_df["archetype"].dropna().unique().tolist())
        sel_arch  = st.selectbox("Filter by Archetype", arch_opts, key="fb_arch_filter")
    with filt_col2:
        fb_type   = st.selectbox("Filter by Feedback", ["All", "👍 Positive", "👎 Negative"], key="fb_type_filter")

    filtered = fb_df.copy()
    if sel_arch != "All":
        filtered = filtered[filtered["archetype"] == sel_arch]
    if fb_type == "👍 Positive":
        filtered = filtered[filtered["feedback"] == "👍"]
    elif fb_type == "👎 Negative":
        filtered = filtered[filtered["feedback"] == "👎"]

    st.markdown(f"Showing **{len(filtered)}** of **{n_total}** feedback records")

    # ── Feedback table ─────────────────────────────────────────────────────────
    display_cols = [c for c in ["timestamp","household_key","commodity_desc","rank",
                                 "utility_score","feedback","archetype","reason"]
                    if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].sort_values("timestamp", ascending=False)
                               .style.map(
            lambda v: "color:#50F08F" if v == "👍" else ("color:#F05050" if v == "👎" else ""),
            subset=["feedback"] if "feedback" in display_cols else []
        ),
        use_container_width=True, hide_index=True
    )

    # ── Summary charts ─────────────────────────────────────────────────────────
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Feedback Sentiment by Archetype")
        if not filtered.empty and "archetype" in filtered.columns:
            arch_fb = (
                filtered.groupby(["archetype", "feedback"])
                .size().reset_index(name="count")
            )
            fig_af = px.bar(
                arch_fb, x="archetype", y="count", color="feedback",
                barmode="group",
                color_discrete_map={"👍": COLOURS["margin"], "👎": COLOURS["danger"]},
                title="Feedback by Archetype"
            )
            fig_af.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_af, use_container_width=True)

    with chart_col2:
        st.markdown("#### Overall Sentiment Pie")
        if n_total > 0:
            fig_pie = make_donut(
                ["👍 Positive", "👎 Negative"],
                [n_positive, n_negative],
                "Feedback Sentiment",
                colours=[COLOURS["margin"], COLOURS["danger"]],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Most-flagged negative items ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚠ Most Frequently Flagged Negative Items")
    neg_fb = fb_df[fb_df["feedback"] == "👎"] if not fb_df.empty else pd.DataFrame()
    if not neg_fb.empty and "commodity_desc" in neg_fb.columns:
        top_neg = (
            neg_fb.groupby("commodity_desc")
            .agg(count=("feedback","size"), reasons=("reason","apply", lambda x: "; ".join(x.dropna().astype(str).tolist())))
            .sort_values("count", ascending=False)
            .head(10)
            .reset_index()
        )
        st.dataframe(top_neg, use_container_width=True, hide_index=True)
    else:
        st.caption("No negative feedback yet.")

    # ── Export button ──────────────────────────────────────────────────────────
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_all = fb_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export All Feedback (CSV)",
            data=csv_all,
            file_name="chimera_feedback_all.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp2:
        neg_for_training = fb_df[fb_df["feedback"] == "👎"].copy() if not fb_df.empty else pd.DataFrame()
        csv_neg = neg_for_training.to_csv(index=False).encode()
        st.download_button(
            "📤 Export Negatives for Retraining",
            data=csv_neg,
            file_name="chimera_retraining_negatives.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export 👎 feedback as negative examples for future model fine-tuning.",
        )
