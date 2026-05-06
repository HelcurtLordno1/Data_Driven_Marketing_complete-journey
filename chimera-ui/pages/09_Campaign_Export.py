"""
pages/09_Campaign_Export.py
Campaign Export Module (Action Center) – convert insights into operational campaigns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime
import json
import pandas as pd
import streamlit as st

from utils.data_loader   import get_data_freshness
from utils.state_manager import (init_session_state, get_staged,
                                  staged_as_dataframe, clear_staging)
from utils.ui_components import inject_css, render_header, PLOTLY_LAYOUT, COLOURS
import plotly.express as px

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Campaign Export (Action Center)", freshness)

st.markdown("""
> **Purpose:** Convert all staged recommendations into an operational campaign package.
> Download as CSV or simulate a webhook POST to your marketing API.
""")

# ── Staging status ─────────────────────────────────────────────────────────────
staged = get_staged()
n_staged = len(staged)

if n_staged == 0:
    st.warning("""
    **No items staged yet.**

    To stage recommendations:
    - Go to **Household Explorer** → click **Stage All 5** or individual **📌 Stage** buttons.
    - Go to **Policy Evaluation → Optimal Targeting List** → select rows and click **Stage**.
    """)
else:
    st.success(f"✅ **{n_staged} recommendation(s) staged** and ready for export.")

# ── Staged items table ────────────────────────────────────────────────────────
st.markdown("### 📋 Staged Recommendations")
df_staged = staged_as_dataframe()

if not df_staged.empty:
    # Editable discount column
    st.markdown("You can edit the discount percentage directly in the table:")
    edited_df = st.data_editor(
        df_staged,
        column_config={
            "household_key":     st.column_config.NumberColumn("Household", disabled=True),
            "commodity_desc":    st.column_config.TextColumn("Item", disabled=True),
            "rank":              st.column_config.NumberColumn("Rank", disabled=True),
            "recommended_margin":st.column_config.NumberColumn("Margin (norm)", format="%.4f", disabled=True),
            "discount_pct":      st.column_config.NumberColumn("Discount %", min_value=0, max_value=50, step=5),
            "incremental_margin_delta": st.column_config.NumberColumn("Δ Margin", format="%.4f", disabled=True),
        },
        use_container_width=True, hide_index=True, key="staged_editor"
    )

    # Recompute incremental_margin_delta based on discount
    if "discount_pct" in edited_df.columns and "recommended_margin" in edited_df.columns:
        edited_df["incremental_margin_delta"] = (
            edited_df["recommended_margin"] * (edited_df["discount_pct"] / 100)
        ).round(4)

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    total_margin = float(edited_df["recommended_margin"].sum())
    total_margin_delta = float(edited_df["incremental_margin_delta"].sum())
    n_hh = int(edited_df["household_key"].nunique())
    avg_disc = float(edited_df["discount_pct"].mean())

    s1.metric("Staged Items",       n_staged)
    s2.metric("Unique Households",  n_hh)
    s3.metric("Total Export Margin", f"{total_margin:.3f}")
    s4.metric("Avg Discount %",      f"{avg_disc:.0f}%")

    # ── Archetype breakdown ───────────────────────────────────────────────────
    st.markdown("---")
    col_chart, col_top = st.columns(2)

    with col_chart:
        if "archetype" in edited_df.columns and not edited_df["archetype"].isna().all():
            arch_counts = edited_df["archetype"].fillna("Unknown").value_counts()
            fig_arch = px.pie(
                names=arch_counts.index.tolist(),
                values=arch_counts.values.tolist(),
                title="Staged Items by Archetype",
                color_discrete_sequence=[COLOURS["relevance"], COLOURS["uplift"],
                                          COLOURS["margin"], COLOURS["context"]],
                hole=0.5,
            )
            fig_arch.update_layout(**PLOTLY_LAYOUT, showlegend=True)
            st.plotly_chart(fig_arch, use_container_width=True)

    with col_top:
        st.markdown("#### Top Staged Items by Frequency")
        if "commodity_desc" in edited_df.columns:
            top_items = (
                edited_df.groupby("commodity_desc")
                .agg(count=("household_key","size"),
                     avg_margin=("recommended_margin","mean"))
                .sort_values("count", ascending=False)
                .head(8).reset_index()
            )
            st.dataframe(top_items.style.format({"avg_margin":"{:.4f}"}),
                         use_container_width=True, hide_index=True)

    # ── Export buttons ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Generate & Export Campaign Pack")

    # Build the full campaign CSV
    campaign_df = edited_df[[c for c in
                              ["household_key","commodity_desc","rank",
                               "recommended_margin","discount_pct",
                               "incremental_margin_delta","archetype"]
                              if c in edited_df.columns]].copy()
    campaign_df.insert(0, "export_timestamp",
                       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    csv_bytes = campaign_df.to_csv(index=False).encode()

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            "⬇ Download Campaign Pack (CSV)",
            data=csv_bytes,
            file_name=f"chimera_campaign_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        # Webhook simulation
        if st.button("📡 Send to Webhook (Simulate)", use_container_width=True):
            payload = campaign_df.to_dict(orient="records")
            st.code(json.dumps({"endpoint": "https://api.example.com/campaigns",
                                 "method": "POST",
                                 "payload_preview": payload[:3],
                                 "total_records": len(payload)}, indent=2),
                    language="json")
            st.success("✅ Webhook POST simulated (not sent – configure endpoint for production).")

    with export_col3:
        if st.button("🗑 Clear All Staging", use_container_width=True, type="secondary"):
            clear_staging()
            st.warning("Staging cleared.")
            st.rerun()

    # ── Campaign notes ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Campaign Notes")
    notes = st.text_area(
        "Add campaign notes (included in JSON export):",
        placeholder="e.g. 'Q2 Margin Drive – focus on Deal-Driven Explorer segment'",
        key="campaign_notes", height=100
    )

    if st.button("💾 Save Campaign with Notes"):
        full_export = {
            "timestamp":  datetime.datetime.now().isoformat(),
            "n_items":    n_staged,
            "notes":      notes,
            "items":      campaign_df.to_dict(orient="records"),
        }
        json_bytes = json.dumps(full_export, indent=2).encode()
        st.download_button(
            "⬇ Download Campaign JSON",
            data=json_bytes,
            file_name=f"chimera_campaign_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )

else:
    st.info("Stage some recommendations first to build the campaign pack.")

# ── Footer instructions ────────────────────────────────────────────────────────
with st.expander("ℹ How to use the Campaign Export Module", expanded=False):
    st.markdown("""
    1. **Stage items** via Household Explorer (📌 Stage buttons) or Policy Evaluation (Optimal Targeting).
    2. **Review** the staged table above and adjust discount percentages as needed.
    3. **Download** the CSV pack or simulate a webhook POST to your CRM system.
    4. **Clear** the staging queue once the campaign is dispatched.

    **CSV Columns:**
    - `household_key` – Target customer identifier
    - `commodity_desc` – Recommended product category
    - `rank` – Recommendation rank (1-5)
    - `recommended_margin` – Normalised margin score
    - `discount_pct` – Suggested coupon/discount depth
    - `incremental_margin_delta` – Estimated incremental margin from discount
    - `archetype` – Customer behavioural archetype
    """)
