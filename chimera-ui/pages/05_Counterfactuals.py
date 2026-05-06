"""
pages/05_Counterfactuals.py
Counterfactual Explorer – "what if" scenarios for marketing nudges.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader   import (load_top5, load_archetype_assignments,
                                  load_candidate_set_scored, get_data_freshness)
from utils.state_manager import init_session_state, get_weights
from utils.ui_components import (inject_css, render_header, make_waterfall,
                                  PLOTLY_LAYOUT, COLOURS, ARCHETYPE_COLOURS)

init_session_state()
inject_css()

freshness = get_data_freshness()
render_header("Counterfactual Explorer", freshness)

st.markdown("""
> **Purpose:** Investigate "what-if" scenarios — discover how much a shopper's behaviour
> would need to change for a specific item to enter their top-5.
""")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    top5       = load_top5()
    archetypes = load_archetype_assignments()
    weights    = get_weights()

if top5.empty:
    st.error("Top-5 data not available.")
    st.stop()

all_hh = sorted(top5["household_key"].unique().tolist())

# ── Household selector ────────────────────────────────────────────────────────
ctrl_col, _ = st.columns([1, 2])
with ctrl_col:
    hh_key = st.selectbox("Select Household", all_hh, key="cf_hh_sel")
    target_rank = st.number_input("Target rank to achieve", min_value=1, max_value=5, value=5)

hh_recs = top5[top5["household_key"] == hh_key].head(5).reset_index(drop=True)
arch_row = archetypes[archetypes["household_key"] == hh_key] if not archetypes.empty else pd.DataFrame()
archetype = str(arch_row["archetype"].iloc[0]) if not arch_row.empty and "archetype" in arch_row else "Unknown"

st.markdown(f"#### Top-5 for HH #{hh_key} &nbsp; <span style='color:{ARCHETYPE_COLOURS.get(archetype,'#D1D5DB')};font-size:0.85rem;'>{archetype}</span>", unsafe_allow_html=True)
if not hh_recs.empty:
    display_cols = [c for c in ["rank", "COMMODITY_DESC", "Utility", "Relevance", "Uplift",
                                 "Normalized_Margin", "Context"] if c in hh_recs.columns]
    st.dataframe(
        hh_recs[display_cols].style.format(
            {c: "{:.4f}" for c in display_cols if c not in ("rank", "COMMODITY_DESC")}
        ).background_gradient(cmap="Blues", subset=[c for c in display_cols if c == "Utility"]),
        use_container_width=True, hide_index=True
    )

st.markdown("---")

# ── Item selector for counterfactual ─────────────────────────────────────────
st.markdown("### 🔍 Select Item for Counterfactual Analysis")
if not hh_recs.empty and "COMMODITY_DESC" in hh_recs.columns:
    sel_item = st.selectbox("Choose item to analyse", hh_recs["COMMODITY_DESC"].tolist(), key="cf_item_sel")

    item_row = hh_recs[hh_recs["COMMODITY_DESC"] == sel_item].iloc[0]
    orig_rank = int(item_row.get("rank", 1))
    orig_util = float(item_row.get("Utility", 0.0))
    orig_upl  = float(item_row.get("Uplift", 0.0))
    orig_rel  = float(item_row.get("Relevance", 0.0))
    orig_mrg  = float(item_row.get("Normalized_Margin", 0.0))
    orig_ctx  = float(item_row.get("Context", 0.0))

    w = weights
    total_w = sum(w.values()) or 1.0
    w_norm = {k: v / total_w for k, v in w.items()}

    # Threshold utility = utility of item currently at target_rank
    if target_rank <= len(hh_recs):
        threshold_row = hh_recs.iloc[target_rank - 1]
        threshold_util = float(threshold_row.get("Utility", orig_util + 0.05))
    else:
        threshold_util = orig_util + 0.05

    delta_needed = max(0.0, threshold_util - orig_util)

    # Solve: new_uplift = (threshold - fixed_part) / w_uplift
    fixed_part = (w_norm["relevance"] * orig_rel
                  + w_norm["margin"]   * orig_mrg
                  + w_norm["context"]  * orig_ctx)
    uplift_needed = (threshold_util - fixed_part) / (w_norm["uplift"] or 0.001)
    uplift_needed = float(np.clip(uplift_needed, 0.0, 1.0))
    habit_needed  = float(np.clip(1.0 - uplift_needed, 0.0, 1.0))
    new_util = (w_norm["relevance"] * orig_rel
                + w_norm["uplift"]   * uplift_needed
                + w_norm["margin"]   * orig_mrg
                + w_norm["context"]  * orig_ctx)

    # ── Result cards ──────────────────────────────────────────────────────────
    st.markdown(f"#### Counterfactual: '{sel_item}' → Rank #{target_rank}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Original Rank",    f"#{orig_rank}")
    r2.metric("Target Rank",      f"#{target_rank}")
    r3.metric("Δ Utility Needed", f"+{delta_needed:.4f}")
    r4.metric("New Utility",      f"{new_util:.4f}", f"{'▲' if new_util > orig_util else '▼'}{abs(new_util - orig_util):.4f}")

    hab_delta = habit_needed - (1.0 - orig_upl)
    st.markdown(f"""
    <div style="background:#1B1F28;border:1px solid #2B3040;border-radius:10px;padding:14px 18px;margin:10px 0;">
      <div style="color:#D1D5DB;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;">Narrative</div>
      <div style="margin-top:8px;line-height:1.6;color:#FFFFFF;">
        '<b>{sel_item}</b>' ranked <b>#{orig_rank}</b> (utility={orig_util:.4f}).
        To reach rank <b>#{target_rank}</b>, utility must be ≥ <b>{threshold_util:.4f}</b> (Δ={delta_needed:.4f}).
        This requires habit strength to drop from <b>{1.0-orig_upl:.3f}</b> → <b>{habit_needed:.3f}</b>
        (raising Uplift from <b>{orig_upl:.3f}</b> → <b>{uplift_needed:.3f}</b>).
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dynamic coupon generator ──────────────────────────────────────────────
    st.markdown("#### 🏷 Dynamic Coupon Generator")
    st.caption("A discount reduces the shopper's purchase frequency (habit strength) → raises Uplift")
    disc_pct = st.slider("Proposed Discount %", 0, 50, 0, 5, key="cf_disc_slider")
    
    # Proxy: each 10% discount → 0.05 habit strength reduction
    habit_disc = max(0.0, (1.0 - orig_upl) - disc_pct * 0.005)
    uplift_disc = 1.0 - habit_disc
    
    # Added margin impact to the simulation
    mrg_disc = max(0.0, orig_mrg - (disc_pct / 100))
    
    util_disc = (w_norm["relevance"] * orig_rel
                 + w_norm["uplift"] * uplift_disc
                 + w_norm["margin"] * mrg_disc
                 + w_norm["context"] * orig_ctx)

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Proposed Discount", f"{disc_pct}%")
    dc2.metric("Projected Uplift",  f"{uplift_disc:.3f}", f"{uplift_disc - orig_upl:+.3f}")
    dc3.metric("Projected Utility", f"{util_disc:.4f}",  f"{util_disc - orig_util:+.4f}")

    # ── Waterfall: Proposed Impact ────────────────────────────────────────────
    st.markdown("#### Waterfall: Utility Shift from Proposed Coupon")
    
    # Calculate components of the shift
    rel_shift = 0.0
    upl_shift = w_norm["uplift"] * (uplift_disc - orig_upl)
    mrg_shift = w_norm["margin"]  * (mrg_disc - orig_mrg)
    ctx_shift = 0.0
    
    labels = ["Original Utility", "Δ Relevance", "Δ Uplift", "Δ Margin", "Δ Context", "Projected Utility"]
    values = [orig_util, rel_shift, upl_shift, mrg_shift, ctx_shift, util_disc]
    
    fig_wf = make_waterfall(labels, values, f"Impact of {disc_pct}% Discount on '{sel_item}'")
    
    # Add a 'Target Line' to the waterfall
    fig_wf.add_hline(y=threshold_util, line_dash="dash", line_color="#F0C850", 
                      annotation_text=f"Target for Rank #{target_rank}", 
                      annotation_position="bottom right")
    
    st.plotly_chart(fig_wf, use_container_width=True)

    needed_disc = max(0, int(np.ceil(max(0, (1.0 - orig_upl) - habit_needed) / 0.005)))
    st.info(f"ℹ Estimated minimum discount to reach Rank #{target_rank}: **~{needed_disc}%**")

else:
    st.info("Select a household with recommendations to begin.")
