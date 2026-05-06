"""
chimera-ui/app.py
Project Chimera – Streamlit Multi-Page Entry Point

Run with:  streamlit run chimera-ui/app.py
           (from the project root directory)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path bootstrap ───────────────────────────────────────────────────────────
_UI_DIR   = Path(__file__).resolve().parent      # chimera-ui/
_PROJ_DIR = _UI_DIR.parent                        # project root

for p in [str(_UI_DIR), str(_PROJ_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Project Chimera",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**Project Chimera** — Utility-Based Hybrid Recommendation Decision Support System",
    },
)

# ── Imports (after path bootstrap) ───────────────────────────────────────────
from utils.state_manager import init_session_state          # noqa: E402
from utils.ui_components import inject_css, render_sidebar_pulse  # noqa: E402
from utils.data_loader   import get_data_freshness           # noqa: E402

# ── Init session state ───────────────────────────────────────────────────────
init_session_state()
inject_css()

# ── Sidebar global controls ──────────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
      <div style="font-size:2rem;font-weight:900;
                  background:linear-gradient(90deg,#4F8FF0,#B050F0);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⬡ Chimera
      </div>
      <div style="font-size:0.72rem;color:#D1D5DB;letter-spacing:0.1em;
                  text-transform:uppercase;margin-top:-4px;">
        Decision Support System
      </div>
    </div>
    <hr style="border-color:#2B3040;margin:8px 0;">
    """, unsafe_allow_html=True)

    # Run mode toggle
    st.session_state["explore_mode"] = st.toggle(
        "⚡ Explore Mode (fast sampling)",
        value=st.session_state.get("explore_mode", True),
        help="In Explore Mode a stratified sample of ~500 households is used for speed. "
             "Disable for full-data Report Mode.",
    )

    # Stock-aware toggle
    st.session_state["stock_aware"] = st.toggle(
        "📦 Stock-Aware Ranking",
        value=st.session_state.get("stock_aware", False),
        help="Applies a utility penalty to items below low-inventory threshold.",
    )

    st.markdown("<hr style='border-color:#2B3040;margin:8px 0;'>", unsafe_allow_html=True)

    # Global weight sliders
    from utils.ui_components import render_sidebar_weights
    render_sidebar_weights()

    st.markdown("<hr style='border-color:#2B3040;margin:8px 0;'>", unsafe_allow_html=True)

    # Scenario manager (quick access)
    from utils.scenario_io import list_scenario_names, load_scenario_from_disk, save_scenario_to_disk
    from utils.state_manager import get_weights

    st.markdown("**📁 Scenario Manager**")
    saved_names = list_scenario_names()
    if saved_names:
        sel = st.selectbox("Load scenario", ["— select —"] + saved_names,
                           key="sidebar_scenario_sel")
        if sel != "— select —":
            sc = load_scenario_from_disk(sel)
            if sc and st.button("Load", key="load_sc_btn"):
                st.session_state["current_weights"] = sc["weights"]
                st.rerun()

    sc_name = st.text_input("Save current as…", key="sidebar_sc_name", placeholder="Scenario name")
    if st.button("💾 Save", key="sidebar_sc_save") and sc_name:
        save_scenario_to_disk(sc_name, get_weights())
        st.success(f"Saved '{sc_name}'")

    st.markdown("<hr style='border-color:#2B3040;margin:8px 0;'>", unsafe_allow_html=True)

    # System Pulse
    freshness = get_data_freshness()
    render_sidebar_pulse(freshness)

    # Staged recommendations counter
    n_staged = len(st.session_state.get("staged_recommendations", []))
    if n_staged:
        st.markdown(
            f'<div style="background:rgba(80,240,143,0.1);border:1px solid #50F08F;'
            f'border-radius:8px;padding:8px 14px;text-align:center;">'
            f'<span style="color:#50F08F;font-weight:700;">📌 {n_staged} items staged</span><br>'
            f'<span style="font-size:0.78rem;color:#D1D5DB;">Visit Campaign Export to download</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Navigation using st.Page ─────────────────────────────────────────────────
pages = [
    st.Page("pages/01_Executive_Dashboard.py",  title="Executive Dashboard",    icon="📊"),
    st.Page("pages/02_Household_Explorer.py",   title="Household Explorer",     icon="🔍"),
    st.Page("pages/03_Archetype_Lens.py",       title="Archetype Lens",         icon="🎯"),
    st.Page("pages/04_Weight_Simulator.py",     title="Weight Simulator",       icon="⚖"),
    st.Page("pages/05_Counterfactuals.py",      title="Counterfactuals",        icon="🔄"),
    st.Page("pages/06_Policy_Evaluation.py",    title="Policy Evaluation",      icon="📈"),
    st.Page("pages/07_Model_Health.py",         title="Model Health",           icon="🩺"),
    st.Page("pages/08_Feedback_Review.py",      title="Feedback Review",        icon="💬"),
    st.Page("pages/09_Campaign_Export.py",      title="Campaign Export",        icon="🚀"),
]

pg = st.navigation(pages)
pg.run()
