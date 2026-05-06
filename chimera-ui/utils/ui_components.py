"""
chimera-ui/utils/ui_components.py
Reusable Streamlit rendering helpers: cards, KPI tiles, utility bars,
radar charts, popovers, gauge charts, and more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Accent palette from spec
COLOURS = {
    "relevance": "#4F8FF0",
    "uplift":    "#F08F50",
    "margin":    "#50F08F",
    "context":   "#B050F0",
    "danger":    "#F05050",
    "warning":   "#F0C850",
    "success":   "#50F08F",
    "muted":     "#D1D5DB",
    "bg":        "#1B1F28",
    "border":    "#2B3040",
}

ARCHETYPE_COLOURS = {
    "Routine Replenisher":  "#4F8FF0",
    "Deal-Driven Explorer": "#F08F50",
    "Premium Discoverer":   "#50F08F",
    "Frugal Loyalist":      "#B050F0",
    "Unknown":              "#D1D5DB",
}

COMPONENT_KEYS    = ["relevance", "uplift", "margin", "context"]
COMPONENT_LABELS  = ["Relevance", "Uplift", "Margin", "Context"]
COMPONENT_COLS    = ["Relevance", "Uplift", "Normalized_Margin", "Context"]


# ── CSS loader ───────────────────────────────────────────────────────────────

def inject_css() -> None:
    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Header banner ────────────────────────────────────────────────────────────

def render_header(page_title: str, freshness: Optional[Dict] = None) -> None:
    max_day   = freshness.get("max_day", "N/A") if freshness else "N/A"
    days_ago  = freshness.get("days_since", 0) if freshness else 0
    if isinstance(days_ago, (int, float)) and days_ago > 7:
        dot_class = "dot-yellow"
        status_text = f"⚠ Data {days_ago}d old"
    else:
        dot_class = "dot-green"
        status_text = "● Live"

    explore_badge = ""
    if st.session_state.get("explore_mode", True):
        explore_badge = '<span style="font-size:0.7rem;background:rgba(240,143,80,0.18);color:#F08F50;padding:2px 10px;border-radius:999px;font-weight:700;margin-left:10px;">EXPLORE MODE</span>'

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#0E1117 0%,#151b2a 50%,#0E1117 100%);
                border-bottom:1px solid #2B3040;padding:12px 4px 12px 0;
                display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="font-size:1.5rem;font-weight:900;
                  background:linear-gradient(90deg,#4F8FF0,#B050F0);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⬡ Project Chimera
      </div>
      <div style="color:#D1D5DB;font-size:0.85rem;margin-left:8px;">{page_title}</div>
      {explore_badge}
      <div style="margin-left:auto;color:#D1D5DB;font-size:0.8rem;">
        <span class="dot {dot_class}"></span>{status_text}&nbsp;|&nbsp;Day max: {max_day}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar_pulse(freshness: Optional[Dict] = None) -> None:
    days = freshness.get("days_since", 0) if freshness else 0
    max_day = freshness.get("max_day", "N/A") if freshness else "N/A"
    col = "#50F08F" if (isinstance(days, (int, float)) and days <= 7) else "#F0C850"
    st.sidebar.markdown(f"""
    <div style="background:#131720;border:1px solid #2B3040;border-radius:8px;padding:10px 14px;margin-bottom:12px;">
      <div style="font-size:0.72rem;color:#D1D5DB;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">System Pulse</div>
      <div style="font-size:0.85rem;"><span style="color:{col};">●</span> Freshness: Day {max_day}</div>
      <div style="font-size:0.85rem;color:#D1D5DB;">Staged: {len(st.session_state.get('staged_recommendations',[]))}</div>
      <div style="font-size:0.85rem;color:#D1D5DB;">Feedback: {len(st.session_state.get('feedback_log',[]))} flags</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_weights() -> Dict[str, float]:
    """Render global weight sliders and return normalised weights."""
    from utils.state_manager import normalise_weights
    st.sidebar.markdown("**⚖ Utility Weights**")
    weights_raw = {}
    current = st.session_state.get("current_weights", {"relevance":0.4,"uplift":0.25,"margin":0.2,"context":0.15})
    for key, colour, label in zip(
        COMPONENT_KEYS,
        [COLOURS["relevance"], COLOURS["uplift"], COLOURS["margin"], COLOURS["context"]],
        COMPONENT_LABELS,
    ):
        weights_raw[key] = st.sidebar.slider(
            f"{label}",
            min_value=0.0, max_value=1.0,
            value=float(current.get(key, 0.25)),
            step=0.05, key=f"sidebar_w_{key}"
        )
    norm = normalise_weights(weights_raw)
    st.session_state["current_weights"] = norm
    # Show normalised values
    parts = " + ".join([f"**{v:.2f}** {l}" for v, l in zip(norm.values(), COMPONENT_LABELS)])
    st.sidebar.caption(f"Σ=1 → {parts}")
    return norm


# ── Utility-decomposition bar ────────────────────────────────────────────────

def utility_bar_html(
    relevance: float, uplift: float, margin: float, context: float,
    weights: Optional[Dict[str, float]] = None,
    height: int = 12,
) -> str:
    """Return an HTML coloured stacked bar representing weighted utility."""
    w = weights or {"relevance":0.4,"uplift":0.25,"margin":0.2,"context":0.15}
    total = sum(w.values()) or 1.0
    segs = [
        (w["relevance"]/total * relevance,  COLOURS["relevance"]),
        (w["uplift"]/total    * uplift,     COLOURS["uplift"]),
        (w["margin"]/total    * margin,     COLOURS["margin"]),
        (w["context"]/total   * context,    COLOURS["context"]),
    ]
    bars = "".join(
        f'<div style="flex:{v:.4f};background:{c};height:{height}px;"></div>'
        for v, c in segs
    )
    return f'<div style="display:flex;border-radius:4px;overflow:hidden;height:{height}px;">{bars}</div>'


# ── Recommendation card ──────────────────────────────────────────────────────

def render_rec_card(
    row: pd.Series,
    rank: int,
    weights: Dict[str, float],
    household_key: int,
    archetype: str = "",
    show_stage_btn: bool = True,
) -> None:
    from utils.state_manager import stage_recommendation, log_feedback
    rel  = float(row.get("Relevance", 0))
    upl  = float(row.get("Uplift", 0))
    mrg  = float(row.get("Normalized_Margin", 0))
    ctx  = float(row.get("Context", 0))
    util = float(row.get("Utility", 0))
    comm = str(row.get("COMMODITY_DESC", "Unknown"))
    src  = str(row.get("source_detail", ""))

    src_badge = ""
    if src in ("ALS", "MBA", "BOTH"):
        badge_col = {"ALS": COLOURS["relevance"], "MBA": COLOURS["uplift"], "BOTH": COLOURS["margin"]}[src]
        src_badge = f'<span style="font-size:0.68rem;background:rgba(0,0,0,0.3);color:{badge_col};padding:1px 8px;border-radius:999px;margin-left:6px;">{src}</span>'

    bar_html = utility_bar_html(rel, upl, mrg, ctx, weights=weights)

    st.markdown(f"""
    <div style="background:#1B1F28;border:1px solid #2B3040;border-radius:10px;
                padding:14px 18px;margin-bottom:10px;">
      <div style="display:flex;align-items:baseline;gap:10px;">
        <span style="font-size:1.4rem;font-weight:900;color:#FFFFFF;">#{rank}</span>
        <span style="font-size:1rem;font-weight:700;color:#FFFFFF;">{comm}</span>
        {src_badge}
        <span style="margin-left:auto;font-size:1.1rem;font-weight:700;
                     background:linear-gradient(90deg,#4F8FF0,#B050F0);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          U={util:.4f}
        </span>
      </div>
      <div style="margin-top:8px;">{bar_html}</div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:0.8rem;color:#D1D5DB;">
        <span><span style="color:#4F8FF0;">●</span> Rel {rel:.3f}</span>
        <span><span style="color:#F08F50;">●</span> Upl {upl:.3f}</span>
        <span><span style="color:#50F08F;">●</span> Mrg {mrg:.3f}</span>
        <span><span style="color:#B050F0;">●</span> Ctx {ctx:.3f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        if st.button("👍", key=f"thumb_up_{household_key}_{rank}_{comm[:8]}"):
            log_feedback(household_key, comm, rank, util, "👍", archetype=archetype)
            st.success("Logged ✓")
    with btn_cols[1]:
        if st.button("👎", key=f"thumb_dn_{household_key}_{rank}_{comm[:8]}"):
            reason = st.text_input("Why not?", key=f"reason_{household_key}_{rank}")
            log_feedback(household_key, comm, rank, util, "👎", reason=reason, archetype=archetype)
            st.warning("Logged ✓")
    with btn_cols[2]:
        if show_stage_btn:
            # Check for discount from slider
            active_disc = st.session_state.get(f"disc_{household_key}_{rank}_{comm[:8]}", 0)
            btn_label = f"📌 Stage #{rank}" + (f" (-{active_disc}%)" if active_disc > 0 else "")
            
            if st.button(btn_label, key=f"stage_{household_key}_{rank}_{comm[:8]}"):
                # Calculate final adjusted margin for the export
                final_mrg = max(0.0, mrg - (active_disc / 100))
                stage_recommendation({
                    "household_key":    household_key,
                    "commodity_desc":   comm,
                    "rank":             rank,
                    "recommended_margin": round(final_mrg, 4),
                    "discount_pct":     active_disc,
                    "incremental_margin_delta": round(final_mrg - mrg, 4),
                    "archetype":        archetype,
                })
                st.success(f"Staged with {active_disc}% discount ✓")

    # Popover explanation (inside expander for compatibility)
    with st.expander(f"🔍 Why? (decomposition)", expanded=False):
        w = weights
        st.markdown(f"""
        | Component | Raw Score | Weight | Contribution |
        |:----------|----------:|-------:|-------------:|
        | Relevance | {rel:.4f} | {w.get('relevance',0):.2f} | {w.get('relevance',0)*rel:.4f} |
        | Uplift    | {upl:.4f} | {w.get('uplift',0):.2f}    | {w.get('uplift',0)*upl:.4f}    |
        | Margin    | {mrg:.4f} | {w.get('margin',0):.2f}    | {w.get('margin',0)*mrg:.4f}    |
        | Context   | {ctx:.4f} | {w.get('context',0):.2f}   | {w.get('context',0)*ctx:.4f}   |
        | **Total** |  | | **{util:.4f}** |
        """)
        st.latex(r"U = w_r \cdot R + w_u \cdot U + w_m \cdot M + w_c \cdot C")

    # Dynamic coupon generator for deal-driven archetypes
    if archetype in ("Deal-Driven Explorer", "Frugal Loyalist"):
        with st.expander("🏷 Suggest Discount (Trade-off Analysis)", expanded=False):
            disc_key = f"disc_{household_key}_{rank}_{comm[:8]}"
            disc = st.slider(
                "Discount %", 0, 40, 0, 5,
                key=disc_key,
                help="A higher discount increases Context score but reduces Margin score."
            )
            
            # Weighted trade-off logic
            # 1. Discount boosts context (deal sensitivity)
            proj_ctx = min(1.0, ctx + (disc / 100) * 2.0) # Assume 10% discount = +0.2 context
            # 2. Discount reduces margin
            proj_mrg = max(0.0, mrg - (disc / 100))
            
            proj_util = (w.get("relevance",0)*rel + w.get("uplift",0)*upl
                         + w.get("margin",0)*proj_mrg + w.get("context",0)*proj_ctx)
            delta = proj_util - util
            
            mc1, mc2 = st.columns(2)
            mc1.metric("Projected Utility", f"{proj_util:.4f}", f"{delta:+.4f}")
            mc2.metric("Net Margin impact", f"{proj_mrg:.4f}", f"{proj_mrg - mrg:+.4f}", delta_color="inverse")
            
            if disc > 0:
                st.info(f"💡 Apply a {disc}% coupon to shift this item's utility.")


# ── Plotly helpers ───────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="Roboto, sans-serif", color="#FFFFFF"),
    # Legend default
    legend=dict(font=dict(color="#D1D5DB", size=11)),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(tickfont=dict(color="#D1D5DB"), title_font=dict(color="#FFFFFF"), gridcolor="#2B3040"),
    yaxis=dict(tickfont=dict(color="#D1D5DB"), title_font=dict(color="#FFFFFF"), gridcolor="#2B3040"),
    coloraxis=dict(colorbar=dict(tickfont=dict(color="#D1D5DB"))),
)


def make_donut(labels: List[str], values: List[float], title: str, colours: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colours or [COLOURS["relevance"], COLOURS["uplift"], COLOURS["margin"], COLOURS["context"]],
        textfont_size=12,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=16)),
        **PLOTLY_LAYOUT,
        showlegend=True
    )
    return fig


def make_radar(categories: List[str], values: List[float], title: str, colour: str = "#4F8FF0") -> go.Figure:
    cats = categories + [categories[0]]
    vals = values + [values[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor=f"rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.25)",
        line_color=colour, line_width=2,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=16)),
        polar=dict(
            bgcolor="#1B1F28",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2B3040", tickfont=dict(color="#D1D5DB")),
            angularaxis=dict(gridcolor="#2B3040", tickfont=dict(color="#D1D5DB")),
        ),
        **PLOTLY_LAYOUT,
    )
    return fig


def make_gauge(value: float, title: str) -> go.Figure:
    if value < 0.20:
        bar_color = COLOURS["margin"]
    elif value < 0.40:
        bar_color = COLOURS["warning"]
    else:
        bar_color = COLOURS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 26, "color": "#FFFFFF"}},
        title={"text": title, "font": {"color": "#D1D5DB", "size": 13}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#D1D5DB", "tickfont": {"color": "#D1D5DB"}},
            "bar":   {"color": bar_color},
            "steps": [
                {"range": [0, 20],   "color": "rgba(80,240,143,0.12)"},
                {"range": [20, 40],  "color": "rgba(240,200,80,0.12)"},
                {"range": [40, 100], "color": "rgba(240,80,80,0.12)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": value * 100},
            "bgcolor": "#1B1F28",
            "bordercolor": "#2B3040",
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def make_waterfall(labels: List[str], values: List[float], title: str) -> go.Figure:
    measures = ["relative"] * (len(labels) - 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures,
        x=labels, y=values,
        connector={"line": {"color": "#2B3040"}},
        increasing={"marker": {"color": COLOURS["margin"]}},
        decreasing={"marker": {"color": COLOURS["danger"]}},
        totals={"marker": {"color": COLOURS["relevance"]}},
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=16)),
        **PLOTLY_LAYOUT,
        showlegend=False
    )
    return fig


def make_bump_chart(rank_data: Dict[str, List]) -> go.Figure:
    """rank_data: {household_key: [rank_at_scenario_A, rank_at_scenario_B]}"""
    fig = go.Figure()
    colours_list = [COLOURS["relevance"], COLOURS["uplift"], COLOURS["margin"],
                    COLOURS["context"], COLOURS["warning"]]
    for i, (hh, ranks) in enumerate(rank_data.items()):
        col = colours_list[i % len(colours_list)]
        fig.add_trace(go.Scatter(
            x=["Scenario A", "Scenario B"], y=ranks,
            mode="lines+markers+text",
            name=f"HH {hh}",
            text=[f"#{r}" for r in ranks],
            textposition="middle right",
            line=dict(color=col, width=2),
            marker=dict(color=col, size=10),
            textfont=dict(color="#FFFFFF", size=11),
        ))
    fig.update_yaxes(autorange="reversed", title="Rank")
    fig.update_layout(
        title=dict(text="Rank Shift Across Scenarios", font=dict(color="#FFFFFF", size=16)),
        **PLOTLY_LAYOUT
    )
    return fig
