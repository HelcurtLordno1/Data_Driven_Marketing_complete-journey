"""
chimera-ui/utils/state_manager.py
Session-state initialisation and convenience helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.data_loader import load_top5, load_archetype_assignments

# Default weight configuration
DEFAULT_WEIGHTS: Dict[str, float] = {
    "relevance": 0.40,
    "uplift":    0.25,
    "margin":    0.20,
    "context":   0.15,
}

_FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback_log.csv"


def init_session_state() -> None:
    """Ensure all required session-state keys exist exactly once."""
    defaults: Dict[str, Any] = {
        "selected_household":      None,
        "current_weights":         dict(DEFAULT_WEIGHTS),
        "explore_mode":            True,
        "staged_recommendations":  [],   # list of dicts {household_key, commodity, rank, margin, discount_pct}
        "feedback_log":            [],   # list of dicts persisted to CSV
        "saved_scenarios":         [],   # list of {name, weights, archetype_filter}
        "active_archetype_filter": None,
        "stock_aware":             False,
        "show_baseline_overlay":   False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Weight helpers ───────────────────────────────────────────────────────────

def get_weights() -> Dict[str, float]:
    return st.session_state.get("current_weights", dict(DEFAULT_WEIGHTS))


def set_weights(weights: Dict[str, float]) -> None:
    total = sum(weights.values())
    if total > 0:
        st.session_state["current_weights"] = {k: v / total for k, v in weights.items()}
    else:
        st.session_state["current_weights"] = dict(DEFAULT_WEIGHTS)


def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in weights.items()}


# ── Staging helpers ──────────────────────────────────────────────────────────

def stage_recommendation(entry: Dict[str, Any]) -> None:
    staged: List[Dict] = st.session_state.get("staged_recommendations", [])
    # De-duplicate by (household_key, commodity_desc)
    key = (entry.get("household_key"), entry.get("commodity_desc"))
    if not any((r.get("household_key"), r.get("commodity_desc")) == key for r in staged):
        staged.append(entry)
        st.session_state["staged_recommendations"] = staged


def clear_staging() -> None:
    st.session_state["staged_recommendations"] = []


def get_staged() -> List[Dict[str, Any]]:
    return st.session_state.get("staged_recommendations", [])


def staged_as_dataframe() -> pd.DataFrame:
    staged = get_staged()
    if not staged:
        return pd.DataFrame(columns=["household_key", "commodity_desc",
                                     "rank", "recommended_margin",
                                     "discount_pct", "incremental_margin_delta"])
    df = pd.DataFrame(staged)
    for col in ["rank", "recommended_margin", "discount_pct", "incremental_margin_delta"]:
        if col not in df.columns:
            df[col] = 0
    return df


# ── Feedback helpers ─────────────────────────────────────────────────────────

def log_feedback(
    household_key: int,
    commodity_desc: str,
    rank: int,
    utility_score: float,
    feedback: str,           # "👍" | "👎"
    reason: str = "",
    archetype: str = "",
) -> None:
    import datetime
    entry = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "household_key": household_key,
        "commodity_desc": commodity_desc,
        "rank":          rank,
        "utility_score": round(utility_score, 4),
        "feedback":      feedback,
        "reason":        reason,
        "archetype":     archetype,
    }
    log: List[Dict] = st.session_state.get("feedback_log", [])
    log.append(entry)
    st.session_state["feedback_log"] = log

    # Append to CSV
    try:
        row_df = pd.DataFrame([entry])
        header = not _FEEDBACK_PATH.exists() or _FEEDBACK_PATH.stat().st_size == 0
        row_df.to_csv(_FEEDBACK_PATH, mode="a", header=header, index=False)
    except Exception:
        pass


def get_feedback_log() -> pd.DataFrame:
    if _FEEDBACK_PATH.exists() and _FEEDBACK_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(_FEEDBACK_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp", "household_key", "commodity_desc",
                                  "rank", "utility_score", "feedback", "reason", "archetype"])


# ── Scenario helpers ─────────────────────────────────────────────────────────

def save_scenario(name: str, weights: Optional[Dict] = None, archetype: Optional[str] = None) -> None:
    scenarios: List[Dict] = st.session_state.get("saved_scenarios", [])
    scenarios = [s for s in scenarios if s.get("name") != name]  # replace if same name
    scenarios.append({
        "name":     name,
        "weights":  weights or get_weights(),
        "archetype": archetype,
    })
    st.session_state["saved_scenarios"] = scenarios


def load_scenario(name: str) -> Optional[Dict]:
    for s in st.session_state.get("saved_scenarios", []):
        if s.get("name") == name:
            return s
    return None


def list_scenarios() -> List[str]:
    return [s["name"] for s in st.session_state.get("saved_scenarios", [])]
