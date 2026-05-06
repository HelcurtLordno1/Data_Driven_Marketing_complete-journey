"""
chimera-ui/utils/data_loader.py
Centralised, cached data-loading layer for the Chimera Streamlit UI.

All expensive I/O is wrapped in @st.cache_data (TTL = session).
A helper draws a stratified sample by archetype for Explore Mode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Resolve project root & add src to import path ───────────────────────────
_UI_DIR   = Path(__file__).resolve().parent.parent          # chimera-ui/
_PROJ_DIR = _UI_DIR.parent                                   # project root
_DATA_DIR = _PROJ_DIR / "data" / "02_processed"

if str(_PROJ_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_PROJ_DIR / "src"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Return a DataFrame or None, silently if the file is a Git-LFS pointer."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            first = fh.read(48)
        if b"git-lfs" in first.lower():
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def _safe_csv(path: Path, **kw) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


# ── Primary loaders (all cached) ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_top5() -> pd.DataFrame:
    """Load the module-3 top-5 recommendations CSV."""
    df = _safe_csv(_DATA_DIR / "top5_recommendations_module3.csv")
    if df is None or df.empty:
        return pd.DataFrame(columns=["household_key", "COMMODITY_DESC",
                                     "Utility", "Relevance", "Uplift",
                                     "Normalized_Margin", "Context",
                                     "source_detail", "rank"])
    # Ensure rank column
    if "rank" not in df.columns:
        df["rank"] = df.groupby("household_key").cumcount() + 1
    df["household_key"] = pd.to_numeric(df["household_key"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_archetype_assignments() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module8_archetype_assignments.csv")
    if df is None or df.empty:
        return pd.DataFrame(columns=["household_key", "archetype",
                                     "deal_sensitivity", "basket_diversity"])
    df["household_key"] = pd.to_numeric(df["household_key"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_archetype_summary() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module8_archetype_summary.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_commodity_margin() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "commodity_margin.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ablation_summary() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module4_ablation_summary.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ab_test_results() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module9_ab_test_results.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_optimal_targeting() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module9_optimal_targeting_top20pct.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_hypothesis_results() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module9_hypothesis_results.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_basket_diversity() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module6_basket_diversity.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_margin_shift() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module6_margin_shift_chimera.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_basket_impact_summary() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module6_basket_impact_summary.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_interpretability_summary() -> pd.DataFrame:
    df = _safe_csv(_DATA_DIR / "module7_interpretability_summary.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_master_transactions() -> Optional[pd.DataFrame]:
    """Load (or skip if too large / LFS pointer)."""
    return _safe_parquet(_DATA_DIR / "master_transactions.parquet")


@st.cache_data(show_spinner=False)
def load_candidate_set_scored() -> Optional[pd.DataFrame]:
    """Module-3 scored candidate set – large, load only when needed."""
    df = _safe_csv(_DATA_DIR / "candidate_set_module3_scored.csv",
                   low_memory=False)
    if df is not None and not df.empty:
        df["household_key"] = pd.to_numeric(df["household_key"],
                                             errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def get_data_freshness() -> Dict[str, object]:
    """Return metadata about the master transactions file."""
    path = _DATA_DIR / "master_transactions.parquet"
    if not path.exists():
        return {"max_day": None, "file_mtime": None, "days_since": None}
    import os, datetime
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    days_since = (datetime.datetime.now() - mtime).days
    df = _safe_parquet(path)
    max_day = None
    if df is not None and "DAY" in df.columns:
        max_day = int(pd.to_numeric(df["DAY"], errors="coerce").max())
    return {"max_day": max_day, "file_mtime": mtime, "days_since": days_since}


# ── Explore-mode sampling ────────────────────────────────────────────────────

def sample_households(
    df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample of households by archetype, preserving distribution."""
    if df.empty or "household_key" not in df.columns:
        return df

    hh_col = "household_key"
    unique_hh = df[hh_col].unique()
    if len(unique_hh) <= n:
        return df

    if archetype_df.empty or "archetype" not in archetype_df.columns:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(unique_hh, size=n, replace=False)
        return df[df[hh_col].isin(sampled)].copy()

    arch_map = archetype_df.set_index("household_key")["archetype"].to_dict()
    groups: Dict[str, list] = {}
    for hh in unique_hh:
        a = arch_map.get(hh, "Unknown")
        groups.setdefault(a, []).append(hh)

    rng = np.random.default_rng(seed)
    selected = []
    for a, members in groups.items():
        k = max(1, round(n * len(members) / len(unique_hh)))
        chosen = rng.choice(members, size=min(k, len(members)), replace=False)
        selected.extend(chosen.tolist())
    return df[df[hh_col].isin(selected)].copy()


# ── Convenience bundle ───────────────────────────────────────────────────────

def load_all_primary() -> Dict[str, pd.DataFrame]:
    """Load all fast-loading artifacts in one call."""
    return {
        "top5":            load_top5(),
        "archetypes":      load_archetype_assignments(),
        "archetype_summary": load_archetype_summary(),
        "margin":          load_commodity_margin(),
        "ablation":        load_ablation_summary(),
        "ab_test":         load_ab_test_results(),
        "optimal":         load_optimal_targeting(),
        "hypothesis":      load_hypothesis_results(),
        "diversity":       load_basket_diversity(),
        "margin_shift":    load_margin_shift(),
        "basket_impact":   load_basket_impact_summary(),
        "interpretability":load_interpretability_summary(),
    }
