"""
chimera-ui/utils/recompute.py
Thin wrapper around src/ utility_scorer for live re-ranking
when the user changes weights in the Weight Simulator page.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_PROJ_DIR / "src"))

from utility_scorer import score_utility, DEFAULT_UTILITY_WEIGHTS  # noqa: E402


def rerank_households(
    candidates: pd.DataFrame,
    weights: Dict[str, float],
    household_keys: Optional[List[int]] = None,
    top_k: int = 5,
    stock_penalty: float = 0.0,
) -> pd.DataFrame:
    """
    Re-score and rank a candidate DataFrame under the given weights.

    Parameters
    ----------
    candidates      : Must contain columns Relevance, Uplift, Normalized_Margin, Context.
    weights         : dict with keys relevance, uplift, margin, context.
    household_keys  : If provided, filter to these households first.
    top_k           : Number of top items to keep per household.
    stock_penalty   : Subtracted from utility for low-inventory items
                      (requires 'low_stock' bool column in candidates).
    """
    required = {"household_key", "COMMODITY_DESC", "Relevance", "Uplift",
                 "Normalized_Margin", "Context"}
    missing = required - set(candidates.columns)
    if missing:
        return pd.DataFrame()

    df = candidates.copy()
    if household_keys:
        df = df[df["household_key"].isin(household_keys)]
    if df.empty:
        return pd.DataFrame()

    # Normalise weights
    total = sum(weights.values())
    w = {k: v / total for k, v in weights.items()} if total > 0 else dict(DEFAULT_UTILITY_WEIGHTS)

    df["Utility_new"] = score_utility(
        relevance=df["Relevance"],
        uplift=df["Uplift"],
        margin=df["Normalized_Margin"],
        context=df["Context"],
        weights=w,
    )

    if stock_penalty > 0 and "low_stock" in df.columns:
        df["Utility_new"] = df["Utility_new"] - df["low_stock"].astype(float) * stock_penalty

    df = df.sort_values(["household_key", "Utility_new"], ascending=[True, False])
    df["new_rank"] = df.groupby("household_key").cumcount() + 1
    return df[df["new_rank"] <= top_k].reset_index(drop=True)


def compute_stability(
    original_top1: pd.Series,      # household_key → top commodity
    new_top1: pd.Series,
) -> float:
    """Fraction of households where rank-1 item is unchanged."""
    merged = original_top1.rename("orig").to_frame().join(
        new_top1.rename("new"), how="inner"
    )
    if merged.empty:
        return 0.0
    return float((merged["orig"] == merged["new"]).mean())
