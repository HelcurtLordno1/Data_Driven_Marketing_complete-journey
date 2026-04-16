"""Uplift and utility scoring helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def calculate_habit_strength(candidate_set: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
	"""Compute item habit strength per household and item."""
	if candidate_set.empty:
		return pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "habit_strength"])
	total_baskets = history.groupby("household_key")["BASKET_ID"].nunique().rename("total_baskets")
	item_baskets = history.groupby(["household_key", "COMMODITY_DESC"])["BASKET_ID"].nunique().rename("item_baskets")
	out = item_baskets.reset_index().merge(total_baskets.reset_index(), on="household_key", how="left")
	out["habit_strength"] = out["item_baskets"] / out["total_baskets"].replace(0, np.nan)
	out["habit_strength"] = out["habit_strength"].fillna(0).clip(0, 1)
	return out[["household_key", "COMMODITY_DESC", "habit_strength"]]


def calculate_uplift_score(habit_strength: pd.Series | np.ndarray) -> np.ndarray:
	"""Convert habit strength into uplift score in [0, 1]."""
	return 1.0 - np.clip(np.asarray(habit_strength, dtype=float), 0.0, 1.0)


def calculate_context_score(deal_sensitivity: float, is_active_campaign: bool, is_promoted_item: bool) -> float:
	"""Apply the project's context score rules."""
	if deal_sensitivity > 0.6 and is_active_campaign and is_promoted_item:
		return 1.0
	if deal_sensitivity > 0.6 and is_active_campaign and not is_promoted_item:
		return 0.5
	if deal_sensitivity < 0.3 and is_promoted_item:
		return 0.2
	if not is_active_campaign and is_promoted_item:
		return 0.5
	return 0.7


def score_utility(
	relevance: pd.Series,
	uplift: pd.Series,
	margin: pd.Series,
	context: pd.Series,
	weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
	"""Compute the unified utility score."""
	active_weights = weights or {"relevance": 0.40, "uplift": 0.25, "margin": 0.20, "context": 0.15}
	return (
		active_weights["relevance"] * relevance
		+ active_weights["uplift"] * uplift
		+ active_weights["margin"] * margin
		+ active_weights["context"] * context
	)


def rank_candidates(candidate_set: pd.DataFrame, utility_column: str = "Utility") -> pd.DataFrame:
	"""Return candidates sorted by utility score in descending order."""
	if candidate_set.empty or utility_column not in candidate_set.columns:
		return candidate_set.copy()
	return candidate_set.sort_values(["household_key", utility_column], ascending=[True, False]).reset_index(drop=True)


def calculate_expected_profit(*args, **kwargs):
	"""Placeholder for downstream profit simulation hooks."""
	raise NotImplementedError


def filter_persuadables(*args, **kwargs):
	"""Placeholder for persuadable-segment filtering."""
	raise NotImplementedError

