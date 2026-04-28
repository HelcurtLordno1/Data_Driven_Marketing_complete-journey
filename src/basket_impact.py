"""Module 6 helpers for basket behavior impact analysis."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


def _series_to_item_set(series: pd.Series) -> set:
	values = series.dropna().astype(str).str.strip()
	return set(item for item in values if item)


def build_item_sets(history: pd.DataFrame) -> Dict[int, set]:
	"""Return household -> set of commodity descriptions."""
	if history.empty or "household_key" not in history.columns or "COMMODITY_DESC" not in history.columns:
		return {}
	series = history.groupby("household_key")["COMMODITY_DESC"].apply(_series_to_item_set)
	return {int(key): set(values) for key, values in series.items()}


def build_recommendation_sets(topk: pd.DataFrame, top_k: int = 5) -> Dict[int, set]:
	"""Return household -> set of recommended commodities (top-k)."""
	if topk.empty or "household_key" not in topk.columns or "COMMODITY_DESC" not in topk.columns:
		return {}
	trimmed = topk.sort_values(["household_key"]).groupby("household_key", as_index=False).head(top_k)
	series = trimmed.groupby("household_key")["COMMODITY_DESC"].apply(_series_to_item_set)
	return {int(key): set(values) for key, values in series.items()}


def build_new_category_lookup(train_items_by_user: Dict[int, set], test_items_by_user: Dict[int, set]) -> Dict[int, set]:
	"""Return household -> new categories purchased in the test period."""
	eligible = sorted(set(train_items_by_user).intersection(test_items_by_user))
	return {int(household): set(test_items_by_user[household]) - set(train_items_by_user[household]) for household in eligible}


def compute_category_expansion_table(
	topk: pd.DataFrame,
	train_items_by_user: Dict[int, set],
	test_items_by_user: Dict[int, set],
	top_k: int = 5,
) -> pd.DataFrame:
	"""Compute household-level category expansion hits."""
	new_categories = build_new_category_lookup(train_items_by_user, test_items_by_user)
	rec_sets = build_recommendation_sets(topk, top_k=top_k)

	rows = []
	for household_key in sorted(new_categories):
		new_items = new_categories.get(household_key, set())
		recs = rec_sets.get(household_key, set())
		hit_count = len(recs.intersection(new_items))
		rows.append(
			{
				"household_key": int(household_key),
				"new_categories": len(new_items),
				"recommended_new_categories": hit_count,
				"expansion_hit": hit_count > 0,
			}
		)

	return pd.DataFrame(rows)


def summarize_category_expansion(expansion_table: pd.DataFrame, label: str) -> pd.DataFrame:
	"""Summarize category expansion metrics for a model."""
	if expansion_table.empty:
		return pd.DataFrame(
			[
				{
					"Model": label,
					"Households": 0,
					"Category_Expansion_Rate": 0.0,
					"Avg_New_Categories": 0.0,
					"Avg_New_Categories_Hit": 0.0,
				}
			]
		)
	return pd.DataFrame(
		[
			{
				"Model": label,
				"Households": int(len(expansion_table)),
				"Category_Expansion_Rate": float(expansion_table["expansion_hit"].mean()),
				"Avg_New_Categories": float(expansion_table["new_categories"].mean()),
				"Avg_New_Categories_Hit": float(expansion_table["recommended_new_categories"].mean()),
			}
		]
	)


def compute_margin_shift_table(
	train_history: pd.DataFrame,
	test_history: pd.DataFrame,
	margin_lookup: pd.DataFrame,
	eligible_households: Optional[list[int]] = None,
) -> pd.DataFrame:
	"""Return per-household train/test average margin and shift."""
	def _average_margin(frame: pd.DataFrame, label: str) -> pd.DataFrame:
		merged = frame.merge(margin_lookup[["COMMODITY_DESC", "Normalized_Margin"]], on="COMMODITY_DESC", how="left")
		merged["Normalized_Margin"] = pd.to_numeric(merged["Normalized_Margin"], errors="coerce").fillna(0.0).clip(0, 1)
		return (
			merged.groupby("household_key")["Normalized_Margin"]
			.mean()
			.reset_index()
			.rename(columns={"Normalized_Margin": label})
		)

	train_margin = _average_margin(train_history, "train_margin")
	test_margin = _average_margin(test_history, "test_margin")
	merged = train_margin.merge(test_margin, on="household_key", how="outer")
	merged["train_margin"] = merged["train_margin"].fillna(0.0)
	merged["test_margin"] = merged["test_margin"].fillna(0.0)
	merged["margin_shift"] = merged["test_margin"] - merged["train_margin"]

	if eligible_households is not None:
		merged = merged[merged["household_key"].isin(eligible_households)].copy()

	return merged.reset_index(drop=True)


def compute_basket_diversity(test_history: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Return basket-level and household-level basket diversity metrics."""
	if test_history.empty:
		empty_baskets = pd.DataFrame(columns=["household_key", "BASKET_ID", "distinct_commodities"])
		empty_households = pd.DataFrame(columns=["household_key", "avg_distinct_commodities"])
		return empty_baskets, empty_households

	basket_diversity = (
		test_history.groupby(["household_key", "BASKET_ID"])["COMMODITY_DESC"]
		.nunique()
		.reset_index(name="distinct_commodities")
	)
	household_diversity = (
		basket_diversity.groupby("household_key")["distinct_commodities"]
		.mean()
		.reset_index(name="avg_distinct_commodities")
	)
	return basket_diversity, household_diversity


def compute_hit_rate_table(topk: pd.DataFrame, test_items_by_user: Dict[int, set], top_k: int = 5) -> pd.DataFrame:
	"""Compute hit-rate per household based on test period purchases."""
	rec_sets = build_recommendation_sets(topk, top_k=top_k)
	rows = []
	for household_key, recs in rec_sets.items():
		test_items = test_items_by_user.get(household_key, set())
		hits = len(set(recs).intersection(test_items))
		rows.append(
			{
				"household_key": int(household_key),
				"hits": int(hits),
				"hit_rate": hits / float(top_k) if top_k else 0.0,
				"has_hit": hits > 0,
			}
		)
	return pd.DataFrame(rows)


def build_tradeoff_table(
	topk: pd.DataFrame,
	train_items_by_user: Dict[int, set],
	test_items_by_user: Dict[int, set],
	model_label: str,
	top_k: int = 5,
) -> pd.DataFrame:
	"""Build a per-household trade-off table for hit-rate vs discovery."""
	new_categories = build_new_category_lookup(train_items_by_user, test_items_by_user)
	hit_rates = compute_hit_rate_table(topk, test_items_by_user, top_k=top_k)

	rows = []
	for row in hit_rates.itertuples(index=False):
		new_count = len(new_categories.get(int(row.household_key), set()))
		rows.append(
			{
				"household_key": int(row.household_key),
				"model": model_label,
				"hit_rate": float(row.hit_rate),
				"hits": int(row.hits),
				"new_categories_purchased": int(new_count),
			}
		)

	return pd.DataFrame(rows)
