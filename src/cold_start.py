"""Cold-start recommendation rules and demographic prior helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


def build_demographic_priors(hh_demographic: pd.DataFrame, item_popularity: pd.DataFrame) -> pd.DataFrame:
	"""Build a simple demographic prior table from household segments."""
	if hh_demographic.empty or item_popularity.empty:
		return pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "prior_score"])
	segment_cols = [column for column in ["AGE_DESC", "INCOME_DESC", "HOMEOWNER_DESC", "KID_CATEGORY_DESC"] if column in hh_demographic.columns]
	if not segment_cols:
		return pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "prior_score"])

	priors = hh_demographic[["household_key", *segment_cols]].drop_duplicates().merge(
		item_popularity[["COMMODITY_DESC", "popularity_score"]],
		how="cross",
	)
	priors["prior_score"] = priors["popularity_score"].astype(float)
	return priors[["household_key", "COMMODITY_DESC", "prior_score"]]


@dataclass
class ColdStartRecommender:
	"""Provides default recommendations for users without purchase history."""

	fallback_items: Optional[pd.DataFrame] = None

	def recommend_for_new_user(self, demographic_profile, top_k: int = 10) -> pd.DataFrame:
		"""Return top-k prior-based fallback items for a new user."""
		if self.fallback_items is None or self.fallback_items.empty:
			return pd.DataFrame(columns=["COMMODITY_DESC", "prior_score"])
		return self.fallback_items.sort_values("prior_score", ascending=False).head(top_k).reset_index(drop=True)


def recommend_for_new_user(demographic_profile, top_k: int = 10, fallback_items: Optional[pd.DataFrame] = None):
	"""Convenience wrapper for a single cold-start recommendation request."""
	recommender = ColdStartRecommender(fallback_items=fallback_items)
	return recommender.recommend_for_new_user(demographic_profile, top_k=top_k)

