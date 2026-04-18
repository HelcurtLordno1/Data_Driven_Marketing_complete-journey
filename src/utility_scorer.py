"""Utility scoring helpers for Module 3 ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_UTILITY_WEIGHTS: Dict[str, float] = {
	"relevance": 0.40,
	"uplift": 0.25,
	"margin": 0.20,
	"context": 0.15,
}


@dataclass
class UtilityArtifacts:
	"""Container for full utility outputs."""

	scored_candidates: pd.DataFrame
	top_recommendations: pd.DataFrame
	deal_sensitivity: pd.DataFrame
	habit_strength: pd.DataFrame
	active_campaigns: pd.DataFrame
	promoted_commodities: pd.DataFrame


def _normalize_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
	active = dict(DEFAULT_UTILITY_WEIGHTS)
	if weights:
		active.update(weights)
	missing = {"relevance", "uplift", "margin", "context"} - set(active)
	if missing:
		raise ValueError(f"Missing utility weights for: {sorted(missing)}")
	return active


def _coerce_boolean(series: pd.Series) -> pd.Series:
	values = series.fillna(False)
	if pd.api.types.is_bool_dtype(values):
		return values.astype(bool)
	text = values.astype(str).str.strip().str.upper()
	return ~text.isin({"", "0", "FALSE", "NAN", "NONE", "NO"})


def calculate_relevance_score(
	relevance_als: pd.Series | np.ndarray,
	relevance_mba: pd.Series | np.ndarray,
) -> np.ndarray:
	"""Take the stronger of ALS and MBA relevance signals."""
	als = np.clip(np.asarray(relevance_als, dtype=float), 0.0, 1.0)
	mba = np.clip(np.asarray(relevance_mba, dtype=float), 0.0, 1.0)
	return np.maximum(als, mba)


def calculate_habit_strength(candidate_set: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
	"""Compute item habit strength per household and commodity."""
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


def calculate_deal_sensitivity(history: pd.DataFrame) -> pd.DataFrame:
	"""Share of baskets containing any promoted item."""
	if history.empty:
		return pd.DataFrame(columns=["household_key", "deal_sensitivity"])

	basket_flags = (
		history.groupby(["household_key", "BASKET_ID"], as_index=False)["Is_Promoted_Item"]
		.any()
		.rename(columns={"Is_Promoted_Item": "basket_has_promo"})
	)
	user_summary = basket_flags.groupby("household_key").agg(
		promo_baskets=("basket_has_promo", "sum"),
		total_baskets=("basket_has_promo", "size"),
	)
	user_summary["deal_sensitivity"] = (
		user_summary["promo_baskets"] / user_summary["total_baskets"].replace(0, np.nan)
	).fillna(0.0)
	user_summary["deal_sensitivity"] = user_summary["deal_sensitivity"].clip(0, 1)
	return user_summary.reset_index()[["household_key", "deal_sensitivity"]]


def calculate_context_score(deal_sensitivity: float, is_active_campaign: bool, is_promoted_item: bool) -> float:
	"""Apply the project context score rules."""
	if deal_sensitivity > 0.6 and is_active_campaign and is_promoted_item:
		return 1.0
	if deal_sensitivity > 0.6 and is_active_campaign and not is_promoted_item:
		return 0.5
	if deal_sensitivity < 0.3 and is_promoted_item:
		return 0.2
	if not is_active_campaign and is_promoted_item:
		return 0.5
	return 0.7


def build_household_campaign_flags(
	campaign_table: pd.DataFrame,
	campaign_desc: pd.DataFrame,
	snapshot_day: int,
) -> pd.DataFrame:
	"""Resolve whether each household is currently inside an active campaign window."""
	if campaign_table.empty or campaign_desc.empty:
		return pd.DataFrame(columns=["household_key", "is_active_campaign_period"])

	active_campaigns = campaign_desc[
		(campaign_desc["START_DAY"] <= snapshot_day) & (campaign_desc["END_DAY"] >= snapshot_day)
	][["CAMPAIGN"]].drop_duplicates()
	if active_campaigns.empty:
		return pd.DataFrame(columns=["household_key", "is_active_campaign_period"])

	household_campaigns = (
		campaign_table.merge(active_campaigns, on="CAMPAIGN", how="inner")[["household_key"]]
		.drop_duplicates()
		.assign(is_active_campaign_period=True)
	)
	return household_campaigns


def resolve_available_snapshot_week(
	causal_data: pd.DataFrame,
	snapshot_week: Optional[int] = None,
) -> Optional[int]:
	"""Choose the latest available causal week at or before the requested snapshot."""
	if causal_data.empty or "WEEK_NO" not in causal_data.columns:
		return None

	available_weeks = (
		pd.to_numeric(causal_data["WEEK_NO"], errors="coerce")
		.dropna()
		.astype(int)
		.drop_duplicates()
		.sort_values()
	)
	if available_weeks.empty:
		return None
	if snapshot_week is None:
		return int(available_weeks.iloc[-1])

	requested_week = int(snapshot_week)
	eligible_weeks = available_weeks[available_weeks <= requested_week]
	if not eligible_weeks.empty:
		return int(eligible_weeks.iloc[-1])
	return int(available_weeks.iloc[0])


def build_promoted_commodity_flags(
	causal_data: pd.DataFrame,
	product_lookup: pd.DataFrame,
	snapshot_week: Optional[int] = None,
) -> pd.DataFrame:
	"""Map currently promoted products to promoted commodities."""
	if causal_data.empty or product_lookup.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "item_is_promoted"])

	causal = causal_data.copy()
	causal["WEEK_NO"] = pd.to_numeric(causal["WEEK_NO"], errors="coerce")
	causal = causal.dropna(subset=["WEEK_NO", "PRODUCT_ID"])
	if causal.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "item_is_promoted"])

	current_week = resolve_available_snapshot_week(causal, snapshot_week=snapshot_week)
	if current_week is None:
		return pd.DataFrame(columns=["COMMODITY_DESC", "item_is_promoted"])
	causal = causal[causal["WEEK_NO"] == current_week].copy()
	if causal.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "item_is_promoted"])

	display_numeric = pd.to_numeric(causal["display"], errors="coerce").fillna(0)
	mailer_flag = _coerce_boolean(causal["mailer"])
	causal["item_is_promoted"] = (display_numeric > 0) | mailer_flag

	promoted_products = causal[causal["item_is_promoted"]][["PRODUCT_ID"]].drop_duplicates()
	if promoted_products.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "item_is_promoted"])

	commodity_flags = (
		promoted_products.merge(product_lookup[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left")
		.dropna(subset=["COMMODITY_DESC"])
		.assign(item_is_promoted=True)
		.groupby("COMMODITY_DESC", as_index=False)["item_is_promoted"]
		.max()
	)
	return commodity_flags


def prepare_margin_lookup(commodity_margin: pd.DataFrame) -> pd.DataFrame:
	"""Return a many-to-one commodity margin lookup with a normalized score."""
	if commodity_margin.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "Normalized_Margin"])

	margin = commodity_margin.copy()
	if "Normalized_Margin" not in margin.columns:
		raw_col = "Raw_Margin" if "Raw_Margin" in margin.columns else None
		if raw_col is None:
			raise ValueError("commodity_margin must include Normalized_Margin or Raw_Margin.")
		raw_values = pd.to_numeric(margin[raw_col], errors="coerce")
		vmin = raw_values.min()
		vmax = raw_values.max()
		if pd.isna(vmin) or pd.isna(vmax) or np.isclose(vmin, vmax):
			margin["Normalized_Margin"] = np.where(raw_values.fillna(0) > 0, 1.0, 0.0)
		else:
			margin["Normalized_Margin"] = (raw_values - vmin) / (vmax - vmin)

	margin["Normalized_Margin"] = pd.to_numeric(margin["Normalized_Margin"], errors="coerce").fillna(0.0).clip(0, 1)
	margin = (
		margin[["COMMODITY_DESC", "Normalized_Margin"]]
		.dropna(subset=["COMMODITY_DESC"])
		.groupby("COMMODITY_DESC", as_index=False)["Normalized_Margin"]
		.max()
	)
	return margin


def build_commodity_margin_table(
	product_lookup: pd.DataFrame,
	private_label_margin: float = 0.40,
	national_brand_margin: float = 0.20,
	other_brand_margin: float = 0.30,
) -> pd.DataFrame:
	"""Create the normalized commodity-margin table from business proxy rules."""
	required_cols = {"COMMODITY_DESC", "BRAND"}
	missing = sorted(required_cols - set(product_lookup.columns))
	if missing:
		raise ValueError(f"product_lookup missing required columns for margin build: {missing}")

	margin = product_lookup[["COMMODITY_DESC", "BRAND"]].dropna(subset=["COMMODITY_DESC"]).copy()
	brand_text = margin["BRAND"].fillna("").astype(str).str.upper()
	margin["Raw_Margin"] = np.select(
		[
			brand_text.str.contains("PRIVATE", na=False),
			brand_text.str.contains("NATIONAL", na=False),
		],
		[private_label_margin, national_brand_margin],
		default=other_brand_margin,
	).astype(float)

	raw_min = float(margin["Raw_Margin"].min())
	raw_max = float(margin["Raw_Margin"].max())
	if np.isclose(raw_min, raw_max):
		margin["Normalized_Margin"] = np.where(margin["Raw_Margin"] > 0, 1.0, 0.0)
	else:
		margin["Normalized_Margin"] = (margin["Raw_Margin"] - raw_min) / (raw_max - raw_min)

	return (
		margin.groupby("COMMODITY_DESC", as_index=False)
		.agg(Raw_Margin=("Raw_Margin", "max"), Normalized_Margin=("Normalized_Margin", "max"))
		.sort_values("COMMODITY_DESC")
		.reset_index(drop=True)
	)


def score_utility(
	relevance: pd.Series,
	uplift: pd.Series,
	margin: pd.Series,
	context: pd.Series,
	weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
	"""Compute the unified utility score."""
	active_weights = _normalize_weights(weights)
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
	return candidate_set.sort_values(
		["household_key", utility_column, "Relevance", "Normalized_Margin", "COMMODITY_DESC"],
		ascending=[True, False, False, False, True],
	).reset_index(drop=True)


def top_k_recommendations(candidate_set: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
	"""Keep the top-k ranked items per household."""
	ranked = rank_candidates(candidate_set)
	if ranked.empty:
		return ranked
	return ranked.groupby("household_key", as_index=False).head(top_k).reset_index(drop=True)


def score_candidate_set(
	candidate_set: pd.DataFrame,
	history: pd.DataFrame,
	commodity_margin: pd.DataFrame,
	campaign_table: pd.DataFrame,
	campaign_desc: pd.DataFrame,
	causal_data: pd.DataFrame,
	product_lookup: pd.DataFrame,
	weights: Optional[Dict[str, float]] = None,
	top_k: int = 5,
	snapshot_day: Optional[int] = None,
	snapshot_week: Optional[int] = None,
) -> UtilityArtifacts:
	"""Build the full Module 3 utility ranking outputs."""
	required_candidate_cols = {"household_key", "COMMODITY_DESC", "relevance_als", "relevance_mba"}
	missing_candidate = sorted(required_candidate_cols - set(candidate_set.columns))
	if missing_candidate:
		raise ValueError(f"candidate_set missing required columns: {missing_candidate}")

	required_history_cols = {"household_key", "BASKET_ID", "COMMODITY_DESC", "Is_Promoted_Item"}
	missing_history = sorted(required_history_cols - set(history.columns))
	if missing_history:
		raise ValueError(f"history missing required columns: {missing_history}")

	scored = candidate_set.copy()
	scored["relevance_als"] = pd.to_numeric(scored["relevance_als"], errors="coerce").fillna(0.0).clip(0, 1)
	scored["relevance_mba"] = pd.to_numeric(scored["relevance_mba"], errors="coerce").fillna(0.0).clip(0, 1)
	scored["Relevance"] = calculate_relevance_score(scored["relevance_als"], scored["relevance_mba"])

	habit_strength = calculate_habit_strength(scored, history)
	scored = scored.merge(habit_strength, on=["household_key", "COMMODITY_DESC"], how="left")
	scored["habit_strength"] = scored["habit_strength"].fillna(0.0).clip(0, 1)
	scored["Uplift"] = calculate_uplift_score(scored["habit_strength"])

	margin_lookup = prepare_margin_lookup(commodity_margin)
	scored = scored.merge(margin_lookup, on="COMMODITY_DESC", how="left")
	scored["Normalized_Margin"] = scored["Normalized_Margin"].fillna(0.0).clip(0, 1)

	deal_sensitivity = calculate_deal_sensitivity(history)
	scored = scored.merge(deal_sensitivity, on="household_key", how="left")
	scored["deal_sensitivity"] = scored["deal_sensitivity"].fillna(0.0).clip(0, 1)

	if snapshot_day is None:
		if "DAY" not in history.columns:
			raise ValueError("history must include DAY when snapshot_day is not provided.")
		snapshot_day = int(pd.to_numeric(history["DAY"], errors="coerce").max())
	if snapshot_week is None:
		if "snapshot_week" in scored.columns:
			snapshot_week = int(pd.to_numeric(scored["snapshot_week"], errors="coerce").dropna().max())
		elif "WEEK_NO" in history.columns:
			snapshot_week = int(pd.to_numeric(history["WEEK_NO"], errors="coerce").max())

	active_campaigns = build_household_campaign_flags(campaign_table, campaign_desc, snapshot_day=snapshot_day)
	scored = scored.merge(active_campaigns, on="household_key", how="left")
	scored["is_active_campaign_period"] = (
		scored["is_active_campaign_period"].astype("boolean").fillna(False).astype(bool)
	)

	promoted_commodities = build_promoted_commodity_flags(
		causal_data=causal_data,
		product_lookup=product_lookup,
		snapshot_week=snapshot_week,
	)
	scored = scored.merge(promoted_commodities, on="COMMODITY_DESC", how="left")
	scored["item_is_promoted"] = scored["item_is_promoted"].astype("boolean").fillna(False).astype(bool)

	scored["Context"] = [
		calculate_context_score(
			deal_sensitivity=float(deal_sensitivity),
			is_active_campaign=bool(is_active_campaign),
			is_promoted_item=bool(is_promoted_item),
		)
		for deal_sensitivity, is_active_campaign, is_promoted_item in scored[
			["deal_sensitivity", "is_active_campaign_period", "item_is_promoted"]
		].itertuples(index=False, name=None)
	]

	scored["Utility"] = score_utility(
		relevance=scored["Relevance"],
		uplift=scored["Uplift"],
		margin=scored["Normalized_Margin"],
		context=scored["Context"],
		weights=weights,
	)

	ranked = rank_candidates(scored)
	top_recommendations = top_k_recommendations(ranked, top_k=top_k)
	return UtilityArtifacts(
		scored_candidates=ranked,
		top_recommendations=top_recommendations,
		deal_sensitivity=deal_sensitivity,
		habit_strength=habit_strength,
		active_campaigns=active_campaigns,
		promoted_commodities=promoted_commodities,
	)


def calculate_expected_profit(*args, **kwargs):
	"""Placeholder for downstream profit simulation hooks."""
	raise NotImplementedError


def filter_persuadables(*args, **kwargs):
	"""Placeholder for persuadable-segment filtering."""
	raise NotImplementedError
