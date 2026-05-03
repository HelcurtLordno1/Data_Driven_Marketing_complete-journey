"""Module 8 helpers for rule-based customer archetype analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .module5_reporting import compute_component_contributions
from .utility_scorer import DEFAULT_UTILITY_WEIGHTS


ARCHETYPE_ORDER = [
	"Routine Replenisher",
	"Deal-Driven Explorer",
	"Premium Discoverer",
	"Frugal Loyalist",
]


@dataclass
class ArchetypeCaseStudy:
	"""Container for one representative household archetype walkthrough."""

	archetype: str
	household_key: int
	deal_sensitivity: float
	basket_diversity: float
	top_recommendations: pd.DataFrame
	purchase_history_summary: pd.DataFrame
	narrative: str


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
	missing = sorted(required - set(frame.columns))
	if missing:
		raise ValueError(f"{frame_name} missing required columns: {missing}")


def compute_household_features(scored_candidates: pd.DataFrame, basket_diversity: pd.DataFrame) -> pd.DataFrame:
	"""Merge household deal sensitivity and average basket diversity features."""
	_require_columns(scored_candidates, {"household_key", "deal_sensitivity"}, "scored_candidates")
	_require_columns(basket_diversity, {"household_key", "avg_basket_diversity"}, "basket_diversity")

	deal = (
		scored_candidates[["household_key", "deal_sensitivity"]]
		.dropna(subset=["household_key"])
		.assign(deal_sensitivity=lambda df: pd.to_numeric(df["deal_sensitivity"], errors="coerce"))
		.groupby("household_key", as_index=False)["deal_sensitivity"]
		.mean()
	)

	diversity = basket_diversity.copy()
	if "treatment" in diversity.columns:
		chimera = diversity[diversity["treatment"].astype(str).str.lower() == "chimera"]
		if not chimera.empty:
			diversity = chimera
	diversity = (
		diversity[["household_key", "avg_basket_diversity"]]
		.dropna(subset=["household_key"])
		.assign(basket_diversity=lambda df: pd.to_numeric(df["avg_basket_diversity"], errors="coerce"))
		.groupby("household_key", as_index=False)["basket_diversity"]
		.mean()
	)

	features = deal.merge(diversity, on="household_key", how="inner")
	features["household_key"] = features["household_key"].astype(int)
	features["deal_sensitivity"] = features["deal_sensitivity"].clip(0, 1)
	return features.dropna(subset=["deal_sensitivity", "basket_diversity"]).sort_values("household_key").reset_index(drop=True)


def assign_archetypes(
	household_features: pd.DataFrame,
	deal_threshold: Optional[float] = None,
	diversity_threshold: Optional[float] = None,
) -> pd.DataFrame:
	"""Assign households to four behavioral archetypes using two thresholded features."""
	_require_columns(household_features, {"household_key", "deal_sensitivity", "basket_diversity"}, "household_features")
	features = household_features.copy()
	features["deal_sensitivity"] = pd.to_numeric(features["deal_sensitivity"], errors="coerce")
	features["basket_diversity"] = pd.to_numeric(features["basket_diversity"], errors="coerce")
	features = features.dropna(subset=["deal_sensitivity", "basket_diversity"]).copy()

	if deal_threshold is None:
		deal_threshold = float(features["deal_sensitivity"].median())
	if diversity_threshold is None:
		diversity_threshold = float(features["basket_diversity"].median())

	high_deal = features["deal_sensitivity"] > deal_threshold
	high_diversity = features["basket_diversity"] > diversity_threshold
	features["archetype"] = np.select(
		[
			~high_deal & ~high_diversity,
			high_deal & high_diversity,
			~high_deal & high_diversity,
			high_deal & ~high_diversity,
		],
		ARCHETYPE_ORDER,
		default="Unclassified",
	)
	features["deal_threshold"] = deal_threshold
	features["diversity_threshold"] = diversity_threshold
	return features[["household_key", "archetype", "deal_sensitivity", "basket_diversity", "deal_threshold", "diversity_threshold"]].reset_index(drop=True)


def compute_archetype_utility_profile(
	top_recommendations: pd.DataFrame,
	archetype_assignments: pd.DataFrame,
	weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
	"""Summarize utility composition and candidate source mix for each archetype."""
	required = {"household_key", "Relevance", "Uplift", "Normalized_Margin", "Context", "source_detail"}
	_require_columns(top_recommendations, required, "top_recommendations")
	_require_columns(archetype_assignments, {"household_key", "archetype"}, "archetype_assignments")

	recs = compute_component_contributions(top_recommendations, weights=weights or DEFAULT_UTILITY_WEIGHTS)
	joined = recs.merge(archetype_assignments[["household_key", "archetype"]], on="household_key", how="inner")
	if joined.empty:
		return pd.DataFrame()

	metrics = (
		joined.groupby("archetype")
		.agg(
			households=("household_key", "nunique"),
			recommendations=("COMMODITY_DESC", "size"),
			avg_relevance=("Relevance", "mean"),
			avg_uplift=("Uplift", "mean"),
			avg_margin=("Normalized_Margin", "mean"),
			avg_context=("Context", "mean"),
			avg_utility=("Utility", "mean"),
			avg_relevance_contribution=("Relevance_Contribution", "mean"),
			avg_uplift_contribution=("Uplift_Contribution", "mean"),
			avg_margin_contribution=("Margin_Contribution", "mean"),
			avg_context_contribution=("Context_Contribution", "mean"),
		)
		.reset_index()
	)

	source_mix = pd.crosstab(joined["archetype"], joined["source_detail"].fillna("UNKNOWN"), normalize="index")
	for source in ["ALS", "MBA", "BOTH", "UNKNOWN"]:
		if source not in source_mix.columns:
			source_mix[source] = 0.0
	source_mix = source_mix[["ALS", "MBA", "BOTH", "UNKNOWN"]].rename(
		columns={
			"ALS": "source_pct_als",
			"MBA": "source_pct_mba",
			"BOTH": "source_pct_both",
			"UNKNOWN": "source_pct_unknown",
		}
	)

	out = metrics.merge(source_mix.reset_index(), on="archetype", how="left")
	out["archetype"] = pd.Categorical(out["archetype"], categories=ARCHETYPE_ORDER, ordered=True)
	return out.sort_values("archetype").reset_index(drop=True)


def compute_archetype_performance(user_metrics: pd.DataFrame, archetype_assignments: pd.DataFrame) -> pd.DataFrame:
	"""Join holdout precision metrics to archetypes and compute lift vs population."""
	required = {"household_key", "incremental_precision_at_5", "avg_recommended_margin"}
	_require_columns(user_metrics, required, "user_metrics")
	_require_columns(archetype_assignments, {"household_key", "archetype"}, "archetype_assignments")

	joined = user_metrics.merge(archetype_assignments[["household_key", "archetype"]], on="household_key", how="inner")
	if joined.empty:
		return pd.DataFrame()

	pop_precision = float(joined["incremental_precision_at_5"].mean())
	pop_margin = float(joined["avg_recommended_margin"].mean())
	perf = (
		joined.groupby("archetype")
		.agg(
			households=("household_key", "nunique"),
			avg_incremental_precision_at_5=("incremental_precision_at_5", "mean"),
			avg_recommended_margin=("avg_recommended_margin", "mean"),
			avg_incremental_hits=("incremental_hits", "mean") if "incremental_hits" in joined.columns else ("incremental_precision_at_5", "mean"),
		)
		.reset_index()
	)
	perf["precision_lift_vs_population"] = perf["avg_incremental_precision_at_5"] / pop_precision - 1 if pop_precision else np.nan
	perf["margin_lift_vs_population"] = perf["avg_recommended_margin"] / pop_margin - 1 if pop_margin else np.nan
	perf["archetype"] = pd.Categorical(perf["archetype"], categories=ARCHETYPE_ORDER, ordered=True)
	return perf.sort_values("archetype").reset_index(drop=True)


def _build_history_summary(history: pd.DataFrame, household_key: int, top_n: int = 6) -> pd.DataFrame:
	if history.empty or "household_key" not in history.columns:
		return pd.DataFrame(columns=["COMMODITY_DESC", "baskets", "revenue", "promoted_rows"])

	hh = history[history["household_key"] == household_key].copy()
	if hh.empty:
		return pd.DataFrame(columns=["COMMODITY_DESC", "baskets", "revenue", "promoted_rows"])
	for column, default in [("Revenue_Retailer", 0.0), ("Is_Promoted_Item", False)]:
		if column not in hh.columns:
			hh[column] = default
	return (
		hh.groupby("COMMODITY_DESC", as_index=False)
		.agg(
			baskets=("BASKET_ID", "nunique"),
			revenue=("Revenue_Retailer", "sum"),
			promoted_rows=("Is_Promoted_Item", "sum"),
		)
		.sort_values(["baskets", "revenue", "COMMODITY_DESC"], ascending=[False, False, True])
		.head(top_n)
		.reset_index(drop=True)
	)


def _case_narrative(archetype: str, household_key: int, recs: pd.DataFrame) -> str:
	if recs.empty:
		return f"Household {household_key} has no top recommendations available for this archetype."

	source_counts = recs["source_detail"].fillna("UNKNOWN").value_counts(normalize=True)
	dominant_source = str(source_counts.index[0])
	component_means = {
		"relevance": float(recs["Relevance"].mean()),
		"uplift": float(recs["Uplift"].mean()),
		"margin": float(recs["Normalized_Margin"].mean()),
		"context": float(recs["Context"].mean()),
	}
	leading_component = max(component_means, key=component_means.get)
	top_item = str(recs.iloc[0]["COMMODITY_DESC"])
	return (
		f"Household {household_key} represents the {archetype} segment. "
		f"The top recommendation is {top_item}, and the top-5 list leans most on {leading_component} "
		f"with {dominant_source} as the largest candidate source. "
		f"Chimera is balancing familiar relevance, discovery uplift, margin, and campaign context rather than applying one universal rule."
	)


def build_archetype_case_study(
	top_recommendations: pd.DataFrame,
	archetype_assignments: pd.DataFrame,
	history: Optional[pd.DataFrame] = None,
	weights: Optional[Dict[str, float]] = None,
	top_k: int = 5,
) -> Dict[str, ArchetypeCaseStudy]:
	"""Pick one representative household per archetype and build case-study payloads."""
	_require_columns(top_recommendations, {"household_key", "COMMODITY_DESC", "Utility"}, "top_recommendations")
	_require_columns(archetype_assignments, {"household_key", "archetype", "deal_sensitivity", "basket_diversity"}, "archetype_assignments")

	recs = compute_component_contributions(top_recommendations, weights=weights or DEFAULT_UTILITY_WEIGHTS)
	joined = recs.merge(archetype_assignments, on="household_key", how="inner")
	if joined.empty:
		return {}

	payloads: Dict[str, ArchetypeCaseStudy] = {}
	history_frame = history if history is not None else pd.DataFrame()
	for archetype in ARCHETYPE_ORDER:
		group = joined[joined["archetype"] == archetype].copy()
		if group.empty:
			continue
		representative = (
			group.groupby("household_key")["Utility"]
			.mean()
			.sort_values(ascending=False)
			.index[0]
		)
		hh_rows = (
			group[group["household_key"] == representative]
			.sort_values(["Utility", "Relevance", "Normalized_Margin"], ascending=[False, False, False])
			.head(top_k)
			.reset_index(drop=True)
		)
		profile = archetype_assignments[archetype_assignments["household_key"] == representative].iloc[0]
		payloads[archetype] = ArchetypeCaseStudy(
			archetype=archetype,
			household_key=int(representative),
			deal_sensitivity=float(profile["deal_sensitivity"]),
			basket_diversity=float(profile["basket_diversity"]),
			top_recommendations=hh_rows,
			purchase_history_summary=_build_history_summary(history_frame, int(representative)),
			narrative=_case_narrative(archetype, int(representative), hh_rows),
		)
	return payloads
