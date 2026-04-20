"""Module 4 validation helpers for temporal holdout and ablation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class HoldoutSplit:
	"""Container for temporal split artifacts."""

	train_history: pd.DataFrame
	test_history: pd.DataFrame
	train_items_by_user: Dict[int, set]
	test_items_by_user: Dict[int, set]
	eligible_households: list[int]
	selected_weeks: list[int]


def build_ablation_weight_templates() -> pd.DataFrame:
	"""Return the README-specified four ablation variants."""
	return pd.DataFrame(
		[
			{"Variant": "Variant 0 - Relevance only", "relevance": 1.0, "uplift": 0.0, "margin": 0.0, "context": 0.0},
			{"Variant": "Variant 1 - Relevance + Uplift", "relevance": 0.75, "uplift": 0.25, "margin": 0.0, "context": 0.0},
			{"Variant": "Variant 2 - Relevance + Uplift + Margin", "relevance": 0.60, "uplift": 0.20, "margin": 0.20, "context": 0.0},
			{"Variant": "Variant 3 - Full Chimera", "relevance": 0.40, "uplift": 0.25, "margin": 0.20, "context": 0.15},
		]
	)


def make_variant_weights(relevance: float, uplift: float = 0.0, margin: float = 0.0, context: float = 0.0) -> Dict[str, float]:
	"""Normalize variant weights to sum to 1."""
	total = relevance + uplift + margin + context
	if total <= 0:
		raise ValueError("At least one component weight must be positive.")
	return {
		"relevance": relevance / total,
		"uplift": uplift / total,
		"margin": margin / total,
		"context": context / total,
	}


def build_temporal_holdout(
	history: pd.DataFrame,
	holdout_weeks: Optional[Iterable[int]] = None,
	day_split: tuple[int, int] = (600, 711),
) -> HoldoutSplit:
	"""Build a holdout split based on weeks when available, otherwise day range fallback."""
	tx = history.copy()
	tx["WEEK_NO"] = pd.to_numeric(tx["WEEK_NO"], errors="coerce")
	tx["DAY"] = pd.to_numeric(tx["DAY"], errors="coerce")

	requested_weeks = list(holdout_weeks) if holdout_weeks is not None else list(range(81, 103))
	available_weeks = sorted(set(tx["WEEK_NO"].dropna().astype(int)).intersection(requested_weeks))

	if available_weeks:
		train_history = tx[~tx["WEEK_NO"].isin(available_weeks)].copy()
		test_history = tx[tx["WEEK_NO"].isin(available_weeks)].copy()
		selected_weeks = available_weeks
	else:
		start_day, end_day = day_split
		train_history = tx[tx["DAY"] <= start_day].copy()
		test_history = tx[(tx["DAY"] > start_day) & (tx["DAY"] <= end_day)].copy()
		selected_weeks = []

	train_items_by_user = train_history.groupby("household_key")["COMMODITY_DESC"].apply(set).to_dict()
	test_items_by_user = test_history.groupby("household_key")["COMMODITY_DESC"].apply(set).to_dict()
	eligible_households = sorted(set(train_items_by_user).intersection(test_items_by_user))

	return HoldoutSplit(
		train_history=train_history,
		test_history=test_history,
		train_items_by_user=train_items_by_user,
		test_items_by_user=test_items_by_user,
		eligible_households=eligible_households,
		selected_weeks=selected_weeks,
	)


def build_variant_topk(candidate_frame: pd.DataFrame, weights: Dict[str, float], top_k: int = 5) -> pd.DataFrame:
	"""Rank candidates by a variant utility formula and keep top-k per user."""
	variant = candidate_frame.copy()
	variant["Utility"] = (
		weights["relevance"] * variant["Relevance"]
		+ weights["uplift"] * variant["Uplift"]
		+ weights["margin"] * variant["Normalized_Margin"]
		+ weights["context"] * variant["Context"]
	)
	variant = variant.sort_values(
		["household_key", "Utility", "Relevance", "Normalized_Margin", "COMMODITY_DESC"],
		ascending=[True, False, False, False, True],
	)
	return variant.groupby("household_key", as_index=False).head(top_k).reset_index(drop=True)


def evaluate_incremental_precision(topk_frame: pd.DataFrame, train_items: Dict[int, set], test_items: Dict[int, set], top_k: int = 5) -> pd.DataFrame:
	"""Compute household-level incremental precision metrics."""
	rows = []
	for household_key, user_recs in topk_frame.groupby("household_key"):
		user_recs = user_recs.head(top_k)
		purchased_train = train_items.get(household_key, set())
		purchased_test = test_items.get(household_key, set())
		incremental_targets = purchased_test - purchased_train
		hits = user_recs["COMMODITY_DESC"].isin(incremental_targets).sum()
		rows.append(
			{
				"household_key": household_key,
				"incremental_hits": int(hits),
				"incremental_precision_at_5": hits / top_k,
				"avg_recommended_margin": float(user_recs["Normalized_Margin"].mean()),
				"top5_recommendations": len(user_recs),
				"incremental_targets": len(incremental_targets),
			}
		)
	return pd.DataFrame(rows)


def build_popularity_baseline_topk(
	train_history: pd.DataFrame,
	eligible_households: list[int],
	train_items_by_user: Dict[int, set],
	top_k: int = 5,
) -> pd.DataFrame:
	"""Build a popularity baseline per household excluding already purchased train items."""
	popularity = (
		train_history.groupby("COMMODITY_DESC")
		.agg(purchase_count=("BASKET_ID", "nunique"), revenue=("Revenue_Retailer", "sum"))
		.reset_index()
		.sort_values(["purchase_count", "revenue", "COMMODITY_DESC"], ascending=[False, False, True])
	)

	baseline_rows = []
	for household_key in eligible_households:
		excluded = train_items_by_user.get(household_key, set())
		candidates = popularity[~popularity["COMMODITY_DESC"].isin(excluded)].head(top_k)
		for _, row in candidates.iterrows():
			baseline_rows.append({"household_key": household_key, "COMMODITY_DESC": row["COMMODITY_DESC"]})
	return pd.DataFrame(baseline_rows)


def attach_margin_to_topk(topk_frame: pd.DataFrame, margin_lookup: pd.DataFrame) -> pd.DataFrame:
	"""Attach normalized margin score to top-k lists."""
	merged = topk_frame.merge(margin_lookup[["COMMODITY_DESC", "Normalized_Margin"]], on="COMMODITY_DESC", how="left")
	merged["Normalized_Margin"] = pd.to_numeric(merged["Normalized_Margin"], errors="coerce").fillna(0.0).clip(0, 1)
	return merged


def run_ablation(
	scored_candidates: pd.DataFrame,
	split: HoldoutSplit,
	weight_templates: pd.DataFrame,
	margin_lookup: pd.DataFrame,
	top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
	"""Run all ablation variants and return summary + per-household metrics + raw outputs."""
	filtered = scored_candidates[scored_candidates["household_key"].isin(split.eligible_households)].copy()

	variant_outputs: dict = {}
	variant_user_metrics = []

	for _, template in weight_templates.iterrows():
		weights = make_variant_weights(template["relevance"], template["uplift"], template["margin"], template["context"])
		variant_topk = build_variant_topk(filtered, weights, top_k=top_k)
		user_metrics = evaluate_incremental_precision(
			topk_frame=variant_topk,
			train_items=split.train_items_by_user,
			test_items=split.test_items_by_user,
			top_k=top_k,
		)
		variant_outputs[str(template["Variant"])] = {
			"Weights": weights,
			"TopK": variant_topk,
			"UserMetrics": user_metrics,
		}
		variant_user_metrics.append(user_metrics.assign(Variant=str(template["Variant"])))

	ablation_summary = pd.DataFrame(
		[
			{
				"Variant": variant,
				"Incremental_Precision@5": data["UserMetrics"]["incremental_precision_at_5"].mean(),
				"Average_Recommended_Margin": data["UserMetrics"]["avg_recommended_margin"].mean(),
				"Average_Incremental_Hits": data["UserMetrics"]["incremental_hits"].mean(),
				"Average_Targets": data["UserMetrics"]["incremental_targets"].mean(),
			}
			for variant, data in variant_outputs.items()
		]
	)

	baseline_precision = float(
		ablation_summary.loc[
			ablation_summary["Variant"] == "Variant 0 - Relevance only", "Incremental_Precision@5"
		].iloc[0]
	)
	baseline_margin = float(
		ablation_summary.loc[
			ablation_summary["Variant"] == "Variant 0 - Relevance only", "Average_Recommended_Margin"
		].iloc[0]
	)

	ablation_summary["Precision_Lift_vs_Baseline"] = (
		ablation_summary["Incremental_Precision@5"] / baseline_precision - 1 if baseline_precision else np.nan
	)
	ablation_summary["Margin_Lift_vs_Baseline"] = (
		ablation_summary["Average_Recommended_Margin"] / baseline_margin - 1 if baseline_margin else np.nan
	)

	popularity_topk = build_popularity_baseline_topk(
		train_history=split.train_history,
		eligible_households=split.eligible_households,
		train_items_by_user=split.train_items_by_user,
		top_k=top_k,
	)
	popularity_topk = attach_margin_to_topk(popularity_topk, margin_lookup)
	popularity_metrics = evaluate_incremental_precision(
		topk_frame=popularity_topk,
		train_items=split.train_items_by_user,
		test_items=split.test_items_by_user,
		top_k=top_k,
	)
	pop_margin = float(popularity_metrics["avg_recommended_margin"].mean()) if not popularity_metrics.empty else np.nan
	ablation_summary["Margin_Lift_vs_Popularity"] = (
		ablation_summary["Average_Recommended_Margin"] / pop_margin - 1 if pop_margin and not np.isnan(pop_margin) else np.nan
	)

	user_metrics_long = pd.concat(variant_user_metrics, ignore_index=True) if variant_user_metrics else pd.DataFrame()
	variant_outputs["Popularity_Baseline"] = {
		"TopK": popularity_topk,
		"UserMetrics": popularity_metrics,
	}

	return ablation_summary, user_metrics_long, variant_outputs