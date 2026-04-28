"""Module 6 basket impact analysis helpers.

These helpers support the notebook that evaluates how Chimera recommendations
change basket composition, margin, diversity, and the discovery vs hit-rate
trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str], frame_name: str) -> None:
	missing = [column for column in required if column not in frame.columns]
	if missing:
		raise ValueError(f"{frame_name} missing required columns: {missing}")


def _as_string_set(values: pd.Series) -> set[str]:
	return set(values.dropna().astype(str).tolist())


def _variant_summary_rows(detail: pd.DataFrame, variant_label: str) -> dict:
	return {
		"variant": variant_label,
		"household_count": int(detail["household_key"].nunique()) if not detail.empty else 0,
		"expansion_rate": float(detail["expanded_category"].mean()) if not detail.empty else 0.0,
		"avg_hit_rate": float(detail["hit_rate"].mean()) if "hit_rate" in detail.columns and not detail.empty else 0.0,
		"avg_discovery_rate": float(detail["discovery_rate"].mean()) if "discovery_rate" in detail.columns and not detail.empty else 0.0,
		"avg_new_categories_purchased": float(detail["new_category_count"].mean()) if "new_category_count" in detail.columns and not detail.empty else 0.0,
	}


def compute_category_expansion_rate_by_variant(
	test_history: pd.DataFrame,
	top_recommendations_chimera: pd.DataFrame,
	top_recommendations_baseline: pd.DataFrame,
	train_items_by_user: dict[int, set],
	) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Compute whether recommendations introduced a new commodity category."""
	_ensure_columns(test_history, ["household_key", "COMMODITY_DESC"], "test_history")
	_ensure_columns(top_recommendations_chimera, ["household_key", "COMMODITY_DESC"], "top_recommendations_chimera")
	_ensure_columns(top_recommendations_baseline, ["household_key", "COMMODITY_DESC"], "top_recommendations_baseline")

	test_items = (
		test_history.groupby("household_key")["COMMODITY_DESC"]
		.apply(lambda series: _as_string_set(series))
		.to_dict()
	)

	def build_detail(recommendations: pd.DataFrame, variant_label: str) -> pd.DataFrame:
		rows = []
		for household_key, recs in recommendations.groupby("household_key"):
			train_items = {str(item) for item in train_items_by_user.get(int(household_key), set())}
			test_set = {str(item) for item in test_items.get(int(household_key), set())}
			new_categories = test_set - train_items
			recommended_items = [str(item) for item in recs["COMMODITY_DESC"].dropna().astype(str).tolist()]
			hit_items = sorted(set(recommended_items).intersection(test_set))
			new_hits = sorted(set(recommended_items).intersection(new_categories))
			rows.append(
				{
					"household_key": int(household_key),
					"variant": variant_label,
					"recommended_count": len(recommended_items),
					"test_category_count": len(test_set),
					"train_category_count": len(train_items),
					"new_category_count": len(new_categories),
					"hit_count": len(hit_items),
					"new_category_hit_count": len(new_hits),
					"hit_rate": len(hit_items) / len(recommended_items) if recommended_items else 0.0,
					"discovery_rate": len(new_hits) / len(test_set) if test_set else 0.0,
					"expanded_category": len(new_hits) > 0,
				}
			)
		return pd.DataFrame(rows)

	chimera_detail = build_detail(top_recommendations_chimera, "Chimera")
	baseline_detail = build_detail(top_recommendations_baseline, "Popularity Baseline")
	detail = pd.concat([chimera_detail, baseline_detail], ignore_index=True)
	summary = pd.DataFrame([
		_variant_summary_rows(chimera_detail, "Chimera"),
		_variant_summary_rows(baseline_detail, "Popularity Baseline"),
	])
	return detail, summary


def compute_margin_shift_index(
	test_history: pd.DataFrame,
	train_history: pd.DataFrame,
	margin_lookup: pd.DataFrame,
	) -> pd.DataFrame:
	"""Compare average margin in the test window versus the training window."""
	_ensure_columns(test_history, ["household_key", "COMMODITY_DESC"], "test_history")
	_ensure_columns(train_history, ["household_key", "COMMODITY_DESC"], "train_history")
	_ensure_columns(margin_lookup, ["COMMODITY_DESC", "Normalized_Margin"], "margin_lookup")

	lookup = margin_lookup[["COMMODITY_DESC", "Normalized_Margin"]].copy()
	lookup["Normalized_Margin"] = pd.to_numeric(lookup["Normalized_Margin"], errors="coerce").fillna(0.0).clip(0, 1)

	def attach_margin(frame: pd.DataFrame) -> pd.DataFrame:
		joined = frame.merge(lookup, on="COMMODITY_DESC", how="left")
		joined["Normalized_Margin"] = pd.to_numeric(joined["Normalized_Margin"], errors="coerce").fillna(0.0).clip(0, 1)
		return joined

	train = attach_margin(train_history)
	test = attach_margin(test_history)

	train_summary = train.groupby("household_key").agg(train_avg_margin=("Normalized_Margin", "mean"), train_items=("COMMODITY_DESC", "nunique"), train_baskets=("BASKET_ID", "nunique"))
	test_summary = test.groupby("household_key").agg(test_avg_margin=("Normalized_Margin", "mean"), test_items=("COMMODITY_DESC", "nunique"), test_baskets=("BASKET_ID", "nunique"))

	result = train_summary.join(test_summary, how="outer").reset_index()
	result["train_avg_margin"] = result["train_avg_margin"].fillna(0.0)
	result["test_avg_margin"] = result["test_avg_margin"].fillna(0.0)
	result["margin_shift"] = result["test_avg_margin"] - result["train_avg_margin"]
	result["margin_shift_pct"] = np.where(
		result["train_avg_margin"] > 0,
		(result["margin_shift"] / result["train_avg_margin"]) * 100.0,
		0.0,
	)
	result["moved_higher_margin"] = result["margin_shift"] > 0
	return result.sort_values("household_key").reset_index(drop=True)


def compute_basket_size_uplift(
	test_history: pd.DataFrame,
	top_recommendations_chimera: pd.DataFrame,
	top_recommendations_baseline: pd.DataFrame,
	train_items_by_user: dict[int, set],
	) -> pd.DataFrame:
	"""Compare basket diversity by treatment type."""
	_ensure_columns(test_history, ["household_key", "BASKET_ID", "COMMODITY_DESC"], "test_history")

	def summarize(treatment: str, recommendations: pd.DataFrame) -> pd.DataFrame:
		rows = []
		for household_key, recs in recommendations.groupby("household_key"):
			hh_test = test_history[test_history["household_key"] == household_key].copy()
			basket_diversity = hh_test.groupby("BASKET_ID")["COMMODITY_DESC"].nunique()
			rows.append(
				{
					"household_key": int(household_key),
					"treatment": treatment,
					"avg_basket_diversity": float(basket_diversity.mean()) if not basket_diversity.empty else 0.0,
					"median_basket_diversity": float(basket_diversity.median()) if not basket_diversity.empty else 0.0,
					"total_baskets_test": int(hh_test["BASKET_ID"].nunique()),
					"total_items_test": int(hh_test["COMMODITY_DESC"].nunique()),
					"recommendation_count": int(recs["COMMODITY_DESC"].nunique()),
				}
			)
		return pd.DataFrame(rows)

	chimera = summarize("Chimera", top_recommendations_chimera)
	baseline = summarize("Baseline", top_recommendations_baseline)
	return pd.concat([chimera, baseline], ignore_index=True)


def compute_hit_rate_discovery_tradeoff(
	test_history: pd.DataFrame,
	top_recommendations: pd.DataFrame,
	train_items_by_user: dict[int, set],
	) -> pd.DataFrame:
	"""Measure recommendation hit-rate and discovery rate per household."""
	_ensure_columns(test_history, ["household_key", "COMMODITY_DESC"], "test_history")
	_ensure_columns(top_recommendations, ["household_key", "COMMODITY_DESC"], "top_recommendations")

	test_items = test_history.groupby("household_key")["COMMODITY_DESC"].apply(lambda series: _as_string_set(series)).to_dict()

	rows = []
	for household_key, recs in top_recommendations.groupby("household_key"):
		recommended_items = [str(item) for item in recs["COMMODITY_DESC"].dropna().astype(str).tolist()]
		test_set = {str(item) for item in test_items.get(int(household_key), set())}
		train_set = {str(item) for item in train_items_by_user.get(int(household_key), set())}
		new_purchased_items = test_set - train_set
		hits = set(recommended_items).intersection(test_set)
		new_hits = set(recommended_items).intersection(new_purchased_items)
		rows.append(
			{
				"household_key": int(household_key),
				"recommended_count": len(recommended_items),
				"purchased_count": len(test_set),
				"new_purchased_count": len(new_purchased_items),
				"hit_count": len(hits),
				"new_hit_count": len(new_hits),
				"hit_rate": len(hits) / len(recommended_items) if recommended_items else 0.0,
				"discovery_rate": len(new_hits) / len(test_set) if test_set else 0.0,
				"baseline_items_in_train": len(train_set),
			}
		)

	return pd.DataFrame(rows).sort_values("household_key").reset_index(drop=True)


def compute_pre_post_summary(
	test_history: pd.DataFrame,
	train_history: pd.DataFrame,
	top_recommendations_chimera: pd.DataFrame,
	top_recommendations_baseline: pd.DataFrame,
	margin_lookup: pd.DataFrame,
	train_items_by_user: dict[int, set],
	) -> pd.DataFrame:
	"""Build a compact executive summary table for Module 6."""
	category_detail, category_summary = compute_category_expansion_rate_by_variant(
		test_history=test_history,
		top_recommendations_chimera=top_recommendations_chimera,
		top_recommendations_baseline=top_recommendations_baseline,
		train_items_by_user=train_items_by_user,
	)
	margin_shift = compute_margin_shift_index(test_history=test_history, train_history=train_history, margin_lookup=margin_lookup)
	basket_diversity = compute_basket_size_uplift(
		test_history=test_history,
		top_recommendations_chimera=top_recommendations_chimera,
		top_recommendations_baseline=top_recommendations_baseline,
		train_items_by_user=train_items_by_user,
	)
	tradeoff = pd.concat(
		[
			compute_hit_rate_discovery_tradeoff(test_history, top_recommendations_chimera, train_items_by_user).assign(variant="Chimera"),
			compute_hit_rate_discovery_tradeoff(test_history, top_recommendations_baseline, train_items_by_user).assign(variant="Popularity Baseline"),
		],
		ignore_index=True,
	)

	def safe_mean(frame: pd.DataFrame, column: str, filter_expr: Optional[pd.Series] = None) -> float:
		if filter_expr is not None:
			frame = frame.loc[filter_expr].copy()
		if frame.empty or column not in frame.columns:
			return 0.0
		return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())

	chimera_category = category_summary.loc[category_summary["variant"] == "Chimera"].iloc[0]
	baseline_category = category_summary.loc[category_summary["variant"] == "Popularity Baseline"].iloc[0]
	chimera_tradeoff = tradeoff.loc[tradeoff["variant"] == "Chimera"]
	baseline_tradeoff = tradeoff.loc[tradeoff["variant"] == "Popularity Baseline"]

	summary = pd.DataFrame(
		[
			{"metric": "category_expansion_rate", "chimera": chimera_category["expansion_rate"], "baseline": baseline_category["expansion_rate"]},
			{"metric": "avg_margin_shift", "chimera": safe_mean(margin_shift, "margin_shift"), "baseline": 0.0},
			{"metric": "avg_basket_diversity", "chimera": safe_mean(basket_diversity, "avg_basket_diversity", basket_diversity["treatment"] == "Chimera"), "baseline": safe_mean(basket_diversity, "avg_basket_diversity", basket_diversity["treatment"] == "Baseline")},
			{"metric": "avg_hit_rate", "chimera": safe_mean(chimera_tradeoff, "hit_rate"), "baseline": safe_mean(baseline_tradeoff, "hit_rate")},
			{"metric": "avg_discovery_rate", "chimera": safe_mean(chimera_tradeoff, "discovery_rate"), "baseline": safe_mean(baseline_tradeoff, "discovery_rate")},
		],
	)
	summary["absolute_lift"] = summary["chimera"] - summary["baseline"]
	summary["relative_lift_pct"] = np.where(summary["baseline"].abs() > 0, (summary["absolute_lift"] / summary["baseline"]) * 100.0, 0.0)

	return summary
