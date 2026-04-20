"""Module 5 helpers for executive narrative, case study, and simulator tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utility_scorer import DEFAULT_UTILITY_WEIGHTS


@dataclass
class CaseStudyArtifacts:
	"""Container for Module 5 narrative tables."""

	household_key: int
	variant0_top5: pd.DataFrame
	fullchimera_top5: pd.DataFrame
	comparison_table: pd.DataFrame
	history_timeline: pd.DataFrame
	component_decomposition: pd.DataFrame


def pick_case_study_household(top5: pd.DataFrame, preferred_household: Optional[int] = None) -> int:
	"""Pick a household for the executive walkthrough."""
	if preferred_household is not None:
		return int(preferred_household)
	if top5.empty:
		raise ValueError("top5 recommendations are empty.")
	return int(top5.groupby("household_key")["Utility"].max().sort_values(ascending=False).index[0])


def compute_component_contributions(scored: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
	"""Add weighted utility contribution columns to scored recommendations."""
	active = dict(DEFAULT_UTILITY_WEIGHTS)
	if weights:
		active.update(weights)
	out = scored.copy()
	out["Relevance_Contribution"] = active["relevance"] * out["Relevance"]
	out["Uplift_Contribution"] = active["uplift"] * out["Uplift"]
	out["Margin_Contribution"] = active["margin"] * out["Normalized_Margin"]
	out["Context_Contribution"] = active["context"] * out["Context"]
	return out


def build_user_purchase_timeline(history: pd.DataFrame, household_key: int) -> pd.DataFrame:
	"""Aggregate user history by day for timeline visuals."""
	timeline = (
		history[history["household_key"] == household_key]
		.groupby(["DAY", "WEEK_NO"], as_index=False)
		.agg(
			baskets=("BASKET_ID", "nunique"),
			commodities=("COMMODITY_DESC", "nunique"),
			revenue=("Revenue_Retailer", "sum"),
			promoted_rows=("Is_Promoted_Item", "sum"),
		)
		.sort_values("DAY")
	)
	timeline["promoted_rows"] = pd.to_numeric(timeline["promoted_rows"], errors="coerce").fillna(0).astype(int)
	return timeline


def build_variant0_from_scored(scored_candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
	"""Re-rank candidates with relevance-only utility (Variant 0 baseline)."""
	variant0 = scored_candidates.copy()
	variant0["Utility_V0"] = variant0["Relevance"]
	variant0 = variant0.sort_values(
		["household_key", "Utility_V0", "Relevance", "Normalized_Margin", "COMMODITY_DESC"],
		ascending=[True, False, False, False, True],
	)
	return variant0.groupby("household_key", as_index=False).head(top_k).reset_index(drop=True)


def build_case_study(
	history: pd.DataFrame,
	scored_candidates: pd.DataFrame,
	top5: pd.DataFrame,
	household_key: Optional[int] = None,
	weights: Optional[Dict[str, float]] = None,
	top_k: int = 5,
) -> CaseStudyArtifacts:
	"""Build the full Module 5 John Smith-style case-study payload."""
	selected_hh = pick_case_study_household(top5=top5, preferred_household=household_key)
	variant0_all = build_variant0_from_scored(scored_candidates=scored_candidates, top_k=top_k)
	variant0_top5 = variant0_all[variant0_all["household_key"] == selected_hh].copy()
	full_top5 = top5[top5["household_key"] == selected_hh].copy()

	contrib = compute_component_contributions(full_top5, weights=weights)
	decomp = contrib[
		[
			"COMMODITY_DESC",
			"Relevance",
			"Uplift",
			"Normalized_Margin",
			"Context",
			"Utility",
			"Relevance_Contribution",
			"Uplift_Contribution",
			"Margin_Contribution",
			"Context_Contribution",
		]
	].copy()

	comparison = (
		full_top5[["COMMODITY_DESC", "Utility"]]
		.rename(columns={"COMMODITY_DESC": "Chimera_Item", "Utility": "Chimera_Utility"})
		.reset_index(drop=True)
	)
	comparison["Variant0_Item"] = variant0_top5["COMMODITY_DESC"].reset_index(drop=True)
	comparison["Variant0_Relevance"] = variant0_top5["Relevance"].reset_index(drop=True)

	history_timeline = build_user_purchase_timeline(history=history, household_key=selected_hh)

	return CaseStudyArtifacts(
		household_key=selected_hh,
		variant0_top5=variant0_top5,
		fullchimera_top5=full_top5,
		comparison_table=comparison,
		history_timeline=history_timeline,
		component_decomposition=decomp,
	)


def build_recommendation_simulator_table(
	hh_demographic: pd.DataFrame,
	top5: pd.DataFrame,
	weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
	"""Prepare a flat table for dashboard simulator usage."""
	active = dict(DEFAULT_UTILITY_WEIGHTS)
	if weights:
		active.update(weights)

	persona_cols = [
		column
		for column in ["household_key", "AGE_DESC", "INCOME_DESC", "HOMEOWNER_DESC", "KID_CATEGORY_DESC", "MARITAL_STATUS_CODE"]
		if column in hh_demographic.columns
	]
	persona = hh_demographic[persona_cols].drop_duplicates("household_key") if persona_cols else pd.DataFrame(columns=["household_key"])

	with_contrib = compute_component_contributions(top5, weights=active)
	simulator = with_contrib.merge(persona, on="household_key", how="left")
	simulator["Utility_Formula_Text"] = (
		active["relevance"].__format__(".2f")
		+ "*R + "
		+ active["uplift"].__format__(".2f")
		+ "*U + "
		+ active["margin"].__format__(".2f")
		+ "*M + "
		+ active["context"].__format__(".2f")
		+ "*C"
	)

	simulator["Utility_Expanded_Text"] = (
		simulator["Relevance_Contribution"].round(3).astype(str)
		+ " + "
		+ simulator["Uplift_Contribution"].round(3).astype(str)
		+ " + "
		+ simulator["Margin_Contribution"].round(3).astype(str)
		+ " + "
		+ simulator["Context_Contribution"].round(3).astype(str)
		+ " = "
		+ simulator["Utility"].round(3).astype(str)
	)

	return simulator.sort_values(["household_key", "Utility"], ascending=[True, False]).reset_index(drop=True)


def build_ablation_proof_table(ablation_summary: pd.DataFrame) -> pd.DataFrame:
	"""Create a compact proof table for the executive dashboard."""
	proof = ablation_summary.copy()
	proof["Precision_Lift_%"] = (proof["Precision_Lift_vs_Baseline"] * 100).round(2)
	proof["Margin_Lift_%"] = (proof["Margin_Lift_vs_Baseline"] * 100).round(2)
	if "Margin_Lift_vs_Popularity" in proof.columns:
		proof["Margin_Lift_vs_Popularity_%"] = (proof["Margin_Lift_vs_Popularity"] * 100).round(2)
	return proof
