from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.archetypes import (  # noqa: E402
	assign_archetypes,
	build_archetype_case_study,
	compute_archetype_performance,
	compute_archetype_utility_profile,
	compute_household_features,
)


def test_compute_household_features_and_assign_archetypes_use_medians():
	scored = pd.DataFrame(
		{
			"household_key": [1, 1, 2, 3, 4],
			"deal_sensitivity": [0.2, 0.2, 0.8, 0.2, 0.8],
		}
	)
	diversity = pd.DataFrame(
		{
			"household_key": [1, 2, 3, 4],
			"treatment": ["Chimera", "Chimera", "Chimera", "Chimera"],
			"avg_basket_diversity": [2.0, 8.0, 8.0, 2.0],
		}
	)

	features = compute_household_features(scored, diversity)
	assignments = assign_archetypes(features, deal_threshold=0.5, diversity_threshold=5.0)

	assert dict(zip(assignments["household_key"], assignments["archetype"])) == {
		1: "Routine Replenisher",
		2: "Deal-Driven Explorer",
		3: "Premium Discoverer",
		4: "Frugal Loyalist",
	}


def test_compute_archetype_utility_profile_includes_source_mix():
	assignments = pd.DataFrame(
		{
			"household_key": [1, 2],
			"archetype": ["Routine Replenisher", "Deal-Driven Explorer"],
		}
	)
	top = pd.DataFrame(
		{
			"household_key": [1, 1, 2],
			"COMMODITY_DESC": ["MILK", "BREAD", "SOAP"],
			"Relevance": [0.5, 0.7, 0.9],
			"Uplift": [0.2, 0.4, 0.8],
			"Normalized_Margin": [0.6, 1.0, 0.5],
			"Context": [0.7, 0.7, 1.0],
			"Utility": [0.47, 0.61, 0.78],
			"source_detail": ["ALS", "MBA", "BOTH"],
		}
	)

	profile = compute_archetype_utility_profile(top, assignments)
	routine = profile[profile["archetype"] == "Routine Replenisher"].iloc[0]

	assert routine["recommendations"] == 2
	assert routine["source_pct_als"] == 0.5
	assert routine["source_pct_mba"] == 0.5


def test_performance_and_case_studies_join_on_archetype():
	assignments = pd.DataFrame(
		{
			"household_key": [1, 2],
			"archetype": ["Routine Replenisher", "Deal-Driven Explorer"],
			"deal_sensitivity": [0.2, 0.8],
			"basket_diversity": [2.0, 8.0],
		}
	)
	metrics = pd.DataFrame(
		{
			"household_key": [1, 2],
			"incremental_precision_at_5": [0.0, 0.2],
			"avg_recommended_margin": [0.5, 1.0],
			"incremental_hits": [0, 1],
		}
	)
	top = pd.DataFrame(
		{
			"household_key": [1, 2],
			"COMMODITY_DESC": ["MILK", "SOAP"],
			"Relevance": [0.5, 0.9],
			"Uplift": [0.2, 0.8],
			"Normalized_Margin": [0.6, 0.5],
			"Context": [0.7, 1.0],
			"Utility": [0.47, 0.78],
			"source_detail": ["ALS", "BOTH"],
		}
	)

	perf = compute_archetype_performance(metrics, assignments)
	cases = build_archetype_case_study(top, assignments)

	assert set(perf["archetype"].astype(str)) == {"Routine Replenisher", "Deal-Driven Explorer"}
	assert cases["Routine Replenisher"].household_key == 1
	assert cases["Deal-Driven Explorer"].top_recommendations.iloc[0]["COMMODITY_DESC"] == "SOAP"
