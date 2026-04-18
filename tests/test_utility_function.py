from pathlib import Path
import sys
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location("utility_scorer", ROOT / "src" / "utility_scorer.py")
UTILITY_SCORER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(UTILITY_SCORER)

build_household_campaign_flags = UTILITY_SCORER.build_household_campaign_flags
build_promoted_commodity_flags = UTILITY_SCORER.build_promoted_commodity_flags
calculate_context_score = UTILITY_SCORER.calculate_context_score
score_candidate_set = UTILITY_SCORER.score_candidate_set


def test_calculate_context_score_rules():
	assert calculate_context_score(0.8, True, True) == 1.0
	assert calculate_context_score(0.8, True, False) == 0.5
	assert calculate_context_score(0.2, True, True) == 0.2
	assert calculate_context_score(0.4, False, True) == 0.5
	assert calculate_context_score(0.4, True, False) == 0.7


def test_campaign_and_promotion_flags_are_resolved_from_inputs():
	campaign_table = pd.DataFrame({"household_key": [1, 2], "CAMPAIGN": [10, 20]})
	campaign_desc = pd.DataFrame(
		{
			"CAMPAIGN": [10, 20],
			"START_DAY": [100, 200],
			"END_DAY": [160, 250],
		}
	)
	active = build_household_campaign_flags(campaign_table, campaign_desc, snapshot_day=120)
	assert active.to_dict("records") == [{"household_key": 1, "is_active_campaign_period": True}]

	causal_data = pd.DataFrame(
		{
			"PRODUCT_ID": [101, 102, 103],
			"WEEK_NO": [20, 20, 19],
			"display": [1, 0, 1],
			"mailer": ["0", "D", "0"],
		}
	)
	product_lookup = pd.DataFrame(
		{
			"PRODUCT_ID": [101, 102, 103],
			"COMMODITY_DESC": ["YOGURT", "COFFEE", "SOAP"],
		}
	)
	promoted = build_promoted_commodity_flags(causal_data, product_lookup, snapshot_week=20)
	assert set(promoted["COMMODITY_DESC"]) == {"YOGURT", "COFFEE"}
	assert promoted["item_is_promoted"].all()


def test_score_candidate_set_builds_top_5_recommendations():
	candidate_set = pd.DataFrame(
		{
			"household_key": [1, 1, 2],
			"COMMODITY_DESC": ["MILK", "YOGURT", "COFFEE"],
			"relevance_als": [0.9, 0.4, 0.6],
			"relevance_mba": [0.2, 0.8, 0.1],
		}
	)
	history = pd.DataFrame(
		{
			"household_key": [1, 1, 1, 2, 2],
			"BASKET_ID": [11, 12, 13, 21, 22],
			"DAY": [100, 101, 102, 100, 101],
			"WEEK_NO": [15, 15, 15, 15, 15],
			"COMMODITY_DESC": ["MILK", "MILK", "BREAD", "TEA", "TEA"],
			"Is_Promoted_Item": [True, True, False, False, False],
		}
	)
	commodity_margin = pd.DataFrame(
		{
			"COMMODITY_DESC": ["MILK", "YOGURT", "COFFEE"],
			"Normalized_Margin": [0.1, 0.9, 0.7],
		}
	)
	campaign_table = pd.DataFrame({"household_key": [1], "CAMPAIGN": [301]})
	campaign_desc = pd.DataFrame({"CAMPAIGN": [301], "START_DAY": [90], "END_DAY": [120]})
	causal_data = pd.DataFrame(
		{
			"PRODUCT_ID": [1001, 1002],
			"WEEK_NO": [15, 15],
			"display": [1, 0],
			"mailer": ["0", "D"],
		}
	)
	product_lookup = pd.DataFrame(
		{
			"PRODUCT_ID": [1001, 1002],
			"COMMODITY_DESC": ["YOGURT", "COFFEE"],
		}
	)

	artifacts = score_candidate_set(
		candidate_set=candidate_set,
		history=history,
		commodity_margin=commodity_margin,
		campaign_table=campaign_table,
		campaign_desc=campaign_desc,
		causal_data=causal_data,
		product_lookup=product_lookup,
		top_k=5,
	)

	top = artifacts.top_recommendations
	assert list(top[top["household_key"] == 1]["COMMODITY_DESC"]) == ["YOGURT", "MILK"]
	assert list(top[top["household_key"] == 2]["COMMODITY_DESC"]) == ["COFFEE"]

	row_yogurt = top[(top["household_key"] == 1) & (top["COMMODITY_DESC"] == "YOGURT")].iloc[0]
	assert row_yogurt["Relevance"] == 0.8
	assert row_yogurt["Uplift"] == 1.0
	assert row_yogurt["Normalized_Margin"] == 0.9
	assert row_yogurt["Context"] == 1.0
