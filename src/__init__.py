"""Core source package for the Chimera utility recommendation project."""

from .cold_start import ColdStartRecommender, build_demographic_priors, recommend_for_new_user
from .data_loader import get_project_root, load_or_build_master_transactions
from .financial_utils import calculate_margin, calculate_true_price, normalize_discount_values
from .utility_scorer import (
	DEFAULT_UTILITY_WEIGHTS,
	UtilityArtifacts,
	build_commodity_margin_table,
	build_household_campaign_flags,
	build_promoted_commodity_flags,
	calculate_deal_sensitivity,
	calculate_context_score,
	calculate_expected_profit,
	calculate_habit_strength,
	calculate_relevance_score,
	calculate_uplift_score,
	filter_persuadables,
	prepare_margin_lookup,
	rank_candidates,
	score_candidate_set,
	score_utility,
	top_k_recommendations,
)

try:
	from .recall_engine import (
		AlternatingLeastSquares,
		CandidateArtifacts,
		build_als_model,
		build_candidate_set,
		build_mba_lookup,
		build_mba_rules,
		build_seed_items_table,
		compute_als_scores_topk,
		minmax_scale,
		normalize_lift_to_unit,
		rowwise_minmax,
		save_als_factors,
	)
except ImportError:
	AlternatingLeastSquares = None
	CandidateArtifacts = None
	build_als_model = None
	build_candidate_set = None
	build_mba_lookup = None
	build_mba_rules = None
	build_seed_items_table = None
	compute_als_scores_topk = None
	minmax_scale = None
	normalize_lift_to_unit = None
	rowwise_minmax = None
	save_als_factors = None

__all__ = [
	"AlternatingLeastSquares",
	"CandidateArtifacts",
	"ColdStartRecommender",
	"DEFAULT_UTILITY_WEIGHTS",
	"build_als_model",
	"build_candidate_set",
	"build_commodity_margin_table",
	"build_demographic_priors",
	"build_household_campaign_flags",
	"build_mba_lookup",
	"build_mba_rules",
	"build_promoted_commodity_flags",
	"build_seed_items_table",
	"UtilityArtifacts",
	"calculate_deal_sensitivity",
	"calculate_context_score",
	"calculate_expected_profit",
	"calculate_habit_strength",
	"calculate_margin",
	"calculate_relevance_score",
	"calculate_true_price",
	"calculate_uplift_score",
	"compute_als_scores_topk",
	"filter_persuadables",
	"get_project_root",
	"load_or_build_master_transactions",
	"minmax_scale",
	"normalize_discount_values",
	"normalize_lift_to_unit",
	"prepare_margin_lookup",
	"rank_candidates",
	"rowwise_minmax",
	"save_als_factors",
	"recommend_for_new_user",
	"score_candidate_set",
	"score_utility",
	"top_k_recommendations",
]
