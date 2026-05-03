"""Core source package for the Chimera utility recommendation project."""

from .cold_start import ColdStartRecommender, build_demographic_priors, recommend_for_new_user
from .recommendation_explainer import (
	GlobalImportanceResult,
	ExplanationCard,
	CounterfactualResult,
	WeightSensitivityResult,
	compute_global_component_importance,
	generate_explanation_card,
	generate_explanation_cards_for_household,
	cards_to_dataframe,
	compute_counterfactual_explanation,
	weight_sensitivity_analysis,
	compute_similar_user_pct,
)
from .data_loader import get_project_root, load_or_build_master_transactions
from .financial_utils import calculate_margin, calculate_true_price, normalize_discount_values
from .module4_validation import (
	HoldoutSplit,
	attach_margin_to_topk,
	build_ablation_weight_templates,
	build_temporal_holdout,
	build_variant_topk,
	evaluate_incremental_precision,
	make_variant_weights,
	run_ablation,
)
from .module5_reporting import (
	CaseStudyArtifacts,
	build_ablation_proof_table,
	build_case_study,
	build_recommendation_simulator_table,
	build_user_purchase_timeline,
	compute_component_contributions,
	pick_case_study_household,
)
from .deployment_plan import (
	DEFAULT_DEPLOYMENT_CONFIG,
	DeploymentRoadmapArtifacts,
	build_dashboard_wireframe_table,
	build_deployment_config,
	build_deployment_roadmap,
	build_retraining_policy_table,
	build_system_architecture_table,
	build_uniqueness_table,
	export_deployment_roadmap,
)
from .archetypes import (
	ARCHETYPE_ORDER,
	ArchetypeCaseStudy,
	assign_archetypes,
	build_archetype_case_study,
	compute_archetype_performance,
	compute_archetype_utility_profile,
	compute_household_features,
)
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
	"GlobalImportanceResult",
	"ExplanationCard",
	"CounterfactualResult",
	"WeightSensitivityResult",
	"compute_global_component_importance",
	"generate_explanation_card",
	"generate_explanation_cards_for_household",
	"cards_to_dataframe",
	"compute_counterfactual_explanation",
	"weight_sensitivity_analysis",
	"compute_similar_user_pct",
	"AlternatingLeastSquares",
	"ARCHETYPE_ORDER",
	"ArchetypeCaseStudy",
	"CandidateArtifacts",
	"ColdStartRecommender",
	"DEFAULT_UTILITY_WEIGHTS",
	"build_als_model",
	"build_ablation_proof_table",
	"build_archetype_case_study",
	"build_ablation_weight_templates",
	"build_case_study",
	"DEFAULT_DEPLOYMENT_CONFIG",
	"DeploymentRoadmapArtifacts",
	"build_dashboard_wireframe_table",
	"build_deployment_config",
	"build_deployment_roadmap",
	"build_candidate_set",
	"build_commodity_margin_table",
	"build_demographic_priors",
	"build_household_campaign_flags",
	"build_retraining_policy_table",
	"build_system_architecture_table",
	"build_uniqueness_table",
	"export_deployment_roadmap",
	"build_mba_lookup",
	"build_mba_rules",
	"build_promoted_commodity_flags",
	"build_recommendation_simulator_table",
	"build_seed_items_table",
	"build_temporal_holdout",
	"build_user_purchase_timeline",
	"build_variant_topk",
	"CaseStudyArtifacts",
	"compute_component_contributions",
	"compute_archetype_performance",
	"compute_archetype_utility_profile",
	"compute_household_features",
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
	"evaluate_incremental_precision",
	"filter_persuadables",
	"get_project_root",
	"HoldoutSplit",
	"load_or_build_master_transactions",
	"make_variant_weights",
	"minmax_scale",
	"normalize_discount_values",
	"normalize_lift_to_unit",
	"pick_case_study_household",
	"prepare_margin_lookup",
	"rank_candidates",
	"rowwise_minmax",
	"run_ablation",
	"attach_margin_to_topk",
	"assign_archetypes",
	"save_als_factors",
	"recommend_for_new_user",
	"score_candidate_set",
	"score_utility",
	"top_k_recommendations",
]
