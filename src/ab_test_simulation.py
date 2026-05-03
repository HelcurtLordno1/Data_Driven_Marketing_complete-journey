"""A/B Test Simulation and Outcome Analysis for Module 9."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_power_analysis(
    historical_margin_data: pd.Series,
    min_detectable_effect: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    """
    Compute power analysis for the A/B test.
    
    Args:
        historical_margin_data: Historical incremental margin per household.
        min_detectable_effect: Minimum effect size to detect (in dollars or percentage).
        alpha: Type I error rate.
        power: Statistical power desired.
    
    Returns:
        Dictionary with sample size and detection details.
    """
    # Remove NaN and infinite values
    data = historical_margin_data.dropna()
    data = data[np.isfinite(data)]
    
    if len(data) < 2:
        raise ValueError("Insufficient historical data for power analysis")
    
    mean_margin = data.mean()
    std_margin = data.std()
    
    # If mean is near zero, use absolute effect; otherwise relative
    if abs(mean_margin) < 1e-6:
        effect_size_absolute = min_detectable_effect
    else:
        effect_size_absolute = abs(mean_margin) * min_detectable_effect
    
    # Standardized effect size (Cohen's d)
    cohens_d = effect_size_absolute / std_margin if std_margin > 0 else 0
    
    # Approximate sample size using Welch's t-test formula
    # n ≈ 2 * (z_alpha/2 + z_beta)^2 / d^2
    z_alpha_half = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    if cohens_d > 0:
        sample_size_per_group = int(
            np.ceil(2 * ((z_alpha_half + z_beta) / cohens_d) ** 2)
        )
    else:
        sample_size_per_group = 10000  # Default large number if effect size is near zero
    
    return {
        "mean_margin": mean_margin,
        "std_margin": std_margin,
        "cohens_d": cohens_d,
        "z_alpha_half": z_alpha_half,
        "z_beta": z_beta,
        "sample_size_per_group": sample_size_per_group,
        "total_sample_size": sample_size_per_group * 2,
        "min_detectable_effect": effect_size_absolute,
    }


def simulate_ab_test(
    control_margins: pd.Series,
    treatment_margins: pd.Series,
    random_seed: int = 42,
) -> Dict:
    """
    Simulate A/B test with control (Popularity) vs. treatment (Chimera).
    
    Args:
        control_margins: Incremental margins for control group (Popularity baseline).
        treatment_margins: Incremental margins for treatment group (Chimera).
        random_seed: Random seed for reproducibility.
    
    Returns:
        Dictionary with test results and statistics.
    """
    np.random.seed(random_seed)
    
    # Remove NaN values
    control = control_margins.dropna().values
    treatment = treatment_margins.dropna().values
    
    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Control or treatment group has no data")
    
    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment)
    control_std = np.std(control, ddof=1)
    treatment_std = np.std(treatment, ddof=1)
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(control) - 1) * control_std ** 2 + (len(treatment) - 1) * treatment_std ** 2)
        / (len(control) + len(treatment) - 2)
    )
    cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
    
    # Absolute lift
    absolute_lift = treatment_mean - control_mean
    relative_lift_pct = (absolute_lift / abs(control_mean) * 100) if control_mean != 0 else 0
    
    # Confidence interval using Welch's method
    se_diff = np.sqrt(control_std ** 2 / len(control) + treatment_std ** 2 / len(treatment))
    df_welch = (
        (control_std ** 2 / len(control) + treatment_std ** 2 / len(treatment)) ** 2
        / (
            (control_std ** 2 / len(control)) ** 2 / (len(control) - 1)
            + (treatment_std ** 2 / len(treatment)) ** 2 / (len(treatment) - 1)
        )
    )
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_lower = absolute_lift - t_crit * se_diff
    ci_upper = absolute_lift + t_crit * se_diff
    
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "control_std": control_std,
        "treatment_std": treatment_std,
        "control_n": len(control),
        "treatment_n": len(treatment),
        "absolute_lift": absolute_lift,
        "relative_lift_pct": relative_lift_pct,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_significant": p_value < 0.05,
    }


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    statistic_func=np.mean,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Data array.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        statistic_func: Function to compute statistic (default: mean).
    
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    np.random.seed(42)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    alpha = 1 - ci
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower_bound, upper_bound


def random_split_households(
    household_ids: pd.Series,
    control_fraction: float = 0.5,
    random_seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Randomly split households into control and treatment groups.
    
    Args:
        household_ids: Household key identifiers.
        control_fraction: Fraction assigned to control group.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (control_ids, treatment_ids).
    """
    np.random.seed(random_seed)
    
    n_total = len(household_ids)
    n_control = int(np.ceil(n_total * control_fraction))
    
    # Randomly shuffle indices
    shuffled_indices = np.random.permutation(n_total)
    control_indices = shuffled_indices[:n_control]
    treatment_indices = shuffled_indices[n_control:]
    
    control_ids = household_ids.iloc[control_indices].values
    treatment_ids = household_ids.iloc[treatment_indices].values
    
    return control_ids, treatment_ids


def compute_incremental_margin(
    baseline_margin: pd.Series,
    treatment_margin: pd.Series,
) -> pd.Series:
    """
    Compute incremental margin (treatment - baseline).
    
    Args:
        baseline_margin: Baseline (control) margin per household.
        treatment_margin: Treatment margin per household.
    
    Returns:
        Incremental margin series.
    """
    return treatment_margin - baseline_margin


def guardrail_checks(
    control_data: pd.DataFrame,
    treatment_data: pd.DataFrame,
    basket_size_col: str = "test_baskets",
    purchase_freq_col: str = "test_baskets",
    tolerance: float = 0.10,
) -> Dict[str, bool]:
    """
    Check guardrails: no significant drop in basket size or purchase frequency.
    
    Args:
        control_data: Control group household metrics.
        treatment_data: Treatment group household metrics.
        basket_size_col: Column name for basket size.
        purchase_freq_col: Column name for purchase frequency.
        tolerance: Acceptable relative drop tolerance (default 10%).
    
    Returns:
        Dictionary with guardrail check results.
    """
    if basket_size_col in control_data.columns and basket_size_col in treatment_data.columns:
        control_basket_size = control_data[basket_size_col].mean()
        treatment_basket_size = treatment_data[basket_size_col].mean()
        basket_size_ok = (
            (treatment_basket_size / control_basket_size) >= (1 - tolerance)
            if control_basket_size > 0
            else True
        )
    else:
        basket_size_ok = True
    
    if purchase_freq_col in control_data.columns and purchase_freq_col in treatment_data.columns:
        control_freq = control_data[purchase_freq_col].mean()
        treatment_freq = treatment_data[purchase_freq_col].mean()
        freq_ok = (
            (treatment_freq / control_freq) >= (1 - tolerance)
            if control_freq > 0
            else True
        )
    else:
        freq_ok = True
    
    return {
        "basket_size_guardrail_ok": basket_size_ok,
        "purchase_freq_guardrail_ok": freq_ok,
        "all_guardrails_ok": basket_size_ok and freq_ok,
    }


def summarize_ab_test_results(test_results: Dict, n_households: int) -> pd.DataFrame:
    """
    Summarize A/B test results into a readable dataframe.
    
    Args:
        test_results: Results dictionary from simulate_ab_test().
        n_households: Total number of households in test.
    
    Returns:
        Summary dataframe.
    """
    summary = pd.DataFrame(
        {
            "Metric": [
                "Control Mean Margin ($)",
                "Treatment Mean Margin ($)",
                "Absolute Lift ($)",
                "Relative Lift (%)",
                "Cohen's d",
                "t-statistic",
                "p-value",
                "95% CI Lower",
                "95% CI Upper",
                "Statistically Significant",
                "Control Sample Size",
                "Treatment Sample Size",
            ],
            "Value": [
                f"${test_results['control_mean']:.2f}",
                f"${test_results['treatment_mean']:.2f}",
                f"${test_results['absolute_lift']:.2f}",
                f"{test_results['relative_lift_pct']:.2f}%",
                f"{test_results['cohens_d']:.3f}",
                f"{test_results['t_stat']:.4f}",
                f"{test_results['p_value']:.4f}",
                f"${test_results['ci_lower']:.2f}",
                f"${test_results['ci_upper']:.2f}",
                "Yes" if test_results["is_significant"] else "No",
                test_results["control_n"],
                test_results["treatment_n"],
            ],
        }
    )
    return summary
