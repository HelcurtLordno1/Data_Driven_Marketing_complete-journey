"""Budget Allocation Optimization and ROI Analysis for Module 9."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_incremental_margin_estimates(
    recommendations_df: pd.DataFrame,
    test_period_purchases: pd.DataFrame,
    commodity_margin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate incremental margin per household from recommendations.
    
    Compares recommended items to actual purchases to estimate the margin
    impact of each recommendation.
    
    Args:
        recommendations_df: Top-5 recommendations with Utility and Margin columns.
        test_period_purchases: Actual purchases in test period.
        commodity_margin_df: Margin data for each commodity.
    
    Returns:
        DataFrame with household-level incremental margin estimates.
    """
    # Group recommendations by household
    rec_by_hh = recommendations_df.groupby("household_key").agg(
        {
            "COMMODITY_DESC": list,
            "Utility": list,
            "Normalized_Margin": list,
        }
    ).reset_index()
    
    # Group actual purchases by household
    purchase_by_hh = test_period_purchases.groupby("household_key").agg(
        {"COMMODITY_DESC": list, "Normalized_Margin": list}
    ).reset_index()
    purchase_by_hh.columns = ["household_key", "purchased_items", "purchase_margins"]
    
    # Merge recommendations and purchases
    merged = rec_by_hh.merge(purchase_by_hh, on="household_key", how="left")
    merged["purchased_items"] = merged["purchased_items"].fillna(
        merged["purchased_items"].apply(lambda x: [])
    )
    
    # Calculate hit rate and incremental margin
    def calc_hit_and_margin(row):
        recommended = set(row["COMMODITY_DESC"])
        purchased = set(row["purchased_items"]) if isinstance(row["purchased_items"], list) else set()
        
        hits = recommended & purchased
        hit_rate = len(hits) / len(recommended) if len(recommended) > 0 else 0
        
        # Average margin of recommended items purchased
        if len(hits) > 0:
            margin_of_hits = [
                row["Normalized_Margin"][i]
                for i, item in enumerate(row["COMMODITY_DESC"])
                if item in hits
            ]
            avg_margin_hit = np.mean(margin_of_hits)
        else:
            avg_margin_hit = 0
        
        # Incremental margin = margin of recommended items purchased
        incremental_margin = avg_margin_hit * len(hits)
        
        return pd.Series(
            {
                "hit_rate": hit_rate,
                "hits": len(hits),
                "incremental_margin": incremental_margin,
                "avg_margin_hit": avg_margin_hit,
            }
        )
    
    results = merged.apply(calc_hit_and_margin, axis=1)
    merged = merged.join(results)
    
    return merged[
        [
            "household_key",
            "hit_rate",
            "hits",
            "incremental_margin",
            "avg_margin_hit",
        ]
    ].copy()


def rank_households_by_incremental_potential(
    incremental_margin_df: pd.DataFrame,
    percentile_cutoff: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank households by predicted incremental margin potential.
    
    Args:
        incremental_margin_df: Household incremental margin data.
        percentile_cutoff: Percentile cutoff for high-potential households (e.g., 0.80 for top 20%).
    
    Returns:
        Tuple of (all_ranked, high_potential_subset).
    """
    # Sort by incremental margin descending
    ranked = incremental_margin_df.sort_values(
        "incremental_margin", ascending=False
    ).reset_index(drop=True)
    
    # Add rank
    ranked["rank"] = ranked.index + 1
    ranked["percentile"] = ranked["rank"] / len(ranked)
    
    # High-potential subset (top percentage)
    threshold_percentile = 1 - (1 - percentile_cutoff)
    high_potential = ranked[
        ranked["percentile"] <= threshold_percentile
    ].copy()
    
    return ranked, high_potential


def cumulative_profit_by_strategy(
    ranked_hh: pd.DataFrame,
    strategy: str = "utility_incremental",
) -> Dict:
    """
    Calculate cumulative profit based on targeting strategy.
    
    Strategies:
    - "random": Random household selection
    - "clv": Target by highest CLV (proxy: sum of historical margins)
    - "utility_incremental": Target by highest utility-estimated incremental margin
    
    Args:
        ranked_hh: Ranked household dataframe with incremental_margin column.
        strategy: Strategy to use for ranking.
    
    Returns:
        Dictionary with cumulative profit curve and summary stats.
    """
    if strategy == "random":
        # Randomize order
        profit = ranked_hh.copy()
        profit = profit.sample(frac=1, random_state=42).reset_index(drop=True)
    elif strategy == "utility_incremental":
        # Already sorted by incremental margin
        profit = ranked_hh.copy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate cumulative profit
    profit["cumulative_margin"] = profit["incremental_margin"].cumsum()
    profit["cumulative_households"] = np.arange(1, len(profit) + 1)
    profit["pct_households_targeted"] = profit["cumulative_households"] / len(profit) * 100
    
    return {
        "strategy": strategy,
        "dataframe": profit,
        "total_incremental_margin": profit["incremental_margin"].sum(),
        "avg_incremental_margin_per_hh": profit["incremental_margin"].mean(),
        "max_profit_targeting_top_20pct": profit[
            profit["pct_households_targeted"] <= 20
        ]["cumulative_margin"].iloc[-1]
        if len(profit[profit["pct_households_targeted"] <= 20]) > 0
        else 0,
    }


def compare_targeting_strategies(
    incremental_margin_df: pd.DataFrame,
    strategies: List[str] = None,
) -> pd.DataFrame:
    """
    Compare cumulative profit across multiple targeting strategies.
    
    Args:
        incremental_margin_df: Household incremental margin data.
        strategies: List of strategies to compare.
    
    Returns:
        Comparison dataframe with profit curves for each strategy.
    """
    if strategies is None:
        strategies = ["random", "utility_incremental"]
    
    results = []
    
    for strategy in strategies:
        if strategy == "random":
            # Randomize for random strategy
            test_data = incremental_margin_df.sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
        elif strategy == "utility_incremental":
            # Sort by incremental margin descending for utility strategy
            test_data = incremental_margin_df.sort_values(
                "incremental_margin", ascending=False
            ).reset_index(drop=True)
        else:
            continue
        
        outcome = cumulative_profit_by_strategy(test_data, strategy=strategy)
        
        results.append(
            {
                "strategy": strategy,
                "total_profit": outcome["total_incremental_margin"],
                "avg_profit_per_hh": outcome["avg_incremental_margin_per_hh"],
                "profit_top_20pct": outcome["max_profit_targeting_top_20pct"],
                "dataframe": outcome["dataframe"],
            }
        )
    
    summary = pd.DataFrame(
        {
            "Strategy": [r["strategy"] for r in results],
            "Total Incremental Margin": [r["total_profit"] for r in results],
            "Avg Margin per Household": [r["avg_profit_per_hh"] for r in results],
            "Profit from Top 20% Targeted": [r["profit_top_20pct"] for r in results],
        }
    )
    
    return summary, results


def budget_allocation_optimization(
    incremental_margin_df: pd.DataFrame,
    budget_per_hh: float = 10.0,
    total_budget: float = None,
) -> Dict:
    """
    Optimize budget allocation based on incremental margin estimates.
    
    Determines how many households should be targeted given a budget
    constraint and per-household intervention cost.
    
    Args:
        incremental_margin_df: Household incremental margin data.
        budget_per_hh: Cost to target one household (e.g., $10).
        total_budget: Total marketing budget available.
    
    Returns:
        Dictionary with optimization results.
    """
    # Sort by incremental margin descending
    ranked = incremental_margin_df.sort_values(
        "incremental_margin", ascending=False
    ).reset_index(drop=True)
    
    ranked["cost_to_target"] = budget_per_hh
    ranked["net_benefit"] = ranked["incremental_margin"] - budget_per_hh
    ranked["cumulative_cost"] = (ranked.index + 1) * budget_per_hh
    ranked["cumulative_benefit"] = ranked["incremental_margin"].cumsum()
    ranked["cumulative_net"] = ranked["net_benefit"].cumsum()
    
    # Determine max households to target within budget
    if total_budget is not None:
        max_hh = int(total_budget / budget_per_hh)
        affordable_hh = ranked[ranked.index < max_hh].copy()
    else:
        # Target all households with positive net benefit
        affordable_hh = ranked[ranked["net_benefit"] > 0].copy()
    
    return {
        "ranked_by_incremental_margin": ranked,
        "optimal_target_count": len(affordable_hh),
        "total_investment": len(affordable_hh) * budget_per_hh,
        "total_projected_benefit": affordable_hh["incremental_margin"].sum(),
        "net_profit": affordable_hh["net_benefit"].sum(),
        "roi": (
            affordable_hh["net_benefit"].sum() / (len(affordable_hh) * budget_per_hh) * 100
            if len(affordable_hh) > 0 and budget_per_hh > 0
            else 0
        ),
        "recommendation": (
            f"Target top {len(affordable_hh)} households with highest incremental potential. "
            f"Projected ROI: {(affordable_hh['net_benefit'].sum() / (len(affordable_hh) * budget_per_hh) * 100):.1f}%"
            if len(affordable_hh) > 0
            else "No households have positive ROI at current budget per household."
        ),
    }


def lifetime_value_proxy(
    household_history: pd.DataFrame,
    margin_col: str = "train_avg_margin",
    items_col: str = "train_items",
) -> pd.DataFrame:
    """
    Compute customer lifetime value proxy based on historical purchasing.
    
    Args:
        household_history: Historical household purchase data.
        margin_col: Column name for average margin.
        items_col: Column name for item count.
    
    Returns:
        DataFrame with CLV proxy per household.
    """
    clv_data = household_history[[margin_col, items_col]].copy()
    clv_data["household_key"] = household_history.index
    
    # CLV proxy = average margin * number of items (proxy for engagement)
    clv_data["clv_proxy"] = clv_data[margin_col] * clv_data[items_col]
    
    return clv_data[["household_key", "clv_proxy"]].sort_values(
        "clv_proxy", ascending=False
    )
