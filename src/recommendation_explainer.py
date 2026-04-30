"""Module 7 – Interpretability & Trust helpers.

Provides:
  - global_component_importance   : permutation-importance proxy via RF classifier
  - generate_explanation_card     : human-readable "Why this recommendation?" text
  - counterfactual_explanation    : shows what would change a mis-recommendation rank
  - weight_sensitivity_analysis   : top-5 list changes as w2/w3 are swept
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utility_scorer import DEFAULT_UTILITY_WEIGHTS, score_utility


# ---------------------------------------------------------------------------
# 1. Global Component Importance
# ---------------------------------------------------------------------------

COMPONENT_COLS = ["Relevance", "Uplift", "Normalized_Margin", "Context"]
COMPONENT_LABELS = {
    "Relevance": "Relevance",
    "Uplift": "Uplift",
    "Normalized_Margin": "Margin",
    "Context": "Context",
}


@dataclass
class GlobalImportanceResult:
    """Container for the global RF importance analysis."""

    importances_mean: pd.Series       # mean permutation importance per feature
    importances_std: pd.Series        # std of permutation importance
    classifier_accuracy: float        # OOB or test-set accuracy
    n_samples: int
    n_purchased: int
    feature_names: List[str]


def _build_purchase_labels(
    scored_candidates: pd.DataFrame,
    test_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a binary 'purchased' label to each scored candidate row."""
    purchased_pairs = (
        test_history[["household_key", "COMMODITY_DESC"]]
        .drop_duplicates()
        .assign(purchased=1)
    )
    labeled = scored_candidates.merge(
        purchased_pairs, on=["household_key", "COMMODITY_DESC"], how="left"
    )
    labeled["purchased"] = labeled["purchased"].fillna(0).astype(int)
    return labeled


def compute_global_component_importance(
    scored_candidates: pd.DataFrame,
    test_history: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
    n_repeats: int = 10,
) -> GlobalImportanceResult:
    """Fit an RF classifier to predict purchase; extract permutation importance.

    Parameters
    ----------
    scored_candidates:
        Full scored candidate set (from Module 3/4) with columns
        [household_key, COMMODITY_DESC, Relevance, Uplift,
         Normalized_Margin, Context, Utility].
    test_history:
        Test-period transactions used to build purchase labels.
    """
    required = {"household_key", "COMMODITY_DESC"} | set(COMPONENT_COLS)
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"scored_candidates missing columns: {sorted(missing)}")

    labeled = _build_purchase_labels(scored_candidates, test_history)
    feature_data = labeled[COMPONENT_COLS].fillna(0.0)
    target = labeled["purchased"]

    if target.sum() < 5:
        raise ValueError(
            "Fewer than 5 purchased items in candidate set – cannot fit classifier."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target, test_size=0.25, random_state=random_state, stratify=target
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,   # avoid joblib disk temp-files on low-disk machines
    )
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    perm = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,   # single-threaded: no disk serialization needed
    )

    feature_names = [COMPONENT_LABELS.get(c, c) for c in COMPONENT_COLS]
    importances_mean = pd.Series(perm.importances_mean, index=feature_names)
    importances_std = pd.Series(perm.importances_std, index=feature_names)

    return GlobalImportanceResult(
        importances_mean=importances_mean.sort_values(ascending=False),
        importances_std=importances_std,
        classifier_accuracy=float(accuracy),
        n_samples=int(len(labeled)),
        n_purchased=int(target.sum()),
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# 2. Per-Recommendation Explanation Card
# ---------------------------------------------------------------------------

@dataclass
class ExplanationCard:
    """Human-readable explanation for a single recommendation."""

    household_key: int
    commodity_desc: str
    rank: int
    utility_score: float
    relevance: float
    uplift: float
    margin: float
    context: float
    weights: Dict[str, float]
    habit_strength: float
    deal_sensitivity: float
    similar_user_pct: float           # fraction of similar users who bought this
    relevance_text: str = field(init=False)
    uplift_text: str = field(init=False)
    margin_text: str = field(init=False)
    context_text: str = field(init=False)
    summary_text: str = field(init=False)

    def __post_init__(self) -> None:
        rel_qual = "high" if self.relevance >= 0.6 else ("moderate" if self.relevance >= 0.3 else "low")
        self.relevance_text = (
            f"Relevance: {rel_qual} ({self.relevance:.2f}) – "
            f"{self.similar_user_pct:.0%} of similar users buy this category."
        )

        uplift_qual = (
            "strong discovery opportunity" if self.uplift >= 0.7
            else "moderate new-item opportunity" if self.uplift >= 0.4
            else "familiar item, low incremental lift"
        )
        self.uplift_text = (
            f"Uplift: {self.uplift:.2f} – habit strength is "
            f"{self.habit_strength:.0%} of trips → {uplift_qual}."
        )

        margin_qual = "high" if self.margin >= 0.6 else ("average" if self.margin >= 0.3 else "low")
        self.margin_text = (
            f"Margin: {margin_qual} ({self.margin:.2f}) – "
            "this item carries a strong retailer margin."
            if self.margin >= 0.6
            else f"Margin: {margin_qual} ({self.margin:.2f})."
        )

        ds = self.deal_sensitivity
        if ds > 0.6:
            ctx_prefix = "Deal-sensitive shopper"
        elif ds < 0.3:
            ctx_prefix = "Non-deal-driven shopper"
        else:
            ctx_prefix = "Moderate deal sensitivity"
        self.context_text = (
            f"Context: {self.context:.2f} – {ctx_prefix} "
            f"(deal_sensitivity={ds:.2f}); current campaign conditions applied."
        )

        w = self.weights
        self.summary_text = (
            f"Overall utility = "
            f"{w['relevance']:.2f}×{self.relevance:.3f} + "
            f"{w['uplift']:.2f}×{self.uplift:.3f} + "
            f"{w['margin']:.2f}×{self.margin:.3f} + "
            f"{w['context']:.2f}×{self.context:.3f} "
            f"= {self.utility_score:.4f}"
        )


def generate_explanation_card(
    household_key: int,
    commodity_desc: str,
    scored_row: pd.Series,
    habit_strength: float,
    deal_sensitivity: float,
    similar_user_pct: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
    rank: int = 1,
) -> ExplanationCard:
    """Build an ExplanationCard for a single (household, item) pair."""
    active_weights = dict(DEFAULT_UTILITY_WEIGHTS)
    if weights:
        active_weights.update(weights)

    return ExplanationCard(
        household_key=int(household_key),
        commodity_desc=str(commodity_desc),
        rank=int(rank),
        utility_score=float(scored_row.get("Utility", 0.0)),
        relevance=float(scored_row.get("Relevance", 0.0)),
        uplift=float(scored_row.get("Uplift", 0.0)),
        margin=float(scored_row.get("Normalized_Margin", 0.0)),
        context=float(scored_row.get("Context", 0.0)),
        weights=active_weights,
        habit_strength=float(habit_strength),
        deal_sensitivity=float(deal_sensitivity),
        similar_user_pct=float(similar_user_pct),
    )


def generate_explanation_cards_for_household(
    household_key: int,
    top5: pd.DataFrame,
    deal_sensitivity_lookup: pd.DataFrame,
    habit_strength_lookup: pd.DataFrame,
    similar_user_pct_lookup: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[ExplanationCard]:
    """Generate all Top-5 explanation cards for a household."""
    hh_recs = top5[top5["household_key"] == household_key].reset_index(drop=True)
    if hh_recs.empty:
        return []

    ds_row = deal_sensitivity_lookup[deal_sensitivity_lookup["household_key"] == household_key]
    deal_sens = float(ds_row["deal_sensitivity"].iloc[0]) if not ds_row.empty else 0.5

    cards = []
    for rank, (_, row) in enumerate(hh_recs.iterrows(), start=1):
        commodity = str(row["COMMODITY_DESC"])

        hs_row = habit_strength_lookup[
            (habit_strength_lookup["household_key"] == household_key)
            & (habit_strength_lookup["COMMODITY_DESC"] == commodity)
        ]
        hs = float(hs_row["habit_strength"].iloc[0]) if not hs_row.empty else 0.0

        sim_pct = 0.5
        if similar_user_pct_lookup is not None:
            sim_row = similar_user_pct_lookup[
                similar_user_pct_lookup["COMMODITY_DESC"] == commodity
            ]
            if not sim_row.empty:
                sim_pct = float(sim_row["similar_user_pct"].iloc[0])

        card = generate_explanation_card(
            household_key=household_key,
            commodity_desc=commodity,
            scored_row=row,
            habit_strength=hs,
            deal_sensitivity=deal_sens,
            similar_user_pct=sim_pct,
            weights=weights,
            rank=rank,
        )
        cards.append(card)
    return cards


def cards_to_dataframe(cards: List[ExplanationCard]) -> pd.DataFrame:
    """Convert a list of ExplanationCards to a tidy DataFrame."""
    rows = []
    for c in cards:
        rows.append({
            "household_key": c.household_key,
            "rank": c.rank,
            "commodity_desc": c.commodity_desc,
            "utility_score": c.utility_score,
            "relevance": c.relevance,
            "uplift": c.uplift,
            "margin": c.margin,
            "context": c.context,
            "habit_strength": c.habit_strength,
            "deal_sensitivity": c.deal_sensitivity,
            "relevance_text": c.relevance_text,
            "uplift_text": c.uplift_text,
            "margin_text": c.margin_text,
            "context_text": c.context_text,
            "summary_text": c.summary_text,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Counterfactual Explanation
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """What would need to change for this item to rank higher."""

    household_key: int
    commodity_desc: str
    original_rank: int
    original_utility: float
    target_rank: int
    target_utility: float             # utility needed to reach target rank
    delta_needed: float               # total utility gap
    counterfactual_habit_strength: float   # new habit_strength that would achieve it
    counterfactual_uplift: float
    counterfactual_utility: float
    narrative: str


def compute_counterfactual_explanation(
    household_key: int,
    commodity_desc: str,
    top5: pd.DataFrame,
    all_candidates: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    target_rank: int = 5,
) -> Optional[CounterfactualResult]:
    """For a mis-recommended item (in top-5 but not purchased), compute
    counterfactual: how much would habit_strength need to drop (→ uplift rise)
    for this item to reach rank `target_rank` among all candidates?

    Returns None if item is not found or already purchased.
    """
    active_weights = dict(DEFAULT_UTILITY_WEIGHTS)
    if weights:
        active_weights.update(weights)

    if target_rank < 1:
        raise ValueError("target_rank must be >= 1.")
    if active_weights.get("uplift", 0.0) <= 0:
        raise ValueError("Uplift weight must be > 0 for counterfactual analysis.")

    required_cols = {"household_key", "COMMODITY_DESC"}
    if not required_cols.issubset(set(top5.columns)):
        raise ValueError("top5 must include household_key and COMMODITY_DESC columns.")
    if not required_cols.issubset(set(all_candidates.columns)):
        raise ValueError(
            "all_candidates must include household_key and COMMODITY_DESC columns."
        )

    hh_top5 = top5[top5["household_key"] == household_key]
    if hh_top5.empty:
        return None
    if commodity_desc not in set(hh_top5["COMMODITY_DESC"].values):
        return None

    hh_all = all_candidates[all_candidates["household_key"] == household_key].copy()
    if hh_all.empty:
        return None

    for col in COMPONENT_COLS:
        if col not in hh_all.columns:
            hh_all[col] = 0.0

    if "Utility" not in hh_all.columns:
        hh_all["Utility"] = score_utility(
            relevance=hh_all.get("Relevance", 0.0),
            uplift=hh_all.get("Uplift", 0.0),
            margin=hh_all.get("Normalized_Margin", 0.0),
            context=hh_all.get("Context", 0.0),
            weights=active_weights,
        )

    hh_all = hh_all.sort_values("Utility", ascending=False).reset_index(drop=True)

    item_row_idx = hh_all[hh_all["COMMODITY_DESC"] == commodity_desc].index
    if len(item_row_idx) == 0:
        return None

    item_row = hh_all.loc[item_row_idx[0]]
    original_rank = int(item_row_idx[0]) + 1
    original_utility = float(item_row["Utility"])

    # Utility threshold needed to be at target_rank among all candidates
    if target_rank <= len(hh_all):
        threshold_utility = float(hh_all.iloc[target_rank - 1]["Utility"])
    else:
        threshold_utility = original_utility + 0.05

    delta_needed = max(0.0, threshold_utility - original_utility)

    # Solve: new_utility = R*relevance + U*new_uplift + M*margin + C*context >= threshold
    # new_uplift = (threshold - fixed_part) / U
    fixed_part = (
        active_weights["relevance"] * float(item_row.get("Relevance", 0.0))
        + active_weights["margin"] * float(item_row.get("Normalized_Margin", 0.0))
        + active_weights["context"] * float(item_row.get("Context", 0.0))
    )
    uplift_needed = (threshold_utility - fixed_part) / active_weights["uplift"]
    uplift_needed = float(np.clip(uplift_needed, 0.0, 1.0))
    habit_needed = float(np.clip(1.0 - uplift_needed, 0.0, 1.0))

    new_utility = float(score_utility(
        relevance=pd.Series([float(item_row.get("Relevance", 0.0))]),
        uplift=pd.Series([uplift_needed]),
        margin=pd.Series([float(item_row.get("Normalized_Margin", 0.0))]),
        context=pd.Series([float(item_row.get("Context", 0.0))]),
        weights=active_weights,
    ).iloc[0])

    narrative = (
        f"'{commodity_desc}' was ranked #{original_rank} (utility={original_utility:.4f}) "
        f"among all candidates but was not purchased. To reach rank #{target_rank}, the "
        f"utility would need to be >= {threshold_utility:.4f} (Δ={delta_needed:.4f}). "
        f"This is achievable if the shopper's habit strength for this category dropped "
        f"from {1.0 - float(item_row.get('Uplift', 0.0)):.2f} to {habit_needed:.2f} "
        f"(i.e., they purchased it less frequently in training), "
        f"raising the Uplift signal from {float(item_row.get('Uplift', 0.0)):.2f} "
        f"to {uplift_needed:.2f}."
    )

    return CounterfactualResult(
        household_key=int(household_key),
        commodity_desc=commodity_desc,
        original_rank=original_rank,
        original_utility=original_utility,
        target_rank=target_rank,
        target_utility=threshold_utility,
        delta_needed=delta_needed,
        counterfactual_habit_strength=habit_needed,
        counterfactual_uplift=uplift_needed,
        counterfactual_utility=new_utility,
        narrative=narrative,
    )


# ---------------------------------------------------------------------------
# 4. Weight Sensitivity Analysis
# ---------------------------------------------------------------------------

@dataclass
class WeightSensitivityResult:
    """How the Top-5 list changes as a weight is swept."""

    household_key: int
    swept_weight: str       # e.g. "uplift" or "margin"
    sweep_values: List[float]
    top5_per_step: List[pd.DataFrame]   # one df per step
    stability_score: float              # fraction of steps where rank-1 item stays #1
    rank_change_matrix: pd.DataFrame    # items × steps → rank


def weight_sensitivity_analysis(
    household_key: int,
    all_candidates: pd.DataFrame,
    swept_weight: str = "uplift",
    sweep_range: Tuple[float, float] = (0.0, 0.5),
    n_steps: int = 11,
    base_weights: Optional[Dict[str, float]] = None,
) -> WeightSensitivityResult:
    """Sweep one utility weight across a range; record how the Top-5 list changes.

    Parameters
    ----------
    swept_weight : one of {"relevance", "uplift", "margin", "context"}
    sweep_range  : (min_val, max_val) for the swept weight
    n_steps      : number of discrete steps
    """
    if swept_weight not in {"relevance", "uplift", "margin", "context"}:
        raise ValueError(f"swept_weight must be one of the four utility components.")

    base = dict(DEFAULT_UTILITY_WEIGHTS)
    if base_weights:
        base.update(base_weights)

    hh_cands = all_candidates[all_candidates["household_key"] == household_key].copy()
    if hh_cands.empty:
        raise ValueError(f"No candidates found for household {household_key}.")

    # Ensure component columns exist
    for col in COMPONENT_COLS:
        if col not in hh_cands.columns:
            hh_cands[col] = 0.0

    sweep_values = list(np.linspace(sweep_range[0], sweep_range[1], n_steps))
    top5_per_step: List[pd.DataFrame] = []

    base_utility = score_utility(
        relevance=hh_cands["Relevance"],
        uplift=hh_cands["Uplift"],
        margin=hh_cands["Normalized_Margin"],
        context=hh_cands["Context"],
        weights=base,
    )
    base_top1 = hh_cands.assign(Utility=base_utility).nlargest(1, "Utility")[
        "COMMODITY_DESC"
    ].iloc[0]

    for val in sweep_values:
        w = dict(base)
        total_others = sum(v for k, v in base.items() if k != swept_weight)
        remaining = 1.0 - val
        if total_others > 0:
            scale = remaining / total_others
        else:
            scale = 1.0
        for k in w:
            if k == swept_weight:
                w[k] = val
            else:
                w[k] = base[k] * scale

        step_cands = hh_cands.copy()
        step_cands["Utility"] = score_utility(
            relevance=step_cands["Relevance"],
            uplift=step_cands["Uplift"],
            margin=step_cands["Normalized_Margin"],
            context=step_cands["Context"],
            weights=w,
        )
        top5 = step_cands.nlargest(5, "Utility")[
            ["COMMODITY_DESC", "Utility", "Relevance", "Uplift", "Normalized_Margin", "Context"]
        ].reset_index(drop=True)
        top5["rank"] = range(1, len(top5) + 1)
        top5["sweep_value"] = round(val, 4)
        top5_per_step.append(top5)

    # Stability: fraction of steps where the #1 item matches the base weights
    stability = sum(
        1 for df in top5_per_step if (not df.empty and df.iloc[0]["COMMODITY_DESC"] == base_top1)
    ) / n_steps

    # Rank-change matrix: items × sweep steps
    all_seen_items = sorted(
        {row["COMMODITY_DESC"] for df in top5_per_step for _, row in df.iterrows()}
    )
    matrix_data = {
        f"step_{i+1}_w={round(v, 2)}": {
            row["COMMODITY_DESC"]: int(row["rank"])
            for _, row in df.iterrows()
        }
        for i, (v, df) in enumerate(zip(sweep_values, top5_per_step))
    }
    rank_matrix = pd.DataFrame(matrix_data, index=all_seen_items).fillna(99).astype(int)

    return WeightSensitivityResult(
        household_key=int(household_key),
        swept_weight=swept_weight,
        sweep_values=sweep_values,
        top5_per_step=top5_per_step,
        stability_score=float(stability),
        rank_change_matrix=rank_matrix,
    )


# ---------------------------------------------------------------------------
# 5. Similar-user percentage helper
# ---------------------------------------------------------------------------

def compute_similar_user_pct(
    history: pd.DataFrame,
    n_similar_users: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate the fraction of users (sampled) who bought each commodity.

    Used to populate the 'similar_user_pct' in explanation cards.
    """
    np.random.seed(random_state)
    all_users = history["household_key"].unique()
    sampled = np.random.choice(
        all_users, size=min(n_similar_users, len(all_users)), replace=False
    )
    subset = history[history["household_key"].isin(sampled)]
    n_users = len(sampled)
    pct = (
        subset.groupby("COMMODITY_DESC")["household_key"]
        .nunique()
        .reset_index()
        .rename(columns={"household_key": "similar_user_pct"})
    )
    pct["similar_user_pct"] = pct["similar_user_pct"] / n_users
    return pct
