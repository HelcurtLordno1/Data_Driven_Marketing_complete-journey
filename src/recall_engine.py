"""Recommendation engine components for Module 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix


def normalize_lift_to_unit(lift_values: np.ndarray | pd.Series) -> np.ndarray:
	"""Map Lift values into [0, 1] using the project rule."""
	x = np.asarray(lift_values, dtype=float)
	return np.where(x <= 1.0, 0.0, np.where(x >= 3.0, 1.0, (x - 1.0) / 2.0))


def minmax_scale(values: np.ndarray) -> np.ndarray:
	arr = np.asarray(values, dtype=float)
	if arr.size == 0:
		return arr
	vmin, vmax = float(arr.min()), float(arr.max())
	if np.isclose(vmin, vmax):
		return np.ones_like(arr) if vmax > 0 else np.zeros_like(arr)
	return (arr - vmin) / (vmax - vmin)


def rowwise_minmax(arr_2d: np.ndarray) -> np.ndarray:
	arr = np.asarray(arr_2d, dtype=float)
	if arr.size == 0:
		return arr
	mins = arr.min(axis=1, keepdims=True)
	maxs = arr.max(axis=1, keepdims=True)
	den = maxs - mins
	out = np.zeros_like(arr, dtype=float)
	has_var = den.squeeze(1) > 0
	if np.any(has_var):
		out[has_var] = (arr[has_var] - mins[has_var]) / den[has_var]
	if np.any(~has_var):
		out[~has_var] = (maxs[~has_var] > 0).astype(float)
	return out


def build_mba_rules(
	organic_tx: pd.DataFrame,
	min_support: float = 0.001,
	max_len: int = 2,
	max_baskets: int = 150_000,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Fit FP-Growth on organic baskets and return long-format rule mappings."""
	filtered = organic_tx[~organic_tx["COMMODITY_DESC"].isin(["", "UNKNOWN_COMMODITY", "NO COMMODITY DESCRIPTION"])].copy()
	basket_items = filtered.groupby("BASKET_ID")["COMMODITY_DESC"].agg(lambda series: sorted(set(series.dropna().astype(str))))
	basket_items = basket_items[basket_items.apply(len) >= 2]

	if len(basket_items) == 0:
		empty_rules = pd.DataFrame(columns=["antecedent_item", "consequent_item", "lift", "confidence", "support", "relevance_mba"])
		empty_raw = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
		return empty_rules, empty_raw

	if len(basket_items) > max_baskets:
		basket_items = basket_items.sample(n=max_baskets, random_state=random_state)

	encoder = TransactionEncoder()
	transactions = basket_items.tolist()
	encoded = encoder.fit(transactions).transform(transactions, sparse=True)
	basket_ohe = pd.DataFrame.sparse.from_spmatrix(encoded, index=basket_items.index, columns=encoder.columns_)

	itemsets = fpgrowth(basket_ohe, min_support=min_support, use_colnames=True, max_len=max_len)
	if itemsets.empty:
		empty_rules = pd.DataFrame(columns=["antecedent_item", "consequent_item", "lift", "confidence", "support", "relevance_mba"])
		empty_raw = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
		return empty_rules, empty_raw

	rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
	if rules.empty:
		empty_rules = pd.DataFrame(columns=["antecedent_item", "consequent_item", "lift", "confidence", "support", "relevance_mba"])
		empty_raw = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
		return empty_rules, empty_raw

	rules = rules[(rules["consequents"].apply(len) == 1) & (rules["antecedents"].apply(len) >= 1)].copy()
	if rules.empty:
		empty_rules = pd.DataFrame(columns=["antecedent_item", "consequent_item", "lift", "confidence", "support", "relevance_mba"])
		empty_raw = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
		return empty_rules, empty_raw

	rules["consequent_item"] = rules["consequents"].apply(lambda value: next(iter(value)))
	rules["relevance_mba"] = normalize_lift_to_unit(rules["lift"])

	rows = []
	for rule in rules.itertuples(index=False):
		for antecedent in rule.antecedents:
			rows.append(
				{
					"antecedent_item": str(antecedent),
					"consequent_item": str(rule.consequent_item),
					"lift": float(rule.lift),
					"confidence": float(rule.confidence),
					"support": float(rule.support),
					"relevance_mba": float(rule.relevance_mba),
				}
			)

	rules_long = pd.DataFrame(rows)
	invalid_tokens = {"", "UNKNOWN_COMMODITY", "NO COMMODITY DESCRIPTION"}
	rules_long = rules_long[
		(~rules_long["antecedent_item"].isin(invalid_tokens))
		& (~rules_long["consequent_item"].isin(invalid_tokens))
		& (rules_long["antecedent_item"] != rules_long["consequent_item"])
	].copy()
	rules_long = (
		rules_long.sort_values(["antecedent_item", "relevance_mba", "lift"], ascending=[True, False, False])
		.drop_duplicates(["antecedent_item", "consequent_item"])
		.reset_index(drop=True)
	)
	return rules_long, rules


def build_als_model(
	all_tx: pd.DataFrame,
	factors: int = 64,
	regularization: float = 0.05,
	iterations: int = 20,
	random_state: int = 42,
):
	"""Fit implicit ALS with confidence weighted by retailer revenue."""
	interactions = all_tx.groupby(["household_key", "COMMODITY_DESC"], as_index=False)["Revenue_Retailer"].sum()
	if interactions.empty:
		raise ValueError("Cannot train ALS on an empty interactions table.")

	users = np.sort(interactions["household_key"].unique())
	items = np.sort(interactions["COMMODITY_DESC"].astype(str).unique())
	user_to_idx = {int(user): int(index) for index, user in enumerate(users)}
	idx_to_user = {int(index): int(user) for index, user in enumerate(users)}
	item_to_idx = {str(item): int(index) for index, item in enumerate(items)}
	idx_to_item = {int(index): str(item) for index, item in enumerate(items)}

	u_idx = interactions["household_key"].map(user_to_idx).to_numpy(dtype=np.int32)
	i_idx = interactions["COMMODITY_DESC"].astype(str).map(item_to_idx).to_numpy(dtype=np.int32)
	confidence = 1.0 + np.log10(interactions["Revenue_Retailer"].to_numpy(dtype=float) + 1.0)
	confidence = np.clip(confidence, 1.0, None).astype(np.float32)
	user_item = csr_matrix((confidence, (u_idx, i_idx)), shape=(len(users), len(items)))

	model = AlternatingLeastSquares(
		factors=factors,
		regularization=regularization,
		iterations=iterations,
		random_state=random_state,
	)
	model.fit(user_item)

	user_factors = np.asarray(model.user_factors, dtype=np.float32)
	item_factors = np.asarray(model.item_factors, dtype=np.float32)
	return (
		model,
		user_item,
		user_to_idx,
		idx_to_user,
		item_to_idx,
		idx_to_item,
		users,
		items,
		user_factors,
		item_factors,
	)


def save_als_factors(
	output_dir: Path,
	users: np.ndarray,
	items: np.ndarray,
	user_factors: np.ndarray,
	item_factors: np.ndarray,
	user_to_idx: Dict,
	item_to_idx: Dict,
) -> tuple[Path, Path]:
	"""Persist ALS factors with deterministic mapping metadata."""
	output_dir.mkdir(parents=True, exist_ok=True)
	user_factor_path = output_dir / "user_factors.npz"
	item_factor_path = output_dir / "item_factors.npz"

	np.savez(
		user_factor_path,
		user_factors=user_factors,
		user_ids=np.asarray(users),
		user_id_to_index_json=np.array(json.dumps({str(key): int(value) for key, value in user_to_idx.items()})),
	)
	np.savez(
		item_factor_path,
		item_factors=item_factors,
		item_names=np.asarray(items, dtype=str),
		item_id_to_index_json=np.array(json.dumps({str(key): int(value) for key, value in item_to_idx.items()})),
	)
	return user_factor_path, item_factor_path


def build_mba_lookup(rules_long: pd.DataFrame) -> dict:
	"""Group MBA consequents by antecedent for fast lookup."""
	lookup = {}
	for antecedent, group in rules_long.groupby("antecedent_item"):
		ranked = group.sort_values(["relevance_mba", "lift"], ascending=[False, False])[["consequent_item", "relevance_mba"]]
		lookup[str(antecedent)] = [(str(item), float(score)) for item, score in ranked.itertuples(index=False, name=None)]
	return lookup


def build_seed_items_table(all_tx: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Build k recency seed items per user."""
	if all_tx.empty:
		empty_long = pd.DataFrame(columns=["household_key", "antecedent_item"])
		empty_summary = pd.DataFrame(columns=["household_key", "seed_items"])
		return empty_long, empty_summary

	tx = all_tx.copy()
	latest_day = tx.groupby("household_key", as_index=False)["DAY"].max().rename(columns={"DAY": "latest_day"})
	tx_latest_day = tx.merge(latest_day, on="household_key", how="inner")
	tx_latest_day = tx_latest_day[tx_latest_day["DAY"] == tx_latest_day["latest_day"]].copy()

	latest_basket = tx_latest_day.groupby("household_key", as_index=False)["BASKET_ID"].max().rename(columns={"BASKET_ID": "latest_basket_id"})
	tx_latest_basket = tx_latest_day.merge(latest_basket, on="household_key", how="inner")
	tx_latest_basket = tx_latest_basket[tx_latest_basket["BASKET_ID"] == tx_latest_basket["latest_basket_id"]].copy()

	latest_items = (
		tx_latest_basket.sort_values(["household_key", "DAY", "BASKET_ID"], ascending=[True, False, False])
		.drop_duplicates(["household_key", "COMMODITY_DESC"])
		.groupby("household_key")["COMMODITY_DESC"]
		.apply(list)
		.reset_index(name="latest_items")
	)
	history_items = (
		tx.sort_values(["household_key", "DAY", "BASKET_ID"], ascending=[True, False, False])
		.drop_duplicates(["household_key", "COMMODITY_DESC"])
		.groupby("household_key")["COMMODITY_DESC"]
		.apply(list)
		.reset_index(name="history_items")
	)

	seeds = history_items.merge(latest_items, on="household_key", how="left")

	def _compose_seed(row):
		base = row["latest_items"] if isinstance(row["latest_items"], list) else []
		history = row["history_items"] if isinstance(row["history_items"], list) else []
		seed_items = []
		seen = set()
		for item in base + history:
			if item not in seen:
				seen.add(item)
				seed_items.append(str(item))
			if len(seed_items) >= k:
				break
		return seed_items

	seeds["seed_items_list"] = seeds.apply(_compose_seed, axis=1)
	seeds = seeds[seeds["seed_items_list"].str.len() > 0].copy()
	seeds["seed_items"] = seeds["seed_items_list"].apply(lambda values: " | ".join(values))

	seed_items_long = seeds[["household_key", "seed_items_list"]].explode("seed_items_list")
	seed_items_long = seed_items_long.rename(columns={"seed_items_list": "antecedent_item"}).dropna().copy()
	seed_summary = seeds[["household_key", "seed_items"]].copy()
	return seed_items_long, seed_summary


def compute_als_scores_topk(
	users_arr: np.ndarray,
	items_arr: np.ndarray,
	user_factors_arr: np.ndarray,
	item_factors_arr: np.ndarray,
	user_indices: np.ndarray,
	top_k: int = 100,
	batch_size: int = 256,
) -> pd.DataFrame:
	"""Compute ALS dot-product scores in batches and keep top-k per user."""
	if len(user_indices) == 0 or len(items_arr) == 0:
		return pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "relevance_als"])

	items_obj = np.asarray(items_arr, dtype=object)
	top_k = int(min(top_k, item_factors_arr.shape[0]))
	if top_k <= 0:
		return pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "relevance_als"])

	frames = []
	for start in range(0, len(user_indices), batch_size):
		batch_idx = user_indices[start : start + batch_size]
		batch_users = users_arr[batch_idx]
		batch_user_factors = user_factors_arr[batch_idx]

		scores = batch_user_factors @ item_factors_arr.T
		top_idx = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
		top_scores = np.take_along_axis(scores, top_idx, axis=1)
		order = np.argsort(-top_scores, axis=1)
		top_idx = np.take_along_axis(top_idx, order, axis=1)
		top_scores = np.take_along_axis(top_scores, order, axis=1)
		top_scores_norm = rowwise_minmax(top_scores)

		frame = pd.DataFrame(
			{
				"household_key": np.repeat(batch_users, top_k),
				"COMMODITY_DESC": items_obj[top_idx.ravel()].astype(str),
				"relevance_als": top_scores_norm.ravel().astype(float),
			}
		)
		frames.append(frame)

	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "relevance_als"])


@dataclass
class CandidateArtifacts:
	candidate_set: pd.DataFrame
	filtered_items_log: pd.DataFrame
	als_scores: pd.DataFrame
	mba_candidates: pd.DataFrame
	als_candidates: pd.DataFrame
	recent_pairs: pd.DataFrame
	seed_summary: pd.DataFrame


def build_candidate_set(
	tx_all: pd.DataFrame,
	mba_rules_long: pd.DataFrame,
	users: np.ndarray,
	user_to_idx: Dict,
	user_factors: np.ndarray,
	item_factors: np.ndarray,
	items: np.ndarray,
	top_als: int = 50,
	top_mba: int = 50,
	seed_items_k: int = 3,
	candidate_users_limit: Optional[int] = None,
	als_score_batch_size: int = 256,
	recent_window_days: int = 28,
	snapshot_week: Optional[int] = None,
) -> CandidateArtifacts:
	"""Build the recall union, apply recency filtering, and return the supporting artifacts."""
	household_scope = users.copy()
	if candidate_users_limit is not None:
		household_scope = household_scope[:candidate_users_limit]

	selected_user_indices = np.array([user_to_idx[int(household)] for household in household_scope], dtype=int)
	als_scores = compute_als_scores_topk(
		users_arr=users,
		items_arr=items,
		user_factors_arr=user_factors,
		item_factors_arr=item_factors,
		user_indices=selected_user_indices,
		top_k=max(top_als, 1),
		batch_size=als_score_batch_size,
	)
	als_scores["relevance_als"] = als_scores["relevance_als"].clip(0, 1)

	seed_items_long, seed_summary = build_seed_items_table(tx_all[tx_all["household_key"].isin(household_scope)], k=seed_items_k)
	if seed_items_long.empty or mba_rules_long.empty:
		mba_candidates = pd.DataFrame(columns=["household_key", "COMMODITY_DESC", "relevance_mba"])
	else:
		mba_candidates = seed_items_long.merge(
			mba_rules_long[["antecedent_item", "consequent_item", "relevance_mba"]],
			on="antecedent_item",
			how="inner",
		)
		mba_candidates = (
			mba_candidates.groupby(["household_key", "consequent_item"], as_index=False)["relevance_mba"]
			.max()
			.rename(columns={"consequent_item": "COMMODITY_DESC"})
			.sort_values(["household_key", "relevance_mba"], ascending=[True, False])
			.groupby("household_key", as_index=False)
			.head(top_mba)
			.reset_index(drop=True)
		)

	als_candidates = (
		als_scores.sort_values(["household_key", "relevance_als"], ascending=[True, False])
		.groupby("household_key", as_index=False)
		.head(top_als)
		.reset_index(drop=True)
	)

	max_day = int(tx_all["DAY"].max())
	recent_cutoff_day = max_day - recent_window_days
	if snapshot_week is None:
		snapshot_week = int(tx_all["WEEK_NO"].max()) if "WEEK_NO" in tx_all.columns else int(np.ceil(max_day / 7))

	recent_pairs = (
		tx_all[(tx_all["DAY"] >= recent_cutoff_day) & (tx_all["household_key"].isin(household_scope))][["household_key", "COMMODITY_DESC"]]
		.drop_duplicates()
		.assign(was_recently_purchased=True)
	)

	unified_candidates = als_candidates.merge(mba_candidates, on=["household_key", "COMMODITY_DESC"], how="outer")
	unified_candidates["relevance_als"] = unified_candidates["relevance_als"].fillna(0.0)
	unified_candidates["relevance_mba"] = unified_candidates["relevance_mba"].fillna(0.0)
	unified_candidates["Relevance"] = unified_candidates[["relevance_als", "relevance_mba"]].max(axis=1)
	unified_candidates["source_detail"] = np.select(
		[
			(unified_candidates["relevance_als"] > 0) & (unified_candidates["relevance_mba"] > 0),
			(unified_candidates["relevance_als"] > 0) & (unified_candidates["relevance_mba"] == 0),
			(unified_candidates["relevance_als"] == 0) & (unified_candidates["relevance_mba"] > 0),
		],
		["BOTH", "ALS", "MBA"],
		default="UNKNOWN",
	)
	unified_candidates["source"] = unified_candidates["source_detail"].str.lower()
	unified_candidates["snapshot_week"] = snapshot_week
	unified_candidates = unified_candidates.merge(seed_summary, on="household_key", how="left")
	unified_candidates = unified_candidates.merge(recent_pairs, on=["household_key", "COMMODITY_DESC"], how="left")
	unified_candidates["was_recently_purchased"] = unified_candidates["was_recently_purchased"].fillna(False).astype(bool)

	filtered_items_log = (
		unified_candidates[unified_candidates["was_recently_purchased"]][["household_key", "COMMODITY_DESC"]]
		.drop_duplicates()
		.assign(filter_reason="recent_purchase", reference_week=snapshot_week)
	)
	candidate_set = unified_candidates.loc[~unified_candidates["was_recently_purchased"]].copy()
	for column in ["relevance_als", "relevance_mba", "Relevance"]:
		candidate_set[column] = candidate_set[column].clip(0, 1)

	candidate_set = candidate_set[
		[
			"household_key",
			"COMMODITY_DESC",
			"seed_items",
			"relevance_als",
			"relevance_mba",
			"Relevance",
			"source",
			"source_detail",
			"snapshot_week",
			"was_recently_purchased",
		]
	]

	return CandidateArtifacts(
		candidate_set=candidate_set,
		filtered_items_log=filtered_items_log,
		als_scores=als_scores,
		mba_candidates=mba_candidates,
		als_candidates=als_candidates,
		recent_pairs=recent_pairs,
		seed_summary=seed_summary,
	)

