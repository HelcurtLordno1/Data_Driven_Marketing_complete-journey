from __future__ import annotations

import base64
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.module4_validation import build_temporal_holdout


PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ChiSquareResult:
	statistic: float
	p_value: float


@dataclass(frozen=True)
class BasketDiversityStats:
	chimera_mean: float
	baseline_mean: float
	chimera_median: float
	baseline_median: float
	mannwhitney_u: float
	p_value: float


@dataclass(frozen=True)
class ExpansionRateStats:
	chimera_rate: float
	baseline_rate: float
	absolute_lift: float
	relative_lift_pct: float
	chi_square: ChiSquareResult


@dataclass(frozen=True)
class TradeoffStats:
	chimera_hit_rate_mean: float
	baseline_hit_rate_mean: float
	chimera_discovery_rate_mean: float
	baseline_discovery_rate_mean: float


@dataclass(frozen=True)
class MarginShiftArmStats:
	train_mean: float
	test_mean: float
	shift_mean: float
	shift_std: float
	n_households: int


@dataclass(frozen=True)
class MarginShiftComparison:
	control: MarginShiftArmStats
	treatment: MarginShiftArmStats
	welch_t: float
	p_value: float
	cohen_d: float


@dataclass(frozen=True)
class GlobalImportance:
	components: list[str]
	importance_mean: list[float]
	importance_std: list[float]


def _decode_plotly_typed_array(obj: Any) -> np.ndarray:
	"""Decode Plotly typed array payloads like {"dtype":"f8","bdata":"..."}."""
	if isinstance(obj, dict) and "dtype" in obj and "bdata" in obj:
		dtype = str(obj["dtype"])
		bdata = base64.b64decode(obj["bdata"])
		# Plotly uses numpy dtype strings (f8, i4, etc). Assume little-endian.
		dt = np.dtype("<" + dtype)
		return np.frombuffer(bdata, dtype=dt)
	return np.asarray(obj)


def extract_global_importance_from_html(html_path: Path) -> GlobalImportance:
	html_text = html_path.read_text(encoding="utf-8")

	# Find the exact trace payload by anchoring on the known component labels.
	x_match = re.search(r'"x"\s*:\s*\[\s*"Relevance"\s*,\s*"Uplift"\s*,\s*"Context"\s*,\s*"Margin"\s*\]', html_text)
	if not x_match:
		raise ValueError("Could not locate component labels in Plotly HTML.")
	components = ["Relevance", "Uplift", "Context", "Margin"]
	window = html_text[x_match.start() : x_match.start() + 5000]

	# y is stored as a typed array object.
	y_match = re.search(r'"y"\s*:\s*\{\s*"dtype"\s*:\s*"(?P<dtype>[^"]+)"\s*,\s*"bdata"\s*:\s*"(?P<bdata>[^"]+)"\s*\}', window)
	if not y_match:
		raise ValueError("Could not locate y typed-array payload in Plotly HTML.")

	def unescape_bdata(value: str) -> str:
		# Plotly sometimes encodes base64 characters using unicode escapes (\u002f, \u002b, ...).
		return value.encode("utf-8").decode("unicode_escape")

	y_dtype = y_match.group("dtype")
	y_bdata = unescape_bdata(y_match.group("bdata"))
	y_values = _decode_plotly_typed_array({"dtype": y_dtype, "bdata": y_bdata}).astype(float).tolist()

	err_match = re.search(
		r'"error_y"\s*:\s*\{\s*"array"\s*:\s*\{\s*"dtype"\s*:\s*"(?P<dtype>[^"]+)"\s*,\s*"bdata"\s*:\s*"(?P<bdata>[^"]+)"',
		window,
	)
	if err_match:
		err_dtype = err_match.group("dtype")
		err_bdata = unescape_bdata(err_match.group("bdata"))
		err_values = _decode_plotly_typed_array({"dtype": err_dtype, "bdata": err_bdata}).astype(float).tolist()
	else:
		err_values = [0.0 for _ in y_values]

	return GlobalImportance(components=components, importance_mean=y_values, importance_std=err_values)


def compute_expansion_rate_stats(detail_path: Path) -> ExpansionRateStats:
	detail = pd.read_csv(detail_path)
	if "variant" not in detail.columns or "expanded_category" not in detail.columns:
		raise ValueError("module6_category_expansion_detail.csv missing required columns")

	# Normalize variant labels
	detail["variant"] = detail["variant"].astype(str)
	variant_map = {
		"Chimera": "Chimera",
		"Popularity Baseline": "Baseline",
		"Baseline": "Baseline",
	}
	detail["variant_norm"] = detail["variant"].map(variant_map).fillna(detail["variant"])

	group = detail.groupby("variant_norm")["expanded_category"].mean()
	chimera = float(group.get("Chimera", np.nan))
	baseline = float(group.get("Baseline", np.nan))

	# Contingency table
	counts = detail.groupby(["variant_norm", "expanded_category"]).size().unstack(fill_value=0)
	obs = np.array(
		[
			[counts.loc["Chimera"].get(False, 0), counts.loc["Chimera"].get(True, 0)],
			[counts.loc["Baseline"].get(False, 0), counts.loc["Baseline"].get(True, 0)],
		]
	)
	chi2, p, *_ = stats.chi2_contingency(obs)

	absolute_lift = chimera - baseline
	relative_lift_pct = (absolute_lift / baseline) * 100.0 if baseline not in (0.0, np.nan) else 0.0
	return ExpansionRateStats(
		chimera_rate=chimera,
		baseline_rate=baseline,
		absolute_lift=absolute_lift,
		relative_lift_pct=relative_lift_pct,
		chi_square=ChiSquareResult(statistic=float(chi2), p_value=float(p)),
	)


def compute_basket_diversity_stats(diversity_path: Path) -> BasketDiversityStats:
	df = pd.read_csv(diversity_path)
	df["treatment"] = df["treatment"].astype(str)
	chimera = pd.to_numeric(df.loc[df["treatment"] == "Chimera", "avg_basket_diversity"], errors="coerce").dropna()
	baseline = pd.to_numeric(df.loc[df["treatment"].str.lower().isin(["baseline"]), "avg_basket_diversity"], errors="coerce").dropna()
	if chimera.empty or baseline.empty:
		raise ValueError("module6_basket_diversity.csv missing Chimera/Baseline rows")

	u_stat, p_value = stats.mannwhitneyu(chimera, baseline, alternative="two-sided")
	return BasketDiversityStats(
		chimera_mean=float(chimera.mean()),
		baseline_mean=float(baseline.mean()),
		chimera_median=float(chimera.median()),
		baseline_median=float(baseline.median()),
		mannwhitney_u=float(u_stat),
		p_value=float(p_value),
	)


def compute_tradeoff_stats(tradeoff_path: Path) -> TradeoffStats:
	df = pd.read_csv(tradeoff_path)
	variant = df.get("variant")
	if variant is None:
		raise ValueError("module6_hit_rate_discovery_tradeoff.csv missing variant")

	df["variant"] = df["variant"].astype(str)
	variant_norm = df["variant"].replace({"Popularity Baseline": "Baseline"})
	df["variant_norm"] = variant_norm

	group = df.groupby("variant_norm").agg(hit_rate_mean=("hit_rate", "mean"), discovery_rate_mean=("discovery_rate", "mean"))
	return TradeoffStats(
		chimera_hit_rate_mean=float(group.loc["Chimera", "hit_rate_mean"]),
		baseline_hit_rate_mean=float(group.loc["Baseline", "hit_rate_mean"]),
		chimera_discovery_rate_mean=float(group.loc["Chimera", "discovery_rate_mean"]),
		baseline_discovery_rate_mean=float(group.loc["Baseline", "discovery_rate_mean"]),
	)


def compute_margin_shift_by_arm(master_transactions_path: Path, assignment_path: Path) -> MarginShiftComparison:
	mt = pd.read_parquet(master_transactions_path)
	assignment = pd.read_csv(assignment_path)
	if "household_key" not in assignment.columns:
		raise ValueError("ab_assignment_mapping.csv missing household_key")

	arm_col = None
	for candidate in ["treatment", "arm", "variant", "group", "treatment_arm"]:
		if candidate in assignment.columns:
			arm_col = candidate
			break
	if arm_col is None:
		raise ValueError("ab_assignment_mapping.csv missing an arm column")

	assignment = assignment[["household_key", arm_col]].copy()
	assignment[arm_col] = assignment[arm_col].astype(str)
	# Normalize labels
	assignment["arm_norm"] = assignment[arm_col].replace(
		{
			"Control": "Control",
			"Popularity": "Control",
			"Popularity Baseline": "Control",
			"Treatment": "Treatment",
			"Chimera": "Treatment",
		}
	)

	holdout = build_temporal_holdout(mt)
	# Restrict to households used in the A/B simulation (present in assignment mapping)
	hh = set(pd.to_numeric(assignment["household_key"], errors="coerce").dropna().astype(int))
	train = holdout.train_history[holdout.train_history["household_key"].isin(hh)].copy()
	test = holdout.test_history[holdout.test_history["household_key"].isin(hh)].copy()

	# Compute per-household average Normalized_Margin (observed purchases)
	train_m = train.groupby("household_key")["Normalized_Margin"].mean().rename("train_mean")
	test_m = test.groupby("household_key")["Normalized_Margin"].mean().rename("test_mean")
	merged = pd.concat([train_m, test_m], axis=1).dropna().reset_index()
	merged["shift"] = merged["test_mean"] - merged["train_mean"]

	merged = merged.merge(assignment[["household_key", "arm_norm"]], on="household_key", how="left")
	merged = merged.dropna(subset=["arm_norm"])

	def arm_stats(frame: pd.DataFrame) -> MarginShiftArmStats:
		return MarginShiftArmStats(
			train_mean=float(frame["train_mean"].mean()),
			test_mean=float(frame["test_mean"].mean()),
			shift_mean=float(frame["shift"].mean()),
			shift_std=float(frame["shift"].std(ddof=1)),
			n_households=int(len(frame)),
		)

	control = merged.loc[merged["arm_norm"] == "Control"].copy()
	treatment = merged.loc[merged["arm_norm"] == "Treatment"].copy()

	t_stat, p_value = stats.ttest_ind(treatment["shift"], control["shift"], equal_var=False)

	# Cohen's d for shift difference
	def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
		na, nb = len(a), len(b)
		sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
		return math.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))

	d = (treatment["shift"].mean() - control["shift"].mean()) / pooled_sd(treatment["shift"].to_numpy(), control["shift"].to_numpy())

	return MarginShiftComparison(
		control=arm_stats(control),
		treatment=arm_stats(treatment),
		welch_t=float(t_stat),
		p_value=float(p_value),
		cohen_d=float(d),
	)


def compute_hit_rate_by_archetype(tradeoff_path: Path, archetype_assignments_path: Path) -> pd.DataFrame:
	tradeoff = pd.read_csv(tradeoff_path)
	arche = pd.read_csv(archetype_assignments_path, usecols=["household_key", "archetype"])
	tradeoff["variant"] = tradeoff["variant"].astype(str).replace({"Popularity Baseline": "Baseline"})
	merged = tradeoff.merge(arche, on="household_key", how="left")
	summary = (
		merged.groupby(["archetype", "variant"], dropna=False)
		.agg(hit_rate_mean=("hit_rate", "mean"), discovery_rate_mean=("discovery_rate", "mean"), households=("household_key", "nunique"))
		.reset_index()
	)
	return summary


def main() -> None:
	results: Dict[str, Any] = {}

	results["module6_expansion"] = asdict(compute_expansion_rate_stats(PROCESSED_DIR / "module6_category_expansion_detail.csv"))
	results["module6_basket_diversity"] = asdict(compute_basket_diversity_stats(PROCESSED_DIR / "module6_basket_diversity.csv"))
	results["module6_tradeoff"] = asdict(compute_tradeoff_stats(PROCESSED_DIR / "module6_hit_rate_discovery_tradeoff.csv"))
	results["module6_margin_shift_by_arm"] = asdict(
		compute_margin_shift_by_arm(
			master_transactions_path=PROCESSED_DIR / "master_transactions.parquet",
			assignment_path=PROCESSED_DIR / "ab_assignment_mapping.csv",
		)
	)

	results["module7_global_importance"] = asdict(extract_global_importance_from_html(REPORTS_DIR / "global_importance_bar.html"))

	# Archetype-level tradeoff table for reporting
	archetype_tradeoff = compute_hit_rate_by_archetype(
		tradeoff_path=PROCESSED_DIR / "module6_hit_rate_discovery_tradeoff.csv",
		archetype_assignments_path=PROCESSED_DIR / "module8_archetype_assignments.csv",
	)
	results["module6_tradeoff_by_archetype_head"] = archetype_tradeoff.head(20).to_dict(orient="records")

	out_path = PROCESSED_DIR / "report_4_5_extended_metrics.json"
	out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

	print("Wrote:", out_path)
	print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
	main()
