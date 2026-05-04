"""Data loading helpers for the project source files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
	from .financial_utils import normalize_discount_values
except ImportError:
	from financial_utils import normalize_discount_values


def get_project_root(reference_path: Optional[Path] = None) -> Path:
	"""Resolve the project root from a file or current working directory."""
	reference = Path(reference_path or Path.cwd()).resolve()
	if reference.name == "notebooks":
		return reference.parent
	if reference.name in {"src", "tests", "data", "reports"}:
		return reference.parent
	return reference


def find_repo_root(start_path: Optional[Path] = None) -> Path:
	"""Locate the project root by searching upward for the expected folders."""
	search_root = (start_path or Path.cwd()).resolve()
	for candidate in [search_root, *search_root.parents]:
		if (candidate / "src").exists() and (candidate / "data").exists() and (candidate / "notebooks").exists():
			return candidate
	return search_root


def _is_git_lfs_pointer(file_path: Path) -> bool:
	if not file_path.exists() or file_path.stat().st_size == 0:
		return False
	try:
		with open(file_path, "r", encoding="utf-8") as handle:
			first_line = handle.readline().strip().lower()
		return first_line.startswith("version https://git-lfs.github.com/spec")
	except UnicodeDecodeError:
		return False


def _safe_read_parquet(file_path: Path) -> Optional[pd.DataFrame]:
	if not file_path.exists() or _is_git_lfs_pointer(file_path):
		return None
	try:
		return pd.read_parquet(file_path)
	except Exception:
		return None


def _normalize_master_schema(df: pd.DataFrame, product_lookup: Optional[pd.DataFrame] = None) -> pd.DataFrame:
	rename_map = {}
	if "HOUSEHOLD_KEY" in df.columns and "household_key" not in df.columns:
		rename_map["HOUSEHOLD_KEY"] = "household_key"
	if "commodity_desc" in df.columns and "COMMODITY_DESC" not in df.columns:
		rename_map["commodity_desc"] = "COMMODITY_DESC"
	if rename_map:
		df = df.rename(columns=rename_map)

	if "COMMODITY_DESC" not in df.columns and product_lookup is not None and "PRODUCT_ID" in df.columns:
		df = df.merge(product_lookup, on="PRODUCT_ID", how="left")

	if "Revenue_Retailer" not in df.columns:
		if "SALES_VALUE" in df.columns:
			df["Revenue_Retailer"] = pd.to_numeric(df["SALES_VALUE"], errors="coerce").fillna(0).clip(lower=0)
		else:
			df["Revenue_Retailer"] = 0.0

	if "Is_Promoted_Item" not in df.columns:
		discount_cols = [column for column in ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"] if column in df.columns]
		if discount_cols:
			df["Is_Promoted_Item"] = (df[discount_cols].fillna(0) != 0).any(axis=1)
		else:
			df["Is_Promoted_Item"] = False

	required_cols = ["household_key", "BASKET_ID", "DAY", "COMMODITY_DESC", "Revenue_Retailer", "Is_Promoted_Item"]
	missing = [column for column in required_cols if column not in df.columns]
	if missing:
		raise ValueError(f"Master transaction data missing required columns: {missing}")

	commodity = df["COMMODITY_DESC"].fillna("UNKNOWN_COMMODITY").astype(str).str.strip()
	df["COMMODITY_DESC"] = np.where(commodity == "", "UNKNOWN_COMMODITY", commodity)
	df["Revenue_Retailer"] = pd.to_numeric(df["Revenue_Retailer"], errors="coerce").fillna(0).clip(lower=0)
	return df


def load_or_build_master_transactions(
	raw_dir: Path,
	processed_dir: Path,
	sample_nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load processed module-1 artifacts or reconstruct them from raw CSV files."""
	all_path = processed_dir / "master_transactions_all.parquet"
	organic_path = processed_dir / "master_transactions_organic_only.parquet"
	product_lookup = pd.read_csv(raw_dir / "product.csv", usecols=["PRODUCT_ID", "COMMODITY_DESC"])

	all_df = _safe_read_parquet(all_path)
	organic_df = _safe_read_parquet(organic_path)
	if all_df is not None and organic_df is not None and not all_df.empty and not organic_df.empty:
		return _normalize_master_schema(all_df, product_lookup), _normalize_master_schema(organic_df, product_lookup)

	txn_cols = [
		"household_key",
		"BASKET_ID",
		"DAY",
		"PRODUCT_ID",
		"QUANTITY",
		"SALES_VALUE",
		"RETAIL_DISC",
		"COUPON_DISC",
		"COUPON_MATCH_DISC",
		"WEEK_NO",
	]
	txn_df = pd.read_csv(raw_dir / "transaction_data.csv", usecols=txn_cols, nrows=sample_nrows)
	merged = txn_df.merge(product_lookup, on="PRODUCT_ID", how="left")
	merged["RETAIL_DISC"] = normalize_discount_values(merged["RETAIL_DISC"])
	merged["COUPON_DISC"] = normalize_discount_values(merged["COUPON_DISC"])
	merged["COUPON_MATCH_DISC"] = normalize_discount_values(merged["COUPON_MATCH_DISC"])
	merged["Is_Promoted_Item"] = (merged[["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]] != 0).any(axis=1)
	merged["Revenue_Retailer"] = merged["SALES_VALUE"].clip(lower=0)

	basket_is_promoted = merged.groupby("BASKET_ID")["Is_Promoted_Item"].transform("any")
	organic_df = merged.loc[~basket_is_promoted].copy()

	all_df = _normalize_master_schema(merged, product_lookup)
	organic_df = _normalize_master_schema(organic_df, product_lookup)

	if sample_nrows is None:
		all_df.to_parquet(all_path, index=False)
		organic_df.to_parquet(organic_path, index=False)

	return all_df, organic_df

