"""Financial utility functions for pricing and margin calculations."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ScalarLike = Union[int, float, np.ndarray, pd.Series]


def normalize_discount_values(values: ScalarLike) -> ScalarLike:
	"""Normalize discount inputs to non-negative numeric values."""
	if isinstance(values, pd.Series):
		return pd.to_numeric(values, errors="coerce").fillna(0).abs()
	return np.abs(np.asarray(values, dtype=float))


def calculate_true_price(
	sales_value: ScalarLike,
	retail_disc: ScalarLike = 0,
	coupon_match_disc: ScalarLike = 0,
) -> ScalarLike:
	"""Compute net price paid after retailer and coupon-match discounts."""
	if isinstance(sales_value, pd.Series):
		sales = pd.to_numeric(sales_value, errors="coerce").fillna(0)
		retail = pd.to_numeric(retail_disc, errors="coerce").fillna(0)
		coupon = pd.to_numeric(coupon_match_disc, errors="coerce").fillna(0)
		return (sales - retail - coupon).astype(float)

	sales = np.asarray(sales_value, dtype=float)
	retail = np.asarray(retail_disc, dtype=float)
	coupon = np.asarray(coupon_match_disc, dtype=float)
	return sales - retail - coupon


def calculate_margin(
	price_paid_customer: ScalarLike,
	cost: ScalarLike | None = None,
	margin_rate: ScalarLike | None = None,
) -> ScalarLike:
	"""Compute gross margin using either direct cost or a margin rate."""
	if cost is None and margin_rate is None:
		raise ValueError("Provide either cost or margin_rate to calculate_margin.")

	if isinstance(price_paid_customer, pd.Series):
		price = pd.to_numeric(price_paid_customer, errors="coerce").fillna(0)
		if cost is not None:
			return price - pd.to_numeric(cost, errors="coerce").fillna(0)
		rate = pd.to_numeric(margin_rate, errors="coerce").fillna(0)
		return price * rate

	price = np.asarray(price_paid_customer, dtype=float)
	if cost is not None:
		return price - np.asarray(cost, dtype=float)
	return price * np.asarray(margin_rate, dtype=float)

