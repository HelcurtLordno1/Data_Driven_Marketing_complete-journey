"""Clustering and feature engineering pipeline for LRFMC segmentation."""


class RFMTransformer:
	"""Transforms transaction-level data into household-level LRFMC features."""

	def fit_transform(self, transactions_df):
		"""Generate model-ready features from cleaned transactions."""
		raise NotImplementedError

