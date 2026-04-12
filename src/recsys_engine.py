"""Recommendation engine components (MBA + ALS fallback logic)."""


class HybridRecommender:
	"""Combines association rules and ALS scoring for product recommendations."""

	def recommend(self, household_id, top_k=10):
		"""Return top-k candidate products for a household."""
		raise NotImplementedError

