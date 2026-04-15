"""Cold-start recommendation rules and demographic prior helpers."""


class ColdStartRecommender:
	"""Provides default recommendations for users without purchase history."""

	def recommend_for_new_user(self, demographic_profile, top_k=10):
		"""Return top-k prior-based fallback items for a new user."""
		raise NotImplementedError

