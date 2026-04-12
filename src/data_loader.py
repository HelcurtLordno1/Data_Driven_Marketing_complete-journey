"""Data loading helpers for dunnhumby source files."""


class DunnhumbyLoader:
	"""Loads raw dunnhumby tables, with support for chunked reads."""

	def load_causal_data_in_chunks(self, file_path, chunk_size=100_000):
		"""Yield causal data chunks to avoid loading very large CSV files at once."""
		raise NotImplementedError

