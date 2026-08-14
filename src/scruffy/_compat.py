"""Compatibility helpers for every supported Python runtime."""

from datetime import timezone

# ``datetime.UTC`` was only added in Python 3.11. Compute nodes and remote
# submitters commonly provide Python 3.10, where ``timezone.utc`` is equivalent.
UTC = timezone.utc
