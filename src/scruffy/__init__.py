"""Scruffy: a small queue for GPUs held inside a Slurm allocation."""

from importlib.metadata import PackageNotFoundError, version

from .client import (
    cancel_job,
    drain_queue,
    explain,
    observe,
    publish_event,
    status,
    submit_job,
    summary,
    wait_for_job,
)
from .models import NodeInventory, ResourceRequest

try:
    __version__ = version("scruffy-gpu")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"

__all__ = [
    "NodeInventory",
    "ResourceRequest",
    "__version__",
    "cancel_job",
    "drain_queue",
    "explain",
    "observe",
    "publish_event",
    "status",
    "submit_job",
    "summary",
    "wait_for_job",
]
