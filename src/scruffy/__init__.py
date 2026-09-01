"""Scruffy: a small queue for GPUs held inside a Slurm allocation."""

from importlib.metadata import PackageNotFoundError, version

from .client import (
    cancel_job,
    drain_queue,
    explain,
    observe,
    publish_event,
    resume_queue,
    status,
    submit_job,
    submit_workflow,
    summary,
    validate_workflow,
    wait_for_event_ack,
    wait_for_job,
)
from .models import ResourceRequest
from .storage import ConflictError

try:
    __version__ = version("scruffy-gpu")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.1"

__all__ = [
    "ConflictError",
    "ResourceRequest",
    "__version__",
    "cancel_job",
    "drain_queue",
    "explain",
    "observe",
    "publish_event",
    "resume_queue",
    "status",
    "submit_job",
    "submit_workflow",
    "summary",
    "validate_workflow",
    "wait_for_event_ack",
    "wait_for_job",
]
