"""Slurm-specific discovery and launch command construction."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import NodeInventory, validate_inventory


@dataclass(frozen=True, slots=True)
class SlurmStep:
    step_id: str
    name: str
    nodes: str


def new_step_name() -> str:
    """Create a persisted launch token which cannot collide with user steps."""

    return f"scruffy-{uuid.uuid4().hex}"


def load_inventory(source: Path) -> dict[str, NodeInventory]:
    """Load the explicit resource pool that Scruffy is allowed to manage."""

    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("inventory must be a JSON object")
    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, dict) or not raw_nodes:
        raise ValueError("inventory must contain a non-empty 'nodes' mapping")
    nodes = []
    for name, values in raw_nodes.items():
        if not isinstance(values, dict) or "name" in values:
            raise ValueError(f"inventory entry {name!r} must not contain 'name'")
        nodes.append(NodeInventory.from_dict({"name": name, **values}))
    return {node.name: node for node in validate_inventory(nodes)}


def _allocation_job(slurm_job_id: str) -> Mapping[str, Any]:
    result = subprocess.run(
        ["scontrol", "--json", "show", "job", slurm_job_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise TypeError("scontrol returned an invalid allocation document")
    jobs = document.get("jobs")
    if document.get("errors") or not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("scontrol returned an invalid allocation document")
    job = jobs[0]
    if not isinstance(job, dict):
        raise TypeError("scontrol returned an invalid allocation job")
    return job


def _tres_values(job: Mapping[str, Any]) -> dict[str, str]:
    encoded = job.get("tres_alloc_str")
    if not isinstance(encoded, str):
        raise TypeError("Slurm allocation has no allocated TRES")
    values: dict[str, str] = {}
    for item in encoded.split(","):
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in values:
            raise ValueError("Slurm allocation has invalid allocated TRES")
        values[key] = value
    return values


def _positive_count(values: Mapping[str, str], key: str) -> int:
    value = values.get(key)
    if value is None or not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"Slurm allocation has no positive {key} count")
    return int(value)


def _gpu_count(values: Mapping[str, str]) -> int:
    if "gres/gpu" in values:
        return _positive_count(values, "gres/gpu")
    typed = [key for key in values if key.startswith("gres/gpu:")]
    if not typed:
        raise ValueError("Slurm allocation has no GPUs")
    return sum(_positive_count(values, key) for key in typed)


def _memory_mib(values: Mapping[str, str]) -> int:
    value = values.get("mem", "")
    match = re.fullmatch(r"([0-9]+)([KMGT])", value, re.IGNORECASE)
    if match is None:
        raise ValueError("Slurm allocation has invalid memory TRES")
    units = {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}
    kib = int(match.group(1)) * units[match.group(2).upper()]
    mib = kib // 1024
    if mib <= 0:
        raise ValueError("Slurm allocation has no usable memory")
    return mib


def _per_node(total: int, nodes: int, label: str) -> int:
    value, remainder = divmod(total, nodes)
    if value <= 0 or remainder:
        raise ValueError(
            f"Slurm allocation has heterogeneous {label}; use --inventory"
        )
    return value


def _managed_capacity(discovered: int, cap: int | None, option: str) -> int:
    if cap is None:
        return discovered
    if cap <= 0:
        raise ValueError(f"{option} must be a positive integer")
    if cap > discovered:
        raise ValueError(f"{option} exceeds the Slurm allocation")
    return cap


def _allocation_hostnames(job: Mapping[str, Any]) -> list[str]:
    node_expression = job.get("nodes")
    if not isinstance(node_expression, str) or not node_expression:
        raise ValueError("Slurm allocation has no assigned nodes")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise ValueError("Slurm returned no nodes for the current allocation")
    return names


def _allocation_capacity(job: Mapping[str, Any], nodes: int) -> tuple[int, int, int]:
    values = _tres_values(job)
    gpus = _per_node(_gpu_count(values), nodes, "GPU counts")
    cpus = _per_node(_positive_count(values, "cpu"), nodes, "CPU counts")
    memory_gb = _per_node(_memory_mib(values), nodes, "memory") // 1024
    return gpus, cpus, memory_gb


def discover_slurm_inventory(
    *,
    slurm_job_id: str | None = None,
    gpus_per_node: int | None = None,
    cpus_per_node: int | None = None,
    memory_gb_per_node: int | None = None,
) -> dict[str, NodeInventory]:
    """Build a homogeneous inventory from resources granted by Slurm."""

    job_id = slurm_job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("--inventory is required outside a Slurm allocation")
    job = _allocation_job(job_id)
    names = _allocation_hostnames(job)
    discovered_gpus, discovered_cpus, discovered_memory = _allocation_capacity(
        job, len(names)
    )
    gpus = _managed_capacity(discovered_gpus, gpus_per_node, "--gpus-per-node")
    cpus = _managed_capacity(discovered_cpus, cpus_per_node, "--cpus-per-node")
    memory = _managed_capacity(
        discovered_memory, memory_gb_per_node, "--memory-gb-per-node"
    )
    nodes = validate_inventory(
        tuple(
            NodeInventory(
                name=name,
                gpu_ids=tuple(range(gpus)),
                cpus=cpus,
                memory_gb=memory,
            )
            for name in names
        )
    )
    return {node.name: node for node in nodes}


def build_srun_argv(
    *,
    slurm_job_id: str,
    name: str,
    assignment_file: Path,
    node_names: list[str],
    cpus_per_node: int,
    memory_gb_per_node: int,
    wait_seconds: int = 0,
) -> list[str]:
    """Build one argv-only Slurm step for a rectangular multi-node job.

    Scruffy deliberately uses ``--overlap`` because the outer allocation owns
    the full GPU pool. Its own per-node ledger selects disjoint GPU IDs, and the
    worker narrows ``CUDA_VISIBLE_DEVICES`` before executing the user command.
    """

    if not slurm_job_id:
        raise ValueError("a Slurm job ID is required for the Slurm launcher")
    nodes = len(node_names)
    return [
        "srun",
        f"--jobid={slurm_job_id}",
        f"--job-name={name}",
        "--overlap",
        "--exact",
        f"--nodes={nodes}",
        f"--nodelist={','.join(node_names)}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus_per_node}",
        f"--mem={memory_gb_per_node}G",
        "--kill-on-bad-exit=1",
        f"--wait={wait_seconds}",
        "--wait-for-children",
        "--label",
        sys.executable,
        "-m",
        "scruffy.worker",
        str(assignment_file),
    ]


def build_srun_environment() -> dict[str, str]:
    """Return the controller environment without a stale worker node override."""

    environment = os.environ.copy()
    environment.pop("SCRUFFY_NODE", None)
    return environment


def live_steps(slurm_job_id: str) -> tuple[SlurmStep, ...]:
    """Return an error-checked live snapshot for one outer allocation.

    Tokyo's ``squeue --steps`` omits regular steps, so release reconciliation
    uses Slurm's structured ``scontrol`` output instead. Any malformed or
    error-bearing response raises: uncertainty must retain GPU reservations.
    """

    result = subprocess.run(
        ["scontrol", "--json", "show", "step", slurm_job_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    document = json.loads(result.stdout)
    if not isinstance(document, dict) or not isinstance(document.get("steps"), list):
        raise RuntimeError("scontrol returned an invalid step document")
    if document.get("errors"):
        raise RuntimeError(f"scontrol reported errors: {document['errors']!r}")
    steps: list[SlurmStep] = []
    for item in document["steps"]:
        if not isinstance(item, dict):
            raise RuntimeError("scontrol returned an invalid step record")
        steps.append(
            SlurmStep(
                step_id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                nodes=str(item.get("nodes", "")),
            )
        )
    return tuple(steps)


def cancel_step(slurm_job_id: str, step_id: str) -> None:
    """Cancel exactly one numeric step, never its outer allocation."""

    prefix = f"{slurm_job_id}."
    suffix = step_id.removeprefix(prefix)
    if (
        not step_id.startswith(prefix)
        or not suffix.isascii()
        or not suffix.isdecimal()
    ):
        raise ValueError(f"refusing unsafe Slurm step ID {step_id!r}")

    subprocess.run(
        ["scancel", "--ctld", "--quiet", step_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def build_local_argv(
    assignment_file: Path, node_name: str
) -> tuple[list[str], dict[str, str]]:
    """Build a single-node local worker command used by tests and development."""

    environment = os.environ.copy()
    environment["SCRUFFY_NODE"] = node_name
    return (
        [sys.executable, "-m", "scruffy.worker", str(assignment_file)],
        environment,
    )


def allocation_metadata(allocation_id: str, launcher: str) -> dict[str, Any]:
    return {
        "id": allocation_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher": launcher,
        "deadline": os.environ.get("SLURM_JOB_END_TIME"),
    }
