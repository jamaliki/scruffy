"""Slurm-specific discovery and launch command construction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
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


def discover_slurm_inventory(
    *, gpus_per_node: int, cpus_per_node: int, memory_gb_per_node: int
) -> dict[str, NodeInventory]:
    """Build a homogeneous inventory from the current Slurm allocation."""

    node_expression = os.environ.get("SLURM_JOB_NODELIST")
    if not node_expression:
        raise ValueError("--inventory is required outside a Slurm allocation")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_expression],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise ValueError("Slurm returned no nodes for the current allocation")
    nodes = validate_inventory(
        tuple(
            NodeInventory(
                name=name,
                gpu_ids=tuple(range(gpus_per_node)),
                cpus=cpus_per_node,
                memory_gb=memory_gb_per_node,
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
    if not step_id.startswith(prefix) or not step_id.removeprefix(prefix).isdigit():
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
