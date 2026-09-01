"""Slurm-specific discovery and launch command construction."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from ._compat import UTC
from .models import NodeInventory, validate_inventory

# SchedMD documents these as outer allocation, array, cluster, or submission
# metadata. ``build_srun_argv`` pins the few that are also accepted as options
# (job ID, job name, and node count), so their inherited values cannot steer the
# child step.
_ALLOCATION_ENVIRONMENT_KEYS = frozenset(
    {
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_STEP",
        "SLURM_CLUSTER_NAME",
        "SLURM_CONF",
        "SLURM_JOBID",  # Legacy spelling of SLURM_JOB_ID.
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_JOB_DEPENDENCY",
        "SLURM_JOB_END_TIME",
        "SLURM_JOB_GPUS",
        "SLURM_JOB_ID",
        "SLURM_JOB_LICENSES",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NODES",
        "SLURM_JOB_NUM_NODES",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_RESERVATION",
        "SLURM_JOB_SEGMENT_SIZE",
        "SLURM_JOB_START_TIME",
        "SLURM_SUBMIT_DIR",
        "SLURM_SUBMIT_HOST",
    }
)
_SLURM_ENVIRONMENT_PREFIXES = ("SLURM_", "SLURMD_", "SRUN_")


@dataclass(frozen=True, slots=True)
class SlurmStep:
    step_id: str
    name: str
    nodes: str


@dataclass(frozen=True, slots=True)
class SlurmStepResult:
    state: str
    returncode: int


@dataclass(frozen=True, slots=True)
class AllocationIncarnation:
    """Immutable identity of one execution of a Slurm allocation job."""

    slurm_job_id: str
    restart_count: int
    inventory: tuple[NodeInventory, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.slurm_job_id, str) or not self.slurm_job_id:
            raise ValueError("allocation incarnation has no Slurm job ID")
        if type(self.restart_count) is not int or self.restart_count < 0:
            raise ValueError("allocation restart count must be a non-negative integer")
        validated = validate_inventory(self.inventory)
        object.__setattr__(
            self, "inventory", tuple(sorted(validated, key=lambda item: item.name))
        )

    @property
    def inventory_sha256(self) -> str:
        return _canonical_sha256([item.to_dict() for item in self.inventory])

    @property
    def fingerprint_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": 1,
                "slurm_job_id": self.slurm_job_id,
                "restart_count": self.restart_count,
                "inventory_sha256": self.inventory_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "slurm_job_id": self.slurm_job_id,
            "restart_count": self.restart_count,
            "inventory": [item.to_dict() for item in self.inventory],
            "inventory_sha256": self.inventory_sha256,
            "fingerprint_sha256": self.fingerprint_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> AllocationIncarnation:
        if not isinstance(value, Mapping):
            raise TypeError("allocation incarnation is not an object")
        expected = {
            "schema",
            "slurm_job_id",
            "restart_count",
            "inventory",
            "inventory_sha256",
            "fingerprint_sha256",
        }
        if (
            set(value) != expected
            or type(value.get("schema")) is not int
            or value.get("schema") != 1
        ):
            raise ValueError("allocation incarnation has invalid keys or schema")
        raw_inventory = value["inventory"]
        if isinstance(raw_inventory, (str, bytes)) or not isinstance(
            raw_inventory, list
        ):
            raise TypeError("allocation incarnation inventory is not a list")
        result = cls(
            slurm_job_id=value["slurm_job_id"],  # type: ignore[arg-type]
            restart_count=value["restart_count"],  # type: ignore[arg-type]
            inventory=tuple(NodeInventory.from_dict(item) for item in raw_inventory),
        )
        if (
            value["inventory_sha256"] != result.inventory_sha256
            or value["fingerprint_sha256"] != result.fingerprint_sha256
        ):
            raise ValueError("allocation incarnation digest differs")
        return result


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return sha256(payload).hexdigest()


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


def _restart_count(job: Mapping[str, Any]) -> int:
    value = job.get("restart_cnt")
    if type(value) is not int or value < 0:
        raise ValueError("Slurm allocation has no valid restart count")
    return value


def _inventories_from_job(
    job: Mapping[str, Any],
    *,
    gpus_per_node: int | None,
    cpus_per_node: int | None,
    memory_gb_per_node: int | None,
) -> tuple[tuple[NodeInventory, ...], tuple[NodeInventory, ...]]:
    names = _allocation_hostnames(job)
    discovered_gpus, discovered_cpus, discovered_memory = _allocation_capacity(
        job, len(names)
    )
    authoritative = validate_inventory(
        tuple(
            NodeInventory(
                name=name,
                gpu_ids=tuple(range(discovered_gpus)),
                cpus=discovered_cpus,
                memory_gb=discovered_memory,
            )
            for name in names
        )
    )
    gpus = _managed_capacity(discovered_gpus, gpus_per_node, "--gpus-per-node")
    cpus = _managed_capacity(discovered_cpus, cpus_per_node, "--cpus-per-node")
    memory = _managed_capacity(
        discovered_memory, memory_gb_per_node, "--memory-gb-per-node"
    )
    managed = validate_inventory(
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
    return authoritative, managed


def discover_slurm_allocation(
    *,
    slurm_job_id: str | None = None,
    gpus_per_node: int | None = None,
    cpus_per_node: int | None = None,
    memory_gb_per_node: int | None = None,
) -> tuple[dict[str, NodeInventory], AllocationIncarnation]:
    """Discover managed capacity and its authoritative Slurm incarnation."""

    job_id = slurm_job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("--inventory is required outside a Slurm allocation")
    job = _allocation_job(job_id)
    returned_job_id = job.get("job_id")
    if type(returned_job_id) not in {int, str} or str(returned_job_id) != job_id:
        raise ValueError("Slurm returned a different allocation job")
    authoritative, managed = _inventories_from_job(
        job,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        memory_gb_per_node=memory_gb_per_node,
    )
    incarnation = AllocationIncarnation(
        slurm_job_id=job_id,
        restart_count=_restart_count(job),
        inventory=authoritative,
    )
    return {node.name: node for node in managed}, incarnation


def discover_slurm_incarnation(slurm_job_id: str) -> AllocationIncarnation:
    """Read an allocation identity even when managed inventory is explicit."""

    _managed, incarnation = discover_slurm_allocation(slurm_job_id=slurm_job_id)
    return incarnation


def discover_slurm_inventory(
    *,
    slurm_job_id: str | None = None,
    gpus_per_node: int | None = None,
    cpus_per_node: int | None = None,
    memory_gb_per_node: int | None = None,
) -> dict[str, NodeInventory]:
    """Build a homogeneous inventory from resources granted by Slurm."""

    inventory, _incarnation = discover_slurm_allocation(
        slurm_job_id=slurm_job_id,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        memory_gb_per_node=memory_gb_per_node,
    )
    return inventory


def build_srun_argv(
    *,
    slurm_job_id: str,
    name: str,
    assignment_file: Path,
    stdout_file: Path,
    stderr_file: Path,
    node_names: list[str],
    gpus_per_node: int,
    cpus_per_node: int,
    memory_gb_per_node: int,
    wait_seconds: int = 0,
    gpu_ids_per_node: Sequence[Sequence[int]] | None = None,
) -> list[str]:
    """Build one argv-only Slurm step for a rectangular multi-node job.

    Slurm owns the physical resources for every step. Scruffy chooses nodes and
    admission slots, while each worker task requests its exact GPUs, CPUs, and
    memory. Task-scoped GPU requests make Slurm bind a device set to every
    worker and populate ``CUDA_VISIBLE_DEVICES``. When ``gpu_ids_per_node`` is
    supplied, its common slot set is converted to an explicit Slurm GRES mask;
    the worker validates the resulting physical mapping before exec.
    ``--exact`` prevents a partial step from inheriting the remaining
    outer-allocation resources.
    """

    if not slurm_job_id:
        raise ValueError("a Slurm job ID is required for the Slurm launcher")
    if type(gpus_per_node) is not int or gpus_per_node < 0:
        raise ValueError("gpus_per_node must be a non-negative integer")
    nodes = len(node_names)
    if gpu_ids_per_node is not None:
        per_node = tuple(tuple(ids) for ids in gpu_ids_per_node)
        if len(per_node) != nodes:
            raise ValueError("GPU binding must provide one slot set per node")
        if any(
            any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in ids)
            or len(set(ids)) != len(ids)
            or len(ids) != gpus_per_node
            for ids in per_node
        ):
            raise ValueError("GPU binding does not match the requested GPU count")
        if per_node and any(ids != per_node[0] for ids in per_node[1:]):
            raise ValueError("exact GPU binding requires one common slot set per node")
    argv = [
        "srun",
        f"--jobid={slurm_job_id}",
        f"--job-name={name}",
        "--exact",
        f"--nodes={nodes}",
        f"--nodelist={','.join(node_names)}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
    ]
    if gpus_per_node == 0:
        # Slurm steps inherit the outer allocation's GRES unless they opt out.
        # CPU-only work must not receive or hide an allocation GPU.
        argv.append("--gres=none")
    else:
        # There is exactly one worker task per selected node. Request GPUs for
        # that task rather than merely reserving node-level GRES: Slurm then
        # owns both exclusivity and the task-visible CUDA device mapping.
        argv.append(f"--gpus-per-task={gpus_per_node}")
        if gpu_ids_per_node is not None:
            mask = sum(1 << gpu_id for gpu_id in gpu_ids_per_node[0])
            argv.append(f"--tres-bind=gres/gpu:mask:0x{mask:x}")
    argv.extend(
        [
            f"--cpus-per-task={cpus_per_node}",
            "--cpu-bind=none",
            f"--mem={memory_gb_per_node}G",
            "--kill-on-bad-exit=1",
            f"--wait={wait_seconds}",
            "--wait-for-children",
            "--export=ALL",
            "--label",
            f"--output={stdout_file}",
            f"--error={stderr_file}",
            sys.executable,
            "-m",
            "scruffy.worker",
            str(assignment_file),
        ]
    )
    return argv


def build_srun_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return allocation-level state safe for launching a nested ``srun``.

    A controller launched inside a Slurm step inherits that step's rank,
    binding, memory, GPU, task, and ``srun`` option variables. Passing those
    variables to a child ``srun`` can override or constrain the child's
    explicit resource request. Preserve only outer-allocation identity,
    capacity, and provenance. The generated command explicitly overrides the
    allocation variables that are also accepted as ``srun`` inputs and asks
    Slurm to export every remaining variable to the worker.
    """

    environment = dict(os.environ if source is None else source)
    environment.pop("SCRUFFY_NODE", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    for name in tuple(environment):
        if (
            name.startswith(_SLURM_ENVIRONMENT_PREFIXES)
            and name not in _ALLOCATION_ENVIRONMENT_KEYS
        ):
            del environment[name]
    return environment


def build_health_srun_argv(
    *,
    slurm_job_id: str,
    name: str,
    root: Path,
    node_names: list[str],
    gpus_per_node: int,
    interval: float,
    allocation_incarnation_sha256: str,
) -> list[str]:
    """Build one overlapping, allocation-wide GPU health monitor step."""

    if not slurm_job_id or not node_names:
        raise ValueError("health monitoring requires a Slurm job and nodes")
    if type(gpus_per_node) is not int or gpus_per_node < 1:
        raise ValueError("health monitoring requires at least one GPU per node")
    if interval <= 0:
        raise ValueError("health interval must be positive")
    if len(allocation_incarnation_sha256) != 64:
        raise ValueError("health monitoring requires an allocation fingerprint")
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
        f"--gpus-per-task={gpus_per_node}",
        "--cpus-per-task=1",
        "--cpu-bind=none",
        "--mem=1G",
        "--kill-on-bad-exit=1",
        "--wait=0",
        "--export=ALL",
        "--output=/dev/null",
        f"--error={root / 'health' / 'monitor-%N.err'}",
        sys.executable,
        "-m",
        "scruffy.health_worker",
        "--root",
        str(root),
        "--interval",
        str(interval),
        "--allocation-incarnation-sha256",
        allocation_incarnation_sha256,
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


def completed_step(step_id: str) -> SlurmStepResult | None:
    """Return one completed step's exit status, or ``None`` while it settles."""

    result = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            f"--jobs={step_id}",
            "--format=JobIDRaw,State,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    for line in result.stdout.splitlines():
        job_id, separator, remainder = line.partition("|")
        if not separator or job_id != step_id:
            continue
        state, separator, encoded_exit = remainder.partition("|")
        if not separator or state.split(maxsplit=1)[0] in {
            "PENDING",
            "RUNNING",
            "COMPLETING",
            "CONFIGURING",
        }:
            return None
        code, separator, signal_number = encoded_exit.partition(":")
        if (
            not separator
            or not code.isascii()
            or not code.isdecimal()
            or not signal_number.isascii()
            or not signal_number.isdecimal()
        ):
            raise RuntimeError(f"sacct returned invalid exit code {encoded_exit!r}")
        signal_value = int(signal_number)
        return SlurmStepResult(
            state=state,
            returncode=-signal_value if signal_value else int(code),
        )
    return None


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


def signal_step(slurm_job_id: str, step_id: str, sig: str = "USR1") -> None:
    """Signal exactly one owned numeric Slurm step, never the allocation."""

    prefix = f"{slurm_job_id}."
    suffix = step_id.removeprefix(prefix)
    if (
        not step_id.startswith(prefix)
        or not suffix.isascii()
        or not suffix.isdecimal()
        or sig != "USR1"
    ):
        raise ValueError(f"refusing unsafe Slurm signal target {step_id!r}")
    subprocess.run(
        ["scancel", "--ctld", "--quiet", f"--signal={sig}", step_id],
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
    deadline = os.environ.get("SLURM_JOB_END_TIME")
    metadata = {
        "id": allocation_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher": launcher,
        "deadline": deadline,
    }
    if deadline and deadline.isascii() and deadline.isdecimal():
        metadata["deadline_at"] = datetime.fromtimestamp(
            int(deadline), UTC
        ).isoformat(timespec="seconds")
    return metadata
