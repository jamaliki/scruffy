"""Satellite Slurm allocations ("leases") and the pure lease autoscaler.

The controller's own allocation is the *primary* allocation. A *lease* is any
other Slurm allocation the controller can launch steps into with
``srun --jobid``. Leases are either *attached* (an allocation the user already
holds and hands to Scruffy) or *elastic* (an allocation Scruffy requests from a
configured pool with ``sbatch`` and releases when idle).

Elastic leases are placeholders: the batch payload is ``scruffy.lease_agent``,
which only waits. Jobs are bound to a lease when it starts, so a queued job
always takes whichever capacity frees first. Everything in this module is pure;
the controller owns I/O and journaling.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ._compat import UTC
from .models import (
    Assignment,
    NodeInventory,
    QueuedJob,
    ResourceRequest,
    normalize_project_id,
)
from .scheduler import choose_first_fitting_job, request_can_ever_fit

LEASE_ID_PREFIX = "scruffy-lease-"
LEASE_KINDS = frozenset({"attached", "elastic"})
# ``requested`` is journaled before ``sbatch`` so an uncertain submission can
# be found again by its job name. ``releasing`` admits no new work.
LIVE_LEASE_STATES = frozenset({"requested", "pending", "running", "releasing"})
LEASE_STATES = LIVE_LEASE_STATES | {"ended"}
PLACEABLE_LEASE_STATES = frozenset({"running"})
MAX_ENDED_LEASES = 100
# A pending lease is only judged after Slurm has had time to evaluate it.
PENDING_SETTLE_SECONDS = 60
# Pending reasons that mean the user's own limits, not cluster load, block the
# lease. Waiting would not get capacity until other work of the user ends.
_LIMIT_REASON = re.compile(r"^(QOS|Assoc)(Grp|Max|JobLimit|ResourceLimit|UsageThreshold)")
_POOL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Options Scruffy itself owns on every lease submission.
_RESERVED_SBATCH_OPTIONS = frozenset(
    {
        "--job-name",
        "-J",
        "--output",
        "-o",
        "--error",
        "-e",
        "--parsable",
        "--wrap",
        "--requeue",
        "--no-requeue",
        "--nodes",
        "-N",
        "--ntasks",
        "-n",
        "--ntasks-per-node",
        "--cpus-per-task",
        "-c",
        "--mem",
        "--mem-per-cpu",
        "--mem-per-gpu",
        "--gpus",
        "-G",
        "--gpus-per-node",
        "--gpus-per-task",
        "--gres",
        "--time",
        "-t",
        "--array",
        "-a",
        "--export",
        "--test-only",
        "--hold",
        "-H",
        "--dependency",
        "-d",
        "--chdir",
        "-D",
    }
)
_POOL_KEYS = {
    "name",
    "nodes",
    "gpus_per_node",
    "cpus_per_node",
    "memory_gb_per_node",
    "time_limit_seconds",
    "partition",
    "qos",
    "account",
    "constraint",
    "gpu_type",
    "max_gpus",
    "max_leases",
    "max_pending_leases",
    "idle_release_seconds",
    "start_within_seconds",
    "cancel_when_limited",
    "projects",
    "time_margin_seconds",
    "backoff_seconds",
    "sbatch_args",
}
_SETTINGS_KEYS = {
    "v",
    "pools",
    "max_total_gpus",
    "heartbeat_timeout_seconds",
    "observe_interval_seconds",
}


class LeaseConfigError(ValueError):
    """Raised for an invalid lease pool configuration."""


def new_lease_id() -> str:
    """Return a lease ID that is also its unique Slurm job name."""

    return f"{LEASE_ID_PREFIX}{uuid.uuid4().hex[:16]}"


def is_limit_reason(reason: object) -> bool:
    """Return whether a pending reason is one of the user's own limits."""

    return isinstance(reason, str) and bool(_LIMIT_REASON.match(reason))


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """One kind of elastic lease Scruffy may request from the main cluster."""

    name: str
    nodes: int
    gpus_per_node: int
    cpus_per_node: int
    memory_gb_per_node: int
    time_limit_seconds: int
    partition: str | None = None
    qos: str | None = None
    account: str | None = None
    constraint: str | None = None
    gpu_type: str | None = None
    max_gpus: int | None = None
    max_leases: int = 1
    max_pending_leases: int = 1
    idle_release_seconds: int = 300
    start_within_seconds: int | None = None
    cancel_when_limited: bool = True
    projects: tuple[str, ...] | None = None
    time_margin_seconds: int = 120
    backoff_seconds: int = 300
    sbatch_args: tuple[str, ...] = ()

    @property
    def lease_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    def shape(self) -> dict[str, int]:
        return {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_node": self.cpus_per_node,
            "memory_gb_per_node": self.memory_gb_per_node,
            "time_limit_seconds": self.time_limit_seconds,
        }

    def virtual_inventory(self, prefix: str) -> tuple[NodeInventory, ...]:
        """Return the capacity one granted lease of this pool would provide."""

        return tuple(
            NodeInventory(
                name=f"{prefix}-{index}",
                gpu_ids=tuple(range(self.gpus_per_node)),
                cpus=self.cpus_per_node,
                memory_gb=self.memory_gb_per_node,
            )
            for index in range(self.nodes)
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            **self.shape(),
            "max_leases": self.max_leases,
            "max_pending_leases": self.max_pending_leases,
            "idle_release_seconds": self.idle_release_seconds,
            "cancel_when_limited": self.cancel_when_limited,
            "time_margin_seconds": self.time_margin_seconds,
            "backoff_seconds": self.backoff_seconds,
        }
        for key in (
            "partition",
            "qos",
            "account",
            "constraint",
            "gpu_type",
            "max_gpus",
            "start_within_seconds",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.projects is not None:
            result["projects"] = list(self.projects)
        if self.sbatch_args:
            result["sbatch_args"] = list(self.sbatch_args)
        return result


@dataclass(frozen=True, slots=True)
class LeaseSettings:
    """Elastic lease pools plus allocation-wide lease limits."""

    pools: tuple[PoolConfig, ...] = ()
    max_total_gpus: int | None = None
    heartbeat_timeout_seconds: int = 600
    observe_interval_seconds: float = 15.0

    def pool(self, name: object) -> PoolConfig | None:
        return next((pool for pool in self.pools if pool.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pools": [pool.to_dict() for pool in self.pools],
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "observe_interval_seconds": self.observe_interval_seconds,
        }
        if self.max_total_gpus is not None:
            result["max_total_gpus"] = self.max_total_gpus
        return result


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise LeaseConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, label: str, *, minimum: int) -> int | None:
    return None if value is None else _integer(value, label, minimum=minimum)


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise LeaseConfigError(f"{label} must be a non-empty string without spaces")
    return value


def _sbatch_args(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise LeaseConfigError(f"{label} must be an array of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.startswith("-") or "\n" in item:
            raise LeaseConfigError(f"{label} entries must be option strings")
        option = item.split("=", 1)[0]
        if option in _RESERVED_SBATCH_OPTIONS:
            raise LeaseConfigError(f"{label} must not set Scruffy-owned option {option}")
        result.append(item)
    return tuple(result)


def pool_from_dict(value: object, index: int = 0) -> PoolConfig:
    label = f"pools[{index}]"
    if not isinstance(value, Mapping):
        raise LeaseConfigError(f"{label} must be an object")
    unknown = set(value) - _POOL_KEYS
    if unknown:
        raise LeaseConfigError(f"{label} has unknown keys {sorted(unknown)!r}")
    name = value.get("name")
    if not isinstance(name, str) or not _POOL_NAME.fullmatch(name):
        raise LeaseConfigError(f"{label}.name must match {_POOL_NAME.pattern}")
    label = f"pool {name!r}"
    projects = value.get("projects")
    if projects is not None:
        if isinstance(projects, (str, bytes)) or not isinstance(projects, list) or not projects:
            raise LeaseConfigError(f"{label}.projects must be a non-empty array")
        try:
            projects = tuple(normalize_project_id(project) for project in projects)
        except ValueError as exc:
            raise LeaseConfigError(f"{label}.projects: {exc}") from exc
    cancel_when_limited = value.get("cancel_when_limited", True)
    if not isinstance(cancel_when_limited, bool):
        raise LeaseConfigError(f"{label}.cancel_when_limited must be a boolean")
    pool = PoolConfig(
        name=name,
        nodes=_integer(value.get("nodes", 1), f"{label}.nodes", minimum=1),
        gpus_per_node=_integer(value.get("gpus_per_node"), f"{label}.gpus_per_node", minimum=1),
        cpus_per_node=_integer(value.get("cpus_per_node"), f"{label}.cpus_per_node", minimum=1),
        memory_gb_per_node=_integer(
            value.get("memory_gb_per_node"), f"{label}.memory_gb_per_node", minimum=1
        ),
        time_limit_seconds=_integer(
            value.get("time_limit_seconds"), f"{label}.time_limit_seconds", minimum=60
        ),
        partition=_optional_text(value.get("partition"), f"{label}.partition"),
        qos=_optional_text(value.get("qos"), f"{label}.qos"),
        account=_optional_text(value.get("account"), f"{label}.account"),
        constraint=_optional_text(value.get("constraint"), f"{label}.constraint"),
        gpu_type=_optional_text(value.get("gpu_type"), f"{label}.gpu_type"),
        max_gpus=_optional_integer(value.get("max_gpus"), f"{label}.max_gpus", minimum=1),
        max_leases=_integer(value.get("max_leases", 1), f"{label}.max_leases", minimum=1),
        max_pending_leases=_integer(
            value.get("max_pending_leases", 1), f"{label}.max_pending_leases", minimum=1
        ),
        idle_release_seconds=_integer(
            value.get("idle_release_seconds", 300),
            f"{label}.idle_release_seconds",
            minimum=0,
        ),
        start_within_seconds=_optional_integer(
            value.get("start_within_seconds"), f"{label}.start_within_seconds", minimum=0
        ),
        cancel_when_limited=cancel_when_limited,
        projects=projects,
        time_margin_seconds=_integer(
            value.get("time_margin_seconds", 120),
            f"{label}.time_margin_seconds",
            minimum=0,
        ),
        backoff_seconds=_integer(
            value.get("backoff_seconds", 300), f"{label}.backoff_seconds", minimum=0
        ),
        sbatch_args=_sbatch_args(value.get("sbatch_args"), f"{label}.sbatch_args"),
    )
    if pool.max_gpus is not None and pool.max_gpus < pool.lease_gpus:
        raise LeaseConfigError(f"{label}.max_gpus is smaller than one lease")
    if pool.time_margin_seconds >= pool.time_limit_seconds:
        raise LeaseConfigError(f"{label}.time_margin_seconds must be below its time limit")
    return pool


def lease_settings_from_dict(value: object) -> LeaseSettings:
    if not isinstance(value, Mapping):
        raise LeaseConfigError("lease configuration must be a JSON object")
    unknown = set(value) - _SETTINGS_KEYS
    if unknown:
        raise LeaseConfigError(f"lease configuration has unknown keys {sorted(unknown)!r}")
    if value.get("v", 1) != 1:
        raise LeaseConfigError("lease configuration version must be 1")
    raw_pools = value.get("pools", [])
    if isinstance(raw_pools, (str, bytes)) or not isinstance(raw_pools, list):
        raise LeaseConfigError("pools must be an array")
    pools = tuple(pool_from_dict(item, index) for index, item in enumerate(raw_pools))
    names = [pool.name for pool in pools]
    if len(set(names)) != len(names):
        raise LeaseConfigError("pool names must be unique")
    observe_interval = value.get("observe_interval_seconds", 15)
    if (
        isinstance(observe_interval, bool)
        or not isinstance(observe_interval, (int, float))
        or not math.isfinite(observe_interval)
        or observe_interval < 1
    ):
        raise LeaseConfigError("observe_interval_seconds must be a number >= 1")
    return LeaseSettings(
        pools=pools,
        max_total_gpus=_optional_integer(value.get("max_total_gpus"), "max_total_gpus", minimum=1),
        heartbeat_timeout_seconds=_integer(
            value.get("heartbeat_timeout_seconds", 600),
            "heartbeat_timeout_seconds",
            minimum=60,
        ),
        observe_interval_seconds=float(observe_interval),
    )


def load_lease_settings(source: Path) -> LeaseSettings:
    """Load a pools file; unknown keys are rejected to catch typos early."""

    with source.open(encoding="utf-8") as handle:
        return lease_settings_from_dict(json.load(handle))


# ---------------------------------------------------------------------------
# Lease records


def parse_time(value: object) -> float | None:
    """Return epoch seconds for a stored ISO timestamp."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def format_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def is_live(lease: Mapping[str, Any]) -> bool:
    return lease.get("state") in LIVE_LEASE_STATES


def lease_inventory(lease: Mapping[str, Any]) -> tuple[NodeInventory, ...]:
    """Return the managed capacity of a started lease, or nothing."""

    raw = lease.get("inventory")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        return ()
    return tuple(NodeInventory.from_dict(item) for item in raw)


def lease_gpus(lease: Mapping[str, Any], settings: LeaseSettings | None = None) -> int:
    """Return GPUs a lease holds, or would hold once granted."""

    inventory = lease_inventory(lease)
    if inventory:
        return sum(len(node.gpu_ids) for node in inventory)
    shape = lease.get("shape")
    if isinstance(shape, Mapping):
        nodes, gpus = shape.get("nodes"), shape.get("gpus_per_node")
        if type(nodes) is int and type(gpus) is int:
            return nodes * gpus
    pool = settings.pool(lease.get("pool")) if settings is not None else None
    return pool.lease_gpus if pool is not None else 0


def lease_remaining_seconds(lease: Mapping[str, Any], now: float) -> float | None:
    """Return a lease's remaining wall time; ``None`` means no known end."""

    end = parse_time(lease.get("end_at"))
    return None if end is None else end - now


def job_project_allowed(projects: Sequence[str] | None, project_id: str) -> bool:
    return projects is None or project_id in projects


def lease_accepts(
    lease: Mapping[str, Any],
    request: ResourceRequest,
    project_id: str,
    now: float,
    *,
    settings: LeaseSettings | None,
    untimed_margin_seconds: float,
) -> bool:
    """Return whether a job may ever be placed on one running lease.

    A satellite lease accepts a timed job only if the job's limit plus a
    margin ends before the lease does. An untimed job may only use an attached
    lease with no known end or more than ``untimed_margin_seconds`` left, the
    same window the primary allocation drains at. Elastic leases always end,
    so they never accept untimed work.
    """

    if lease.get("state") not in PLACEABLE_LEASE_STATES or lease.get("draining"):
        return False
    pool = settings.pool(lease.get("pool")) if settings is not None else None
    allowed = lease.get("projects")
    if isinstance(allowed, list) and project_id not in allowed:
        return False
    if pool is not None and not job_project_allowed(pool.projects, project_id):
        return False
    margin = pool.time_margin_seconds if pool is not None else 120
    remaining = lease_remaining_seconds(lease, now)
    if request.time_limit_seconds is None:
        if lease.get("kind") != "attached":
            return False
        if remaining is not None and remaining <= untimed_margin_seconds:
            return False
    elif remaining is not None and request.time_limit_seconds + margin > remaining:
        return False
    inventory = lease_inventory(lease)
    return bool(inventory) and request_can_ever_fit(inventory, request)


def pool_accepts(pool: PoolConfig, request: ResourceRequest, project_id: str) -> bool:
    """Return whether a fresh lease from this pool could ever run a job.

    A fresh lease's node names are unknown until Slurm grants it, so a job
    with an ``allowed_nodes`` list never drives a new lease. Excluded nodes
    are enforced when the granted lease is placed on.
    """

    return (
        request.allowed_nodes is None
        and request.time_limit_seconds is not None
        and request.time_limit_seconds + pool.time_margin_seconds <= pool.time_limit_seconds
        and job_project_allowed(pool.projects, project_id)
        and request_can_ever_fit(pool.virtual_inventory("shape"), request)
    )


def lease_view(lease: Mapping[str, Any], now: float, used_gpus: int = 0) -> dict[str, Any]:
    """Return a compact public view of one lease."""

    inventory = lease_inventory(lease)
    remaining = lease_remaining_seconds(lease, now)
    view = {
        "lease_id": lease.get("lease_id"),
        "kind": lease.get("kind"),
        "pool": lease.get("pool"),
        "state": lease.get("state"),
        "slurm_job_id": lease.get("slurm_job_id"),
        "nodes": [node.name for node in inventory],
        "gpus": sum(len(node.gpu_ids) for node in inventory),
        "used_gpus": used_gpus,
        "remaining_seconds": None if remaining is None else max(0, int(remaining)),
        "end_at": lease.get("end_at"),
        "reason": lease.get("reason"),
    }
    if lease.get("state") in {"requested", "pending"}:
        view["pending_reason"] = lease.get("pending_reason")
        view["estimated_start_at"] = lease.get("estimated_start_at")
    if lease.get("draining"):
        view["draining"] = True
    return view


def prune_ended_leases(leases: dict[str, Any], keep: int = MAX_ENDED_LEASES) -> list[str]:
    """Drop the oldest ended lease records beyond ``keep``; return their IDs."""

    ended = sorted(
        (
            (str(lease.get("ended_at") or ""), lease_id)
            for lease_id, lease in leases.items()
            if isinstance(lease, Mapping) and lease.get("state") == "ended"
        ),
    )
    removed = [lease_id for _, lease_id in ended[: max(0, len(ended) - keep)]]
    for lease_id in removed:
        leases.pop(lease_id, None)
    return removed


# ---------------------------------------------------------------------------
# Autoscaler


@dataclass(frozen=True, slots=True)
class LeaseDemand:
    """One queued job that could not be placed on any current capacity."""

    job_id: str
    project_id: str
    request: ResourceRequest


@dataclass(frozen=True, slots=True)
class LeasePlan:
    """Lease transitions for the controller to journal and then perform."""

    submit: tuple[str, ...] = ()
    cancel: tuple[tuple[str, str], ...] = ()
    release: tuple[tuple[str, str], ...] = ()
    backoff: tuple[tuple[str, str], ...] = ()
    # Demand that fits some pool but has no lease budget left.
    unplaced: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.submit or self.cancel or self.release or self.backoff)


@dataclass(slots=True)
class _Bin:
    inventory: tuple[NodeInventory, ...]
    lease_id: str | None
    assignments: list[Assignment] = field(default_factory=list)


def _try_bin(bin_: _Bin, demand: LeaseDemand) -> bool:
    choice = choose_first_fitting_job(
        bin_.inventory,
        bin_.assignments,
        [QueuedJob(demand.job_id, demand.request)],
    )
    if choice is None:
        return False
    bin_.assignments.append(replace(choice[1], lease_id=bin_.lease_id))
    return True


def plan_leases(
    settings: LeaseSettings,
    leases: Mapping[str, Mapping[str, Any]],
    demand: Iterable[LeaseDemand],
    *,
    busy_lease_ids: Iterable[str],
    idle_since: Mapping[str, float],
    pool_status: Mapping[str, Mapping[str, Any]],
    now: float,
    allow_growth: bool = True,
    committed_gpus: int | None = None,
) -> LeasePlan:
    """Decide which elastic leases to request, withdraw, or release.

    ``demand`` is queued work in priority order that the scheduler could not
    place on current capacity. Pending leases are counted as capacity that is
    already on its way, so a burst of submissions requests each lease once.
    ``committed_gpus`` is the user's GPU footprint known to Slurm (pending and
    running); without it only Scruffy's own leases count toward
    ``max_total_gpus``. ``idle_since`` holds the controller's own observation
    of when each running lease last became empty; it is deliberately not
    persisted, so a restarted controller never releases a lease at once.
    """

    busy = set(busy_lease_ids)
    submit: list[str] = []
    cancel: list[tuple[str, str]] = []
    release: list[tuple[str, str]] = []
    backoff: dict[str, str] = {}
    unplaced: list[str] = []
    pools = {pool.name: pool for pool in settings.pools}
    elastic = {
        lease_id: lease
        for lease_id, lease in leases.items()
        if lease.get("kind") == "elastic" and is_live(lease) and lease.get("pool") in pools
    }

    # 1. Withdraw pending leases blocked by the user's own limits, or whose
    #    estimated start is too far away. Their pool then backs off.
    withdrawn: set[str] = set()
    for lease_id, lease in sorted(elastic.items()):
        pool = pools[str(lease["pool"])]
        if lease.get("state") != "pending":
            continue
        submitted = parse_time(lease.get("submitted_at")) or now
        settled = now - submitted >= PENDING_SETTLE_SECONDS
        reason = lease.get("pending_reason")
        if pool.cancel_when_limited and is_limit_reason(reason):
            cancel.append((lease_id, "limit_saturated"))
            backoff.setdefault(pool.name, f"limit_saturated:{reason}")
            withdrawn.add(lease_id)
            continue
        estimated = parse_time(lease.get("estimated_start_at"))
        if (
            settled
            and pool.start_within_seconds is not None
            and estimated is not None
            and estimated - now > pool.start_within_seconds
        ):
            cancel.append((lease_id, "start_too_late"))
            backoff.setdefault(pool.name, "start_too_late")
            withdrawn.add(lease_id)

    # 2. Release idle running elastic leases.
    for lease_id, lease in sorted(elastic.items()):
        if lease.get("state") != "running" or lease_id in busy or lease.get("draining"):
            continue
        pool = pools[str(lease["pool"])]
        idle_at = idle_since.get(lease_id)
        remaining = lease_remaining_seconds(lease, now)
        if remaining is not None and remaining <= pool.time_margin_seconds + 60:
            release.append((lease_id, "expiring"))
        elif idle_at is not None and now - idle_at >= pool.idle_release_seconds:
            release.append((lease_id, "idle"))

    # 3. Pack demand into pending leases first, then into new leases.
    remaining_leases = {
        lease_id: lease for lease_id, lease in elastic.items() if lease_id not in withdrawn
    }
    bins: dict[str, list[_Bin]] = {pool.name: [] for pool in settings.pools}
    for lease_id, lease in sorted(
        remaining_leases.items(), key=lambda item: str(item[1].get("requested_at") or "")
    ):
        if lease.get("state") in {"requested", "pending"}:
            pool = pools[str(lease["pool"])]
            bins[pool.name].append(_Bin(pool.virtual_inventory(lease_id), lease_id))
    released_ids = {lease_id for lease_id, _ in release}
    held = {
        lease_id: lease
        for lease_id, lease in remaining_leases.items()
        if lease_id not in released_ids
    }
    held_gpus = {
        name: sum(
            lease_gpus(lease, settings) for lease in held.values() if lease.get("pool") == name
        )
        for name in pools
    }
    total_gpus = committed_gpus if committed_gpus is not None else sum(held_gpus.values())
    budget: dict[str, int] = {}
    for pool in settings.pools:
        live = [lease for lease in held.values() if lease.get("pool") == pool.name]
        pending = sum(lease.get("state") in {"requested", "pending"} for lease in live)
        allowed = min(pool.max_leases - len(live), pool.max_pending_leases - pending)
        if pool.max_gpus is not None:
            allowed = min(allowed, (pool.max_gpus - held_gpus[pool.name]) // pool.lease_gpus)
        status = pool_status.get(pool.name) or {}
        backoff_until = parse_time(status.get("backoff_until"))
        if (
            not allow_growth
            or pool.name in backoff
            or (backoff_until is not None and backoff_until > now)
        ):
            allowed = 0
        budget[pool.name] = max(0, allowed)

    for item in demand:
        eligible = [
            pool for pool in settings.pools if pool_accepts(pool, item.request, item.project_id)
        ]
        if not eligible:
            continue
        if any(_try_bin(bin_, item) for pool in eligible for bin_ in bins[pool.name]):
            continue
        for pool in eligible:
            if budget[pool.name] <= 0:
                continue
            if (
                settings.max_total_gpus is not None
                and total_gpus + pool.lease_gpus > settings.max_total_gpus
            ):
                continue
            new_bin = _Bin(pool.virtual_inventory(f"new-{len(submit)}"), None)
            if not _try_bin(new_bin, item):
                continue
            bins[pool.name].append(new_bin)
            budget[pool.name] -= 1
            total_gpus += pool.lease_gpus
            submit.append(pool.name)
            break
        else:
            unplaced.append(item.job_id)

    # 4. Withdraw pending leases that no queued job needs any more.
    for pool_bins in bins.values():
        for bin_ in pool_bins:
            if bin_.lease_id is None or bin_.assignments:
                continue
            lease = remaining_leases[bin_.lease_id]
            requested = parse_time(lease.get("submitted_at") or lease.get("requested_at"))
            if requested is not None and now - requested < PENDING_SETTLE_SECONDS:
                continue
            if lease.get("state") == "pending":
                cancel.append((bin_.lease_id, "no_demand"))

    return LeasePlan(
        submit=tuple(submit),
        cancel=tuple(cancel),
        release=tuple(release),
        backoff=tuple(sorted(backoff.items())),
        unplaced=tuple(unplaced),
    )
