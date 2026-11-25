"""Small, validated data types shared by Scruffy's queue and scheduler.

The scheduler only needs to understand resource shapes.  Commands, logs, and
process state deliberately live elsewhere so placement remains a pure and easy
to-test transformation of these values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

JsonObject = Mapping[str, object]

ACTIVE_JOB_STATES = frozenset({"starting", "running", "finishing", "cancelling"})
TERMINAL_JOB_STATES = frozenset(
    {"succeeded", "failed", "cancelled", "lost", "rejected", "skipped"}
)
DEFAULT_PROJECT = "default"
_PROJECT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_NODE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}\Z")
MAX_NODE_CONSTRAINT_NAMES = 256


class ModelError(ValueError):
    """Raised when persisted or programmatic model data is invalid."""


def normalize_project_id(value: object | None = None) -> str:
    """Return a small, filesystem-safe project name; missing means default."""

    if value is None:
        return DEFAULT_PROJECT
    if not isinstance(value, str) or _PROJECT_PATTERN.fullmatch(value) is None:
        raise ModelError(
            "project_id must be 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return value


def job_project(job: Mapping[str, object]) -> str:
    """Read a job's project while treating legacy records as ``default``."""

    return normalize_project_id(job.get("project_id"))


def node_name_key(name: str) -> str:
    """Return the case-insensitive short hostname used to match node names.

    Inventories reject duplicate short names, so ``gpu-6`` and
    ``gpu-6.cluster`` always identify the same node.
    """

    return name.lower().split(".", 1)[0]


def _node_names(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelError(f"{label} must be a list of node names")
    names = tuple(value)
    if not names:
        raise ModelError(f"{label} must name at least one node; omit it for no constraint")
    if len(names) > MAX_NODE_CONSTRAINT_NAMES:
        raise ModelError(f"{label} may name at most {MAX_NODE_CONSTRAINT_NAMES} nodes")
    for name in names:
        if not isinstance(name, str) or _NODE_NAME_PATTERN.fullmatch(name) is None:
            raise ModelError(
                f"{label} entries must be hostnames of letters, digits, '.', '_' or '-'"
            )
    if len({node_name_key(name) for name in names}) != len(names):
        raise ModelError(f"{label} must not repeat a node")
    return tuple(sorted(names))


def _mapping(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ModelError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: JsonObject, expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ModelError(f"invalid {label}: {', '.join(details)}")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ModelError(f"{label} must not have leading or trailing whitespace")
    return value


def _positive_int(value: object, label: str) -> int:
    # bool is an int subclass, but accepting True as one GPU is never useful.
    if type(value) is not int or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ModelError(f"{label} must be a non-negative integer")
    return value


def _gpu_ids(
    value: object, label: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelError(f"{label} must be a sequence of GPU IDs")
    ids = tuple(value)
    if not ids and not allow_empty:
        raise ModelError(f"{label} must contain at least one GPU ID")
    if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in ids):
        raise ModelError(f"{label} must contain non-negative integers")
    if len(set(ids)) != len(ids):
        raise ModelError(f"{label} must not contain duplicate GPU IDs")
    return tuple(sorted(ids))


@dataclass(frozen=True, slots=True)
class NodeInventory:
    """The resources Scruffy is allowed to manage on one physical node."""

    name: str
    gpu_ids: tuple[int, ...]
    cpus: int
    memory_gb: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "node name"))
        object.__setattr__(self, "gpu_ids", _gpu_ids(self.gpu_ids, "gpu_ids"))
        object.__setattr__(self, "cpus", _positive_int(self.cpus, "cpus"))
        object.__setattr__(
            self,
            "memory_gb",
            _positive_int(self.memory_gb, "memory_gb"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gpu_ids": list(self.gpu_ids),
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
        }

    @classmethod
    def from_dict(cls, value: object) -> NodeInventory:
        data = _mapping(value, "node inventory")
        _exact_keys(data, {"name", "gpu_ids", "cpus", "memory_gb"}, "node inventory")
        return cls(
            name=data["name"],  # type: ignore[arg-type]
            gpu_ids=data["gpu_ids"],  # type: ignore[arg-type]
            cpus=data["cpus"],  # type: ignore[arg-type]
            memory_gb=data["memory_gb"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class NodeAvailability:
    """Resources currently free on a node; every quantity may be zero."""

    name: str
    gpu_ids: tuple[int, ...]
    cpus: int
    memory_gb: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "node name"))
        object.__setattr__(
            self,
            "gpu_ids",
            _gpu_ids(self.gpu_ids, "gpu_ids", allow_empty=True),
        )
        object.__setattr__(self, "cpus", _nonnegative_int(self.cpus, "cpus"))
        object.__setattr__(
            self,
            "memory_gb",
            _nonnegative_int(self.memory_gb, "memory_gb"),
        )


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """A rectangular request: the same resources are needed on every node.

    GPU count may be zero for controller-accounted CPU work.  CPU and memory
    remain positive because every launched process must own a real node slice.
    The optional time limit is an execution contract rather than a placement
    dimension, but keeping it in the immutable request makes retries and
    provenance unambiguous.

    ``allowed_nodes`` and ``exclude_nodes`` are optional placement
    constraints: when set, every selected node must be allowed and none may be
    excluded. Names match case-insensitively by short hostname. Unset fields
    are omitted from the serialized request, so requests without constraints
    keep their identity digests.
    """

    nodes: int
    gpus_per_node: int
    cpus_per_node: int
    memory_gb_per_node: int
    time_limit_seconds: int | None = None
    allowed_nodes: tuple[str, ...] | None = None
    exclude_nodes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in ("nodes", "cpus_per_node", "memory_gb_per_node"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "gpus_per_node",
            _nonnegative_int(self.gpus_per_node, "gpus_per_node"),
        )
        if self.time_limit_seconds is not None:
            object.__setattr__(
                self,
                "time_limit_seconds",
                _positive_int(self.time_limit_seconds, "time_limit_seconds"),
            )
        for field_name in ("allowed_nodes", "exclude_nodes"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _node_names(value, field_name))
        if self.allowed_nodes is not None:
            usable = [name for name in self.allowed_nodes if self.allows_node(name)]
            if len(usable) < self.nodes:
                raise ModelError(
                    f"allowed_nodes leaves {len(usable)} usable nodes for a "
                    f"{self.nodes}-node request"
                )

    @property
    def has_node_constraints(self) -> bool:
        return self.allowed_nodes is not None or self.exclude_nodes is not None

    def allows_node(self, name: str) -> bool:
        """Return whether this request may place a reservation on ``name``."""

        key = node_name_key(name)
        if self.allowed_nodes is not None and key not in {
            node_name_key(item) for item in self.allowed_nodes
        }:
            return False
        return self.exclude_nodes is None or key not in {
            node_name_key(item) for item in self.exclude_nodes
        }

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_node": self.cpus_per_node,
            "memory_gb_per_node": self.memory_gb_per_node,
        }
        if self.time_limit_seconds is not None:
            result["time_limit_seconds"] = self.time_limit_seconds
        if self.allowed_nodes is not None:
            result["allowed_nodes"] = list(self.allowed_nodes)
        if self.exclude_nodes is not None:
            result["exclude_nodes"] = list(self.exclude_nodes)
        return result

    @classmethod
    def from_dict(cls, value: object) -> ResourceRequest:
        data = _mapping(value, "resource request")
        required = {"nodes", "gpus_per_node", "cpus_per_node", "memory_gb_per_node"}
        optional = {"time_limit_seconds", "allowed_nodes", "exclude_nodes"}
        missing = required - data.keys()
        extra = data.keys() - (required | optional)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)!r}")
            if extra:
                details.append(f"unexpected {sorted(extra)!r}")
            raise ModelError(f"invalid resource request: {', '.join(details)}")
        for field_name in ("allowed_nodes", "exclude_nodes"):
            if field_name in data and data[field_name] is None:
                raise ModelError(f"{field_name} must be omitted rather than null")
        return cls(
            nodes=data["nodes"],  # type: ignore[arg-type]
            gpus_per_node=data["gpus_per_node"],  # type: ignore[arg-type]
            cpus_per_node=data["cpus_per_node"],  # type: ignore[arg-type]
            memory_gb_per_node=data["memory_gb_per_node"],  # type: ignore[arg-type]
            time_limit_seconds=data.get("time_limit_seconds"),  # type: ignore[arg-type]
            allowed_nodes=data.get("allowed_nodes"),  # type: ignore[arg-type]
            exclude_nodes=data.get("exclude_nodes"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class QueuedJob:
    """The scheduling portion of a queued job.

    A sequence of ``QueuedJob`` objects is ordered by scheduling priority.
    Keeping policy metadata out of this model leaves ordering with the
    single-writer controller.
    """

    job_id: str
    request: ResourceRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _nonempty_string(self.job_id, "job_id"))
        if not isinstance(self.request, ResourceRequest):
            raise ModelError("request must be a ResourceRequest")


@dataclass(frozen=True, slots=True)
class NodeReservation:
    """Resources reserved for one job on one node."""

    node: str
    gpu_ids: tuple[int, ...]
    cpus: int
    memory_gb: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node", _nonempty_string(self.node, "node"))
        object.__setattr__(
            self,
            "gpu_ids",
            _gpu_ids(self.gpu_ids, "gpu_ids", allow_empty=True),
        )
        object.__setattr__(self, "cpus", _positive_int(self.cpus, "cpus"))
        object.__setattr__(
            self,
            "memory_gb",
            _positive_int(self.memory_gb, "memory_gb"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "gpu_ids": list(self.gpu_ids),
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
        }

    @classmethod
    def from_dict(cls, value: object) -> NodeReservation:
        data = _mapping(value, "node reservation")
        _exact_keys(data, {"node", "gpu_ids", "cpus", "memory_gb"}, "node reservation")
        return cls(
            node=data["node"],  # type: ignore[arg-type]
            gpu_ids=data["gpu_ids"],  # type: ignore[arg-type]
            cpus=data["cpus"],  # type: ignore[arg-type]
            memory_gb=data["memory_gb"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    """An atomic, multi-node reservation belonging to one job."""

    job_id: str
    request: ResourceRequest
    reservations: tuple[NodeReservation, ...]
    # ``None`` is the controller's own (primary) allocation. A lease ID names
    # another Slurm allocation the controller launches into. Primary
    # assignments serialize without the key so their digests are unchanged.
    lease_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _nonempty_string(self.job_id, "job_id"))
        if self.lease_id is not None:
            object.__setattr__(
                self, "lease_id", _nonempty_string(self.lease_id, "lease_id")
            )
        if not isinstance(self.request, ResourceRequest):
            raise ModelError("request must be a ResourceRequest")
        reservations = tuple(self.reservations)
        if not reservations:
            raise ModelError("an assignment must contain at least one reservation")
        if not all(isinstance(item, NodeReservation) for item in reservations):
            raise ModelError("reservations must contain NodeReservation values")
        if len({item.node for item in reservations}) != len(reservations):
            raise ModelError("an assignment cannot reserve a node twice")
        object.__setattr__(self, "reservations", reservations)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "job_id": self.job_id,
            "request": self.request.to_dict(),
            "reservations": [item.to_dict() for item in self.reservations],
        }
        if self.lease_id is not None:
            result["lease_id"] = self.lease_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> Assignment:
        data = _mapping(value, "assignment")
        required = {"job_id", "request", "reservations"}
        _exact_keys(
            data,
            required | ({"lease_id"} if "lease_id" in data else set()),
            "assignment",
        )
        raw_reservations = data["reservations"]
        if isinstance(raw_reservations, (str, bytes)) or not isinstance(
            raw_reservations, Sequence
        ):
            raise ModelError("reservations must be a JSON array")
        lease_id = data.get("lease_id")
        if "lease_id" in data and lease_id is None:
            raise ModelError("assignment lease_id must be a non-empty string")
        return cls(
            job_id=data["job_id"],  # type: ignore[arg-type]
            request=ResourceRequest.from_dict(data["request"]),
            reservations=tuple(
                NodeReservation.from_dict(item) for item in raw_reservations
            ),
            lease_id=lease_id,  # type: ignore[arg-type]
        )


def validate_inventory(
    inventory: Sequence[NodeInventory],
) -> tuple[NodeInventory, ...]:
    """Reject duplicate hostnames, including short/FQDN aliases."""

    nodes = tuple(inventory)
    if not nodes:
        raise ModelError("inventory must contain at least one node")
    if not all(isinstance(node, NodeInventory) for node in nodes):
        raise ModelError("inventory must contain NodeInventory values")
    names = [node.name.lower() for node in nodes]
    short_names = [name.split(".", 1)[0] for name in names]
    if len(set(names)) != len(names):
        raise ModelError("inventory node names must be unique")
    if len(set(short_names)) != len(short_names):
        raise ModelError("inventory short node names must be unique")
    return nodes
