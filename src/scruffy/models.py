"""Small, validated data types shared by Scruffy's queue and scheduler.

The scheduler only needs to understand resource shapes.  Commands, logs, and
process state deliberately live elsewhere so placement remains a pure and easy
to-test transformation of these values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


JsonObject = Mapping[str, object]


class ModelError(ValueError):
    """Raised when persisted or programmatic model data is invalid."""


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
    """A rectangular request: the same resources are needed on every node."""

    nodes: int
    gpus_per_node: int
    cpus_per_node: int
    memory_gb_per_node: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gpus_per_node",
            _nonnegative_int(self.gpus_per_node, "gpus_per_node"),
        )
        for field_name in ("nodes", "cpus_per_node", "memory_gb_per_node"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_node": self.cpus_per_node,
            "memory_gb_per_node": self.memory_gb_per_node,
        }

    @classmethod
    def from_dict(cls, value: object) -> ResourceRequest:
        data = _mapping(value, "resource request")
        keys = {"nodes", "gpus_per_node", "cpus_per_node", "memory_gb_per_node"}
        _exact_keys(data, keys, "resource request")
        return cls(
            nodes=data["nodes"],  # type: ignore[arg-type]
            gpus_per_node=data["gpus_per_node"],  # type: ignore[arg-type]
            cpus_per_node=data["cpus_per_node"],  # type: ignore[arg-type]
            memory_gb_per_node=data["memory_gb_per_node"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class QueuedJob:
    """The scheduling portion of a queued job.

    A sequence of ``QueuedJob`` objects is ordered oldest first.  Keeping age
    out of this model avoids trusting timestamps supplied by concurrent clients.
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _nonempty_string(self.job_id, "job_id"))
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
        return {
            "job_id": self.job_id,
            "request": self.request.to_dict(),
            "reservations": [item.to_dict() for item in self.reservations],
        }

    @classmethod
    def from_dict(cls, value: object) -> Assignment:
        data = _mapping(value, "assignment")
        _exact_keys(data, {"job_id", "request", "reservations"}, "assignment")
        raw_reservations = data["reservations"]
        if isinstance(raw_reservations, (str, bytes)) or not isinstance(
            raw_reservations, Sequence
        ):
            raise ModelError("reservations must be a JSON array")
        return cls(
            job_id=data["job_id"],  # type: ignore[arg-type]
            request=ResourceRequest.from_dict(data["request"]),
            reservations=tuple(
                NodeReservation.from_dict(item) for item in raw_reservations
            ),
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
