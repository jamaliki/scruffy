from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import attach_lease as client_attach_lease
from scruffy.client import release_lease as client_release_lease
from scruffy.controller import _initialize_controller
from scruffy.lease_agent import should_exit, worker_steps_live
from scruffy.lease_runtime import (
    _end_lease,
    attach_lease,
    lease_command_outcome,
    observe_leases,
    plan_and_apply_leases,
    release_lease,
)
from scruffy.leases import (
    LEASE_ID_PREFIX,
    LeaseConfigError,
    LeaseDemand,
    LeaseSettings,
    format_time,
    is_limit_reason,
    lease_accepts,
    lease_settings_from_dict,
    plan_leases,
    pool_accepts,
    pool_from_dict,
)
from scruffy.lifecycle import schedule
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.slurm import (
    AllocationIncarnation,
    SlurmJobObservation,
    SlurmStep,
    build_lease_environment,
    build_lease_sbatch_argv,
    parse_squeue_jobs,
    tres_gpu_count,
)
from scruffy.state import apply_lease_event, group_commit
from scruffy.storage import list_commands, read_events, utc_now, write_state
from scruffy.submissions import job_from_spec
from scruffy.summary import lease_summary

NOW = 1_800_000_000.0
POOL = {
    "name": "h100",
    "nodes": 1,
    "gpus_per_node": 4,
    "cpus_per_node": 32,
    "memory_gb_per_node": 256,
    "time_limit_seconds": 4 * 3600,
}


def _settings(**pool_overrides: object) -> LeaseSettings:
    return lease_settings_from_dict({"v": 1, "pools": [{**POOL, **pool_overrides}]})


def _request(gpus: int = 1, time_limit: int | None = 3600) -> ResourceRequest:
    return ResourceRequest(1, gpus, 4, 16, time_limit_seconds=time_limit)


def _running_lease(
    lease_id: str = "scruffy-lease-a",
    *,
    kind: str = "attached",
    pool: str | None = None,
    remaining: float | None = 7200,
    gpus: int = 4,
    projects: list[str] | None = None,
) -> dict[str, object]:
    lease: dict[str, object] = {
        "lease_id": lease_id,
        "kind": kind,
        "pool": pool,
        "state": "running",
        "slurm_job_id": "5001",
        "end_at": format_time(NOW + remaining) if remaining is not None else None,
        "inventory": [NodeInventory("n1", tuple(range(gpus)), 32, 256).to_dict()],
    }
    if projects is not None:
        lease["projects"] = projects
    return lease


class LeaseConfigTests(unittest.TestCase):
    def test_valid_pool_file_round_trips(self) -> None:
        settings = lease_settings_from_dict(
            {
                "v": 1,
                "max_total_gpus": 16,
                "pools": [{**POOL, "qos": "normal", "sbatch_args": ["--exclusive"]}],
            }
        )
        self.assertEqual(16, settings.max_total_gpus)
        pool = settings.pool("h100")
        assert pool is not None
        self.assertEqual(4, pool.lease_gpus)
        self.assertEqual(("--exclusive",), pool.sbatch_args)
        self.assertEqual(settings, lease_settings_from_dict({"v": 1, **settings.to_dict()}))

    def test_invalid_pool_files_are_rejected(self) -> None:
        cases = [
            {"v": 1, "pools": [{**POOL, "typo": 1}]},
            {"v": 2, "pools": []},
            {"v": 1, "pools": [POOL, POOL]},
            {"v": 1, "pools": [{**POOL, "sbatch_args": ["--time=1:00:00"]}]},
            {"v": 1, "pools": [{**POOL, "sbatch_args": ["--job-name=x"]}]},
            {"v": 1, "pools": [{**POOL, "max_gpus": 2}]},
            {"v": 1, "pools": [{**POOL, "time_margin_seconds": 4 * 3600}]},
            {"v": 1, "pools": [{**POOL, "partition": "has space"}]},
            {"v": 1, "heartbeat_timeout_seconds": 5, "pools": []},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(LeaseConfigError):
                lease_settings_from_dict(case)

    def test_limit_reasons_are_the_users_own_limits(self) -> None:
        self.assertTrue(is_limit_reason("QOSMaxGRESPerUser"))
        self.assertTrue(is_limit_reason("AssocGrpGRES"))
        self.assertFalse(is_limit_reason("Priority"))
        self.assertFalse(is_limit_reason("Resources"))
        self.assertFalse(is_limit_reason(None))


class LeaseFitTests(unittest.TestCase):
    def _accepts(
        self, lease: dict[str, object], request: ResourceRequest, **kwargs: object
    ) -> bool:
        return lease_accepts(
            lease,
            request,
            str(kwargs.get("project", "default")),
            NOW,
            settings=kwargs.get("settings"),  # type: ignore[arg-type]
            untimed_margin_seconds=900,
        )

    def test_timed_job_must_end_before_the_lease(self) -> None:
        lease = _running_lease(remaining=7200)
        self.assertTrue(self._accepts(lease, _request(time_limit=3600)))
        self.assertFalse(self._accepts(lease, _request(time_limit=7200)))
        self.assertFalse(self._accepts(lease, _request(gpus=8)))

    def test_untimed_job_needs_an_attached_lease_with_time_left(self) -> None:
        self.assertTrue(self._accepts(_running_lease(remaining=None), _request(time_limit=None)))
        self.assertTrue(self._accepts(_running_lease(remaining=7200), _request(time_limit=None)))
        self.assertFalse(self._accepts(_running_lease(remaining=600), _request(time_limit=None)))
        elastic = _running_lease(kind="elastic", pool="h100")
        self.assertFalse(self._accepts(elastic, _request(time_limit=None), settings=_settings()))

    def test_lease_and_pool_project_restrictions(self) -> None:
        lease = _running_lease(projects=["alpha"])
        self.assertTrue(self._accepts(lease, _request(), project="alpha"))
        self.assertFalse(self._accepts(lease, _request(), project="beta"))
        settings = _settings(projects=["alpha"])
        elastic = _running_lease(kind="elastic", pool="h100")
        self.assertFalse(self._accepts(elastic, _request(), project="beta", settings=settings))

    def test_only_running_undrained_leases_accept_work(self) -> None:
        for change in ({"state": "pending"}, {"state": "releasing"}, {"draining": True}):
            with self.subTest(change=change):
                self.assertFalse(self._accepts({**_running_lease(), **change}, _request()))

    def test_pool_accepts_only_timed_work_that_fits_one_lease(self) -> None:
        pool = pool_from_dict(POOL)
        self.assertTrue(pool_accepts(pool, _request(gpus=4, time_limit=3600), "default"))
        self.assertFalse(pool_accepts(pool, _request(time_limit=None), "default"))
        self.assertFalse(pool_accepts(pool, _request(time_limit=4 * 3600), "default"))
        self.assertFalse(pool_accepts(pool, _request(gpus=8), "default"))


class LeasePlannerTests(unittest.TestCase):
    def _plan(
        self, settings: LeaseSettings, leases: dict[str, dict[str, object]], demand, **kwargs
    ):
        options = {
            "busy_lease_ids": (),
            "idle_since": {},
            "pool_status": {},
            "now": NOW,
        }
        options.update(kwargs)
        return plan_leases(settings, leases, demand, **options)

    def test_demand_is_packed_into_as_few_new_leases_as_fit(self) -> None:
        demand = [LeaseDemand(f"job-{index}", "default", _request(gpus=2)) for index in range(3)]
        plan = self._plan(_settings(max_leases=4, max_pending_leases=4), {}, demand)
        self.assertEqual(("h100", "h100"), plan.submit)
        self.assertEqual((), plan.unplaced)

    def test_pending_lease_absorbs_demand_before_new_requests(self) -> None:
        pending = {
            "lease-p": {
                "lease_id": "lease-p",
                "kind": "elastic",
                "pool": "h100",
                "state": "pending",
                "submitted_at": format_time(NOW - 10),
                "requested_at": format_time(NOW - 10),
            }
        }
        plan = self._plan(
            _settings(max_leases=4, max_pending_leases=4),
            pending,
            [LeaseDemand("job", "default", _request(gpus=4))],
        )
        self.assertEqual((), plan.submit)
        self.assertEqual((), plan.cancel)

    def test_budgets_and_backoff_stop_growth(self) -> None:
        demand = [LeaseDemand("job", "default", _request())]
        self.assertEqual((), self._plan(_settings(), {}, demand, allow_growth=False).submit)
        backed_off = {"h100": {"backoff_until": format_time(NOW + 60)}}
        self.assertEqual(
            ("job",), self._plan(_settings(), {}, demand, pool_status=backed_off).unplaced
        )
        limited = lease_settings_from_dict({"v": 1, "max_total_gpus": 8, "pools": [POOL]})
        self.assertEqual((), self._plan(limited, {}, demand, committed_gpus=6).submit)
        self.assertEqual(("h100",), self._plan(limited, {}, demand, committed_gpus=4).submit)

    def test_limit_blocked_pending_lease_is_withdrawn_and_pool_backs_off(self) -> None:
        pending = {
            "lease-p": {
                "lease_id": "lease-p",
                "kind": "elastic",
                "pool": "h100",
                "state": "pending",
                "pending_reason": "QOSMaxGRESPerUser",
                "submitted_at": format_time(NOW - 5),
            }
        }
        plan = self._plan(_settings(), pending, [LeaseDemand("job", "default", _request())])
        self.assertEqual((("lease-p", "limit_saturated"),), plan.cancel)
        self.assertEqual("h100", plan.backoff[0][0])
        self.assertEqual((), plan.submit)

    def test_late_start_estimate_withdraws_only_after_settling(self) -> None:
        lease = {
            "lease_id": "lease-p",
            "kind": "elastic",
            "pool": "h100",
            "state": "pending",
            "pending_reason": "Priority",
            "estimated_start_at": format_time(NOW + 7200),
        }
        settings = _settings(start_within_seconds=600)
        demand = [LeaseDemand("job", "default", _request())]
        fresh = {"lease-p": {**lease, "submitted_at": format_time(NOW - 5)}}
        self.assertEqual((), self._plan(settings, fresh, demand).cancel)
        settled = {"lease-p": {**lease, "submitted_at": format_time(NOW - 120)}}
        self.assertEqual(
            (("lease-p", "start_too_late"),), self._plan(settings, settled, demand).cancel
        )

    def test_idle_and_expiring_running_leases_are_released(self) -> None:
        settings = _settings(idle_release_seconds=300)
        idle = _running_lease("lease-i", kind="elastic", pool="h100")
        expiring = _running_lease("lease-e", kind="elastic", pool="h100", remaining=100)
        busy = _running_lease("lease-b", kind="elastic", pool="h100")
        leases = {"lease-i": idle, "lease-e": expiring, "lease-b": busy}
        plan = self._plan(
            settings,
            leases,
            [],
            busy_lease_ids={"lease-b"},
            idle_since={"lease-i": NOW - 400, "lease-b": NOW - 4000},
        )
        self.assertEqual({("lease-i", "idle"), ("lease-e", "expiring")}, set(plan.release))
        recent = self._plan(settings, {"lease-i": idle}, [], idle_since={"lease-i": NOW - 10})
        self.assertEqual((), recent.release)
        # A restarted controller has no idle observation and never releases at once.
        self.assertEqual((), self._plan(settings, {"lease-i": idle}, []).release)

    def test_pending_lease_without_demand_is_withdrawn_after_settling(self) -> None:
        pending = {
            "lease-p": {
                "lease_id": "lease-p",
                "kind": "elastic",
                "pool": "h100",
                "state": "pending",
                "submitted_at": format_time(NOW - 120),
            }
        }
        self.assertEqual((("lease-p", "no_demand"),), self._plan(_settings(), pending, []).cancel)

    def test_attached_leases_are_never_planned(self) -> None:
        leases = {"lease-a": _running_lease("lease-a")}
        plan = self._plan(_settings(), leases, [], idle_since={"lease-a": NOW - 99999})
        self.assertTrue(plan.empty)


class SlurmLeaseHelperTests(unittest.TestCase):
    def test_squeue_documents_from_several_releases_parse(self) -> None:
        jobs = parse_squeue_jobs(
            {
                "jobs": [
                    {
                        "job_id": 101,
                        "name": "scruffy-lease-abc",
                        "job_state": ["PENDING"],
                        "state_reason": "QOSMaxGRESPerUser",
                        "start_time": {"set": True, "infinite": False, "number": 1800000600},
                        "end_time": {"set": False, "infinite": False, "number": 0},
                        "tres_req_str": "cpu=32,mem=256G,node=1,gres/gpu=4",
                    },
                    {
                        "job_id": {"set": True, "number": 102},
                        "name": "interactive",
                        "job_state": "RUNNING",
                        "state_reason": "None",
                        "nodes": "gpu-[1-2]",
                        "tres_alloc_str": "cpu=16,gres/gpu:h100=8",
                        "end_time": 1800003600,
                        "restart_cnt": 0,
                    },
                    {"job_id": 103, "name": "done", "job_state": "COMPLETED"},
                ]
            }
        )
        pending, running, done = jobs
        self.assertEqual(
            ("101", "PENDING", "QOSMaxGRESPerUser", 4),
            (pending.job_id, pending.state, pending.reason, pending.gpus),
        )
        self.assertEqual(1800000600.0, pending.start_at)
        self.assertIsNone(pending.end_at)
        self.assertEqual(
            ("102", 8, None, "gpu-[1-2]", 0),
            (running.job_id, running.gpus, running.reason, running.nodes, running.restart_count),
        )
        self.assertTrue(done.terminal)
        self.assertFalse(running.terminal)
        with self.assertRaises(RuntimeError):
            parse_squeue_jobs({"jobs": [], "errors": ["boom"]})

    def test_tres_gpu_count_prefers_the_generic_total(self) -> None:
        self.assertEqual(4, tres_gpu_count("cpu=8,gres/gpu=4,gres/gpu:h100=4"))
        self.assertEqual(6, tres_gpu_count("gres/gpu:a100=2,gres/gpu:h100=4"))
        self.assertEqual(0, tres_gpu_count(None))

    def test_lease_sbatch_is_argv_only_and_never_requeues(self) -> None:
        argv = build_lease_sbatch_argv(
            lease_id="scruffy-lease-0123abcd",
            root=Path("/shared/queue"),
            nodes=2,
            gpus_per_node=4,
            cpus_per_node=32,
            memory_gb_per_node=256,
            time_limit_seconds=90061,
            heartbeat_timeout_seconds=600,
            qos="normal",
            gpu_type="h100",
            extra_args=("--exclusive",),
        )
        self.assertEqual(["sbatch", "--parsable", "--job-name=scruffy-lease-0123abcd"], argv[:3])
        for expected in (
            "--no-requeue",
            "--nodes=2",
            "--gpus-per-node=h100:4",
            "--mem=256G",
            "--time=1-01:01:00",
            "--qos=normal",
            "--exclusive",
            "--output=/shared/queue/leases/logs/scruffy-lease-0123abcd.out",
        ):
            self.assertIn(expected, argv)
        self.assertTrue(argv[-1].startswith("--wrap=exec "))
        self.assertIn("scruffy.lease_agent", argv[-1])
        with self.assertRaises(ValueError):
            build_lease_sbatch_argv(
                lease_id="other; rm -rf /",
                root=Path("/q"),
                nodes=1,
                gpus_per_node=1,
                cpus_per_node=1,
                memory_gb_per_node=1,
                time_limit_seconds=60,
                heartbeat_timeout_seconds=60,
            )

    def test_lease_environment_drops_the_controllers_allocation(self) -> None:
        environment = build_lease_environment(
            {
                "PATH": "/bin",
                "SLURM_JOB_ID": "1",
                "SLURM_CONF": "/etc/slurm.conf",
                "SBATCH_PARTITION": "x",
                "SCRUFFY_NODE": "n1",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        self.assertEqual({"PATH": "/bin", "SLURM_CONF": "/etc/slurm.conf"}, environment)


class LeaseModelTests(unittest.TestCase):
    def test_assignment_lease_id_round_trips_and_keeps_primary_digests(self) -> None:
        primary = Assignment("job", _request(), (NodeReservation("n1", (0,), 4, 16),))
        self.assertNotIn("lease_id", primary.to_dict())
        satellite = Assignment(
            "job", _request(), (NodeReservation("n1", (0,), 4, 16),), lease_id="scruffy-lease-a"
        )
        self.assertEqual(satellite, Assignment.from_dict(satellite.to_dict()))
        with self.assertRaises(ValueError):
            Assignment.from_dict({**primary.to_dict(), "lease_id": None})

    def test_lease_events_replay_into_state(self) -> None:
        state: dict[str, object] = {"leases": {}, "lease_pools": {}}
        lease = {
            "lease_id": "scruffy-lease-a",
            "kind": "elastic",
            "pool": "h100",
            "state": "requested",
        }
        apply_lease_event(state, {"kind": "lease.requested", "data": {"lease": lease}})
        apply_lease_event(
            state, {"kind": "lease.running", "data": {"lease": {**lease, "state": "running"}}}
        )
        self.assertEqual("running", state["leases"]["scruffy-lease-a"]["state"])  # type: ignore[index]
        status = {"backoff_until": format_time(NOW), "reason": "limit"}
        apply_lease_event(
            state, {"kind": "lease.pool_backoff", "data": {"pool": "h100", "pool_status": status}}
        )
        self.assertEqual(status, state["lease_pools"]["h100"])  # type: ignore[index]

    def test_command_outcomes_summarize_receipts(self) -> None:
        lease = {"lease_id": "scruffy-lease-a", "slurm_job_id": "5001"}
        self.assertEqual(
            {"state": "attached", "lease_id": "scruffy-lease-a", "slurm_job_id": "5001"},
            lease_command_outcome("lease.attached", {"lease": lease}),
        )
        self.assertEqual(
            {"state": "rejected", "reason": "unknown_lease"},
            lease_command_outcome("command.rejected", {"reason": "unknown_lease"}),
        )

    def test_summary_counts_lease_capacity_and_use(self) -> None:
        lease = _running_lease(gpus=4)
        assignment = Assignment(
            "job",
            _request(gpus=2),
            (NodeReservation("n1", (0, 1), 4, 16),),
            lease_id="scruffy-lease-a",
        )
        state = {
            "leases": {"scruffy-lease-a": lease},
            "jobs": {"job": {"id": "job", "state": "running", "assignment": assignment.to_dict()}},
        }
        summary = lease_summary(state)
        self.assertEqual(4, summary["totals"]["gpus_total"])
        self.assertEqual(2, summary["totals"]["gpus_used"])
        self.assertEqual("scruffy-lease-a", summary["live"][0]["lease_id"])


class LeaseAgentTests(unittest.TestCase):
    def test_agent_exits_only_when_silent_and_idle(self) -> None:
        self.assertFalse(
            should_exit(heartbeat_age=30, started_age=9999, timeout=600, steps_live=False)
        )
        self.assertFalse(
            should_exit(heartbeat_age=900, started_age=9999, timeout=600, steps_live=True)
        )
        self.assertTrue(
            should_exit(heartbeat_age=900, started_age=10, timeout=600, steps_live=False)
        )
        # Without any heartbeat the agent's own age stands in for it.
        self.assertFalse(
            should_exit(heartbeat_age=None, started_age=10, timeout=600, steps_live=False)
        )
        self.assertTrue(
            should_exit(heartbeat_age=None, started_age=700, timeout=600, steps_live=False)
        )

    def test_only_numbered_steps_hold_the_lease_and_errors_keep_it(self) -> None:
        idle = (SlurmStep("77.batch", "batch", "n1"), SlurmStep("77.extern", "extern", "n1"))
        busy = (*idle, SlurmStep("77.3", "scruffy-x", "n1"))
        with mock.patch("scruffy.slurm.live_steps", return_value=idle):
            self.assertFalse(worker_steps_live("77"))
        with mock.patch("scruffy.slurm.live_steps", return_value=busy):
            self.assertTrue(worker_steps_live("77"))
        with mock.patch("scruffy.slurm.live_steps", side_effect=RuntimeError("down")):
            self.assertTrue(worker_steps_live("77"))


class LeaseClientTests(unittest.TestCase):
    def test_attach_and_release_write_idempotent_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            with self.assertRaises(ValueError):
                client_attach_lease(root, "12a")
            first = client_attach_lease(
                root, "5001", gpus_per_node=2, projects=["beta", "alpha"], request_id="attach-1"
            )
            again = client_attach_lease(
                root, "5001", gpus_per_node=2, projects=["alpha", "beta"], request_id="attach-1"
            )
            self.assertEqual(first, again)
            client_release_lease(root, "5001", request_id="release-1")
            commands = {command["request_id"]: command for _, command in list_commands(root)}
            self.assertEqual(
                {
                    "kind": "lease.attach",
                    "slurm_job_id": "5001",
                    "gpus_per_node": 2,
                    "projects": ["alpha", "beta"],
                    "request_id": "attach-1",
                },
                commands["attach-1"],
            )
            self.assertEqual("lease.release", commands["release-1"]["kind"])


class HubCliTests(unittest.TestCase):
    def test_serve_hub_runs_a_controller_without_a_primary_allocation(self) -> None:
        from scruffy.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            pools = Path(temporary) / "pools.json"
            pools.write_text(json.dumps({"v": 1, "pools": [POOL]}))
            with (
                mock.patch.dict("os.environ", {"SLURM_JOB_ID": "999"}, clear=True),
                mock.patch("scruffy.cli.run_controller") as run_controller,
            ):
                self.assertEqual(
                    0, main(["--root", str(root), "serve", "--hub", "--pools", str(pools)])
                )
            options = run_controller.call_args.kwargs
            self.assertEqual((), options["inventory"])
            self.assertIsNone(options["slurm_job_id"])
            self.assertEqual("hub", options["allocation_id"])
            self.assertEqual("off", options["gpu_health_mode"])
            self.assertEqual("h100", options["lease_settings"].pools[0].name)
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                rejected = main(["--root", str(root), "serve", "--hub", "--slurm-job-id", "1"])
            self.assertNotEqual(0, rejected)
            self.assertIn("--hub has no primary allocation", stderr.getvalue())


# ---------------------------------------------------------------------------
# Controller integration with mocked Slurm


LEASE_INVENTORY = (NodeInventory("lease-node", (0, 1), 8, 64),)


def _queued_job(
    root: Path, job_id: str, *, time_limit: int | None = None, retry: bool = False
) -> dict[str, object]:
    spec: dict[str, object] = {
        "v": 1,
        "job_id": job_id,
        "request_id": f"request-{job_id}",
        "name": job_id,
        "submitted_at": utc_now(),
        "argv": ["true"],
        "cwd": str(root),
        "env": {},
        "resources": ResourceRequest(1, 1, 1, 1, time_limit_seconds=time_limit).to_dict(),
        "workflow_id": "flow",
        "task_id": job_id,
        "needs": [],
        "wait_for": [],
    }
    if retry:
        spec["recovery"] = {
            "max_attempts": 2,
            "retry_on": ["lease_ended"],
            "evacuation": {"signal": "USR1", "grace_seconds": 0},
        }
    return job_from_spec(spec, 1)


def _write_hub_queue(root: Path, jobs: list[dict[str, object]]) -> None:
    write_state(
        root,
        {
            "v": 1,
            "queue_id": "queue",
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "gpu_health": None,
            "jobs": {str(job["id"]): job for job in jobs},
            "next_queue_order": len(jobs) + 1,
            "archived_jobs": 0,
            "archived_counts": {},
            "archived_project_counts": {},
            "draining": False,
            "drain_requested": False,
            "launches_paused": False,
            "updated_at": utc_now(),
        },
    )


def _hub(root: Path, settings: LeaseSettings | None = None):
    return _initialize_controller(
        root=root,
        inventory=(),
        launcher="slurm",
        allocation_id="hub",
        slurm_job_id=None,
        poll_interval=0.1,
        cancel_grace=0,
        gpu_health_mode="off",
        lease_settings=settings,
    )


def _observation(
    job_id: str, name: str, state: str = "RUNNING", **values: object
) -> SlurmJobObservation:
    return SlurmJobObservation(
        job_id=job_id,
        name=name,
        state=state,
        reason=values.get("reason"),  # type: ignore[arg-type]
        nodes="lease-node" if state == "RUNNING" else None,
        start_at=None,
        end_at=values.get("end_at", time.time() + 7200),  # type: ignore[arg-type]
        gpus=2,
        restart_count=0,
    )


def _process() -> mock.MagicMock:
    process = mock.MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    return process


class HubControllerTests(unittest.TestCase):
    def _attach(
        self, controller, slurm_job_id: str = "5001", **command: object
    ) -> dict[str, object]:
        incarnation = AllocationIncarnation(slurm_job_id, 0, LEASE_INVENTORY)
        with (
            mock.patch(
                "scruffy.lease_runtime.discover_slurm_allocation",
                return_value=({node.name: node for node in LEASE_INVENTORY}, incarnation),
            ),
            mock.patch(
                "scruffy.lease_runtime.lease_job",
                return_value=_observation(slurm_job_id, "interactive"),
            ),
            group_commit(controller),
        ):
            return attach_lease(
                controller,
                {
                    "kind": "lease.attach",
                    "slurm_job_id": slurm_job_id,
                    "request_id": "attach",
                    **command,
                },
            )

    def test_hub_places_work_on_an_attached_allocation_and_recovers_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            _write_hub_queue(
                root,
                [_queued_job(root, "untimed"), _queued_job(root, "too-long", time_limit=4 * 3600)],
            )
            controller = _hub(root)
            try:
                self.assertTrue(controller.hub)
                self.assertEqual("hub", controller.state["allocation"]["mode"])
                self.assertEqual("queued", controller.state["jobs"]["too-long"]["state"])
                outcome = self._attach(controller)
                self.assertEqual("attached", outcome["state"])
                lease_id = str(outcome["lease_id"])
                self.assertTrue(lease_id.startswith(LEASE_ID_PREFIX))
                duplicate = self._attach(controller)
                self.assertEqual("already_attached", duplicate["state"])

                with mock.patch(
                    "scruffy.lifecycle.subprocess.Popen", return_value=_process()
                ) as popen:
                    schedule(controller)
                popen.assert_called_once()
                argv = popen.call_args.args[0]
                environment = popen.call_args.kwargs["env"]
                self.assertTrue(any("5001" in item for item in argv), argv)
                self.assertNotIn("SLURM_JOB_ID", environment)
                job = controller.state["jobs"]["untimed"]
                self.assertEqual(lease_id, job["lease_id"])
                self.assertEqual(lease_id, job["assignment"]["lease_id"])
                # A 4-hour job cannot finish inside a lease with 2 hours left.
                self.assertEqual("queued", controller.state["jobs"]["too-long"]["state"])
                self.assertEqual(1, lease_summary(controller.state)["totals"]["gpus_used"])
            finally:
                controller.journal.close()

            restarted = _hub(root)
            try:
                self.assertIn("untimed", restarted.running)
                self.assertTrue(restarted.state["launches_paused"])
                self.assertEqual("running", restarted.state["leases"][lease_id]["state"])
            finally:
                restarted.journal.close()

    def test_ended_lease_loses_its_job_and_admits_a_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            _write_hub_queue(root, [_queued_job(root, "train", retry=True)])
            controller = _hub(root)
            try:
                lease_id = str(self._attach(controller)["lease_id"])
                with mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=_process()):
                    schedule(controller)
                running = controller.running["train"]
                with (
                    mock.patch("scruffy.lease_runtime.completed_step", return_value=None),
                    mock.patch("scruffy.lease_runtime.stop_launcher"),
                    group_commit(controller),
                ):
                    _end_lease(controller, controller.state["leases"][lease_id], "allocation_gone")
                self.assertEqual("lost", running.final_state)
                self.assertEqual("lease_ended", running.final_reason)
                self.assertEqual("ended", controller.state["leases"][lease_id]["state"])
            finally:
                controller.journal.close()

            # A controller that restarts after the lease ended resolves the job.
            restarted = _hub(root)
            try:
                job = restarted.state["jobs"]["train"]
                self.assertEqual(("lost", "lease_ended"), (job["state"], job["reason"]))
                successor = job.get("successor_job_id")
                self.assertIsNotNone(successor)
                self.assertEqual("queued", restarted.state["jobs"][successor]["state"])
            finally:
                restarted.journal.close()

    def test_attach_command_from_the_inbox_leaves_a_receipt_with_its_lease(self) -> None:
        from scruffy.client import wait_for_command
        from scruffy.controller import _ingest_commands

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            _write_hub_queue(root, [])
            controller = _hub(root)
            try:
                request = client_attach_lease(root, "5001", request_id="attach-inbox")
                incarnation = AllocationIncarnation("5001", 0, LEASE_INVENTORY)
                with (
                    mock.patch(
                        "scruffy.lease_runtime.discover_slurm_allocation",
                        return_value=({node.name: node for node in LEASE_INVENTORY}, incarnation),
                    ),
                    mock.patch(
                        "scruffy.lease_runtime.lease_job",
                        return_value=_observation("5001", "interactive"),
                    ),
                ):
                    _ingest_commands(controller)
                receipt = wait_for_command(root, request["request_id"], timeout=1)
                self.assertEqual("attached", receipt["outcome"]["state"])
                self.assertIn(receipt["outcome"]["lease_id"], controller.state["leases"])
                self.assertEqual([], list(list_commands(root)))
            finally:
                controller.journal.close()

    def test_release_detaches_attached_lease_and_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            _write_hub_queue(root, [])
            controller = _hub(root)
            try:
                lease_id = str(self._attach(controller)["lease_id"])
                with group_commit(controller):
                    unknown = release_lease(controller, {"lease_id": "nope", "request_id": "r0"})
                    outcome = release_lease(controller, {"lease_id": "5001", "request_id": "r1"})
                self.assertEqual("rejected", unknown["state"])
                self.assertEqual(("releasing", lease_id), (outcome["state"], outcome["lease_id"]))
                with (
                    mock.patch("scruffy.lease_runtime.cancel_lease") as cancel,
                    group_commit(controller),
                ):
                    from scruffy.lease_runtime import advance_releases

                    advance_releases(controller)
                cancel.assert_not_called()
                lease = controller.state["leases"][lease_id]
                self.assertEqual(("ended", "operator"), (lease["state"], lease["reason"]))
            finally:
                controller.journal.close()

    def test_autoscaler_requests_a_pool_lease_and_places_work_once_it_runs(self) -> None:
        settings = _settings(gpus_per_node=2, cpus_per_node=8, memory_gb_per_node=64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            _write_hub_queue(root, [_queued_job(root, "burst", time_limit=3600)])
            controller = _hub(root, settings)
            try:
                with (
                    mock.patch("scruffy.lease_runtime.user_jobs", return_value=()),
                    group_commit(controller),
                ):
                    observe_leases(controller)
                with (
                    mock.patch("scruffy.lease_runtime.submit_lease", return_value="7001") as submit,
                    group_commit(controller),
                ):
                    plan_and_apply_leases(controller)
                submit.assert_called_once()
                (lease,) = controller.state["leases"].values()
                self.assertEqual(
                    ("pending", "7001", "h100"),
                    (lease["state"], lease["slurm_job_id"], lease["pool"]),
                )
                self.assertTrue((root / "leases" / "logs").is_dir())
                events = [event["kind"] for event in read_events(root)]
                self.assertLess(events.index("lease.requested"), events.index("lease.submitted"))

                incarnation = AllocationIncarnation("7001", 0, LEASE_INVENTORY)
                controller.last_lease_observation = 0.0
                with (
                    mock.patch(
                        "scruffy.lease_runtime.user_jobs",
                        return_value=(
                            _observation(
                                "7001", str(lease["lease_id"]), end_at=time.time() + 4 * 3600
                            ),
                        ),
                    ),
                    mock.patch(
                        "scruffy.lease_runtime.discover_slurm_allocation",
                        return_value=({node.name: node for node in LEASE_INVENTORY}, incarnation),
                    ),
                    group_commit(controller),
                ):
                    observe_leases(controller)
                self.assertEqual("running", lease["state"])
                with mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=_process()):
                    schedule(controller)
                self.assertEqual(lease["lease_id"], controller.state["jobs"]["burst"]["lease_id"])
            finally:
                controller.journal.close()


if __name__ == "__main__":
    unittest.main()
