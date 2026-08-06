from __future__ import annotations

import unittest

from scruffy.workflows import (
    WorkflowError,
    resolve_blocked_jobs,
    resolve_dependencies,
    select_task_attempts,
    validate_workflows,
)


def task(
    task_id: str,
    *,
    workflow_id: str = "pipeline",
    project_id: str = "default",
    state: str = "queued",
    needs: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "state": state,
        "needs": [] if needs is None else needs,
    }


def need(task_id: str, condition: str = "succeeded") -> dict[str, str]:
    return {"task_id": task_id, "condition": condition}


class WorkflowValidationTests(unittest.TestCase):
    def test_standalone_jobs_and_empty_workflow_are_valid(self) -> None:
        jobs = [{"state": "queued"}, task("root")]

        validate_workflows(jobs)

        self.assertEqual(
            [resolve_dependencies(job, jobs) for job in jobs],
            [
                {"decision": "ready", "reason": None, "blockers": []},
                {"decision": "ready", "reason": None, "blockers": []},
            ],
        )

    def test_identifiers_must_be_paired_nonempty_and_trimmed(self) -> None:
        invalid_jobs = (
            {"workflow_id": "flow"},
            {"task_id": "one"},
            {"workflow_id": None, "task_id": "one"},
            {"workflow_id": "flow", "task_id": ""},
            {"workflow_id": " flow", "task_id": "one"},
        )

        for job in invalid_jobs:
            with self.subTest(job=job), self.assertRaises(WorkflowError):
                validate_workflows([job])

        with self.assertRaisesRegex(WorkflowError, "must not contain ':'"):
            validate_workflows([task("eval:terminal")])

    def test_standalone_job_cannot_have_dependencies(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "cannot declare needs"):
            validate_workflows([{"needs": [need("root")]}])

    def test_needs_shape_and_condition_are_strictly_validated(self) -> None:
        malformed_needs: tuple[object, ...] = (
            None,
            "root",
            ["root"],
            [{"task_id": "root"}],
            [{"task_id": "root", "condition": "ready"}],
            [{"task_id": "root", "condition": "succeeded", "extra": True}],
        )

        for raw_needs in malformed_needs:
            job = task("child")
            job["needs"] = raw_needs
            with self.subTest(needs=raw_needs), self.assertRaises(WorkflowError):
                validate_workflows([task("root"), job])

    def test_task_ids_are_unique_within_but_not_across_workflows(self) -> None:
        validate_workflows([task("train", workflow_id="a"), task("train", workflow_id="b")])
        validate_workflows(
            [task("train", project_id="a"), task("train", project_id="b")]
        )

        with self.assertRaisesRegex(WorkflowError, "duplicate task_id"):
            validate_workflows([task("train"), task("train")])

    def test_missing_and_cross_workflow_dependencies_remain_open(self) -> None:
        missing = task("child", needs=[need("absent")])
        validate_workflows([missing])
        self.assertEqual(
            "dependency_missing",
            resolve_dependencies(missing, [missing])["blockers"][0]["reason"],
        )

        cross_workflow = task("child", workflow_id="a", needs=[need("root")])
        root_elsewhere = task("root", workflow_id="b")
        jobs = [cross_workflow, root_elsewhere]
        validate_workflows(jobs)
        self.assertEqual(
            "dependency_missing",
            resolve_dependencies(cross_workflow, jobs)["blockers"][0]["reason"],
        )

    def test_self_and_duplicate_edges_are_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowError, "depend on itself"):
            validate_workflows([task("same", needs=[need("same")])])

        duplicate = task(
            "child",
            needs=[need("root", "succeeded"), need("root", "terminal")],
        )
        with self.assertRaisesRegex(WorkflowError, "duplicate dependency"):
            validate_workflows([task("root"), duplicate])

    def test_direct_and_indirect_cycles_are_rejected(self) -> None:
        direct = [task("a", needs=[need("b")]), task("b", needs=[need("a")])]
        indirect = [
            task("a", needs=[need("b")]),
            task("b", needs=[need("c")]),
            task("c", needs=[need("a")]),
        ]

        for jobs in (direct, indirect):
            with self.subTest(jobs=jobs), self.assertRaisesRegex(
                WorkflowError, "dependency cycle"
            ):
                validate_workflows(jobs)

    def test_terminal_task_edges_do_not_keep_a_repaired_graph_cyclic(self) -> None:
        cancelled = task("a", state="cancelled", needs=[need("b")])
        replacement = task("b", needs=[need("a")])

        validate_workflows([cancelled, replacement])

        self.assertEqual(
            "skipped",
            resolve_dependencies(replacement, [cancelled, replacement])["decision"],
        )

    def test_task_past_its_dependency_gate_does_not_race_retry_validation(self) -> None:
        running = {
            **task("b", state="running", needs=[need("a")]),
            "dependency_gate_passed": True,
        }
        retry = task("a", needs=[need("b", "terminal")])

        validate_workflows([running, retry])


class WorkflowResolutionTests(unittest.TestCase):
    def test_newest_valid_retry_shadows_failed_and_invalid_attempts(self) -> None:
        attempts = [
            {**task("train", state="failed"), "queue_order": 1},
            {
                **task("train", state="rejected"),
                "queue_order": 2,
                "workflow_invalid": True,
            },
            {**task("train", state="running"), "queue_order": 3},
        ]

        selected = select_task_attempts(attempts)

        self.assertIs(attempts[2], selected[("default", "pipeline", "train")])

    def test_batch_resolver_reaches_a_fixed_point_in_topological_order(self) -> None:
        jobs = [
            task("consumer", state="blocked", needs=[need("cleanup")]),
            task(
                "cleanup",
                state="blocked",
                needs=[need("child", "terminal")],
            ),
            task("child", state="blocked", needs=[need("root")]),
            task("root", state="failed"),
        ]

        resolutions = resolve_blocked_jobs(jobs)

        self.assertEqual(
            ["child", "cleanup", "consumer"],
            [task_id for _, _, task_id in resolutions],
        )
        self.assertEqual(
            "skipped", resolutions[("default", "pipeline", "child")]["decision"]
        )
        self.assertEqual(
            "ready", resolutions[("default", "pipeline", "cleanup")]["decision"]
        )
        self.assertEqual(
            "blocked", resolutions[("default", "pipeline", "consumer")]["decision"]
        )
        self.assertEqual(
            "queued",
            resolutions[("default", "pipeline", "consumer")]["blockers"][0][
                "state"
            ],
        )

    def test_dependencies_never_cross_project_boundaries(self) -> None:
        child = task("infer", project_id="project-a", needs=[need("train")])
        other = task("train", project_id="project-b", state="succeeded")

        resolution = resolve_dependencies(child, [child, other])

        self.assertEqual("blocked", resolution["decision"])
        self.assertEqual("dependency_missing", resolution["blockers"][0]["reason"])

    def test_open_workflow_treats_not_yet_submitted_task_as_pending(self) -> None:
        child = task(
            "infer",
            workflow_id="workflow-a",
            needs=[{"task_id": "train", "condition": "succeeded"}],
        )

        validate_workflows([child])
        resolution = resolve_dependencies(child, [child])

        self.assertEqual("blocked", resolution["decision"])
        self.assertEqual(
            {
                "task_id": "train",
                "condition": "succeeded",
                "state": "missing",
                "reason": "dependency_missing",
            },
            resolution["blockers"][0],
        )

    def test_pending_succeeded_dependency_blocks(self) -> None:
        root = task("root", state="running")
        child = task("child", needs=[need("root")])

        self.assertEqual(
            resolve_dependencies(child, [root, child]),
            {
                "decision": "blocked",
                "reason": None,
                "blockers": [
                    {
                        "task_id": "root",
                        "condition": "succeeded",
                        "state": "running",
                        "reason": "dependency_pending",
                    }
                ],
            },
        )

    def test_succeeded_dependency_becomes_ready_only_on_success(self) -> None:
        successful = task("root", state="succeeded")
        failed = task("root", state="failed")
        child = task("child", needs=[need("root")])

        self.assertEqual(
            resolve_dependencies(child, [successful, child])["decision"], "ready"
        )
        self.assertEqual(
            resolve_dependencies(child, [failed, child]),
            {
                "decision": "skipped",
                "reason": "dependency_unsatisfied",
                "blockers": [
                    {
                        "task_id": "root",
                        "condition": "succeeded",
                        "state": "failed",
                        "reason": "dependency_unsatisfied",
                    }
                ],
            },
        )

    def test_all_non_success_terminal_states_make_succeeded_unsatisfied(self) -> None:
        child = task("child", needs=[need("root")])

        for state in ("failed", "cancelled", "lost", "rejected", "skipped"):
            with self.subTest(state=state):
                root = task("root", state=state)
                resolution = resolve_dependencies(child, [root, child])
                self.assertEqual(resolution["decision"], "skipped")
                self.assertEqual(resolution["reason"], "dependency_unsatisfied")

    def test_terminal_dependency_accepts_success_or_failure_but_not_running(self) -> None:
        child = task("child", needs=[need("root", "terminal")])

        for state in ("succeeded", "failed", "cancelled", "skipped"):
            with self.subTest(state=state):
                root = task("root", state=state)
                self.assertEqual(
                    resolve_dependencies(child, [root, child])["decision"], "ready"
                )

        running = task("root", state="running")
        self.assertEqual(
            resolve_dependencies(child, [running, child])["decision"], "blocked"
        )

    def test_unsatisfied_dependency_wins_over_another_pending_dependency(self) -> None:
        failed = task("failed", state="failed")
        running = task("running", state="running")
        child = task("child", needs=[need("failed"), need("running")])

        resolution = resolve_dependencies(child, [failed, running, child])

        self.assertEqual(resolution["decision"], "skipped")
        self.assertEqual(
            [blocker["reason"] for blocker in resolution["blockers"]],
            ["dependency_unsatisfied", "dependency_pending"],
        )

    def test_blocker_order_matches_declared_dependency_order(self) -> None:
        first = task("first", state="queued")
        second = task("second", state="starting")
        child = task("child", needs=[need("second"), need("first", "terminal")])

        blockers = resolve_dependencies(child, [first, second, child])["blockers"]

        self.assertEqual([item["task_id"] for item in blockers], ["second", "first"])

    def test_resolution_target_must_belong_to_the_validated_jobs(self) -> None:
        outsider = task("outsider")

        with self.assertRaisesRegex(WorkflowError, "not present in jobs"):
            resolve_dependencies(outsider, [task("root")])

    def test_generator_input_is_supported(self) -> None:
        jobs = [task("root", state="succeeded"), task("child", needs=[need("root")])]

        validate_workflows(job for job in jobs)
        resolutions = [resolve_dependencies(job, jobs) for job in jobs]

        self.assertEqual([item["decision"] for item in resolutions], ["ready", "ready"])


if __name__ == "__main__":
    unittest.main()
