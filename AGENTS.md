# Scruffy agent guide

Scruffy is a cooperative GPU queue inside one multi-node Slurm allocation.

## Safe agent workflow

```bash
export SCRUFFY_ROOT=/shared/run/scruffy

scruffy submit --request-id "$UNIQUE_REQUEST_ID" --gpus-per-node 1 -- command arg...
scruffy summary        # bounded current state, progress, failures, and blockers
scruffy explain JOB_ID # one job plus its resolved dependency states
scruffy observe --after "$CURSOR" --wait 30
scruffy logs JOB_ID --tail 200
scruffy cancel JOB_ID
```

- `submit` returns after the immutable request is durable. It never waits for a
  controller, free resources, or another job.
- Always use a stable, globally unique `--request-id` when a submission may be
  retried. Reusing it with a different specification is an error.
- Commands are argv arrays after `--`. Do not construct nested shell strings;
  explicitly submit `bash -lc ...` only when shell behavior is genuinely needed.
- Every observer owns its cursor. Reading events never consumes them for another
  agent.
- Submit workflow tasks with `--workflow-id`, `--task-id`, and repeated
  `--needs TASK[:succeeded|terminal]`. Submission never waits for dependencies,
  and dependencies may be submitted later by another agent.
- `blocked` means an upstream task is pending or missing. `skipped` is terminal
  and means a required successful dependency ended unsuccessfully.
- Prefer `summary` for orientation, `explain` for one chain, and `observe` with
  your own saved cursor for incremental monitoring.
- Match asynchronous cancel and drain outcomes by the returned `request_id`.
- Treat the snapshot as current truth and lifecycle events as notifications.
  Never infer job success or failure from log text.

## Workload messages

- Inside a worker, `SCRUFFY_ROOT`, `SCRUFFY_JOB_ID`, `SCRUFFY_EVENT_DIR`, and
  `SCRUFFY_NODE` are controller-owned. Do not override or redirect them.
- Publish bounded semantic state through `scruffy report` or
  `scruffy.publish_event`; keep detailed telemetry and artifacts in their normal
  stores. Use a stable event ID when retrying.
- Workload events are annotations only. The queue lifecycle and assignment in
  the snapshot remain authoritative.

## Resource invariants

- There is exactly one controller for a queue.
- GPU identities are `(node, gpu_id)`, never bare global ordinals.
- A rectangular multi-node placement is reserved completely before launch.
- `starting`, `running`, `finishing`, and `cancelling` jobs hold their resources.
- Local resources are released only after the launcher exits and both output
  streams close. Slurm resources additionally require a fresh successful
  `scontrol` snapshot proving the named step is absent.
- CPU and memory are cooperative admission budgets. GPU IDs are the exclusive
  units managed by the queue API.
- Scruffy guarantees non-overlap only for work submitted through Scruffy. Do not
  launch out-of-band GPU work against its configured inventory.

The shared queue contains argv and environment overrides in plaintext. Do not
submit secret values directly; reference protected files instead.
