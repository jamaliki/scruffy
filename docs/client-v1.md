# Scruffy client contract v1

This document defines the JSON-compatible interface used by CLI and Python
clients. The controller snapshot is authoritative for lifecycle and resource
ownership. Journal events are ordered notifications; the snapshot may already
include the effects of returned events.

## Operations

| Purpose | CLI | Python |
| --- | --- | --- |
| Submit | `scruffy submit -- ...` | `submit_job(...)` |
| Full state or one job | `scruffy status [JOB_ID]` | `status(root, job_id=None)` |
| Bounded orientation | `scruffy summary` | `summary(root)` |
| Dependency explanation | `scruffy explain JOB_ID` | `explain(root, job_id)` |
| Snapshot and events | `scruffy observe` | `observe(root, ...)` |
| Wait for one terminal job | `scruffy wait JOB_ID` | `wait_for_job(root, job_id)` |
| Publish workload state | `scruffy report KIND` | `publish_event(...)` |
| Request cancellation | `scruffy cancel JOB_ID` | `cancel_job(root, job_id)` |
| Disable new launches | `scruffy drain` | `drain_queue(root)` |

All operations use `--root ROOT` or `SCRUFFY_ROOT`. Every participant must see
that directory at the same absolute path. CLI commands emit JSON except `logs`
and `observe --follow`, which are streaming interfaces.

## Job states

| State | Meaning | Holds resources | Terminal |
| --- | --- | :---: | :---: |
| `submitted` | Durable request not yet admitted by the controller | no | no |
| `queued` | Admitted and ready for placement | no | no |
| `blocked` | Waiting for a missing or unfinished dependency | no | no |
| `starting` | Placement reserved; launcher is starting | yes | no |
| `running` | Launcher step is confirmed active | yes | no |
| `finishing` | Process exited; output or Slurm reconciliation remains | yes | no |
| `cancelling` | Cancellation requested; release is not yet proven safe | yes | no |
| `succeeded` | Process exited successfully | no | yes |
| `failed` | Process or launch failed | no | yes |
| `cancelled` | Cancellation completed | no | yes |
| `lost` | Allocation or controller ended with unresolved work | no | yes |
| `rejected` | Request or workflow could not be admitted | no | yes |
| `skipped` | A `succeeded` dependency ended unsuccessfully | no | yes |

Never infer a terminal result from output text or workload progress.

## State views

`status(root)` returns the full current state, including durable `submitted`
requests not yet admitted by the controller. `status(root, job_id)` returns one
job or raises `KeyError` if it does not exist.

Important job fields are:

- `id`, `name`, `state`, and `submitted_at`.
- `request` and, while resources are held, `assignment`.
- `started_at`, `finished_at`, `exit_code`, `signal`, `reason`, and `error`.
- Optional `workflow_id`, `task_id`, `needs`, and `blockers`.
- Optional bounded `workload` projection.
- Optional `stdout` and `stderr` paths relative to the queue root.

The full job image may contain argv and environment overrides. Treat it as
sensitive. `summary(root, limit=20)` deliberately returns smaller job views,
grouped as `submitted`, `active`, `queued`, `blocked`, `requires_attention`, and
`recent_terminal`. It also returns exact state `counts`, node availability, and
an `as_of_cursor` suitable for starting incremental observation.

`explain(root, job_id)` returns the job, its resolved upstream job IDs and
states, current blockers, and a short explanation.

## Submission

A successful submission returns:

```json
{
  "job_id": "job-...",
  "state": "submitted",
  "deduplicated": false
}
```

The response means the immutable request is durable, not that a controller or
GPU is available. Submission never waits for admission, dependencies, earlier
jobs, or startup.

`request_id` is an idempotency key scoped to the queue. Reusing it with an
identical specification returns the same job ID with `deduplicated: true`.
Reusing it with a different specification raises `ConflictError`. Omitting it
creates a new job on every call.

CLI resource defaults are one node, one GPU, 14 CPUs per GPU, and 128 GB per
GPU. Python callers provide an explicit `ResourceRequest`.

## Observation and cursors

One-shot `observe` returns:

```json
{
  "snapshot": {"jobs": {}, "nodes": {}},
  "events": [],
  "next_cursor": "queue-...:42:12345",
  "latest_cursor": "queue-...:47:13789",
  "more": false,
  "reset": false
}
```

Cursors are opaque and private to one reader:

1. Start from `summary.as_of_cursor`, or call `observe` without `after` to begin
   at the current journal tail.
2. Process returned events in sequence order, then persist `next_cursor`.
3. While `more` is true, request the next page immediately with that cursor.
4. Otherwise, long-poll with `wait_seconds` or CLI `--wait`.
5. If `reset` is true, the cursor belongs to another queue. Rebuild from the
   returned full snapshot and save the new cursor.

Calling `observe` without a cursor returns current state but does not replay old
journal events. `latest_cursor` indicates the journal tail at response time.
`next_cursor == latest_cursor` with `more: false` means the reader is caught up.

`include_output=True` or CLI `--output` expands each `job.output` reference by
adding `data.text`. `observe --follow` maintains its cursor internally and emits
an initial snapshot followed by events as JSON Lines; use one-shot observation
when the caller must persist resumable state.

## Journal events

Controller events contain these common fields:

- `v`, `queue_id`, `seq`, `event_id`, `kind`, and `allocation_id`.
- `recorded_at`, the controller recording time.
- Optional `job_id`, complete `job` image, and event-specific `data`.

Lifecycle kinds use `allocation.*` and `job.*`. A `job.output` event has this
`data` object:

```json
{
  "job_id": "job-...",
  "stream": "stdout",
  "log": "jobs/job-.../stdout.log",
  "offset": 0,
  "length": 128
}
```

Workload events additionally preserve producer `occurred_at`, `source`, and
`source_event_id`; their contract is [events-v1.md](events-v1.md). Consumers
must ignore unknown fields and event kinds so compatible additions do not break
v1 readers.

## Commands and process exit

`cancel_job` and `drain_queue` durably spool a command and return immediately:

```json
{"job_id": "job-...", "request_id": "...", "state": "cancel_requested"}
```

```json
{"request_id": "...", "state": "drain_requested"}
```

The corresponding outcome event repeats `request_id`. Cancellation retains its
assignment until launcher exit, output closure, and Slurm reconciliation prove
release safe.

Drain disables new launches for the lifetime of the current controller. Running
jobs continue and queued jobs remain durable; restart the controller to resume.

CLI exit conventions are:

- `0` for a successful client operation and for `wait` on a succeeded job.
- `1` for `wait` on a non-success terminal state without a process exit code.
- `1..125` for `wait` on a failed process, derived from its exit code.
- `2` for invalid input, unknown jobs, conflicts, storage errors, or wait timeout.
- `130` when Ctrl-C reaches a client command such as `observe --follow`.
  `serve` handles SIGINT/SIGTERM as a graceful shutdown and returns `0`.

Python callers should expect `ValueError` for invalid input, `KeyError` for an
unknown job, `ConflictError` for idempotency-key reuse, and `TimeoutError` from a
timed-out `wait_for_job`.

## Trust and durability

Clients only create immutable requests, commands, and workload reports. The
controller alone assigns resources and appends globally sequenced events. Queue
contents, argv, environment overrides, output, and annotations are plaintext;
protect filesystem access and pass secret-file paths rather than secret values.
