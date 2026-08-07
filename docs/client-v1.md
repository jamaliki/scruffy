# Scruffy client contract v1

This document defines the JSON-compatible interface used by CLI and Python
clients. The controller snapshot is authoritative for lifecycle and resource
ownership. Journal events are ordered notifications; the snapshot may already
include the effects of returned events.

## Operations

| Purpose | CLI | Python / MCP |
| --- | --- | --- |
| Submit | `scruffy submit -- ...` | `submit_job(...)` |
| Full state or one job | `scruffy status [JOB_ID]` | `status(root, job_id=None)` |
| Bounded orientation | `scruffy summary` | `summary(root)` |
| Resource availability | `scruffy resources` | MCP `resources()` |
| Resource queue | `scruffy queue` | MCP `queue()` |
| Running jobs | `scruffy running` | MCP `running_jobs()` |
| Dependency-blocked jobs | `scruffy blocked` | MCP `blocked_jobs()` |
| Dependency explanation | `scruffy explain JOB_ID` | `explain(root, job_id)` |
| Snapshot and events | `scruffy observe` | `observe(root, ...)` |
| Wait for one terminal job | `scruffy wait JOB_ID` | `wait_for_job(root, job_id)` |
| Publish workload state | `scruffy report KIND` | `publish_event(...)` |
| Request cancellation | `scruffy cancel JOB_ID` | `cancel_job(root, job_id)` |
| Disable new launches | `scruffy drain` | `drain_queue(root)` |
| Resume after recovery | `scruffy resume` | `resume_queue(root)` |

All operations use `--root ROOT` or `SCRUFFY_ROOT`. Every participant must see
that directory at the same absolute path. CLI commands emit JSON except `logs`
and `observe --follow`, which are streaming interfaces.

Submission, full-state views, summaries, explanations, and observations accept
a project selector through `--project`, `SCRUFFY_PROJECT`, or the Python
`project_id` argument. Submission defaults to `default`; read operations remain
allocation-wide when no project is selected.

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

`status(root)` returns the full hot state, including durable `submitted`
requests not yet admitted by the controller. `status(root, job_id)` also looks
up compactly archived terminal jobs, and raises `KeyError` if the ID does not
exist. An archived result carries `"archived": true` and has the reduced field
set described under Retention.

Important hot-job fields are:

- `id`, `name`, `state`, and `submitted_at`.
- `project_id`, with legacy records normalized to `default`.
- `request` and, while resources are held, `assignment`.
- `started_at`, `finished_at`, `exit_code`, `signal`, `reason`, and `error`.
- Optional `workflow_id`, `task_id`, `needs`, and `blockers`.
- Optional bounded `workload` projection.
- Optional `stdout` and `stderr` paths relative to the queue root.

The hot job image may contain argv and environment overrides. Treat it as
sensitive. `summary(root, limit=20)` deliberately returns smaller job views,
grouped as `submitted`, `active`, `queued`, `blocked`, `requires_attention`, and
`recent_terminal`. It also returns exact state `counts`, including archived
terminal jobs, an `archived_jobs` total, node availability, and an
`as_of_cursor` suitable for starting incremental observation.

The focused CLI and MCP views return that same cursor with compact job
identities. `queue` contains `submitted` and `queued`; `running` contains every
resource-holding active state; `blocked` contains only dependency-blocked jobs.
`resources` reports aggregate and per-node free and total GPU, CPU, and memory
capacity without assignment details. Paginate job views while `more` is true,
then inspect only the selected job that needs detail.

`explain(root, job_id)` returns the job, its resolved upstream job IDs and
states, current blockers, and a short explanation. Compact archived job and
workflow metadata remains sufficient for older terminal-job lookup and
dependency explanation.

## Submission

A successful submission returns:

```json
{
  "job_id": "job-...",
  "project_id": "koochak",
  "state": "submitted",
  "deduplicated": false
}
```

With `deduplicated: false`, the response means a new immutable request is
durable. With `deduplicated: true`, an identical pending request or persistent
identity receipt was found; `state: "submitted"` is an acknowledgement, not the
job's current lifecycle state. Query `status` when that distinction matters.
Submission never waits for admission, dependencies, earlier jobs, or startup.

`request_id` is an idempotency key scoped to the project. Reusing it with an
identical specification returns the same job ID with `deduplicated: true`.
Reusing it with a different specification raises `ConflictError`. Omitting it
creates a new job on every call. Its exact identity digest is retained in the
compact archive even after detailed job state expires.

A transient I/O failure while the controller reads an immutable request defers
admission to a later poll without deleting the request or consuming its ID.
Decodable but invalid requests, including a `job_id` that disagrees with its
directory, are rejected and permanently consume that ID.

CLI resource defaults are one node, one GPU, 14 CPUs per GPU, and 128 GB per
GPU. Python callers provide an explicit `ResourceRequest`.

Workflow identity is `(project_id, workflow_id, task_id)`; dependencies only
resolve inside that project. Task IDs cannot contain `:`. A succeeded task
identity remains unique for the workflow and project. A terminal non-success attempt (`failed`, `cancelled`, `lost`,
`rejected`, or `skipped`) may be replaced by a new job using the same
`workflow_id` and `task_id` with a new `request_id`. Resolution and explanation
use the newest valid attempt. Scruffy does not retry skipped dependants
automatically; submit their next attempts explicitly.

## Observation and cursors

One-shot `observe` returns:

```json
{
  "snapshot": {"jobs": {}, "nodes": {}},
  "events": [],
  "next_cursor": "queue-...:3:42:12345",
  "latest_cursor": "queue-...:3:47:13789",
  "more": false,
  "reset": false
}
```

Cursors are opaque and private to one reader:

1. Start from `summary.as_of_cursor`, or call `observe` without `after` to begin
   at the current committed journal tail.
2. Process returned events in sequence order, then persist `next_cursor`.
3. While `more` is true, request the next page immediately with that cursor.
4. Otherwise, long-poll with `wait_seconds` or CLI `--wait`.
5. If `reset` is true, the cursor belongs to another queue or to an expired
   journal generation. Rebuild from the returned full hot snapshot and save the
   new cursor.

Calling `observe` without a cursor returns current state but does not replay old
journal events. `latest_cursor` indicates the committed journal tail at response time.
`next_cursor == latest_cursor` with `more: false` means the reader is caught up.
The active and immediately previous journal generations are retained, but
observation never replays across a generation boundary; a stale cursor resets.

`include_output=True` or CLI `--output` expands each `job.output` reference by
adding `data.text`. `observe --follow` maintains its cursor internally and emits
an initial snapshot followed by events as JSON Lines. It emits another snapshot
after a cursor reset. Use one-shot observation when the caller must persist
resumable state.

A project-filtered observer receives job events only for that project, while
allocation-wide events remain visible. Its cursor still advances over every
suppressed global-journal record, so unrelated project traffic is not replayed.
Node capacity and free resources remain allocation-wide.

## Journal events

Controller events contain these common fields:

- `v`, `queue_id`, `seq`, `event_id`, `kind`, and `allocation_id`.
- `recorded_at`, the controller recording time.
- Optional `job_id`, its authoritative `project_id`, complete `job` image, and
  event-specific `data`.

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
v1 readers. A contained per-item storage failure is published as a `notice`
whose `data.kind` is `storage.item_skipped`; the controller continues serving
unrelated work.

## Retention

After compaction, hot state contains every nonterminal job and the newest 1,000
terminal jobs. Older terminal jobs move to records marked `archived: true`.
These retain identity, lifecycle results and timestamps, and workflow metadata;
they drop the resource request, cwd, argv, environment, assignments, blockers,
workload projection, output paths, and per-job logs. The state exposes per-state
`archived_counts`; `summary.counts` combines these with hot counts, while
detailed summary lists and unqualified `status(root)` remain hot views.

Journal history and workload-report idempotency receipts retain the active and
immediately previous generations. Request idempotency is different: its compact
receipt persists after the full job image is evicted.

Compact request receipts and workflow archive entries persist for the lifetime
of the queue root. They therefore grow as O(total jobs) small files and inodes,
even though retained terminal-job detail and journal history are bounded.

## Commands and process exit

`cancel_job`, `drain_queue`, and `resume_queue` durably spool a command and
return immediately:

```json
{"job_id": "job-...", "request_id": "...", "state": "cancel_requested"}
```

```json
{"request_id": "...", "state": "drain_requested"}
```

```json
{"request_id": "...", "state": "resume_requested"}
```

The corresponding outcome event repeats `request_id`. Cancellation retains its
assignment until launcher exit, output closure, and Slurm reconciliation prove
release safe. Cancelling any terminal job, including an archived one, produces
`job.cancel_ignored` rather than `command.rejected`.

Drain disables new launches for the current allocation. Running jobs continue,
queued jobs remain durable, and controller restarts preserve the drain. A
replacement allocation clears it.

A same-allocation controller restart reattaches existing Slurm steps but starts
with new launches paused. Queued jobs and dependency transitions remain durable
without consuming resources. `scruffy resume` asynchronously clears this
recovery pause after the operator has checked the snapshot; it never overrides
an explicit drain. Its outcome is `allocation.launches_resumed` or
`allocation.resume_ignored`, correlated by `request_id`.

CLI exit conventions are:

- `0` for a successful client operation and for `wait` on a succeeded job.
- `1` for `wait` on a non-success terminal state without a process exit code.
- `1..125` for `wait` on a failed process, derived from its exit code.
- `2` for invalid input, unknown jobs, conflicts, storage errors, or wait timeout.
- `130` when Ctrl-C reaches a client command such as `observe --follow`.
  `serve` handles SIGINT/SIGTERM as a graceful shutdown and returns `0`. In
  Slurm mode it leaves worker steps running for the next controller to reattach;
  replacing the outer allocation still makes unresolved active jobs `lost`.

Python callers should expect `ValueError` for invalid input, `KeyError` for an
unknown job, `ConflictError` for idempotency-key reuse, and `TimeoutError` from a
timed-out `wait_for_job`.

## Trust and durability

Clients only create immutable requests, commands, and workload reports. The
controller alone assigns resources and appends globally sequenced events. Queue
contents, argv, environment overrides, output, and annotations are plaintext.
Protect filesystem access and pass secret-file paths rather than secret values.

The shared filesystem must provide atomic rename and cluster-coherent `flock`
across all nodes. Lustre `localflock` is not sufficient, and Scruffy cannot
detect a filesystem that silently treats these locks as node-local.

The controller deliberately executes submitted jobs; it does not invent
retries, dynamically fan out workflow tasks, or store artifact bytes. Clients
must submit those jobs explicitly and keep artifacts elsewhere.
