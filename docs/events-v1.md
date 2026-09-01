# Workload reports and GPU health events v1

Scruffy jobs may publish bounded semantic updates into the same non-consuming
journal as queue lifecycle, GPU health, and output-reference events. Queue and
Slurm state remain authoritative for execution outcomes and GPU ownership; the
controller remains authoritative for health-based scheduling eligibility.

## Producer envelope

Every report is one JSON object with exactly these keys:

```json
{
  "v": 1,
  "event_id": "producer-unique-id",
  "job_id": "job-...",
  "occurred_at": "2026-08-03T12:34:56.789+00:00",
  "kind": "workload.progress",
  "source": {"name": "koochak", "node": "gpu-3"},
  "data": {
    "phase": "training",
    "completed": 12000,
    "total": 50000,
    "unit": "steps",
    "rate": 4.8,
    "metrics": {"loss": 1.42, "lr": 0.00017}
  }
}
```

Reports emitted by a running Scruffy worker should include the `launch_token`
in `source`; the worker receives it as `SCRUFFY_LAUNCH_TOKEN`. The controller
verifies a supplied token against the admitted launch identity for hot and
archived jobs. Reports without a token remain supported for external observers
because queue-root write access is the legacy trust boundary; a job ID alone is
not an authentication credential.

Allowed kinds are:

- `workload.phase`
- `workload.progress`
- `workload.milestone`
- `workload.artifact`
- `workload.notice`

The encoded envelope is limited to 64 KiB. It must contain only finite JSON
values. Producers must put raw logs, configurations, checkpoint bytes, and full
metric histories elsewhere and publish only bounded summaries or references.

### Artifact publication

An ordinary `workload.artifact` remains an observation. It gains scheduling
meaning only when `data.publication` has this exact shape:

```json
{
  "artifact_type": "checkpoint",
  "publication": {
    "v": 1,
    "artifact_id": "checkpoint/step000100000.pt",
    "path": "/shared/runs/step000100000.pt",
    "size_bytes": 1827364512,
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "manifest_path": "/shared/runs/step000100000.pt.ready.json"
  }
}
```

Both paths must be absolute, the digest must be lowercase SHA256, and the
producer must publish the ready manifest only after atomically installing and
syncing the immutable artifact. A matching `wait_for` condition is scoped to
the producer task's project and workflow. Scruffy records the exact evidence on
the consumer and emits `condition.satisfied`; it never reads or hashes artifact
bytes in the controller loop.

Complete atomic workflows must declare artifact waits before publication. Scruffy
retains exact truth for those declared conditions, but it is not an unbounded
catalog for arbitrary future late-attached consumers after bounded observations
expire.

`event_id` is an idempotency key scoped to one job. Reusing it with the same
job, kind, source, and data is safe; the first occurrence timestamp is retained,
so a retry may provide a newly generated timestamp. Reusing an ID for different
content is an error while its receipt is retained. Receipts span the active and
immediately previous journal generations. After that horizon, the same report
may be accepted again, so this is recent retry protection rather than a
permanent exactly-once ledger.

With `deduplicated: false`, a successful `scruffy report` or
`publish_event(...)` call means a new immutable report is durably spooled. With
`deduplicated: true`, a matching pending report or durable retained receipt was
found, so no new report was written. A retained rejection is a conflict, never a
successful deduplication. Neither successful result proves that the job exists
or that the controller has accepted and published the report yet. A transient
read error leaves the report pending for a later controller poll.

## Controller envelope

The controller validates and sequences a report, then adds its queue identity,
allocation identity, global sequence, recording timestamp, canonical event ID,
the authoritative job `project_id`, and the original `event_id` as
`source_event_id`. The project is derived from the admitted job; producers
cannot choose or change it. Every observer has an
independent cursor; reading never consumes an event for another observer.
Every accepted report is preserved in the retained journal generations, while
the bounded current projection compares producer timestamps and does not let a
late older report regress newer progress.

Reports accepted during one controller tick are appended as a batch, followed
by one journal sync and one cumulative snapshot commit. Observers see only data
at or before that committed snapshot watermark, never a partially committed
batch. The active and immediately previous journal generations are retained;
an observer cursor from any earlier generation resets to the current snapshot.

## Controller-owned GPU health events

GPU monitor samples are not workload reports and do not use the producer
envelope. Node-local samplers atomically replace their latest evidence under
`health/samples/`; the controller validates the allocation-incarnation
fingerprint and physical `(node, NVIDIA UUID)` identity before accepting it.
Periodic metric updates remain outside the journal.

The controller emits `resource.gpu_health_changed` only for health status
transitions and operator quarantine/clear/reprobe actions. Its `data.transitions`
contains the bounded changes and `data.gpu_health` contains the complete health
projection required for replay. An operator-command outcome also includes the
correlated `data.request_id`. Producers cannot publish this event kind or
directly change quarantine state.

## Authority

Workload events update `job.workload` in the current snapshot. They cannot
directly change lifecycle state, assignments, resources, exit codes, or terminal
results. A strict artifact publication can satisfy only an explicit workflow
condition declared before admission; arbitrary report data has no such power.
Only Scruffy's process and Slurm reconciliation path can do that.
Likewise, GPU evidence can change future placement only after the controller's
health policy accepts it; workload reports and Koochak observations have no
scheduling authority.
