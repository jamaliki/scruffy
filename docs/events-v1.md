# Workload events v1

Scruffy jobs may publish bounded semantic updates into the same non-destructive
journal as queue lifecycle and output-reference events. Queue and Slurm state
remain authoritative for execution outcomes and GPU ownership.

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

Allowed kinds are:

- `workload.phase`
- `workload.progress`
- `workload.milestone`
- `workload.artifact`
- `workload.notice`

The encoded envelope is limited to 64 KiB. It must contain only finite JSON
values. Producers must put raw logs, configurations, checkpoint bytes, and full
metric histories elsewhere and publish only bounded summaries or references.

`event_id` is an idempotency key scoped to one job. Reusing it with the same
job, kind, source, and data is safe; the first occurrence timestamp is retained,
so a retry may provide a newly generated timestamp. Reusing an ID for different
content is an error.

## Controller envelope

The controller validates and sequences a report, then adds its queue identity,
allocation identity, global sequence, recording timestamp, canonical event ID,
and the original `event_id` as `source_event_id`. Every observer has an
independent cursor; reading never consumes an event for another observer.
Every report is preserved in the journal, while the bounded current projection
compares producer timestamps and does not let a late older report regress newer
progress.

## Authority

Workload events update only `job.workload` in the current snapshot. They cannot
change lifecycle state, assignments, resources, exit codes, or terminal results.
Only Scruffy's process and Slurm reconciliation path can do that.
