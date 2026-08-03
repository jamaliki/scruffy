# Scruffy agent protocol

Scruffy is a cooperative GPU queue inside one multi-node Slurm allocation. The
canonical response, state, cursor, and exit-code contract is
[`docs/client-v1.md`](docs/client-v1.md).

## Operating loop

```bash
export SCRUFFY_ROOT=/shared/run/scruffy

# 1. Orient and save the returned as_of_cursor.
scruffy summary --limit 20

# 2. Submit durably without waiting for resources or dependencies.
scruffy submit \
  --request-id agent-name/campaign/task/attempt-1 \
  --gpus-per-node 1 -- command arg...

# 3. Poll with this agent's private cursor; persist next_cursor each time.
scruffy observe --after "$CURSOR" --wait 30

# 4. Diagnose only when needed.
scruffy explain JOB_ID
scruffy logs JOB_ID --stream stderr --tail 200
```

While `observe.more` is true, request the next page immediately. If
`observe.reset` is true, discard the old local view, accept the returned full
hot snapshot, and save the returned cursor. Reset is expected when compaction
makes the cursor's journal generation stale. Never share cursor state between
agents; reading does not consume events for anyone else. Use
`observe --follow --output` only for interactive streaming.

## Rules

- Use a stable, queue-wide `request_id` for every retryable submission. A useful
  convention is `<agent>/<campaign>/<task>/<attempt>`. Identical reuse is safe;
  different reuse raises `ConflictError`. This exact identity survives detailed
  job eviction through a compact archive receipt.
- Commands are argv after `--`. Avoid nested shell strings; submit `bash -lc`
  explicitly only when shell behavior is required.
- `cwd`, executables, and referenced files must exist at the same paths on the
  selected worker nodes.
- Workflow tasks use `--workflow-id`, `--task-id`, and repeated
  `--needs TASK[:succeeded|terminal]`. Dependencies may be submitted later by
  another agent. Task IDs cannot contain `:`. A succeeded task identity is
  final; after any other terminal result, retry it with a new `request_id` in
  the same workflow.
- Prefer `summary` for bounded orientation, `explain` for one dependency chain,
  and `observe` for incremental monitoring.
- Hot state keeps all nonterminal jobs and, after compaction, the newest 1,000
  terminal jobs. Older lookups carry `archived: true` and retain lifecycle and
  workflow metadata, but resource requests, cwd, assignments, logs, argv,
  environment, and workload expire. `summary.counts` includes them and
  `archived_jobs` reports their total.
- Compact request receipts and workflow indexes persist for the queue root's
  lifetime, so they grow as O(total jobs) small files and inodes.
- The snapshot is authoritative. Never infer lifecycle success or failure from
  stdout, stderr, or a workload annotation.
- `blocked` means an upstream task is missing or unfinished. `skipped` is
  terminal and means a required successful dependency ended unsuccessfully.
- Match asynchronous cancel and drain outcomes using the returned `request_id`.
  `drain` survives controller restarts and disables launches until the outer
  allocation is replaced.
- GPU identity is `(node, gpu_id)`, never a bare global ordinal.
- `starting`, `running`, `finishing`, and `cancelling` jobs hold their resources.
- CPU and memory are cooperative budgets. GPU exclusivity covers only work
  submitted through Scruffy; do not launch out-of-band work on its inventory.
- `SCRUFFY_ROOT` must provide atomic rename and cluster-coherent `flock` across
  every node. Lustre `localflock` is not sufficient for a multi-node queue.
- `SCRUFFY_ROOT`, `SCRUFFY_JOB_ID`, `SCRUFFY_EVENT_DIR`, and `SCRUFFY_NODE` are
  controller-owned inside a worker.
- Workload reports belong in `scruffy report` or `scruffy.publish_event`; keep
  detailed telemetry and artifact bytes in their normal stores. Reports from
  one controller tick are committed with one journal sync and snapshot.
- A workload report `event_id` deduplicates only across the active and previous
  journal generations. Do not use it as a permanent exactly-once record.
- Scruffy does not create retries, dynamically fan out workflow tasks, or store
  artifacts. Agents must submit those jobs explicitly and retain artifacts
  elsewhere.
- Queue state contains argv and environment overrides in plaintext. Reference
  protected secret files instead of submitting secret values.
