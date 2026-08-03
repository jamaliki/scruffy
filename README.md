# Scruffy

Scruffy is a small, cooperative GPU queue that runs inside one multi-node Slurm
allocation. One controller divides the allocation while any number of humans or
agents submit work asynchronously and observe the same non-consuming event
journal.

It is named after the quiet caretaker from *Futurama*. This project is not
affiliated with the show or its owners.

> No two nonterminal jobs submitted through Scruffy own the same `(node, GPU)`.

This guarantee covers only work launched through Scruffy. Do not run out-of-band
GPU processes against its configured inventory.

## Requirements and boundaries

- Python 3.11 or newer, with no Python package dependencies.
- A POSIX shared filesystem visible at the same absolute `SCRUFFY_ROOT` on every
  client and compute node. Atomic rename and cluster-coherent `flock` are hard
  requirements. On Lustre, verify that distributed `flock` is enabled;
  `localflock` is insufficient. Scruffy cannot detect silently local-only locks.
- Slurm mode requires `srun`, `scancel`, and `scontrol --json` on the controller.
- Submitted working directories, executables, and referenced files must exist on
  every selected worker node.

GPU IDs are node-local values passed through `CUDA_VISIBLE_DEVICES`. CPU and
memory values are cooperative admission budgets, not physical isolation. The
queue trusts every process able to write `SCRUFFY_ROOT`; it is not a multi-user
security boundary.

## Quick start

Install into the Python environment available throughout the allocation:

```bash
python -m pip install .
scruffy --version
```

For a shared source checkout instead:

```bash
export PYTHONPATH=/shared/code/scruffy/src
python -m scruffy --version
```

Set one shared queue root and start the controller as the foreground payload of
the outer allocation. [`examples/scruffy.sbatch`](examples/scruffy.sbatch) is a
complete template.

```bash
export SCRUFFY_ROOT=/shared/runs/scruffy
scruffy --root "$SCRUFFY_ROOT" serve \
  --gpus-per-node 8 --cpus-per-node 112 --memory-gb-per-node 1024
```

Automatic inventory discovery reads `SLURM_JOB_NODELIST` and assumes every node
contributes the same contiguous GPU pool `0..N-1`. Use an explicit inventory for
selected, non-contiguous, or heterogeneous resources:

```bash
scruffy --root "$SCRUFFY_ROOT" serve --inventory inventory.json
```

Any process with access to the shared root can now submit without waiting for
the controller, dependencies, or free GPUs:

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --name probe \
  --request-id agent-a/campaign-a/probe-001 \
  --nodes 1 --gpus-per-node 1 \
  --cpus-per-node 14 --memory-gb-per-node 128 \
  -- python -m my_project.probe --config probe.yaml
```

Orient and monitor from any number of independent readers:

```bash
scruffy --root "$SCRUFFY_ROOT" summary --limit 20
scruffy --root "$SCRUFFY_ROOT" observe
scruffy --root "$SCRUFFY_ROOT" observe --after "$CURSOR" --wait 30
scruffy --root "$SCRUFFY_ROOT" observe --follow --output
```

`summary` is the bounded agent-oriented view and includes an `as_of_cursor`.
One-shot `observe` returns the full current snapshot, journal events after its
cursor, and a `next_cursor`. Each reader must persist its own opaque cursor and
continue immediately while `more` is true. `observe --follow` is the interactive
JSON Lines view. Cursors include a journal generation; compaction makes an older
generation return `reset: true`, at which point the reader rebuilds from the
returned snapshot. See [the client contract](docs/client-v1.md).

## Submission and resources

`submit` returns after its immutable request is durable. A stable `request_id`
is an idempotency key scoped to the queue: retrying the same specification
returns the original job ID, while changing it raises a conflict. Without a
request ID, every call creates a new job. Exact request idempotency survives hot
job eviction through a compact receipt in the archive.

CLI defaults are one node, one GPU, 14 CPUs per GPU, and 128 GB per GPU. Requests
are rectangular: every selected node receives the same resource shape and runs
the same argv once. Distributed rendezvous and application ranks remain the
workload's responsibility; Slurm rank and node variables are available.

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --nodes 2 --gpus-per-node 8 \
  --cpus-per-node 112 --memory-gb-per-node 512 \
  -- ./run_distributed.sh
```

The Python API uses the same JSON-compatible contracts:

```python
from pathlib import Path

from scruffy import ResourceRequest, submit_job

result = submit_job(
    Path("/shared/runs/scruffy"),
    argv=["/shared/env/bin/python", "train.py"],
    name="train",
    cwd=Path("/shared/code/project"),
    environment={},
    request=ResourceRequest(
        nodes=1,
        gpus_per_node=1,
        cpus_per_node=14,
        memory_gb_per_node=128,
    ),
    request_id="agent-a/experiment-7/train",
    workflow_id="experiment-7",
    task_id="train",
)
```

## Workflows

A task may depend on tasks that have not been submitted yet. All submissions
still return immediately.

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id infer \
  --needs train:succeeded -- python infer.py

scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id train -- python train.py
```

`succeeded` requires a successful upstream result. `terminal` accepts any
terminal result. Missing or unfinished dependencies leave a job `blocked`; an
unsatisfied `succeeded` edge makes it terminal `skipped`. Task IDs cannot contain
`:`, which keeps `--needs TASK[:CONDITION]` unambiguous. A succeeded task ID is
final. A failed, cancelled, lost, rejected, or skipped task ID may be reclaimed
by a new job with a new `request_id`; retry skipped dependants the same way.
Active duplicate task IDs, self-dependencies, and cycles are rejected. Use
`scruffy explain JOB_ID` for the resolved dependency state.

## Workload progress and output

Workers receive controller-owned `SCRUFFY_ROOT`, `SCRUFFY_JOB_ID`,
`SCRUFFY_EVENT_DIR`, and `SCRUFFY_NODE`. A workload can publish a bounded semantic
annotation without contacting the controller process:

```bash
scruffy report workload.progress \
  --event-id train/step-12000 \
  --source name=my-trainer \
  --data-json '{"phase":"training","step":12000,"metrics":{"loss":1.42}}'
```

Workload reports are capped at 64 KiB and cannot change lifecycle state, exit
status, assignments, or GPU ownership. Keep raw logs, metric histories,
configuration, and artifact bytes elsewhere. See
[the workload event contract](docs/events-v1.md).

Reports accepted in one controller tick are group-committed with one journal
sync and one cumulative snapshot. Report `event_id` receipts follow journal
retention: the active and immediately previous generations are retained, so
this key is for recent retry safety rather than permanent exactly-once delivery.

Raw output is stored once in per-job files. Journal events contain ordered byte
ranges; `observe --output` expands those references into text. For direct
diagnosis:

```bash
scruffy logs JOB_ID --stream stderr --tail 200 --follow
```

`--tail` reads at most the final 1 MiB per stream, including for newline-free or
carriage-return progress output.

## Lifecycle and operations

The snapshot is authoritative. Never infer success or failure from log text.
Terminal states are `succeeded`, `failed`, `cancelled`, `lost`, `rejected`, and
`skipped`.

```bash
scruffy status [JOB_ID]
scruffy explain JOB_ID
scruffy wait JOB_ID
scruffy cancel JOB_ID
scruffy drain
```

Cancellation and drain requests are asynchronous and return a `request_id` that
appears on their journal outcome. `drain` disables new launches for the current
allocation; running jobs continue and queued jobs remain durable. The drain
survives controller restarts and clears when a replacement allocation starts.

The hot snapshot keeps every nonterminal job and, after compaction, the newest
1,000 terminal jobs. Older terminal jobs remain addressable by job ID with
`archived: true`; compact records keep lifecycle and workflow metadata, but
resource requests, cwd, assignments, logs, argv, environment, and workload state
expire. `summary.counts` includes archived terminal jobs;
`summary.archived_jobs` and `status.archived_counts` expose the archived totals.
The active and immediately previous journal generations are retained.
Compact request receipts and workflow indexes last for the queue root's lifetime,
so archive storage grows as O(total jobs) small files and inodes.

Placement is deterministic best-fit with simple backfilling, but does not
guarantee fairness for a large queued request. Queued and dependency-blocked jobs
can survive a replacement allocation. Active jobs from a replaced allocation
become `lost` and are never retried automatically. Resource assignments remain
reserved whenever Slurm reconciliation is uncertain.

The controller deliberately runs one submitted argv per job. It is not a retry
engine, dynamic workflow fan-out engine, or artifact store: submit retries and
new tasks explicitly, and keep artifacts in their normal storage.

The shared queue stores argv, working directories, environment overrides, state,
events, and logs in plaintext. Protect its permissions and pass paths to secret
files rather than secret values.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m ruff check src tests
```

The local launcher exercises one-node lifecycle behavior without Slurm or GPUs:

```bash
scruffy --root /tmp/scruffy serve \
  --launcher local --inventory examples/local-inventory.json
```
