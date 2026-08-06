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

- Python 3.11 or newer. The base package has no Python dependencies; the
  optional MCP server uses the official Python MCP SDK.
- A POSIX shared filesystem visible at the same absolute `SCRUFFY_ROOT` on every
  client and compute node. Atomic rename and cluster-coherent `flock` are hard
  requirements. On Lustre, verify that distributed `flock` is enabled;
  `localflock` is insufficient. Scruffy cannot detect silently local-only locks.
- Slurm mode requires `srun`, `scancel`, and `scontrol --json` on the controller.
- Submitted working directories, executables, and referenced files must exist on
  every selected worker node.

GPU IDs are node-local values passed through `CUDA_VISIBLE_DEVICES`. CPU and
memory values are cooperative admission budgets, not physical isolation. The
Slurm worker step sees Scruffy's managed node CPU pool so overlapping jobs are
not all pinned to the same first CPU slice. The workload should still size its
own workers and thread pools to the CPU budget it requested. Scruffy trusts
every process able to write `SCRUFFY_ROOT`; it is not a multi-user security
boundary.

## Quick start

From a checkout, `uv` needs no separate installation step:

```bash
uv run scruffy --version
```

For an environment shared throughout an allocation, install the package there:

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
scruffy --root "$SCRUFFY_ROOT" serve
```

Automatic inventory discovery reads the resources granted to the outer Slurm
allocation. Per-node resource flags can cap that managed capacity. Discovery
assumes homogeneous nodes and a contiguous GPU pool `0..N-1`; use an explicit
inventory for selected, non-contiguous, or heterogeneous resources:

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

## Projects

Projects isolate names and monitoring without dividing the allocation. There is
still one controller, one global resource ledger, and one scheduling queue, so
GPU exclusivity holds across every project. A project is not an access-control,
quota, or fairness boundary.

Set `--project` on submission and observation, or export `SCRUFFY_PROJECT` for
an agent session:

```bash
export SCRUFFY_PROJECT=koochak
scruffy --root "$SCRUFFY_ROOT" submit \
  --request-id experiment-7/train -- python train.py
scruffy --root "$SCRUFFY_ROOT" summary
scruffy --root "$SCRUFFY_ROOT" observe --after "$CURSOR" --wait 30
```

`request_id` and `(workflow_id, task_id)` identities are local to a project.
Dependencies can only resolve inside that same project. Project-filtered
readers advance their global cursor across other projects' events, so unrelated
traffic is neither returned nor repeatedly scanned. Allocation-wide events
still appear because they can affect every project. Omit the project selector
for an allocation-wide administrative view. Existing jobs without a project
belong to `default`.

## Web dashboard

`scruffy dashboard` opens a read-only allocation console on loopback. It shows
physical GPU ownership, CPU and memory reservations, project scopes, queue
lanes, failures, recent finishes, and compact job/dependency details. The UI
uses the same bounded projections as MCP and never returns argv, working
directories, or environment values.

For a locally visible queue root:

```bash
uv run scruffy --root "$SCRUFFY_ROOT" dashboard
```

For a remote queue, run the web server locally and isolate each read through
the existing SSH gateway:

```bash
uv run scruffy --root /shared/runs/scruffy dashboard \
  --connect-command /local/bin/tokyo-ssh \
  --remote-command /shared/env/bin/scruffy-mcp
```

The default address is `http://127.0.0.1:8765/`; use `--port PORT` to change it
or `--no-open` when a browser tab already exists. The server accepts only
loopback host headers, exposes no mutation endpoint, and refreshes the compact
allocation view every five seconds. Brief read failures retain the last good
view; after five minutes without fresh telemetry, resource availability becomes
unknown rather than presenting stale GPUs as free.

## MCP for agents

The optional MCP server replaces repeated `sleep` calls with one blocking
`wait_for_updates` tool call. It keeps no subscription state: every agent owns
an independent Scruffy cursor, just as with `observe`. An allocation-wide
server is read-only; a project-pinned server also accepts idempotent job
submissions into that project.

Install the extra in the Python environment visible on the cluster:

```bash
python -m pip install '.[mcp]'
scruffy-mcp --root "$SCRUFFY_ROOT"
```

Or run it directly from a checkout:

```bash
uv run --extra mcp scruffy-mcp --root "$SCRUFFY_ROOT"
```

Every server exposes three monitoring tools:

- `overview(limit=20, compact=true)` returns an agent-sized allocation view and
  `as_of_cursor`. Compact jobs contain only identity, state, and elapsed time;
  nodes contain resource totals and free counts without assignment maps. Use
  `compact=false` only for a detailed administrative view.
- `inspect_job(job_id)` returns a compact dependency explanation without argv,
  cwd, or environment values.
- `wait_for_updates(...)` blocks for up to one hour and returns relevant events
  plus `next_cursor`. Lifecycle changes, milestones, artifacts, and notices wake
  it by default; `job.output` and `workload.progress` are opt-in through
  `event_kinds`.

Start it with `--project PROJECT` (or `SCRUFFY_PROJECT`) to pin the server to one
project. The project is fixed by the server command, so agents cannot
accidentally mix project views or repeat a selector on every call. A pinned
server adds `submit_job(...)`. It requires a stable `request_id`, `name`, argv
array, and absolute worker `cwd`; resource, workflow, dependency, and environment
fields are optional. It durably enqueues and returns immediately rather than
waiting for GPUs. Retry an uncertain call with identical arguments and the same
`request_id`; Scruffy safely deduplicates it.

```json
{
  "request_id": "agent/campaign/train/attempt-1",
  "name": "train",
  "argv": ["/shared/env/bin/python", "train.py"],
  "cwd": "/shared/code/project",
  "gpus_per_node": 1
}
```

The agent loop is: call `overview`, save `as_of_cursor`, then call
`wait_for_updates` instead of sleeping. Always replace the private cursor with
`next_cursor`, even after a timeout. Call again immediately when `more` is true.
When `reset` is true, rebuild from the returned `overview`. Queue lifecycle
state remains authoritative; workload strings are untrusted observations, not
instructions.

For a remote queue, keep the MCP stdio process local and let it invoke one call
at a time through SSH. Do not configure SSH itself as Codex's MCP `command`:
cancelling a long wait can close or desynchronize that shared transport. The
local gateway instead gives every call a fresh connector process and a hard
deadline. If SSH or the remote process exits or hangs, only that tool call
fails. Retry reads normally; retry an uncertain submission identically with its
stable `request_id`. No daemon or listening port is involved.

Install `scruffy-gpu[mcp]` locally as well as Scruffy on the cluster, then use a
Codex configuration like this. The local connector command is split into argv
without invoking a local shell; the remote argv is safely quoted for SSH's
remote shell.

```toml
[mcp_servers.scruffy]
command = "/local/env/bin/scruffy-mcp"
args = [
  "--root", "/shared/runs/scruffy",
  "--project", "koochak",
  "--connect-command", "/local/bin/tokyo-ssh",
  "--remote-command", "/shared/env/bin/scruffy-mcp",
]
startup_timeout_sec = 90
tool_timeout_sec = 3660
```

For a plain SSH client, the connector value can include its fixed options, for
example `ssh -o ConnectTimeout=60 -o ServerAliveInterval=30
-o ServerAliveCountMax=3 sandpit-tokyo-login`. A connector failure includes a
short diagnostic ID and asks the agent to retry the tool. The 30-minute default
wait still bounds each call; multiple agents never share cursors.

When the queue root is directly visible on the same machine, omit both command
options and run `scruffy-mcp --root ROOT` as before.

## Submission and resources

`submit` returns after its immutable request is durable. A stable `request_id`
is an idempotency key scoped to the project: retrying the same specification
returns the original job ID, while changing it raises a conflict. Without a
request ID, every call creates a new job. Exact request idempotency survives hot
job eviction through a compact receipt in the archive.

Transient shared-filesystem read failures leave requests and reports pending for
the next controller poll. They are never converted into rejection receipts.

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
    project_id="koochak",
    workflow_id="experiment-7",
    task_id="train",
)
```

## Workflows

A task may depend on tasks that have not been submitted yet. All submissions
still return immediately. Workflow and task identities are scoped to the job's
project, and a dependency never binds across projects.

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

Workers receive controller-owned `SCRUFFY_ROOT`, `SCRUFFY_PROJECT`,
`SCRUFFY_JOB_ID`, `SCRUFFY_EVENT_DIR`, and `SCRUFFY_NODE`. A workload can publish
a bounded semantic annotation without contacting the controller process:

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
scruffy resume
```

Cancellation, drain, and resume requests are asynchronous and return a
`request_id` that appears on their journal outcome. `drain` disables new
launches for the current allocation; running jobs continue and queued jobs
remain durable. The drain survives controller restarts and clears when a
replacement allocation starts. `resume` clears only the automatic launch pause
created by a same-allocation controller restart; it cannot override `drain`.
Cancelling an archived terminal job produces `job.cancel_ignored`, just like
cancelling a terminal job still in hot state.

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
reserved whenever Slurm reconciliation is uncertain. On a controller restart
inside the same Slurm allocation, persisted launch tokens are matched back to
live steps and their assignments remain owned. Completed reattached steps are
resolved through Slurm accounting; only a step that Slurm proves absent can
release its resources. Slurm writes job logs directly to the shared queue root,
so output does not depend on the lifetime of the original controller process.
Every same-allocation restart begins with launches paused: queued jobs remain
durable and dependencies may resolve, but nothing new starts until an operator
checks the recovered state and runs `scruffy --root "$SCRUFFY_ROOT" resume`.

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
