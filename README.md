# Scruffy

Scruffy is a small, cooperative GPU queue that runs inside one multi-node Slurm
allocation. One controller divides the allocation while any number of humans or
agents submit work asynchronously and observe the same non-consuming event
journal.

It is named after the quiet caretaker from *Futurama*. This project is not
affiliated with the show or its owners.

> In Slurm mode, every job receives an explicit step-level GPU allocation; Slurm
> prevents two live steps from owning the same GPU.

This guarantee covers only work launched through Scruffy. Do not run out-of-band
GPU processes against its configured inventory.

## Architecture at a Glance

### From Submission to GPU

```mermaid
flowchart TB
    producers["Humans · agents · Python clients
submit asynchronously"]
    inboxes[["Shared queue root
requests · workflows · commands · reports · health samples"]]
    controller["Single controller
one lock · one writer · resource and health authority"]
    scheduler["Pure scheduler
dependency gates · fair priority · health-filtered best fit"]
    reserve["Durable reservation
job.starting enters the journal first"]
    workers["Slurm worker steps
explicit GPU · CPU · memory ownership"]
    health["Allocation-wide health step
one sampler per node · all managed GPUs"]
    payload["Job payload
stdout · stderr · provenance · progress"]
    state[("Recovery and observation
journal · snapshot · archived jobs · GPU health")]
    observers["Independent readers
CLI · MCP hub · clickable GPU dashboard"]

    producers -->|non-blocking write| inboxes --> controller --> scheduler
    scheduler --> reserve --> workers --> payload
    controller -.->|launch · reattach| health
    health -->|atomic latest samples| inboxes
    controller --> state --> observers
    payload -.->|reports via inbox| controller

    classDef producer fill:#F4F0FF,stroke:#6D5BD0,color:#241C3A,stroke-width:1.5px;
    classDef storage fill:#F4EDC9,stroke:#9C7B21,color:#352B10,stroke-width:1.5px;
    classDef core fill:#5B4B8A,stroke:#C7B9FF,color:#FFFFFF,stroke-width:1.5px;
    classDef schedule fill:#DDF7F3,stroke:#168B83,color:#123B38,stroke-width:1.5px;
    classDef reservation fill:#FFF4E8,stroke:#D97745,color:#3A2117,stroke-width:1.5px;
    classDef runtime fill:#E8F0FF,stroke:#4977B8,color:#172D4D,stroke-width:1.5px;
    classDef observer fill:#FCE8DE,stroke:#D9674B,color:#44231A,stroke-width:1.5px;

    class producers producer;
    class inboxes,state storage;
    class controller core;
    class scheduler schedule;
    class reserve reservation;
    class workers,health,payload runtime;
    class observers observer;
    linkStyle default stroke:#88859A,stroke-width:1.5px;
```

Producers never coordinate with each other and do not wait for resources. They
write immutable, idempotent requests; the controller is the only process that
turns those requests into queue state. A launch is fail-closed: Scruffy commits
the assignment to its recovery journal before creating the worker step, and
Slurm enforces the worker's physical resource ownership.
The controller also owns the overlapping health step. Node samplers report
physical identity and CUDA, thermal, and ECC evidence but cannot change
placement themselves; only the controller converts those samples into sticky
health state and scheduler eligibility.

### Inside the Controller Loop

```mermaid
flowchart TB
    recover["Start or recover
lock root · replay journal · reattach workload and health steps"]
    ingest["Ingest immutable inboxes
requests · commands · reports"]
    reconcile["Reconcile reality
deadlines · output · processes · Slurm steps"]
    health["Maintain GPU health
validate incarnation · ingest samples · update sticky state"]
    dependencies["Refresh workflow gates
queue ready tasks · skip impossible descendants"]
    maintain["Bound retained state
compact journal · archive cold jobs"]
    fit{"Eligible job fits
free health-eligible resources?"}
    reserve["Persist assignment
update ledger · emit job.starting"]
    launch["Launch worker step
Slurm supplies physical GPU tokens"]
    commit["Publish committed view when due
durable event or heartbeat"]
    wait["Short asynchronous tick
new submissions may arrive meanwhile"]

    recover --> ingest --> reconcile --> health --> dependencies --> maintain --> fit
    fit -->|yes| reserve --> launch --> fit
    fit -->|no| commit --> wait --> ingest

    classDef recovery fill:#F4F0FF,stroke:#6D5BD0,color:#241C3A,stroke-width:1.5px;
    classDef core fill:#5B4B8A,stroke:#C7B9FF,color:#FFFFFF,stroke-width:1.5px;
    classDef reconcileNode fill:#E8F0FF,stroke:#4977B8,color:#172D4D,stroke-width:1.5px;
    classDef workflow fill:#DDF7F3,stroke:#168B83,color:#123B38,stroke-width:1.5px;
    classDef choice fill:#FFF4E8,stroke:#D97745,color:#3A2117,stroke-width:1.5px;
    classDef launchNode fill:#FCE8DE,stroke:#D9674B,color:#44231A,stroke-width:1.5px;
    classDef storage fill:#F4EDC9,stroke:#9C7B21,color:#352B10,stroke-width:1.5px;

    class recover recovery;
    class ingest core;
    class reconcile reconcileNode;
    class health reconcileNode;
    class dependencies workflow;
    class maintain,commit storage;
    class fit choice;
    class reserve,launch launchNode;
    class wait recovery;
    linkStyle default stroke:#88859A,stroke-width:1.5px;
```

The loop reconciles durable intent with live worker state before admitting more
work. Scheduling is a pure calculation over the current inventory and active
assignments; all filesystem writes, lifecycle transitions, and process launches
remain in the single-writer controller.
Periodic metrics remain replaceable evidence rather than journal traffic. Only
health transitions and operator actions are durable events, while enforce mode
removes stale or quarantined nodes from the scheduler's eligible GPU inventory.

## Requirements and boundaries

- Python 3.10 or newer. The base package has no Python dependencies; the
  optional MCP server uses the official Python MCP SDK.
- A POSIX shared filesystem visible at the same absolute `SCRUFFY_ROOT` on every
  client and compute node. Atomic rename and cluster-coherent `flock` are hard
  requirements. On Lustre, verify that distributed `flock` is enabled;
  `localflock` is insufficient. Scruffy cannot detect silently local-only locks.
- Slurm mode requires `srun`, `scancel`, `scontrol --json`, `nvidia-smi`, and a
  working NVIDIA Driver API (`libcuda.so.1`) on the compute nodes.
- Submitted working directories, executables, and referenced files must exist on
  every selected worker node.

The Slurm controller retries transient shared-filesystem failures without
releasing its singleton lock or GPU ledger. A blocked filesystem syscall must
still return before Python can retry, so controller diagnostics should be
written to a filesystem independent of `SCRUFFY_ROOT`.

In Slurm mode, Scruffy requests each job's GPU, CPU, and memory budget on its
worker step without `--overlap`. Slurm chooses the physical GPUs and sets
`CUDA_VISIBLE_DEVICES`; Scruffy preserves that value instead of replacing it.
The controller's per-node GPU IDs remain deterministic admission slots, while
`SCRUFFY_GPU_IDS` contains the authoritative tokens visible to the workload.
Local mode remains cooperative. Scruffy trusts every process able to write
`SCRUFFY_ROOT`; it is not a multi-user security boundary.

## GPU health and quarantine

In Slurm mode Scruffy can own one allocation-wide, overlapping monitor step.
Every 10 seconds, one task on each node records:

- stable NVIDIA UUID, node, Scruffy slot, NVIDIA index, Linux minor number, PCI
  bus ID, serial, model, driver, and VBIOS;
- current temperature, power, uncorrectable volatile ECC count, and NVIDIA's
  software/hardware thermal-slowdown reasons; and
- a CUDA Driver API initialization, device-count, UUID, and context-creation
  probe for every visible GPU. This catches failures where `nvidia-smi` works
  but CUDA initialization does not.

Use `--gpu-health off`, `observe`, or `enforce` when starting the controller.
The default is `observe`: automatic failures become visible without changing
placement. An explicit `gpu-quarantine` always withholds the node.
In `enforce`, three consecutive bad samples make the affected UUID's
quarantine sticky; missing or older-than-45-second samples fail closed. An
operator must explicitly clear a quarantine after investigating it:

```bash
scruffy --root "$SCRUFFY_ROOT" gpus
scruffy --root "$SCRUFFY_ROOT" gpu gpu-3 5
scruffy --root "$SCRUFFY_ROOT" gpu-quarantine gpu-3 GPU-... \
  --reason "Scientific Computing ticket SC-1234"
scruffy --root "$SCRUFFY_ROOT" gpu-clear gpu-3 GPU-...
```

Current worker steps request GPU counts and Slurm chooses their physical
devices. Scruffy therefore cannot safely reuse the healthy GPUs on a node while
guaranteeing exclusion of one physical UUID. Enforced quarantine marks the bad
GPU `STOPPED`, marks its healthy peers `NODE HELD`, and admits no new GPU job on
that node. Existing jobs are not killed automatically; their assignment stays
owned until normal exit or explicit cancellation. Exact healthy-peer reuse
requires a separately validated Slurm GRES/device-isolation contract.

Latest samples are replaceable files under `health/samples/`; only status
transitions and operator actions enter the durable journal. The dashboard makes
every GPU tile clickable and provides a copy-ready identity and health report
for Scientific Computing.

Koochak remains responsible only for workload-local response: it may checkpoint
and exit after detecting a bad CUDA device. When `SCRUFFY_JOB_ID` or
`SCRUFFY_ROOT` is present, Koochak must not requeue or cancel the parent Slurm
allocation. Scruffy is the only health authority allowed to change future
placement.

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

After observing health telemetry on the target cluster, enable fail-closed
placement:

```bash
scruffy --root "$SCRUFFY_ROOT" serve --gpu-health enforce
```

Prefer running the controller as the outer batch script's foreground process.
If it must be a nested management step, launch that controller step with
`--overlap --gres=none` and a small CPU/memory request so it does not consume a
worker GPU or permanently remove one worker CPU from the managed pool.

Automatic inventory discovery reads the resources granted to the outer Slurm
allocation. Per-node resource flags can cap that managed capacity. Discovery
assumes homogeneous nodes and a contiguous GPU pool `0..N-1`; use an explicit
inventory for selected, non-contiguous, or heterogeneous resources:

```bash
scruffy --root "$SCRUFFY_ROOT" serve --inventory inventory.json
```

When Slurm exposes the allocation deadline, the controller stops launching new
jobs 15 minutes before it. Change the window with
`--drain-before-end-seconds SECONDS`, or set it to `0` to disable automatic
draining. Running jobs are not killed by the drain.

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
scruffy --root "$SCRUFFY_ROOT" resources
scruffy --root "$SCRUFFY_ROOT" gpus
scruffy --root "$SCRUFFY_ROOT" running
scruffy --root "$SCRUFFY_ROOT" queue
scruffy --root "$SCRUFFY_ROOT" blocked
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
GPU exclusivity holds across every project. Projects provide a small dynamic
fair-share signal, not a quota or access-control boundary: the scheduler prefers
the project currently holding fewer GPUs, then preserves FIFO order.

Set `--project` on submission and observation, or export `SCRUFFY_PROJECT` for
an agent session:

```bash
export SCRUFFY_PROJECT=koochak
scruffy --root "$SCRUFFY_ROOT" submit \
  --request-id experiment-7/train --gpus-per-node 1 -- python train.py
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
physical GPU ownership and health, CPU and memory reservations, project scopes,
queue lanes, failures, recent finishes, and compact job/dependency details.
Every GPU tile is clickable and exposes a copy-ready Scientific Computing
report with UUID, PCI bus ID, serial, NVIDIA index, Linux minor, model, driver,
VBIOS, metrics, and quarantine reasons. The UI uses the same bounded
projections as MCP and never returns argv, working directories, or environment
values.

For a locally visible queue root:

```bash
uv run scruffy --root "$SCRUFFY_ROOT" dashboard
```

For a remote queue, run the web server beside the queue and forward its loopback
port through the existing SSH gateway:

```bash
scruffy --root /shared/runs/scruffy dashboard --port 18765 --no-open
# Forward local 8765 to remote loopback 18765 with your connection supervisor.
```

The default address is `http://127.0.0.1:8765/`; use `--port PORT` to change it
or `--no-open` when a browser tab already exists. The server accepts only
loopback host headers, exposes no mutation endpoint, and refreshes the compact
allocation view every five seconds. Brief read failures retain the last good
view; after five minutes without fresh telemetry, resource availability becomes
unknown rather than presenting stale GPUs as free.

## MCP for agents

The optional MCP server replaces repeated `sleep` calls with one blocking
`wait_for_updates` tool call. Every agent owns an independent Scruffy cursor,
just as with `observe`. A project scope also enables idempotent job submissions
into that project.

Install the extra in the Python environment visible on the cluster:

```bash
python -m pip install '.[mcp]'
scruffy-mcp --root "$SCRUFFY_ROOT"
```

Or run it directly from a checkout:

```bash
uv run --extra mcp scruffy-mcp --root "$SCRUFFY_ROOT"
```

Every server exposes focused monitoring tools:

- `overview()` returns only allocation health, lifecycle counts, aggregate free
  and total resources, and `as_of_cursor`. It never includes job rows.
- `queue(...)` lists submitted and schedulable queued jobs.
- `running_jobs(...)` lists every resource-holding active state, including
  starting, finishing, and cancelling transitions.
- `blocked_jobs(...)` lists dependency-blocked work separately from the resource
  queue.
- `resources()` returns aggregate and per-node GPU, CPU, and memory availability
  without assignment details.
- `gpus()` returns every physical GPU mapping and its scheduler/health state.
- `inspect_gpu(node, slot)` returns the copy-ready identity, CUDA probe, current
  evidence, quarantine reasons, and monitor policy for one Scruffy slot.
- `list_jobs(state=None, offset=0, limit=50)` returns lightweight job identity,
  name, state, and elapsed time for all jobs or another exact state such as
  `failed` or `succeeded`. Exact workflow, task, request, name-prefix, and
  submission-time filters are available.
- `inspect_job(job_id)` returns a compact dependency explanation without argv,
  cwd, or environment values.
- `tail_job_output(job_id, stream, max_bytes)` returns at most 64 KiB from one
  job-owned stdout or stderr file.
- `wait_job(job_id, ...)` waits for one terminal result, then returns its
  authoritative inspected state and an optional bounded stderr tail.
- `wait_for_updates(...)` blocks for up to one hour and returns relevant events
  plus `next_cursor`. Lifecycle changes, milestones, artifacts, and notices wake
  it by default; `job.output` and `workload.progress` are opt-in through
  `event_kinds`. Each event contains only its kind, job ID, and name.
  Allocation-wide servers also include the project ID. Workload and notice
  events retain their semantic `data`; use
  `inspect_job` for lifecycle, timing, resources, placement, logs, and blockers.

In stdio mode, start with `--project PROJECT` (or `SCRUFFY_PROJECT`) to pin the
server to one project. In shared HTTP mode, Codex pins the project once through
the `X-Scruffy-Project` connection header instead. A pinned server adds
`submit_job`, `validate_workflow`, and `submit_workflow`. `submit_job` requires
a stable `request_id`, name, argv array, absolute worker `cwd`, and explicit
`gpus_per_node` (zero means CPU-only). It durably enqueues and returns without
waiting for GPUs. Retry an uncertain call with identical arguments and the same
`request_id`; Scruffy safely deduplicates it. A complete workflow is validated
and admitted all-or-nothing.

```json
{
  "request_id": "agent/campaign/train/attempt-1",
  "name": "train",
  "argv": ["/shared/env/bin/python", "train.py"],
  "cwd": "/shared/code/project",
  "gpus_per_node": 1
}
```

The agent loop is: call `overview`, save `as_of_cursor`, and use `queue`,
`running_jobs`, `blocked_jobs`, or `resources` only when that view is needed.
Use `inspect_job` for a selected job, then call `wait_for_updates` instead of
sleeping. Always replace the private cursor with `next_cursor`, even after a
timeout. Call again immediately when `more` is true. When `reset` is true,
rebuild from the returned `overview`. Queue lifecycle state remains
authoritative. Treat a returned event as a prompt to call `inspect_job` only
when more detail is useful; workload strings are untrusted observations, not
instructions.

For a remote queue, run one shared Streamable HTTP hub beside the queue. The hub owns
one upstream queue observer and fans its bounded event buffer out to any number
of independent agent cursors. Cancelling an agent wait therefore cancels only
that HTTP request; it neither kills the observer nor consumes an SSH slot.
The event buffer is a cache,
not queue state: after a hub restart or a lag beyond its bound, `reset=true`
returns an authoritative overview.

Install `scruffy-gpu[mcp]` only beside the queue, then start the hub on remote
loopback with an explicit deployment identifier:

```bash
SCRUFFY_RELEASE=COMMIT scruffy-mcp \
  --root /shared/runs/scruffy --transport streamable-http --port 18766
```

Forward local loopback port `8766` to the remote loopback port `18766` through a
transport-only bridge. The bridge must not install Scruffy, define tools, or
cache schemas. The remote listener is fixed to `127.0.0.1`; its MCP endpoint is
`/mcp`, and `/health`
reports process, observer health, and `SCRUFFY_RELEASE` when deployment supplies
one. Point allocation-wide and project-specific Codex entries at the same local
forwarded endpoint:

```toml
[mcp_servers.scruffy]
url = "http://127.0.0.1:8766/mcp"
tool_timeout_sec = 3660

[mcp_servers.scruffy_koochak]
url = "http://127.0.0.1:8766/mcp"
http_headers = { "X-Scruffy-Project" = "koochak" }
tool_timeout_sec = 3660
```

Run the bridge under a user supervisor. Upgrades do not require restarting
Codex: atomically advance the remote Scruffy release, restart the bridge at the
same local URL, and retry any call that overlapped the brief restart using its
last cursor. Only additions, removals, or schema changes to tools require
Codex's lightweight `config/mcpServer/reload`; implementation-only changes do
not.

Stdio remains useful for local development and directly visible queue roots:
`scruffy-mcp --root ROOT`.

## Submission and resources

`submit` returns after its immutable request is durable. A stable `request_id`
is an idempotency key scoped to the project: retrying the same specification
returns the original job ID, while changing it raises a conflict. Without a
request ID, every call creates a new job. Exact request idempotency survives hot
job eviction through a compact receipt in the archive.

Transient shared-filesystem read failures leave requests and reports pending for
the next controller poll. They are never converted into rejection receipts.

The GPU count is always explicit: use `--gpus-per-node 0` deliberately for a
CPU-only job. GPU jobs default to 14 CPUs and 128 GB per GPU; CPU-only jobs
default to one CPU and 4 GB. `--time-limit-seconds` is enforced by the
controller and survives a same-allocation controller restart. Requests are
rectangular: every selected node receives the same resource shape and runs the
same argv once. Distributed rendezvous and application ranks remain the
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

For a complete DAG, prefer one atomic workflow document. `validate-workflow`
runs the same schema, dependency, cycle, identity, and allocation-fit checks
without writing queue state; allocation fit is checked when a live inventory is
available and always rechecked by the controller. `submit-workflow` then
publishes one immutable envelope; the controller creates every task or none.

```json
{
  "request_id": "agent/experiment-7",
  "workflow_id": "experiment-7",
  "project_id": "koochak",
  "tasks": [
    {
      "task_id": "train",
      "argv": ["/shared/env/bin/python", "train.py"],
      "cwd": "/shared/code/project",
      "resources": {
        "nodes": 1, "gpus_per_node": 1,
        "cpus_per_node": 14, "memory_gb_per_node": 128
      }
    },
    {
      "task_id": "infer",
      "argv": ["/shared/env/bin/python", "infer.py"],
      "cwd": "/shared/code/project",
      "resources": {
        "nodes": 1, "gpus_per_node": 1,
        "cpus_per_node": 14, "memory_gb_per_node": 128
      },
      "needs": [{"task_id": "train", "condition": "succeeded"}]
    }
  ]
}
```

```bash
scruffy validate-workflow workflow.json
scruffy submit-workflow workflow.json
```

Single-task submissions remain useful when a dependency is not yet known. They
also return immediately. Workflow and task identities are scoped to the job's
project, and a dependency never binds across projects.

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id infer \
  --needs train:succeeded --gpus-per-node 1 -- python infer.py

scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id train \
  --gpus-per-node 1 -- python train.py
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
`SCRUFFY_JOB_ID`, `SCRUFFY_EVENT_DIR`, `SCRUFFY_NODE`, `SCRUFFY_GPU_IDS`, and
workflow/task/attempt identity. In Slurm mode, `SCRUFFY_GPU_IDS` is copied from
the step's `CUDA_VISIBLE_DEVICES`; `SCRUFFY_SLURM_GPU_IDS` also exposes Slurm's
global step IDs when available. `SCRUFFY_PROVENANCE_PATH` names a mode-0444
launch record containing exact argv, cwd, resources, resolved dependency
attempts, allocation, and scheduler reservation. Its sibling request and result
records survive log and hot-state compaction; environment values are represented
by a digest rather than copied into provenance. `SCRUFFY_ASSIGNMENT_SHA256` is a
stable digest of that reservation.

A workload can publish a bounded semantic annotation without contacting the
controller process:

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
scruffy resources
scruffy running
scruffy queue
scruffy blocked
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
replacement allocation starts. An explicit `resume` also reverses a drain or
clears a recovery or `--start-paused` launch pause.
Cancelling an archived terminal job produces `job.cancel_ignored`, just like
cancelling a terminal job still in hot state.

The hot snapshot keeps every nonterminal job and, after compaction, the newest
1,000 terminal jobs. Older terminal jobs remain addressable by job ID with
`archived: true`; compact records keep lifecycle and workflow identity,
resource requests, final placement, and provenance references. Cwd, argv,
environment, live assignment, blockers, logs, and workload state expire.
`summary.counts` includes archived terminal jobs;
`summary.archived_jobs` and `status.archived_counts` expose the archived totals.
The active and immediately previous journal generations are retained.
Compact request receipts and workflow indexes last for the queue root's lifetime,
so archive storage grows as O(total jobs) small files and inodes.

Placement is deterministic best-fit with simple backfilling. Before every
placement, queued jobs are ordered by their project's currently assigned GPU
count and then by controller-owned queue order. This balances concurrent GPU use
without storing historical usage or blocking smaller jobs that fit. It does not
reserve a fixed share or prevent a large request from waiting for enough GPUs.
Queued and dependency-blocked jobs survive a replacement allocation, including
requests that do not fit the replacement inventory; such jobs remain queued
until a later compatible allocation is attached. Active jobs from a replaced
allocation become `lost` and are never retried automatically. Resource
assignments remain reserved whenever Slurm reconciliation is uncertain. On a
controller restart inside the same Slurm allocation, persisted launch tokens
are matched back to live steps and their assignments remain owned. Completed
reattached steps are resolved through Slurm accounting; only a step that Slurm proves absent can
release its resources. Slurm writes job logs directly to the shared queue root,
so output does not depend on the lifetime of the original controller process.
Every same-allocation restart begins with launches paused: queued jobs remain
durable and dependencies may resolve, but nothing new starts until an operator
checks the recovered state and runs `scruffy --root "$SCRUFFY_ROOT" resume`.

### Moving to a replacement allocation

Keep `SCRUFFY_ROOT` at a stable home-backed location rather than naming it after
the Slurm allocation. The queue state does not need to be copied. A Slurm
requeue that reuses the same numeric job ID is still a replacement when its
restart count or inventory changes. Before the old allocation ends, drain it
and let checkpointing workloads finish when possible:

```bash
scruffy --root "$SCRUFFY_ROOT" drain
```

Start the successor allocation against the same root with launches paused:

```bash
scruffy --root "$SCRUFFY_ROOT" serve --start-paused
scruffy --root "$SCRUFFY_ROOT" summary
scruffy --root "$SCRUFFY_ROOT" resume
```

The summary and MCP overview include a bounded `handover` count of lost,
queued, blocked, and currently inventory-ineligible jobs. Inspect it before
resuming. Only an exact allocation-incarnation match may reattach active steps
or inherit a drain. New submissions are still rejected when they cannot fit the
current inventory; only durable work inherited from an older allocation is
retained.

The controller deliberately runs one submitted argv per job. It numbers
immutable attempts but is not an automatic retry engine, dynamic workflow
fan-out engine, or artifact store: submit retries and new tasks explicitly, and
keep artifact bytes in their normal storage.

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
