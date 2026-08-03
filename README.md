# Scruffy

Scruffy is a small, cooperative GPU queue that runs **inside** a multi-node
Slurm allocation. One controller cares for the allocation while any number of
humans or agents submit work asynchronously and observe the same event stream.

It is named after the quiet caretaker from *Futurama*. This project is not
affiliated with the show or its owners.

## Why

Large Slurm allocations are useful when many short or independent GPU jobs
would otherwise spend their lives waiting in the cluster queue. Scruffy divides
an already-held allocation into node-qualified GPU units and makes this true:

> No two nonterminal jobs submitted through Scruffy own the same `(node, GPU)`.

This is a cooperative boundary. Scruffy does not stop the allocation owner from
launching an unrelated process outside its API. Do not mix out-of-band GPU work
with the GPU IDs assigned to Scruffy.

## Shape

```text
many clients -> immutable requests on shared storage -> one controller
                                                       -> concurrent srun steps
many readers <- atomic snapshot + broadcast journal  <- across allocated nodes
```

- Python 3.11 or newer, with no runtime dependencies.
- A durable filesystem inbox makes `submit` fast even when no GPU or controller
  is available.
- One allocation-wide controller performs atomic per-node placement.
- Slurm launches one worker per selected node; the worker sets that node's
  `CUDA_VISIBLE_DEVICES` and `exec`s the argv without a shell.
- A unique Slurm step name is persisted before launch. Scruffy cancels only the
  resolved `job.step` and releases GPUs only after fresh `scontrol --json`
  state proves that step is gone.
- Raw logs are stored once. The journal contains ordered lifecycle events and
  output-range references, so independent readers can replay it without
  consuming messages from one another.
- Jobs can publish bounded semantic progress into the same journal, while
  dependency metadata turns independently submitted jobs into simple workflows.

## Install

```bash
python -m pip install .
scruffy --version
```

For a checkout used directly on shared storage:

```bash
export PYTHONPATH=/shared/code/scruffy/src
python -m scruffy --version
```

## Start a controller

The safest production shape is for the outer sbatch script to `exec scruffy
serve`, as shown in [`examples/scruffy.sbatch`](examples/scruffy.sbatch). When
the controller exits, the outer allocation ends and Slurm cleans up its steps.

Scruffy can derive a homogeneous inventory from `SLURM_JOB_NODELIST`:

```bash
export SCRUFFY_ROOT=/shared/runs/scruffy
scruffy --root "$SCRUFFY_ROOT" serve \
  --gpus-per-node 8 --cpus-per-node 112 --memory-gb-per-node 1024
```

Automatic discovery deliberately requires the GPU count. It assumes every node
contributes a full, contiguous pool numbered `0..N-1`. Use an explicit inventory
to manage selected or non-contiguous GPUs, or non-homogeneous nodes:

```bash
scruffy --root "$SCRUFFY_ROOT" serve --inventory inventory.json
```

CPU and memory values are cooperative admission budgets, not physical isolation.
Scruffy uses overlapping Slurm steps because the outer allocation already owns
the GPUs; only its `(node, GPU)` ledger and forced `CUDA_VISIBLE_DEVICES` divide
the GPU pool. The outer allocation must provide the declared CPU and memory,
and submitted commands must cooperate with those limits.

## Submit asynchronously

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --name probe \
  --request-id campaign-a/probe-001 \
  --nodes 1 \
  --gpus-per-node 1 \
  --cpus-per-node 14 \
  --memory-gb-per-node 128 \
  -- python -m my_project.probe --config probe.yaml
```

`submit` returns as soon as the request directory is durable. It does not wait
for the controller, free GPUs, earlier jobs, or job startup. Concurrent callers
are safe. Retrying the same request ID and specification returns the original
job ID; changing the specification produces a conflict.

The same operations are small Python functions for agent integrations:

```python
from pathlib import Path
from scruffy import ResourceRequest, submit_job

result = submit_job(
    Path("/shared/runs/scruffy"),
    argv=["python", "train.py"],
    name="train",
    cwd=Path.cwd(),
    environment={},
    request=ResourceRequest(1, 1, 14, 128),
    request_id="agent-a/train-001",
    workflow_id="experiment-7",
    task_id="train",
)
```

Multi-node requests are rectangular and explicit:

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --nodes 2 --gpus-per-node 8 \
  --cpus-per-node 112 --memory-gb-per-node 512 \
  -- ./run_distributed.sh
```

The same command runs once on every selected node. Slurm provides rank and node
environment variables; distributed launch recipes remain application-owned.

## Chain jobs without waiting

Tasks are still independent, durable submissions. A task can name upstream
tasks in the same workflow and return immediately, even when those tasks have
not been submitted yet:

```bash
scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id infer \
  --needs train:succeeded -- python infer.py

scruffy --root "$SCRUFFY_ROOT" submit \
  --workflow-id experiment-7 --task-id train -- python train.py
```

`succeeded` runs the dependent only after a successful upstream result;
`terminal` runs it after any terminal result. Missing dependencies remain
visibly `blocked`. An unsatisfied `succeeded` edge makes the dependent `skipped`,
and duplicate task IDs, self-dependencies, and cycles are rejected. Task IDs are
unique only within a workflow, so multiple agents can safely use separate
workflow namespaces.

Use `scruffy explain JOB_ID` to see the resolved upstream job IDs and current
blockers. Python callers pass `workflow_id`, `task_id`, and a `needs` sequence of
`{"task_id": ..., "condition": "succeeded" | "terminal"}` objects to
`submit_job`.

## Publish semantic workload progress

Workers receive reserved `SCRUFFY_ROOT`, `SCRUFFY_JOB_ID`, `SCRUFFY_EVENT_DIR`,
and `SCRUFFY_NODE` environment variables. They can durably publish a small
semantic update without talking to the controller process:

```bash
scruffy report workload.progress \
  --event-id train/step-12000 \
  --source name=my-trainer \
  --data-json '{"phase":"training","step":12000,"metrics":{"loss":1.42}}'
```

```python
from pathlib import Path
from scruffy import publish_event

publish_event(
    Path("/shared/runs/scruffy"),
    job_id="job-...",
    event_id="train/step-12000",
    kind="workload.progress",
    source={"name": "my-trainer"},
    data={"phase": "training", "step": 12000, "metrics": {"loss": 1.42}},
)
```

The public kinds are `workload.phase`, `workload.progress`,
`workload.milestone`, `workload.artifact`, and `workload.notice`. Events are
finite JSON objects capped at 64 KiB. Producer IDs are idempotent within a job;
raw logs, full metric histories, configs, and checkpoint bytes belong elsewhere.
Reports update only the bounded `job.workload` view and can never change queue
lifecycle or GPU ownership. See [the v1 event contract](docs/events-v1.md).

## Observe from many agents

Every agent keeps its own opaque cursor:

```bash
scruffy --root "$SCRUFFY_ROOT" observe --after "$CURSOR" --wait 30
```

The response contains a complete allocation snapshot, events after the cursor,
and `next_cursor`. Cursors encode queue identity, sequence, and a byte offset;
treat the whole value as opaque. Add `--output` to expand output references into
text.

```bash
scruffy --root "$SCRUFFY_ROOT" status
scruffy --root "$SCRUFFY_ROOT" summary --limit 20
scruffy --root "$SCRUFFY_ROOT" explain JOB_ID
scruffy --root "$SCRUFFY_ROOT" watch --follow --output
scruffy --root "$SCRUFFY_ROOT" logs JOB_ID --tail 200 --follow
scruffy --root "$SCRUFFY_ROOT" wait JOB_ID
scruffy --root "$SCRUFFY_ROOT" cancel JOB_ID
scruffy --root "$SCRUFFY_ROOT" drain
```

`summary` includes durable submissions immediately, even before a controller
has admitted them into its snapshot. This lets one agent submit while another
orients itself without a synchronization handshake.

Cancellation is also asynchronous. A cancelling job retains its resources until
the local `srun` client has exited, both output streams have closed, and a fresh
Slurm snapshot proves the uniquely named step is absent. Query failure is
uncertainty, so Scruffy keeps the GPUs reserved rather than risk overlap. Current
reconciliation failures appear in the allocation or job snapshot and as `notice`
events. Cancel and drain responses include a request ID which is repeated on the
resulting or ignored event, so concurrent agents can correlate their commands.

## Scheduling and failure semantics

- Jobs request `nodes x (GPUs, CPU budget, memory budget)` and receive all nodes
  or none.
- The controller uses deterministic best-fit placement and starts the oldest job
  that currently fits. This simple backfilling policy does not yet guarantee
  fairness for a large blocked job.
- Compatible queued jobs survive replacement allocations. A request which can
  never fit the current inventory is rejected. Jobs active when an allocation
  disappears become `lost` and are never retried automatically.
- A controller will not recover unresolved jobs from the same Slurm allocation:
  it fails closed instead of risking a duplicate launch. Local development mode
  refuses any restart with unresolved work because it cannot prove old children
  are gone.
- Terminal states are `succeeded`, `failed`, `cancelled`, `lost`, and `rejected`.
  Dependency failure also produces terminal `skipped`. Agents never need to
  infer failure from stderr.

## State on disk

```text
ROOT/
  requests/<job-id>/spec.json
  commands/<request-id>.json
  reports/<job-id>/<event-id-hash>.json
  reports/.accepted/<job-id>/<event-id-hash>.json
  jobs/<job-id>/assignment.json
  jobs/<job-id>/stdout.log
  jobs/<job-id>/stderr.log
  state.json
  events.jsonl
  controller.lock
```

Clients only create immutable requests, commands, and workload reports. The
controller alone writes state and assigns the globally ordered event sequence.
Accepted report receipts retain per-job idempotency for the queue's lifetime.

The queue directory is trusted shared state. Job argv, working directories, and
environment overrides are stored there in plaintext and appear in snapshots and
lifecycle records. Protect its permissions and pass secret-file paths rather
than secret values where possible.

## Development

The test suite uses only the standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The local launcher supports single-node lifecycle tests without Slurm or GPUs:

```bash
scruffy --root /tmp/scruffy serve \
  --launcher local --inventory examples/local-inventory.json
```
