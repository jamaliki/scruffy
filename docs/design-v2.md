# Scruffy architecture: submission, execution, and observation

Scruffy deliberately remains a filesystem queue with one controller writer. It
does not need a database or a distributed scheduler consensus layer: the outer
Slurm allocation already establishes the failure and resource boundary, and a
single controller is what makes GPU ownership easy to audit.

The original implementation used one mutable job dictionary for four different
concerns. Version 2 separates those concerns at the points where atomicity and
immutability matter, while retaining the v1 state and journal fields needed to
recover existing queue roots.

## Boundaries

### Submission

A submission is the atomic unit entering the inbox.

- A single-job request remains a one-job submission for compatibility.
- `submit_workflow` writes one immutable envelope containing the complete DAG.
- The producer publishes the envelope with one directory rename.
- The controller validates every task, dependency, identity, and resource
  request before changing authoritative state.
- Accepted DAGs enter the recovery journal in one `submission.admitted` record.
  A crash can therefore recover all task images or none of them.
- Per-job lifecycle notifications follow that record, but they are observations;
  the atomic admission record and snapshot are authoritative.

Matrix expansion, canary patterns, and reusable templates belong in clients.
They should resolve to explicit task specifications before this boundary.

### Execution

An accepted job has three immutable records under `provenance/JOB_ID/`:

- `request.json`: accepted command, cwd, resource contract, logical task
  identity, dependency declarations, and an environment digest.
- `launch.json`: concrete dependency attempts, allocation, node/resource
  reservation, start/deadline, and a digest of the request record.
- `result.json`: terminal state, structured reason, exit status, final scheduler
  reservation, and output byte counts.

The workload receives the absolute launch-record path as
`SCRUFFY_PROVENANCE_PATH` and its concrete reservation digest as
`SCRUFFY_ASSIGNMENT_SHA256`. It also receives controller-owned job, project,
workflow, task, attempt, node, and GPU identity variables. In Slurm mode the
workload GPU tokens come from the step allocation rather than the scheduler's
admission-slot IDs.

Raw environment values remain in the protected live queue state only; immutable
provenance stores their canonical digest. Provenance stays outside the removable
job-log directory and therefore survives hot-state and log compaction.

### Resource ownership

`assignment` means one thing only: a live lease that still consumes resources.
It is cleared only after process and launcher reconciliation prove release safe.
`last_assignment` is historical placement and never participates in scheduling.

This distinction permits terminal inspection and provenance without risking a
false live reservation or weakening the no-overlap invariant.

In Slurm mode the controller requests each reservation as a non-overlapping job
step with explicit GPU, CPU, and memory counts. Slurm selects the physical GPUs
and sets `CUDA_VISIBLE_DEVICES`; the worker validates its count and preserves
it. Scruffy's per-node GPU IDs are deterministic admission slots used to decide
whether a request fits, not a second physical-device allocator.

CPU-only work uses the same ledger with an empty GPU tuple. CPU and memory remain
positive reservations, so CPU jobs can never consume node resources invisibly.
Every mutation API requires an explicit GPU count; zero must be intentional.

`time_limit_seconds` is part of the immutable resource contract. The controller
stores an absolute deadline at launch, reconstructs its remaining duration after
a same-allocation restart, and terminates the job with reason `timeout`.

### Workflow attempts

`(project_id, workflow_id, task_id)` is a logical task identity. Every admitted
execution has a monotonically increasing `attempt`, and dependencies are bound
to concrete job IDs in `resolved_dependencies` before launch.

Old attempts are immutable. A new attempt may replace a terminal non-success,
but a succeeded task identity remains final. Scruffy never reopens an already
terminal descendant; a future cascade-retry operation must create new descendant
attempts transactionally.

### Observation

Read interfaces project the same authoritative state for different costs:

- `overview` contains only health, counts, capacity, scheduler explanation, and
  a cursor.
- queue/running/blocked/list operations return identities, with exact filters.
- `inspect_job` exposes lifecycle, placement, provenance, and dependencies but
  never raw argv or environment values.
- `tail_job_output` reads one job-owned stream with a hard 64 KiB bound.
- `wait_job` composes the event cursor, terminal inspection, and optional stderr
  tail without adding controller state.

Cursors remain bounded recovery cursors rather than an infinite event database.
A reset includes a reason and an authoritative overview. Agents rebuild their
view instead of asking the controller to retain unbounded history.

## Deliberate non-goals

- Artifact bytes remain in normal project storage. Scruffy may later retain a
  bounded manifest of declared and worker-validated artifacts.
- Scientific concepts such as canary promotion and “no candidates” are client
  workflow or workload-outcome concepts, not scheduler lifecycle states.
- GPU quarantine and automatic infrastructure retries require trustworthy
  failure evidence. They should not be driven by stderr string matching.
- Start-time estimates are intentionally absent. Scruffy reports deterministic
  blockers and resource eligibility instead of an unreliable ETA.

These boundaries keep the single-writer loop boring: validate and commit a
submission, own live leases, record execution transitions, and publish bounded
views. Rich experiment behavior can be built above those facts without making
resource recovery depend on a workflow framework.
