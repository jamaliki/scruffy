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

`assignment` means one thing only: a live reservation that still consumes resources.
It is cleared only after process and launcher reconciliation prove release safe.
`last_assignment` is historical placement and never participates in scheduling.

This distinction permits terminal inspection and provenance without risking a
false live reservation or weakening the no-overlap invariant.

In Slurm mode the controller requests each reservation as a non-overlapping job
step with explicit GPU, CPU, and memory counts. Slurm selects the physical GPUs
and sets `CUDA_VISIBLE_DEVICES`; the worker validates its count and preserves
it. Scruffy's per-node GPU IDs are deterministic admission slots used to decide
whether a request fits, not a second physical-device allocator.

Because the slots are not physical devices, the worker reports which devices
Slurm bound. Before exec it runs a context-free CUDA driver probe with the
step's environment and writes `jobs/JOB_ID/gpu-identity-N.json` (UUID, name,
PCI bus ID, and driver minor per device, plus the step and launch token). The
controller copies those records into the job's `gpu_devices` with a
`job.gpu_identified` event while the job runs, and once more when it finishes.
The records are observations, not authority: they never fail a job, and the
runtime placement record remains the launch contract.

CPU-only work uses the same ledger with an empty GPU tuple. CPU and memory remain
positive reservations, so CPU jobs can never consume node resources invisibly.
Every mutation API requires an explicit GPU count; zero must be intentional.

`time_limit_seconds` is part of the immutable resource contract. The controller
stores an absolute deadline at launch, reconstructs its remaining duration after
a same-allocation restart, and terminates the job with reason `timeout`.

`allowed_nodes` and `exclude_nodes` are optional placement constraints in the
same immutable request. The pure scheduler filters candidate nodes by them
before best fit, and the ever-fit check applies the same filter, so admission
and placement agree. They are serialized only when set, which keeps every
existing request digest unchanged.

### GPU health authority

One controller-owned Slurm step overlaps all managed nodes and GPUs. Node-local
workers publish atomic latest samples containing NVIDIA physical identity and
thermal/ECC telemetry, plus CUDA Driver API context probes for GPUs not
currently assigned by Scruffy. The controller alone turns those observations
into `healthy`, `suspect`, or sticky `quarantined` state. Assigned GPUs remain
passively observed, and context OOM races are explicitly inconclusive rather
than failures; assigned software thermal throttling is observational while
uncorrectable ECC and hardware thermal faults remain actionable. Every sample
also carries the exact health-worker fingerprint and reservation-snapshot
provenance, so a controller rejects samples from a stale worker or allocation.
High-rate metrics stay in replaceable samples and the current snapshot; only
state transitions and operator commands enter the journal.
Samples are bound to the outer allocation-incarnation fingerprint, so a requeue
cannot reuse healthy freshness from the previous physical execution.

The health monitor step name includes the worker fingerprint. On controller
upgrade, a live monitor with an older fingerprint is replaced only after its
step identity is matched to the current allocation and node; workload steps
and the outer allocation are never cancelled. Health-step absence requires
bounded consecutive Slurm observations before a live monitor is treated as
gone. Quarantine debounce requires three bad samples within the sampling
window; inconclusive observations neither increment bad streaks nor count as
clean evidence, and materially future-dated samples are rejected.

The durable key is `(node, NVIDIA UUID)`. Scheduler slot, NVIDIA index, Linux
minor, and PCI bus ID are reportable mappings, not stable identity. Missing or
stale evidence fails closed only in enforce mode. The default `gpu` isolation
maps a quarantined slot out of the scheduler. Jobs on healthy nodes use
count-based Slurm allocation; jobs touching a node with a mappable quarantined
GPU use an explicit GRES mask and verify the physical mapping before exec, so a
Slurm substitution cannot silently use the stopped GPU. A multi-node exact step
uses one common slot set on every node; when that contract cannot be
represented, Scruffy does not construct the unsafe placement. `node` isolation
remains available as an explicit conservative fallback. Existing reservations
remain owned and are not terminated implicitly.

An operator quarantine under `gpu` isolation on a Slurm primary-allocation
node is enforced differently, because slot masks cannot pin a physical device:
inside an allocation Slurm picks a count-bound step's devices, `--gpu-bind`
only orders devices a step already has, and GRES cannot name one device per
step. The controller instead admits controller-owned one-GPU *sentinel* jobs
(`system: "gpu_hold"`, project `scruffy-system`, placed before all user work,
excluded from fair share, user cancellation, bulk cancellation, and
evacuation). One sentinel is submitted per free GPU on the node at once, so
together they cover every free device; each worker reports the UUID it was
bound to. The sentinel on the target holds it until `gpu-clear`; the others
park and are cancelled once the target is held (or exit on their own after ten
minutes). Only while sentinels are being placed and identified are the node's
free GPUs withheld from user jobs. A target under an identified running job
waits without withholding until that job ends; a target that may be under an
unidentified job waits until one of those jobs ends or is identified; three
unidentifiable sentinels, 64 launches without a hold, or a target absent from
every identified GPU end in `failed`. The hold record (`gpu_health.holds`) is
journaled with the health image, so replay restores it with the sentinels it
tracks; a sentinel admitted in a batch whose hold update was lost is adopted on
the next tick. The quarantined device record carries `enforcement: "hold"` and
withholds no slot.

### Workflow attempts

`(project_id, workflow_id, task_id)` is a logical task identity. Every admitted
execution has a monotonically increasing `attempt`, and dependencies are bound
to concrete job IDs in `resolved_dependencies` before launch.

Old attempts are immutable. A new attempt may replace a terminal non-success,
but a succeeded task identity remains final. Scruffy never reopens an already
terminal descendant; a future cascade-retry operation must create new descendant
attempts transactionally.

Workflow tasks may declare a strict `recovery` policy with `max_attempts`
(including the first attempt, capped at 10), `retry_on`, and the nested
`evacuation` signal/grace fields. Wave 1 admits only the documented reasons and
automatically creates deterministic successors for allocation replacement,
incarnation change, or an acknowledged evacuation. The Phase 2 evacuation
state machine journals the signal decision before sending `USR1`, verifies the
launch capability and current allocation identity, and tracks bounded target
outcomes through grace deadlines without auto-kill. A successor copies the
predecessor's immutable task inputs and records both lineage links plus the
retry reason. Replay checks the deterministic successor identity before
admission, so controller recovery cannot fan out duplicate attempts.

### Group commit

The journal is the write-ahead log and `state.json` is a replaceable
checkpoint of it. One controller poll iteration is one group commit: every
admission, command, report, and dependency transition in the iteration is
appended without a sync; the iteration then syncs the journal once, replaces
the snapshot once, and only afterwards retires request, command, and report
inbox files. A failure before that commit acknowledges nothing, so a
restarted controller either replays the journaled outcome or applies the
still-pending inbox item. Transitions that guard an external effect (the
`job.starting` reservation before a launch, `job.cancelling` before a signal,
and the evacuation signal decision) force a commit before that effect. The
cost of N queued commands is therefore one snapshot write per iteration rather
than N, and a per-iteration bound keeps one iteration short when thousands of
command files are pending.

### Retention

The snapshot is sized by live work, not by history. Terminal jobs leave hot
state through the same journal as every other transition: compact per-job
archive records (and per-workflow indexes) are written first, then one
`jobs.archived` record removes the jobs and adds their counts, replayed like
any other record after a crash. Only an in-progress evacuation's targets and
`lost` tasks awaiting an automatic retry stay pinned in hot state, plus a
small recent window for summaries. Journal rotation is independent and occurs
only when the journal itself grows large. Log directories follow a separate
bounded retention list so archiving never makes a just-finished job's output
disappear.

### Observation

Read interfaces project the same authoritative state for different costs:

- `overview` contains only health, counts, capacity, scheduler explanation, and
  a cursor.
- queue/running/blocked/list operations return identities, with exact filters.
- `inspect_job` exposes lifecycle, placement, provenance, and dependencies but
  never raw argv or environment values.
- `gpus` and `inspect_gpu` expose physical identity, health evidence, scheduler
  withholding, and a Scientific Computing report without mutation.
- `reprobe_gpu` is the guarded operational exception: it releases only an
  automatic quarantine backed by recent clean controller evidence.
- `tail_job_output` reads one job-owned stream with a hard 64 KiB bound.
- `wait_job` composes the event cursor, terminal inspection, and optional stderr
  tail without adding controller state.

Cursors remain bounded recovery cursors rather than an infinite event database.
A reset includes a reason and an authoritative overview. Agents rebuild their
view instead of asking the controller to retain unbounded history.

### Leases: capacity beyond one allocation

A controller may place work on more than its own (primary) allocation. Every
other Slurm allocation it uses is a *lease*, and each worker step runs inside
exactly one allocation, with `srun --jobid` naming the primary or a lease.

- An **attached** lease is an allocation the user already holds. Scruffy
  discovers its inventory and incarnation exactly as it does for the primary,
  never cancels it, and releasing it only stops placement.
- An **elastic** lease is requested from a configured pool with `sbatch`. Its
  batch payload is `scruffy.lease_agent`, a placeholder that keeps the
  allocation while workers run as steps. The Slurm job name is the lease ID.
  Elastic leases never requeue, because a requeued allocation loses its steps.
- A **hub** is a controller with no primary allocation (`serve --hub`). It owns
  no inventory, runs no health monitor, and places all work on leases.

Lease records are hot state (`state.leases`). Every transition is journaled
with the complete lease image before the controller acts on Slurm, so replay
needs no Slurm query. That includes a `requested` image before `sbatch` and a
`releasing` image before `scancel`. A crash after `sbatch` is resolved by
finding the job by its name; a crash before `scancel` repeats the cancel.
States run `requested → pending → running → releasing → ended`. Only `running`
leases accept placements, and only after two consecutive `squeue` absences (or
a terminal Slurm state) does a lease end.

Ending a lease resolves the jobs still bound to it. A step that Slurm
accounting reports `COMPLETED` keeps its exit code. Anything else becomes
`lost` with reason `lease_ended`, which workflow recovery policies may retry.
A requeue or restart-count change of a running lease also ends it. Lease jobs
are unaffected by a change of the primary allocation and are recovered against
their own lease incarnation at controller restart.

Placement keeps the single-writer scheduler: it tries the primary first and
then running leases, soonest-ending first, choosing the highest-priority job
that fits each. A timed job fits a lease only if its time limit plus the
pool's margin (120 seconds for attached leases) ends before the lease does. An
untimed job may use an attached lease with no known end, or with more than
`--drain-before-end-seconds` left, but never an elastic lease. A multi-node job
must fit within one allocation. A node with an enforced slot-policy GPU
quarantine from the primary's health state is withheld from GPU work on leases
as well; a held quarantine needs no such withholding, because the held device
belongs to the primary allocation.

The elastic autoscaler is a pure function of queued demand, lease records,
pool backoff, and the user's Slurm GPU footprint. Each pass it:

1. withdraws pending leases blocked by the user's own QOS or association
   limits, or whose estimated start exceeds the pool's `start_within_seconds`,
   and backs that pool off;
2. releases running elastic leases that have been idle for
   `idle_release_seconds` or are about to expire;
3. packs queued work, in priority order, into pending leases first and then
   into new ones, within `max_leases`, `max_pending_leases`, `max_gpus`, and
   `max_total_gpus`;
4. withdraws settled pending leases that no queued job needs any more.

Idle timing is kept in controller memory, so a restarted controller never
releases a lease immediately. Growth stops while launches are paused or
draining, and while `squeue` observation is failing.

The lease agent is a dead-man's switch. It exits, ending the allocation, only
when the controller's lease heartbeat (`leases/heartbeat.json`) has been stale
for longer than `heartbeat_timeout_seconds` *and* no numbered worker step is
live. Any uncertainty keeps the allocation.

## Deliberate non-goals

- Artifact bytes remain in normal project storage. Scruffy may later retain a
  bounded manifest of declared and worker-validated artifacts.
- Scientific concepts such as canary promotion and “no candidates” are client
  workflow or workload-outcome concepts, not scheduler lifecycle states.
- Infrastructure retry remains a workload/client policy. GPU quarantine uses
  controller-owned CUDA, thermal, and ECC evidence, never stderr matching.
- Start-time estimates are intentionally absent. Scruffy reports deterministic
  blockers and resource eligibility instead of an unreliable ETA.

These boundaries keep the single-writer loop boring: validate and commit a
submission, own live reservations, record execution transitions, and publish bounded
views. Rich experiment behavior can be built above those facts without making
resource recovery depend on a workflow framework.
