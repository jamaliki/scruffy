const PROJECT_COLORS = [
  "#ff6b42", "#58c8bf", "#8aa6ff", "#d78ac7", "#91c96f", "#ff8b78",
  "#f4c15d", "#6fd9e7", "#b69cff", "#72d6a0", "#ff8bd8", "#b8d66a",
];
const TERMINAL = new Set(["succeeded", "failed", "cancelled", "lost", "rejected", "skipped"]);
export const TELEMETRY_STALE_AFTER_MS = 5 * 60 * 1000;

export function projectColor(project) {
  if (!project || project === "default") return "#78979a";
  let hash = 2166136261;
  for (const character of project) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return PROJECT_COLORS[hash % PROJECT_COLORS.length];
}

export function uniqueJobs(snapshot) {
  const jobs = new Map();
  for (const section of ["submitted", "active", "queued", "blocked", "requires_attention", "recent_terminal"]) {
    for (const job of snapshot?.[section] || []) jobs.set(job.id, job);
  }
  return jobs;
}

export function projects(snapshot) {
  return [...new Set([...uniqueJobs(snapshot).values()].map((job) => job.project_id || "default"))].sort();
}

export function projectSummaries(snapshot) {
  const summaries = new Map();
  const summaryFor = (project) => {
    if (!summaries.has(project)) {
      summaries.set(project, {project, gpus: 0, active: [], queued: [], blocked: []});
    }
    return summaries.get(project);
  };
  const add = (job, lane) => {
    const project = job.project_id || "default";
    const summary = summaryFor(project);
    if (!summary[lane].some((candidate) => candidate.id === job.id)) summary[lane].push(job);
  };
  for (const job of snapshot?.active || []) add(job, "active");
  for (const job of [...(snapshot?.queued || []), ...(snapshot?.submitted || [])]) add(job, "queued");
  for (const job of snapshot?.blocked || []) add(job, "blocked");

  const jobs = uniqueJobs(snapshot);
  for (const node of Object.values(snapshot?.nodes || {})) {
    for (const [jobId, assignment] of Object.entries(node.assignments || {})) {
      const job = jobs.get(jobId);
      if (!job) continue;
      const project = job.project_id || "default";
      summaryFor(project).gpus += assignment.gpu_ids?.length || 0;
    }
  }
  return [...summaries.values()].sort((left, right) =>
    right.gpus - left.gpus
    || (right.active.length + right.queued.length + right.blocked.length) - (left.active.length + left.queued.length + left.blocked.length)
    || left.project.localeCompare(right.project),
  );
}

export function resourceTotals(nodes = {}) {
  const totals = {gpus: 0, freeGpus: 0, cpus: 0, freeCpus: 0, memory: 0, freeMemory: 0};
  for (const node of Object.values(nodes)) {
    totals.gpus += node.capacity?.gpu_ids?.length || 0;
    totals.freeGpus += node.free?.gpu_ids?.length || 0;
    totals.cpus += Number(node.capacity?.cpus || 0);
    totals.freeCpus += Number(node.free?.cpus || 0);
    totals.memory += Number(node.capacity?.memory_gb || 0);
    totals.freeMemory += Number(node.free?.memory_gb || 0);
  }
  return totals;
}

export function gpuOwners(node) {
  const owners = new Map();
  for (const [jobId, reservation] of Object.entries(node.assignments || {})) {
    for (const gpu of reservation.gpu_ids || []) owners.set(gpu, jobId);
  }
  return owners;
}

export function visibleJobs(jobs, project, search) {
  const needle = search.trim().toLowerCase();
  return (jobs || []).filter((job) => {
    if (project !== "all" && (job.project_id || "default") !== project) return false;
    if (!needle) return true;
    return [job.name, job.id, job.workflow_id, job.task_id]
      .filter(Boolean).some((value) => String(value).toLowerCase().includes(needle));
  });
}

export function workflowChoices(snapshot, project = "all", search = "") {
  const needle = search.trim().toLowerCase();
  const grouped = new Map();
  for (const job of uniqueJobs(snapshot).values()) {
    const workflow = job.workflow_id;
    const jobProject = job.project_id || "default";
    if (!workflow || (project !== "all" && jobProject !== project)) continue;
    if (needle && ![job.name, job.id, workflow, job.task_id]
      .filter(Boolean).some((value) => String(value).toLowerCase().includes(needle))) continue;
    const key = JSON.stringify([jobProject, workflow]);
    const current = grouped.get(key) || {key, project: jobProject, workflow_id: workflow, jobs: 0, active: 0, attention: 0, newest: ""};
    current.jobs += 1;
    if (["running", "starting", "finishing", "queued", "submitted", "blocked"].includes(job.state)) current.active += 1;
    if (["failed", "lost", "rejected", "skipped"].includes(job.state)) current.attention += 1;
    current.newest = [current.newest, job.finished_at, job.started_at, job.submitted_at].filter(Boolean).sort().at(-1) || "";
    grouped.set(key, current);
  }
  return [...grouped.values()].sort((left, right) =>
    right.active - left.active || right.attention - left.attention
    || right.newest.localeCompare(left.newest) || left.workflow_id.localeCompare(right.workflow_id),
  );
}

export function workflowLayout(tasks = []) {
  const nodes = new Map(tasks.map((task) => [task.task_id, {...task, missing: false}]));
  for (const task of tasks) {
    for (const need of task.needs || []) {
      if (need.task_id && !nodes.has(need.task_id)) {
        nodes.set(need.task_id, {task_id: need.task_id, name: "Missing dependency", state: "missing", needs: [], missing: true});
      }
    }
  }
  const visiting = new Set();
  const depths = new Map();
  const depthOf = (taskId) => {
    if (depths.has(taskId)) return depths.get(taskId);
    if (visiting.has(taskId)) return 0;
    visiting.add(taskId);
    const task = nodes.get(taskId);
    const parents = (task?.needs || []).map((need) => need.task_id).filter((taskId) => nodes.has(taskId));
    const depth = parents.length ? Math.max(...parents.map(depthOf)) + 1 : 0;
    visiting.delete(taskId); depths.set(taskId, depth); return depth;
  };
  for (const taskId of nodes.keys()) depthOf(taskId);
  const columns = new Map();
  for (const task of nodes.values()) {
    const depth = depths.get(task.task_id) || 0;
    if (!columns.has(depth)) columns.set(depth, []);
    columns.get(depth).push(task);
  }
  const nodeWidth = 210, nodeHeight = 82, columnGap = 92, rowGap = 18;
  const positioned = [];
  for (const [depth, column] of [...columns.entries()].sort(([left], [right]) => left - right)) {
    column.sort((left, right) => String(left.task_id).localeCompare(String(right.task_id), undefined, {numeric: true}));
    column.forEach((task, row) => positioned.push({...task, x: 24 + depth * (nodeWidth + columnGap), y: 24 + row * (nodeHeight + rowGap)}));
  }
  const positionedById = new Map(positioned.map((task) => [task.task_id, task]));
  const edges = [];
  for (const task of positioned) {
    for (const need of task.needs || []) {
      const source = positionedById.get(need.task_id);
      if (source) edges.push({source, target: task, condition: need.condition || "succeeded"});
    }
  }
  return {
    nodes: positioned, edges, nodeWidth, nodeHeight,
    width: Math.max(520, ...positioned.map((task) => task.x + nodeWidth + 24)),
    height: Math.max(190, ...positioned.map((task) => task.y + nodeHeight + 24)),
  };
}

export function focusedWorkflowTasks(tasks = [], maximum = 120) {
  if (tasks.length <= maximum) return {tasks, omitted: 0};
  const byId = new Map(tasks.map((task) => [task.task_id, task]));
  let seeds = tasks.filter((task) =>
    ["submitted", "blocked", "queued", "starting", "running", "finishing", "failed", "lost", "rejected", "skipped"].includes(task.state),
  );
  if (!seeds.length) {
    seeds = [...tasks].sort((left, right) => String(right.submitted_at || "").localeCompare(String(left.submitted_at || ""))).slice(0, 8);
  }
  seeds.sort((left, right) => String(right.submitted_at || "").localeCompare(String(left.submitted_at || "")));
  const included = new Set();
  const pending = seeds.map((task) => task.task_id);
  while (pending.length && included.size < maximum) {
    const taskId = pending.shift();
    if (included.has(taskId) || !byId.has(taskId)) continue;
    included.add(taskId);
    for (const need of byId.get(taskId).needs || []) pending.push(need.task_id);
  }
  const selected = tasks.filter((task) => included.has(task.task_id));
  return {tasks: selected, omitted: tasks.length - selected.length};
}

export function formatNumber(value) {
  return new Intl.NumberFormat("en-GB", {maximumFractionDigits: 1}).format(Number(value || 0));
}

export function formatDuration(start, finish = null) {
  if (!start) return "Not started";
  const started = new Date(start).getTime();
  const end = finish ? new Date(finish).getTime() : Date.now();
  if (!Number.isFinite(started) || !Number.isFinite(end)) return "Unknown time";
  const seconds = Math.max(0, Math.floor((end - started) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

export function formatAge(timestamp) {
  if (!timestamp) return "unknown age";
  const recorded = new Date(timestamp).getTime();
  if (!Number.isFinite(recorded)) return "unknown age";
  const seconds = Math.max(0, Math.floor((Date.now() - recorded) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export function resourceLabel(job) {
  const request = job?.request || {};
  return `${request.nodes || 0}n · ${request.gpus_per_node || 0} GPU/n · ${request.cpus_per_node || 0} CPU/n`;
}

export function progressLabel(job) {
  const workload = job?.workload || {};
  const progress = workload.progress || {};
  const step = progress.step ?? progress.current_step;
  const target = progress.total_steps ?? progress.target_step ?? progress.max_steps;
  if (step != null && target != null) return `${formatNumber(step)} / ${formatNumber(target)} steps`;
  if (step != null) return `Step ${formatNumber(step)}`;
  if (progress.percent != null) return `${formatNumber(progress.percent)}%`;
  return workload.phase || workload.status || job?.reason || "No workload report";
}

export function stateTone(state) {
  if (state === "succeeded") return "success";
  if (["failed", "lost", "rejected", "skipped"].includes(state)) return "danger";
  if (["blocked", "cancelled"].includes(state)) return "warning";
  if (["running", "starting"].includes(state)) return "active";
  return "neutral";
}

export function allocationIsStale(snapshot, now = Date.now()) {
  const heartbeat = snapshot?.allocation?.heartbeat_at;
  const recorded = heartbeat ? new Date(heartbeat).getTime() : Number.NaN;
  return !Number.isFinite(recorded) || now - recorded > TELEMETRY_STALE_AFTER_MS;
}

export function scalarTelemetry(workload = {}) {
  const rows = [];
  const add = (label, value) => {
    if (["string", "number", "boolean"].includes(typeof value)) rows.push([label, String(value)]);
  };
  add("Phase", workload.phase);
  add("Status", workload.status);
  for (const [key, value] of Object.entries(workload.progress || {})) add(key.replaceAll("_", " "), value);
  add("Last update", workload.last_update_at);
  return rows.slice(0, 12);
}

export function isTerminal(job) {
  return TERMINAL.has(job?.state);
}
