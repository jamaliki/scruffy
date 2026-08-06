const PROJECT_COLORS = ["#f97316", "#0E6E66", "#3F4FA8", "#7E3F70", "#5A7A3C", "#B7472A"];
const TERMINAL = new Set(["succeeded", "failed", "cancelled", "lost", "rejected", "skipped"]);
export const TELEMETRY_STALE_AFTER_MS = 5 * 60 * 1000;

export function projectColor(project) {
  if (!project || project === "default") return "#64748b";
  let hash = 0;
  for (const character of project) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
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
