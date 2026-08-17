import {
  allocationIsStale, dependencyLinkedTasks, formatAge, formatDuration, formatNumber, gpuOwners,
  progressLabel, projectColor, projects, projectSummaries, resourceLabel, resourceTotals,
  scalarTelemetry, stateTone, TELEMETRY_STALE_AFTER_MS, uniqueJobs, visibleJobs,
  focusedWorkflowTasks, workflowChoices, workflowLayout,
} from "/assets/model.js";

const byId = (id) => document.getElementById(id);
const LOG_CHUNK_BYTES = 128 * 1024;
const MAX_VISIBLE_LOG_BYTES = 2 * 1024 * 1024;
const LIVE_JOB_STATES = new Set(["starting", "running", "finishing", "cancelling"]);
const view = {
  snapshot: null, project: "all", search: "", connected: false, loading: false,
  lastSuccessAt: null,
  workflowKey: "", workflowData: null, workflowDataKey: "", workflowLoading: false,
  gpuReport: "",
  log: {
    jobId: "", name: "", stream: "stdout", state: "", chunks: [],
    loading: false, following: false, retained: true, generation: 0,
  },
};

function telemetryIsStale(snapshot) {
  const refreshExpired = view.lastSuccessAt != null
    && Date.now() - view.lastSuccessAt > TELEMETRY_STALE_AFTER_MS;
  return refreshExpired || allocationIsStale(snapshot);
}

function element(tag, className = "", text = null) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text != null) item.textContent = text;
  return item;
}

function replace(target, children) {
  target.replaceChildren(...children);
}

function metric(label, value, detail) {
  const card = element("article", "metric");
  card.append(element("span", "metric-label", label), element("strong", "metric-value", value));
  card.append(element("span", "metric-detail", detail));
  return card;
}

function renderMetrics(snapshot, stale) {
  const totals = resourceTotals(snapshot.nodes);
  const counts = snapshot.counts || {};
  const usedGpus = Object.values(snapshot.nodes || {}).reduce((total, node) => total + gpuOwners(node).size, 0);
  const stoppedGpus = Math.max(0, totals.gpus - totals.freeGpus - usedGpus);
  const usedCpus = totals.cpus - totals.freeCpus;
  const usedMemory = totals.memory - totals.freeMemory;
  replace(byId("metrics"), [
    metric("GPU allocation", `${stale ? "?" : usedGpus} / ${totals.gpus}`, stale ? "availability unknown" : `${totals.freeGpus} available · ${stoppedGpus} stopped/held`),
    metric("CPU allocation", `${stale ? "?" : usedCpus} / ${totals.cpus}`, stale ? "availability unknown" : `${totals.freeCpus} cores available`),
    metric("Memory allocation", `${stale ? "?" : formatNumber(usedMemory)} / ${formatNumber(totals.memory)}`, stale ? "availability unknown" : `${formatNumber(totals.freeMemory)} GB available`),
    metric("Waiting", String(Number(counts.submitted || 0) + Number(counts.queued || 0)), `${counts.blocked || 0} dependency-blocked`),
  ]);
}

function progressBar(label, used, total) {
  const wrapper = element("div", "resource-bar");
  const header = element("div", "resource-bar-label");
  header.append(element("span", "", label), element("span", "mono", `${formatNumber(used)} / ${formatNumber(total)}`));
  const track = element("div", "bar-track");
  const fill = element("i", "bar-fill");
  fill.style.width = total ? `${Math.min(100, used / total * 100)}%` : "0%";
  track.append(fill); wrapper.append(header, track);
  return wrapper;
}

function renderNode(name, nodeState, jobs, stale) {
  const card = element("article", "node-card");
  const header = element("header", "node-header");
  header.append(element("h3", "node-name", name));
  const gpuIds = nodeState.capacity?.gpu_ids || [];
  const free = new Set(nodeState.free?.gpu_ids || []);
  const unavailable = new Set(nodeState.unavailable_gpu_ids || []);
  const devices = new Map((nodeState.gpu_devices || []).map((device) => [device.slot, device]));
  header.append(element("span", "node-free", stale ? "GPU availability unknown" : `${free.size} / ${gpuIds.length} GPUs free`));
  const rack = element("div", "gpu-rack");
  const owners = gpuOwners(nodeState);
  for (const gpuId of gpuIds) {
    const jobId = owners.get(gpuId);
    const job = jobs.get(jobId);
    const project = job?.project_id || "default";
    const device = devices.get(gpuId) || {slot: gpuId, status: "unknown"};
    const available = free.has(gpuId);
    let schedulerState = "unavailable", label = "UNAVAILABLE";
    if (jobId) { schedulerState = "occupied"; label = job?.name || jobId || "ASSIGNED"; }
    else if (unavailable.has(gpuId) && device.status === "quarantined") { schedulerState = "stopped"; label = "STOPPED"; }
    else if (unavailable.has(gpuId) && device.status === "healthy") { schedulerState = "node-held"; label = "NODE HELD"; }
    else if (unavailable.has(gpuId)) { schedulerState = "unknown"; label = "HEALTH UNKNOWN"; }
    else if (device.status === "quarantined") { schedulerState = "flagged"; label = "FLAGGED"; }
    else if (available) { schedulerState = stale ? "unknown" : "free"; label = stale ? "UNKNOWN" : "FREE"; }
    const tile = element("button", `gpu ${schedulerState}${device.status === "quarantined" ? " unhealthy" : ""}`);
    tile.type = "button";
    tile.append(element("span", "gpu-id", `GPU ${gpuId}`));
    tile.append(element("strong", "gpu-owner", label));
    tile.dataset.gpuNode = name;
    tile.dataset.gpuSlot = String(gpuId);
    if (jobId) {
      tile.style.setProperty("--project-color", projectColor(project));
      tile.dataset.jobId = jobId || "";
      tile.classList.toggle("foreign", view.project !== "all" && project !== view.project);
    }
    const realId = device.uuid || "UUID not observed";
    tile.title = `${realId} · PCI ${device.pci_bus_id || "unknown"}${jobId ? ` · ${job?.name || jobId} · ${project}` : ""}`;
    rack.append(tile);
  }
  const capacity = nodeState.capacity || {};
  const available = nodeState.free || {};
  const bars = element("div", "node-bars");
  bars.append(
    progressBar("CPU", Number(capacity.cpus || 0) - Number(available.cpus || 0), Number(capacity.cpus || 0)),
    progressBar("Memory GB", Number(capacity.memory_gb || 0) - Number(available.memory_gb || 0), Number(capacity.memory_gb || 0)),
  );
  card.append(header, rack, bars);
  return card;
}

function renderNodes(snapshot) {
  const jobs = uniqueJobs(snapshot);
  const entries = Object.entries(snapshot.nodes || {}).sort(([left], [right]) => left.localeCompare(right, undefined, {numeric: true}));
  const stale = telemetryIsStale(snapshot);
  replace(byId("nodes"), entries.length ? entries.map(([name, state]) => renderNode(name, state, jobs, stale)) : [empty("No managed nodes are visible.")]);
  const totals = resourceTotals(snapshot.nodes);
  const withheld = entries.reduce((total, [, node]) => total + (node.unavailable_gpu_ids || []).length, 0);
  byId("resource-note").textContent = stale ? `${entries.length} nodes / ${totals.gpus} GPUs / availability unknown` : `${entries.length} nodes / ${totals.gpus} GPUs / ${totals.freeGpus} available / ${withheld} health-withheld`;
}

function legendItem(label, color = null, project = null) {
  const interactive = project !== null;
  const item = element(interactive ? "button" : "span", "legend-item");
  if (interactive) {
    const selected = view.project === project;
    item.type = "button";
    item.dataset.projectFilter = project;
    item.style.setProperty("--project-color", color || "var(--teal)");
    item.classList.toggle("selected", selected);
    item.setAttribute("aria-pressed", String(selected));
    item.title = project === "all"
      ? "Show all projects"
      : selected ? "Clear project filter" : `Filter dashboard to ${project}`;
  }
  const swatchClass = project === "all"
    ? "legend-swatch all"
    : color ? "legend-swatch" : "legend-swatch free";
  const swatch = element("i", swatchClass);
  if (color) swatch.style.setProperty("--project-color", color);
  item.append(swatch, element("span", "", label));
  return item;
}

function renderProjectLegend(snapshot) {
  const activeProjects = projectSummaries(snapshot).filter((summary) => summary.gpus > 0);
  replace(byId("project-legend"), [
    legendItem("All projects", null, "all"),
    legendItem("Free"),
    ...activeProjects.map((summary) => legendItem(summary.project, projectColor(summary.project), summary.project)),
  ]);
}

function projectTag(project) {
  const tag = element("span", "project-tag", project || "default");
  tag.style.setProperty("--project-color", projectColor(project));
  return tag;
}

function jobRow(job, compact = false) {
  const row = element("button", `job-row ${compact ? "compact" : ""}`);
  row.type = "button"; row.dataset.jobId = job.id;
  const marker = element("i", `state-marker ${stateTone(job.state)}`);
  const main = element("span", "job-main");
  const top = element("span", "job-top");
  top.append(element("strong", "job-name", job.name || job.id), projectTag(job.project_id));
  const secondary = compact ? `${job.state} · ${formatAge(job.finished_at || job.submitted_at)}` : `${resourceLabel(job)} · ${progressLabel(job)}`;
  main.append(top, element("span", "job-secondary", secondary));
  const tail = element("span", "job-tail");
  tail.append(element("span", `state-text ${stateTone(job.state)}`, String(job.state || "unknown")));
  if (!compact) tail.append(element("span", "mono", formatDuration(job.started_at, job.finished_at)));
  row.append(marker, main, tail);
  return row;
}

function empty(message) {
  return element("p", "empty", message);
}

function renderList(targetId, jobs, emptyMessage, compact = false) {
  const selected = visibleJobs(jobs, view.project, view.search);
  replace(byId(targetId), selected.length ? selected.map((job) => jobRow(job, compact)) : [empty(emptyMessage)]);
  const counter = byId(`${targetId}-count`);
  if (counter) counter.textContent = String(selected.length);
  return selected.length;
}

function projectLane(label, jobs, emptyMessage) {
  const lane = element("section", "project-lane");
  const heading = element("h4", "");
  heading.append(element("span", "", label), element("strong", "", String(jobs.length)));
  const list = element("div", "job-list");
  list.append(...(jobs.length ? jobs.map((job) => jobRow(job, true)) : [empty(emptyMessage)]));
  lane.append(heading, list);
  return lane;
}

function projectCard(summary) {
  const card = element("article", "project-card");
  card.style.setProperty("--project-color", projectColor(summary.project));
  const header = element("header", "project-card-header");
  const identity = element("div", "project-identity");
  identity.append(element("span", "project-swatch"));
  const name = element("button", "project-name", summary.project);
  name.type = "button"; name.dataset.projectFilter = summary.project;
  identity.append(name);
  const stats = element("div", "project-stats");
  stats.append(
    element("span", "", `${summary.gpus} GPU${summary.gpus === 1 ? "" : "s"}`),
    element("span", "", `${summary.active.length} running`),
    element("span", "", `${summary.queued.length} queued`),
    element("span", "", `${summary.blocked.length} blocked`),
  );
  header.append(identity, stats);
  const lanes = element("div", "project-lanes");
  lanes.append(
    projectLane("Running", summary.active, "No running jobs."),
    projectLane("Queued", summary.queued, "No jobs awaiting placement."),
    projectLane("Blocked", summary.blocked, "No dependency-blocked jobs."),
  );
  card.append(header, lanes);
  return card;
}

function renderProjectBoard(snapshot) {
  const summaries = projectSummaries(snapshot)
    .filter((summary) => view.project === "all" || summary.project === view.project)
    .map((summary) => ({
      ...summary,
      active: visibleJobs(summary.active, "all", view.search),
      queued: visibleJobs(summary.queued, "all", view.search),
      blocked: visibleJobs(summary.blocked, "all", view.search),
    }))
    .filter((summary) => summary.active.length || summary.queued.length || summary.blocked.length);
  byId("project-count").textContent = String(summaries.length);
  replace(byId("project-board"), summaries.length
    ? summaries.map(projectCard)
    : [empty("No project has matching active, queued or blocked work.")]);
}

function renderProjects(snapshot) {
  const select = byId("project-filter");
  const choices = [element("option", "", "All projects")];
  choices[0].value = "all";
  for (const project of projects(snapshot)) {
    const option = element("option", "", project); option.value = project; choices.push(option);
  }
  replace(select, choices);
  if (![...select.options].some((option) => option.value === view.project)) view.project = "all";
  select.value = view.project;
}

function renderWorkflowGraph() {
  const canvas = byId("workflow-canvas");
  const viewport = byId("workflow-viewport");
  const data = view.workflowData;
  const linkedTasks = dependencyLinkedTasks(data?.tasks || []);
  if (!data || !linkedTasks.length) {
    canvas.style.width = "100%"; canvas.style.height = "190px";
    const message = view.workflowLoading
      ? "Reading workflow…"
      : data ? "This workflow has no dependency-linked tasks." : "No workflow tasks are available.";
    replace(canvas, [empty(message)]);
    viewport.scrollTo(0, 0);
    return;
  }
  const focused = focusedWorkflowTasks(linkedTasks);
  const layout = workflowLayout(focused.tasks);
  if (focused.omitted) {
    byId("workflow-status").textContent = `${data.project_id} / showing ${focused.tasks.length} relevant tasks of ${linkedTasks.length} dependency-linked / ${data.attempt_count} attempts`;
  }
  canvas.style.width = `${layout.width}px`; canvas.style.height = `${layout.height}px`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("workflow-edges"); svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  for (const edge of layout.edges) {
    const startX = edge.source.x + layout.nodeWidth, startY = edge.source.y + layout.nodeHeight / 2;
    const endX = edge.target.x, endY = edge.target.y + layout.nodeHeight / 2;
    const bend = Math.max(36, (endX - startX) / 2);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", `M ${startX} ${startY} C ${startX + bend} ${startY}, ${endX - bend} ${endY}, ${endX} ${endY}`);
    line.classList.add("workflow-edge", edge.condition === "terminal" ? "terminal" : edge.condition === "artifact" ? "artifact" : "success");
    svg.append(line);
  }
  const children = [svg];
  for (const task of layout.nodes) {
    const node = element("button", `workflow-node ${stateTone(task.state)}${task.missing ? " missing" : ""}`);
    node.type = "button"; node.style.left = `${task.x}px`; node.style.top = `${task.y}px`;
    node.style.width = `${layout.nodeWidth}px`; node.style.height = `${layout.nodeHeight}px`;
    if (task.job_id) node.dataset.jobId = task.job_id; else node.disabled = true;
    const header = element("span", "workflow-node-header");
    header.append(element("strong", "", task.task_id), element("span", "state-text", task.state || "unknown"));
    const attemptCount = task.attempts?.length || 0;
    node.append(header, element("span", "workflow-node-name", task.name || task.job_id || "Unknown task"));
    if (attemptCount > 1) node.append(element("span", "workflow-attempts", `${attemptCount} attempts · current ${task.attempt ?? attemptCount}`));
    children.push(node);
  }
  replace(canvas, children);
}

async function loadWorkflow() {
  const choice = workflowChoices(view.snapshot, view.project, view.search).find((item) => item.key === view.workflowKey);
  if (!choice) {
    view.workflowData = null; view.workflowDataKey = ""; renderWorkflowGraph(); return;
  }
  const requestKey = choice.key;
  const replacing = view.workflowDataKey !== requestKey;
  view.workflowLoading = true;
  if (replacing) {
    view.workflowData = null; view.workflowDataKey = ""; renderWorkflowGraph();
  }
  byId("workflow-status").textContent = `${choice.project} / ${choice.workflow_id}${replacing ? "" : " / refreshing"}`;
  try {
    const response = await fetch(`/api/workflows/${encodeURIComponent(choice.workflow_id)}?project=${encodeURIComponent(choice.project)}`, {cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Workflow lookup failed");
    if (view.workflowKey !== requestKey) return;
    view.workflowData = data; view.workflowDataKey = requestKey;
    const linked = dependencyLinkedTasks(data.tasks || []).length;
    byId("workflow-status").textContent = `${data.project_id} / ${linked} dependency-linked tasks / ${data.attempt_count} attempts`;
  } catch (error) {
    if (view.workflowKey === requestKey) byId("workflow-status").textContent = error.message;
  } finally {
    if (view.workflowKey === requestKey) { view.workflowLoading = false; renderWorkflowGraph(); }
  }
}

function workflowChoiceFromKey(key) {
  try {
    const [project, workflowId] = JSON.parse(key);
    return project && workflowId ? {key, project, workflow_id: workflowId, jobs: 0, retained: true} : null;
  } catch {
    return null;
  }
}

function renderWorkflowChoices(snapshot, preserveCurrent = true) {
  const select = byId("workflow-filter");
  const choices = workflowChoices(snapshot, view.project, view.search);
  if (preserveCurrent && view.workflowKey && !choices.some((choice) => choice.key === view.workflowKey)) {
    const retained = workflowChoiceFromKey(view.workflowKey);
    if (retained) choices.unshift(retained);
  }
  if (!choices.some((choice) => choice.key === view.workflowKey)) {
    view.workflowKey = choices[0]?.key || "";
    view.workflowData = null; view.workflowDataKey = "";
  }
  const options = choices.map((choice) => {
    const suffix = choice.retained ? "selected" : `${choice.jobs} tasks visible`;
    const option = element("option", "", `${choice.workflow_id} · ${choice.project} · ${suffix}`);
    option.value = choice.key; return option;
  });
  if (!options.length) { const option = element("option", "", "No workflows visible"); option.value = ""; options.push(option); }
  replace(select, options); select.value = view.workflowKey;
  if (!view.workflowKey) {
    byId("workflow-status").textContent = "No workflows match the current project and search filters.";
    view.workflowData = null; view.workflowDataKey = ""; renderWorkflowGraph();
  }
}

function renderConnection(snapshot) {
  const allocation = snapshot.allocation || {};
  const stale = telemetryIsStale(snapshot);
  const paused = Boolean(snapshot.launches_paused);
  byId("allocation-state").textContent = stale ? "Telemetry stale" : paused ? "Recovery paused" : String(allocation.state || "No allocation");
  byId("allocation-id").textContent = allocation.id || "Waiting for controller";
  const connectionTone = stale ? "stale" : view.connected ? "live" : "degraded";
  byId("connection-dot").className = `connection-dot ${connectionTone}`;
  byId("stale-banner").hidden = !stale;
  byId("stale-banner").textContent = "Queue telemetry has not refreshed for five minutes. Resource availability is unknown.";
  byId("freshness").textContent = `Heartbeat ${formatAge(allocation.heartbeat_at)}${paused ? " · launches paused" : ""}`;
  byId("queue-id").textContent = snapshot.queue_id || "Queue not identified";
  document.body.classList.toggle("data-stale", stale);
}

function render(preserveWorkflow = true) {
  const snapshot = view.snapshot;
  if (!snapshot) return;
  const stale = telemetryIsStale(snapshot);
  renderConnection(snapshot); renderMetrics(snapshot, stale); renderProjects(snapshot); renderNodes(snapshot);
  renderProjectLegend(snapshot); renderProjectBoard(snapshot);
  renderWorkflowChoices(snapshot, preserveWorkflow);
  renderList("active", snapshot.active, "No jobs are using resources.");
  renderList("queued", [...(snapshot.queued || []), ...(snapshot.submitted || [])], "No jobs are waiting for placement.");
  renderList("blocked", snapshot.blocked, "No workflows are dependency-blocked.");
  const attention = renderList("attention", snapshot.requires_attention, "Nothing requires attention.", true);
  byId("attention-count").textContent = String(attention);
  renderList("recent", snapshot.recent_terminal, "No recent terminal jobs.", true);
}

function detailSection(title, rows) {
  const section = element("section", "detail-section");
  section.append(element("h3", "", title));
  const list = element("dl", "detail-grid");
  for (const [label, value] of rows) {
    const description = element("dd");
    if (value instanceof Node) description.append(value);
    else description.textContent = value == null || value === "" ? "—" : String(value);
    list.append(element("dt", "", label), description);
  }
  section.append(list); return section;
}

function logPathButton(job, stream) {
  const button = element("button", "log-path", job[stream] || `Open ${stream}`);
  button.type = "button";
  button.dataset.logJobId = job.id;
  button.dataset.logStream = stream;
  button.dataset.logJobName = job.name || job.id;
  return button;
}

async function openJob(jobId) {
  if (!jobId) return;
  const dialog = byId("job-dialog");
  byId("dialog-title").textContent = jobId;
  byId("dialog-project").textContent = "Reading job state";
  replace(byId("job-detail"), [empty("Loading compact dependency view…")]);
  if (!dialog.open) dialog.showModal();
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const explanation = await response.json();
    if (!response.ok) throw new Error(explanation.error || "Job lookup failed");
    const job = explanation.job;
    byId("dialog-title").textContent = job.name || job.id;
    byId("dialog-project").textContent = `${job.project_id || "default"} // ${job.state}`;
    const sections = [
      detailSection("Lifecycle", [["Job", job.id], ["State", job.state], ["Reason", job.reason], ["Runtime", formatDuration(job.started_at, job.finished_at)], ["Exit code", job.exit_code]]),
      detailSection("Placement", [["Request", resourceLabel(job)], ["Assignment", JSON.stringify(job.assignment || "Not assigned")], ["Stdout", logPathButton(job, "stdout")], ["Stderr", logPathButton(job, "stderr")]]),
      detailSection("Workflow", [["Workflow", job.workflow_id], ["Task", job.task_id], ["Explanation", explanation.explanation], ["Blockers", JSON.stringify(explanation.blockers || job.blockers || [])]]),
      detailSection("Workload telemetry", scalarTelemetry(job.workload || {})),
    ];
    replace(byId("job-detail"), sections);
  } catch (error) {
    replace(byId("job-detail"), [empty(error.message)]);
  }
}

function visibleLogBytes() {
  return view.log.chunks.reduce((total, chunk) => total + chunk.bytes, 0);
}

function trimLogChunks(prepending) {
  while (view.log.chunks.length > 1 && visibleLogBytes() > MAX_VISIBLE_LOG_BYTES) {
    if (prepending) view.log.chunks.pop();
    else view.log.chunks.shift();
  }
}

function renderLog(scrollToEnd = false) {
  const log = view.log;
  const text = byId("log-text");
  const body = log.chunks.map((chunk) => chunk.text).join("");
  text.textContent = body;
  byId("log-empty").hidden = body.length > 0 || log.loading;
  const first = log.chunks[0];
  const last = log.chunks.at(-1);
  const start = first?.start || 0;
  const end = last?.end || 0;
  const total = last?.total_bytes ?? first?.total_bytes ?? 0;
  byId("log-position").textContent = log.retained
    ? `${start.toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()} bytes`
    : "Output is not retained on this controller";
  byId("log-older").disabled = log.loading || !first?.more_before;
  byId("log-latest").disabled = log.loading || (!last?.more_after && end >= total);
  byId("log-follow").disabled = !LIVE_JOB_STATES.has(log.state);
  byId("log-follow").classList.toggle("active", log.following);
  byId("log-follow").setAttribute("aria-pressed", String(log.following));
  byId("log-follow").textContent = log.following ? "Following live" : "Follow live";
  for (const button of document.querySelectorAll(".log-stream")) {
    const selected = button.dataset.logStream === log.stream;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  }
  if (scrollToEnd) text.scrollTop = text.scrollHeight;
}

async function loadLogRange(mode = "tail") {
  const log = view.log;
  if (!log.jobId || log.loading) return;
  const generation = log.generation;
  let offset;
  let limit = LOG_CHUNK_BYTES;
  if (mode === "older") {
    const start = log.chunks[0]?.start || 0;
    offset = Math.max(0, start - LOG_CHUNK_BYTES);
    limit = start - offset;
    if (!limit) return;
  } else if (mode === "append") {
    offset = log.chunks.at(-1)?.end || 0;
  }
  const query = new URLSearchParams({limit: String(limit)});
  if (offset != null) query.set("offset", String(offset));
  log.loading = true;
  renderLog();
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(log.jobId)}/output/${log.stream}?${query}`, {cache: "no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Output read failed");
    if (generation !== log.generation) return;
    log.state = result.state || log.state;
    log.retained = result.retained;
    const chunk = {...result};
    if (mode === "tail") log.chunks = [chunk];
    else if (mode === "older") log.chunks.unshift(chunk);
    else if (result.start < offset) log.chunks = [chunk]; // The file was replaced or truncated.
    else if (chunk.bytes || chunk.total_bytes !== log.chunks.at(-1)?.total_bytes) log.chunks.push(chunk);
    trimLogChunks(mode === "older");
    if (!LIVE_JOB_STATES.has(log.state) && !result.more_after) log.following = false;
    renderLog(mode !== "older" && log.following);
  } catch (error) {
    if (generation !== log.generation) return;
    byId("log-position").textContent = error.message;
    log.following = false;
  } finally {
    if (generation === log.generation) {
      log.loading = false;
      renderLog(mode !== "older" && log.following);
    }
  }
}

function openLog(jobId, name, stream) {
  const log = view.log;
  log.generation += 1;
  Object.assign(log, {jobId, name, stream, state: "", chunks: [], loading: false, following: false, retained: true});
  byId("log-dialog-title").textContent = name || jobId;
  byId("log-dialog-state").textContent = `${stream} // ${jobId}`;
  const dialog = byId("log-dialog");
  if (!dialog.open) dialog.showModal();
  renderLog();
  loadLogRange("tail");
}

function closeLog() {
  view.log.generation += 1;
  view.log.following = false;
  byId("log-dialog").close();
}

function gpuReport(device) {
  const metrics = device.metrics || {};
  return [
    "Scruffy GPU health report",
    `Node: ${device.node}`,
    `Scruffy slot: ${device.slot}`,
    `NVIDIA UUID: ${device.uuid || "unknown"}`,
    `NVIDIA index: ${device.nvidia_index ?? "unknown"}`,
    `Linux minor number: ${device.minor_number ?? "unknown"}`,
    `PCI bus ID: ${device.pci_bus_id || "unknown"}`,
    `Serial: ${device.serial || "unknown"}`,
    `Model: ${device.name || "unknown"}`,
    `Driver: ${device.driver_version || "unknown"}`,
    `VBIOS: ${device.vbios_version || "unknown"}`,
    `Health status: ${device.status || "unknown"}`,
    `Scheduler state: ${device.scheduler_state || "unknown"}`,
    `Assigned job: ${device.assigned_job_id || "none"}`,
    `Last sample: ${device.last_sample_at || "never"}`,
    `Controller received: ${device.last_received_at || "never"}`,
    `Quarantined at: ${device.quarantined_at || "not quarantined"}`,
    `Quarantine reason: ${JSON.stringify(device.quarantine_reason || [])}`,
    `Last reasons: ${JSON.stringify(device.last_reasons || [])}`,
    `Temperature C: ${metrics.temperature_c ?? "unknown"}`,
    `Thermal slowdown: ${metrics.thermal_slowdown ?? "unknown"}`,
    `Uncorrectable ECC errors: ${metrics.uncorrectable_ecc_errors ?? "unknown"}`,
    `CUDA probe: ${JSON.stringify(device.cuda_probe || {})}`,
    `Policy: ${JSON.stringify(device.policy || {})}`,
  ].join("\n");
}

function readableLabel(value) {
  if (value == null || value === "") return "Unknown";
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function compactTime(value) {
  if (!value) return "Never";
  const time = element("time", "detail-time", formatAge(value));
  time.dateTime = value;
  time.title = value;
  return time;
}

function gpuStat(label, value, detail, tone = "") {
  const item = element("div", `gpu-stat ${tone}`.trim());
  item.append(
    element("span", "gpu-stat-label", label),
    element("strong", "gpu-stat-value", value),
    element("span", "gpu-stat-detail", detail),
  );
  return item;
}

function cudaSummary(device) {
  const probe = device.cuda_probe || {};
  const probes = Array.isArray(probe.devices) ? probe.devices : [];
  const selected = probes.find((item) => item.uuid === device.uuid)
    || probes.find((item) => item.nvidia_index === device.nvidia_index);
  const passed = probes.filter((item) => item.ok).length;
  const total = Number(probe.device_count ?? probes.length);
  const ok = selected?.ok ?? probe.ok;
  return {
    label: ok === true ? "Ready" : ok === false ? "Failed" : "Not sampled",
    detail: total ? `${passed} of ${total} node GPUs passed` : "No CUDA context result",
    error: selected?.error || probe.error,
    tone: ok === true ? "good" : ok === false ? "bad" : "muted",
  };
}

function gpuSummary(device) {
  const metrics = device.metrics || {};
  const cuda = cudaSummary(device);
  const section = element("section", "gpu-summary");
  const headline = element("div", "gpu-summary-headline");
  const copy = element("div");
  copy.append(
    element("span", `health-badge ${device.status || "unknown"}`, readableLabel(device.status)),
    element(
      "p",
      "gpu-summary-copy",
      device.status === "healthy"
        ? "This GPU is healthy and ready for scheduler use."
        : "This GPU needs attention before it should receive work.",
    ),
  );
  const freshness = element("div", "gpu-freshness");
  freshness.append(element("span", "gpu-stat-label", "Latest observation"), compactTime(device.last_sample_at));
  headline.append(copy, freshness);

  const stats = element("div", "gpu-stats");
  const temperature = metrics.temperature_c == null ? "—" : `${metrics.temperature_c}°C`;
  const power = metrics.power_draw_w == null ? "—" : `${Math.round(metrics.power_draw_w)} W`;
  const powerDetail = metrics.power_limit_w == null ? "Power draw" : `${Math.round(metrics.power_limit_w)} W limit`;
  const ecc = metrics.uncorrectable_ecc_errors == null ? "—" : String(metrics.uncorrectable_ecc_errors);
  stats.append(
    gpuStat("Temperature", temperature, metrics.thermal_slowdown ? "Thermal slowdown active" : "No thermal slowdown", metrics.thermal_slowdown ? "bad" : "good"),
    gpuStat("Power", power, powerDetail),
    gpuStat("CUDA context", cuda.label, cuda.detail, cuda.tone),
    gpuStat("Uncorrectable ECC", ecc, ecc === "0" ? "No errors observed" : "Errors observed", ecc === "0" ? "good" : "bad"),
  );
  section.append(headline, stats);
  return section;
}

async function openGpu(node, slot) {
  const dialog = byId("gpu-dialog");
  byId("gpu-dialog-title").textContent = `${node} / GPU ${slot}`;
  byId("gpu-dialog-state").textContent = "Reading physical identity";
  byId("gpu-report-copy").textContent = "Copy report";
  view.gpuReport = "";
  replace(byId("gpu-detail"), [empty("Loading GPU health and identity…")]);
  if (!dialog.open) dialog.showModal();
  try {
    const response = await fetch(`/api/gpus/${encodeURIComponent(node)}/${encodeURIComponent(slot)}`, {cache: "no-store"});
    const device = await response.json();
    if (!response.ok) throw new Error(device.error || "GPU lookup failed");
    byId("gpu-dialog-title").textContent = `${device.node} / GPU ${device.slot}`;
    byId("gpu-dialog-state").textContent = `${readableLabel(device.status)} · ${readableLabel(device.scheduler_state)}`;
    view.gpuReport = gpuReport(device);
    const cuda = cudaSummary(device);
    const reasons = device.last_reasons || [];
    const policy = device.policy || {};
    replace(byId("gpu-detail"), [
      gpuSummary(device),
      detailSection("Physical identity", [["Model", device.name], ["NVIDIA UUID", device.uuid], ["PCI bus", device.pci_bus_id], ["Serial", device.serial], ["Device", `index ${device.nvidia_index ?? "?"} · Linux minor ${device.minor_number ?? "?"}`]]),
      detailSection("Runtime & policy", [["CUDA", cuda.error ? `${cuda.label} · ${cuda.error}` : `${cuda.label} · ${cuda.detail}`], ["Driver", device.driver_version], ["VBIOS", device.vbios_version], ["Scheduling", `${readableLabel(policy.mode)} · ${readableLabel(policy.isolation)} isolation`], ["Assigned job", device.assigned_job_id || "None"]]),
      detailSection("Observation", [["Sampled", compactTime(device.last_sample_at)], ["Received", compactTime(device.last_received_at)], ["Health notes", reasons.length ? reasons.map(readableLabel).join(" · ") : "No health warnings"], ...(device.quarantined_at ? [["Quarantined", compactTime(device.quarantined_at)], ["Source", readableLabel(device.quarantine_source)]] : [])]),
    ]);
  } catch (error) {
    replace(byId("gpu-detail"), [empty(error.message)]);
  }
}

async function loadOverview(refreshWorkflow = false) {
  if (view.loading) return;
  view.loading = true; byId("refresh").disabled = true;
  try {
    const response = await fetch("/api/overview", {cache: "no-store"});
    const snapshot = await response.json();
    if (!response.ok) throw new Error(snapshot.error || "Queue read failed");
    view.snapshot = snapshot; view.connected = true; view.lastSuccessAt = Date.now(); render();
    if (view.workflowKey && (refreshWorkflow || view.workflowDataKey !== view.workflowKey) && !view.workflowLoading) loadWorkflow();
  } catch (error) {
    view.connected = false;
    if (view.snapshot) {
      render();
    } else {
      byId("stale-banner").hidden = false;
      byId("stale-banner").textContent = `Dashboard unavailable: ${error.message}`;
      byId("connection-dot").className = "connection-dot stale";
    }
  } finally {
    view.loading = false; byId("refresh").disabled = false;
  }
}

document.addEventListener("click", (event) => {
  const logTarget = event.target.closest("[data-log-job-id]");
  if (logTarget) {
    openLog(logTarget.dataset.logJobId, logTarget.dataset.logJobName, logTarget.dataset.logStream);
    return;
  }
  const logStream = event.target.closest(".log-stream");
  if (logStream && view.log.jobId && logStream.dataset.logStream !== view.log.stream) {
    openLog(view.log.jobId, view.log.name, logStream.dataset.logStream);
    return;
  }
  const projectFilter = event.target.closest("[data-project-filter]");
  if (projectFilter) {
    const requested = projectFilter.dataset.projectFilter;
    const previousWorkflow = view.workflowKey;
    view.project = requested === "all" || view.project === requested ? "all" : requested;
    render(false);
    if (view.workflowKey && view.workflowKey !== previousWorkflow) loadWorkflow();
    return;
  }
  const gpuTarget = event.target.closest("[data-gpu-node]");
  if (gpuTarget) {
    openGpu(gpuTarget.dataset.gpuNode, gpuTarget.dataset.gpuSlot);
    return;
  }
  const target = event.target.closest("[data-job-id]");
  if (target) openJob(target.dataset.jobId);
});
byId("project-filter").addEventListener("change", (event) => { const previous = view.workflowKey; view.project = event.target.value; render(false); if (view.workflowKey && view.workflowKey !== previous) loadWorkflow(); });
byId("job-search").addEventListener("input", (event) => { const previous = view.workflowKey; view.search = event.target.value; render(false); if (view.workflowKey && view.workflowKey !== previous) loadWorkflow(); });
byId("workflow-filter").addEventListener("change", (event) => { view.workflowKey = event.target.value; loadWorkflow(); });
byId("refresh").addEventListener("click", () => loadOverview(true));
byId("dialog-close").addEventListener("click", () => byId("job-dialog").close());
byId("job-dialog").addEventListener("click", (event) => { if (event.target === byId("job-dialog")) byId("job-dialog").close(); });
byId("gpu-dialog-close").addEventListener("click", () => byId("gpu-dialog").close());
byId("gpu-dialog").addEventListener("click", (event) => { if (event.target === byId("gpu-dialog")) byId("gpu-dialog").close(); });
byId("log-dialog-close").addEventListener("click", closeLog);
byId("log-dialog").addEventListener("click", (event) => { if (event.target === byId("log-dialog")) closeLog(); });
byId("log-older").addEventListener("click", () => {
  view.log.following = false;
  loadLogRange("older");
});
byId("log-latest").addEventListener("click", () => {
  view.log.generation += 1;
  view.log.chunks = [];
  view.log.loading = false;
  loadLogRange("tail");
});
byId("log-follow").addEventListener("click", () => {
  if (!LIVE_JOB_STATES.has(view.log.state)) return;
  view.log.following = !view.log.following;
  renderLog(view.log.following);
  if (view.log.following) loadLogRange("append");
});
byId("log-text").addEventListener("scroll", () => {
  const text = byId("log-text");
  if (view.log.following && text.scrollHeight - text.scrollTop - text.clientHeight > 80) {
    view.log.following = false;
    renderLog();
  }
});
byId("log-text").addEventListener("keydown", (event) => {
  const text = event.currentTarget;
  const page = Math.max(1, text.clientHeight - 40);
  const positions = {
    End: text.scrollHeight,
    Home: 0,
    PageDown: text.scrollTop + page,
    PageUp: text.scrollTop - page,
  };
  if (!(event.key in positions)) return;
  event.preventDefault();
  text.scrollTop = positions[event.key];
});
byId("log-copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(byId("log-text").textContent);
  byId("log-copy").textContent = "Copied";
  setTimeout(() => { byId("log-copy").textContent = "Copy visible"; }, 1_200);
});
byId("gpu-report-copy").addEventListener("click", async () => {
  if (!view.gpuReport) return;
  await navigator.clipboard.writeText(view.gpuReport);
  byId("gpu-report-copy").textContent = "Copied";
});

loadOverview(false);
setInterval(() => loadOverview(false), 5_000);
setInterval(() => {
  if (byId("log-dialog").open && view.log.following) loadLogRange("append");
}, 1_500);
setInterval(() => { if (view.snapshot) renderConnection(view.snapshot); }, 1_000);
