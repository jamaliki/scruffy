import {
  allocationIsStale, formatAge, formatDuration, formatNumber, gpuOwners,
  progressLabel, projectColor, projects, projectSummaries, resourceLabel, resourceTotals,
  scalarTelemetry, stateTone, TELEMETRY_STALE_AFTER_MS, uniqueJobs, visibleJobs,
  focusedWorkflowTasks, workflowChoices, workflowLayout,
} from "/assets/model.js";

const byId = (id) => document.getElementById(id);
const view = {
  snapshot: null, project: "all", search: "", connected: false, loading: false,
  lastSuccessAt: null,
  workflowKey: "", workflowData: null, workflowLoading: false,
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
  const usedGpus = totals.gpus - totals.freeGpus;
  const usedCpus = totals.cpus - totals.freeCpus;
  const usedMemory = totals.memory - totals.freeMemory;
  replace(byId("metrics"), [
    metric("GPU occupancy", `${stale ? "?" : usedGpus} / ${totals.gpus}`, stale ? "availability unknown" : `${totals.freeGpus} available`),
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
  header.append(element("span", "node-free", stale ? "GPU availability unknown" : `${free.size} / ${gpuIds.length} GPUs free`));
  const rack = element("div", "gpu-rack");
  const owners = gpuOwners(nodeState);
  for (const gpuId of gpuIds) {
    const jobId = owners.get(gpuId);
    const job = jobs.get(jobId);
    const project = job?.project_id || "default";
    const available = free.has(gpuId);
    const tile = element("button", available ? `gpu ${stale ? "unknown" : "free"}` : "gpu occupied");
    tile.type = "button";
    tile.append(element("span", "gpu-id", `GPU ${gpuId}`));
    tile.append(element("strong", "gpu-owner", available ? (stale ? "UNKNOWN" : "FREE") : (job?.name || jobId || "ASSIGNED")));
    if (!available) {
      tile.style.setProperty("--project-color", projectColor(project));
      tile.dataset.jobId = jobId || "";
      tile.classList.toggle("foreign", view.project !== "all" && project !== view.project);
      tile.title = `${job?.name || jobId} · ${project}`;
    } else {
      tile.disabled = true;
    }
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
  byId("resource-note").textContent = stale ? `${entries.length} nodes / ${totals.gpus} GPUs / availability unknown` : `${entries.length} nodes / ${totals.gpus} GPUs / ${totals.freeGpus} available`;
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
  if (!data || !data.tasks?.length) {
    canvas.style.width = "100%"; canvas.style.height = "190px";
    replace(canvas, [empty(view.workflowLoading ? "Reading workflow…" : "No workflow tasks are available.")]);
    viewport.scrollTo(0, 0);
    return;
  }
  const focused = focusedWorkflowTasks(data.tasks);
  const layout = workflowLayout(focused.tasks);
  if (focused.omitted) {
    byId("workflow-status").textContent = `${data.project_id} / showing ${focused.tasks.length} relevant tasks of ${data.task_count} / ${data.attempt_count} attempts`;
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
    line.classList.add("workflow-edge", edge.condition === "terminal" ? "terminal" : "success");
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
    view.workflowData = null; renderWorkflowGraph(); return;
  }
  const requestKey = choice.key;
  view.workflowLoading = true; view.workflowData = null; renderWorkflowGraph();
  byId("workflow-status").textContent = `${choice.project} / ${choice.workflow_id}`;
  try {
    const response = await fetch(`/api/workflows/${encodeURIComponent(choice.workflow_id)}?project=${encodeURIComponent(choice.project)}`, {cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Workflow lookup failed");
    if (view.workflowKey !== requestKey) return;
    view.workflowData = data;
    byId("workflow-status").textContent = `${data.project_id} / ${data.task_count} tasks / ${data.attempt_count} attempts`;
  } catch (error) {
    if (view.workflowKey === requestKey) byId("workflow-status").textContent = error.message;
  } finally {
    if (view.workflowKey === requestKey) { view.workflowLoading = false; renderWorkflowGraph(); }
  }
}

function renderWorkflowChoices(snapshot) {
  const select = byId("workflow-filter");
  const choices = workflowChoices(snapshot, view.project, view.search);
  if (!choices.some((choice) => choice.key === view.workflowKey)) {
    view.workflowKey = choices[0]?.key || "";
    view.workflowData = null;
  }
  const options = choices.map((choice) => {
    const option = element("option", "", `${choice.workflow_id} · ${choice.project} · ${choice.jobs} visible`);
    option.value = choice.key; return option;
  });
  if (!options.length) { const option = element("option", "", "No workflows visible"); option.value = ""; options.push(option); }
  replace(select, options); select.value = view.workflowKey;
  if (!view.workflowKey) {
    byId("workflow-status").textContent = "No workflows match the current project and search filters.";
    view.workflowData = null; renderWorkflowGraph();
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

function render() {
  const snapshot = view.snapshot;
  if (!snapshot) return;
  const stale = telemetryIsStale(snapshot);
  renderConnection(snapshot); renderMetrics(snapshot, stale); renderProjects(snapshot); renderNodes(snapshot);
  renderProjectLegend(snapshot); renderProjectBoard(snapshot);
  renderWorkflowChoices(snapshot);
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
    list.append(element("dt", "", label), element("dd", "", value == null || value === "" ? "—" : String(value)));
  }
  section.append(list); return section;
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
      detailSection("Placement", [["Request", resourceLabel(job)], ["Assignment", JSON.stringify(job.assignment || "Not assigned")], ["Stdout", job.stdout], ["Stderr", job.stderr]]),
      detailSection("Workflow", [["Workflow", job.workflow_id], ["Task", job.task_id], ["Explanation", explanation.explanation], ["Blockers", JSON.stringify(explanation.blockers || job.blockers || [])]]),
      detailSection("Workload telemetry", scalarTelemetry(job.workload || {})),
    ];
    replace(byId("job-detail"), sections);
  } catch (error) {
    replace(byId("job-detail"), [empty(error.message)]);
  }
}

async function loadOverview() {
  if (view.loading) return;
  view.loading = true; byId("refresh").disabled = true;
  try {
    const response = await fetch("/api/overview", {cache: "no-store"});
    const snapshot = await response.json();
    if (!response.ok) throw new Error(snapshot.error || "Queue read failed");
    view.snapshot = snapshot; view.connected = true; view.lastSuccessAt = Date.now(); render();
    if (view.workflowKey) loadWorkflow();
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
  const projectFilter = event.target.closest("[data-project-filter]");
  if (projectFilter) {
    const requested = projectFilter.dataset.projectFilter;
    view.project = requested === "all" || view.project === requested ? "all" : requested;
    render();
    if (view.workflowKey) loadWorkflow();
    return;
  }
  const target = event.target.closest("[data-job-id]");
  if (target) openJob(target.dataset.jobId);
});
byId("project-filter").addEventListener("change", (event) => { view.project = event.target.value; render(); if (view.workflowKey) loadWorkflow(); });
byId("job-search").addEventListener("input", (event) => { view.search = event.target.value; render(); if (view.workflowKey) loadWorkflow(); });
byId("workflow-filter").addEventListener("change", (event) => { view.workflowKey = event.target.value; loadWorkflow(); });
byId("refresh").addEventListener("click", loadOverview);
byId("dialog-close").addEventListener("click", () => byId("job-dialog").close());
byId("job-dialog").addEventListener("click", (event) => { if (event.target === byId("job-dialog")) byId("job-dialog").close(); });

loadOverview();
setInterval(loadOverview, 5_000);
setInterval(() => { if (view.snapshot) renderConnection(view.snapshot); }, 1_000);
