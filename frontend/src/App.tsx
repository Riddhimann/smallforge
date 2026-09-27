import {
  Activity,
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Beaker,
  Bot,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  Code2,
  Database,
  FlaskConical,
  Gauge,
  History,
  Layers3,
  ListChecks,
  LoaderCircle,
  Menu,
  Play,
  RefreshCw,
  Route,
  Server,
  ShieldCheck,
  Sparkles,
  SquareTerminal,
  TriangleAlert,
  Wrench,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api } from "./api";
import type {
  AblationResult,
  BenchmarkDescriptor,
  Experiment,
  ExperimentRequest,
  ExperimentResults,
  Health,
  MetricComparison,
  ModelDescriptor,
  TaskResult,
  TraceEvent,
} from "./types";

type View = "overview" | "configure" | "runs" | "experiment";
type SourceKind = "synthetic" | "real" | "unknown";

type ComparisonRow = {
  key: string;
  label: string;
  baseline: number;
  harness: number;
  delta: number;
  relativeDelta?: number;
  unit: string;
  lowerIsBetter: boolean;
};

type NormalizedAblation = {
  id: string;
  label: string;
  score: number;
  delta: number;
  tasks?: number;
  metric: string;
};

const MODE_OPTIONS = [
  {
    id: "baseline",
    title: "Direct baseline",
    detail: "Model response without reliability controls.",
  },
  {
    id: "structured",
    title: "Structured output",
    detail: "Schema-constrained generation and validation only.",
  },
  {
    id: "tools",
    title: "Tool execution",
    detail: "Planning and tool use without retrieval or verification.",
  },
  {
    id: "retrieval",
    title: "Retrieval grounded",
    detail: "Evidence retrieval and citation-aware generation.",
  },
  {
    id: "verified",
    title: "Verified response",
    detail: "Post-generation checks with bounded corrective retries.",
  },
  {
    id: "full",
    title: "Full reliability harness",
    detail: "Planner, tools, retrieval, validation, verifier, and retries.",
  },
] as const;

const METRIC_LABELS: Record<string, string> = {
  task_accuracy: "Task accuracy",
  success_rate: "Task success",
  schema_validity: "Schema validity",
  schema_valid_rate: "Schema validity",
  tool_selection_accuracy: "Tool selection",
  tool_argument_accuracy: "Tool arguments",
  groundedness: "Groundedness",
  citation_precision: "Citation precision",
  citation_recall: "Citation recall",
  abstention_accuracy: "Safe abstention",
  verification_accuracy: "Verification",
  recovery_rate: "Recovery rate",
  retry_count: "Mean retries",
  retries: "Mean retries",
  latency_ms: "Latency",
  cost_usd: "Estimated cost",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const record = asRecord(value);
  if (!record) return null;
  for (const key of ["value", "mean", "score", "rate", "candidate_value", "harness_value"]) {
    const candidate = record[key];
    if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  }
  return null;
}

function humanize(value: string) {
  return value
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricLabel(key: string) {
  return METRIC_LABELS[key] ?? humanize(key);
}

function lowerIsBetter(key: string, direction?: unknown) {
  if (direction === "lower") return true;
  if (direction === "higher") return false;
  return /latency|retry|cost|error|failure|tokens/.test(key);
}

function metricUnit(key: string, unit?: unknown) {
  if (typeof unit === "string") return unit;
  if (key.includes("latency")) return "ms";
  if (key.includes("cost")) return "$";
  if (/accuracy|valid|ground|precision|recall|rate|verification|abstention/.test(key)) return "%";
  return "";
}

function formatMetric(value: number, key: string, unit = metricUnit(key)) {
  if (unit === "%") {
    const percentage = Math.abs(value) <= 1 ? value * 100 : value;
    return `${percentage.toFixed(1)}%`;
  }
  if (unit === "ms") return `${value >= 1000 ? (value / 1000).toFixed(2) + "s" : value.toFixed(0) + "ms"}`;
  if (unit === "$") return `$${value.toFixed(value < 1 ? 4 : 2)}`;
  return value.toFixed(Number.isInteger(value) ? 0 : 2);
}

function formatDelta(row: ComparisonRow) {
  const sign = row.delta > 0 ? "+" : "";
  return `${sign}${formatMetric(row.delta, row.key, row.unit)}`;
}

function isPositiveDelta(row: ComparisonRow) {
  return row.lowerIsBetter ? row.delta < 0 : row.delta > 0;
}

function comparisonFromRecord(
  key: string,
  value: MetricComparison | Record<string, unknown>,
): ComparisonRow | null {
  const baseline = numberValue(value.baseline ?? value.baseline_value);
  const harness = numberValue(
    value.harness ?? value.harness_value ?? value.candidate_value,
  );
  if (baseline === null || harness === null) return null;
  const delta = numberValue(value.delta ?? value.absolute_delta) ?? harness - baseline;
  const relativeDelta = numberValue(value.delta_pct ?? value.relative_delta_percent) ?? undefined;
  return {
    key,
    label:
      (typeof value.label === "string" && value.label) ||
      (typeof value.name === "string" && value.name) ||
      metricLabel(key),
    baseline,
    harness,
    delta,
    relativeDelta,
    unit: metricUnit(key, value.unit),
    lowerIsBetter: lowerIsBetter(key, value.direction),
  };
}

function normalizeComparisons(results?: ExperimentResults | null): ComparisonRow[] {
  if (!results) return [];
  const rows: ComparisonRow[] = [];
  const raw = results.comparisons;

  if (Array.isArray(raw)) {
    raw.forEach((item, index) => {
      const key = String(item.key ?? item.metric ?? item.name ?? `metric_${index + 1}`);
      const row = comparisonFromRecord(key, item);
      if (row) rows.push(row);
    });
  } else {
    const record = asRecord(raw);
    if (record) {
      Object.entries(record).forEach(([key, value]) => {
        const item = asRecord(value);
        if (!item) return;
        const row = comparisonFromRecord(key, item);
        if (row) rows.push(row);
      });
    }
  }

  if (rows.length) return rows;
  const summary = asRecord(results.summary);
  const baseline = asRecord(summary?.baseline ?? results.baseline);
  const harness = asRecord(summary?.harness ?? summary?.candidate ?? results.harness);
  const baselineMetrics = asRecord(baseline?.aggregates) ?? baseline;
  const harnessMetrics = asRecord(harness?.aggregates) ?? harness;
  if (baselineMetrics && harnessMetrics) {
    for (const key of Object.keys(baselineMetrics)) {
      const baselineValue = numberValue(baselineMetrics[key]);
      const harnessValue = numberValue(harnessMetrics[key]);
      if (baselineValue === null || harnessValue === null) continue;
      rows.push({
        key,
        label: metricLabel(key),
        baseline: baselineValue,
        harness: harnessValue,
        delta: harnessValue - baselineValue,
        unit: metricUnit(key),
        lowerIsBetter: lowerIsBetter(key),
      });
    }
  }
  return rows;
}

function normalizeAblations(results?: ExperimentResults | null): NormalizedAblation[] {
  if (!results) return [];
  const direct = Array.isArray(results.ablations)
    ? results.ablations
    : Array.isArray(results.ablation)
      ? results.ablation
      : null;

  if (direct) {
    return direct.flatMap((item: AblationResult, index) => {
      const score = numberValue(item.success_rate ?? item.score);
      if (score === null) return [];
      return [{
        id: String(item.id ?? item.mode ?? item.name ?? index),
        label: String(item.label ?? item.name ?? item.mode ?? `Variant ${index + 1}`),
        score,
        delta: numberValue(item.delta) ?? 0,
        tasks: typeof item.tasks === "number" ? item.tasks : undefined,
        metric: "task_accuracy",
      }];
    });
  }

  const ablationObject = asRecord(results.ablation) ?? asRecord(asRecord(results.summary)?.ablation);
  const comparisons = ablationObject?.comparisons ?? (
    Array.isArray(results.comparisons) && results.comparisons.some((item) => asRecord(item)?.deltas)
      ? results.comparisons
      : undefined
  );
  if (!Array.isArray(comparisons)) return [];
  return comparisons.flatMap((candidate, index) => {
    const comparison = asRecord(candidate);
    const deltas = asRecord(comparison?.deltas);
    if (!comparison || !deltas) return [];
    const preferredKey = ["task_accuracy", "groundedness", "verification_accuracy"]
      .find((key) => asRecord(deltas[key])) ?? Object.keys(deltas)[0];
    if (!preferredKey) return [];
    const metric = asRecord(deltas[preferredKey]);
    const score = numberValue(metric?.candidate_value);
    if (score === null) return [];
    return [{
      id: String(comparison.variant ?? index),
      label: humanize(String(comparison.variant ?? `Variant ${index + 1}`)),
      score,
      delta: numberValue(metric?.absolute_delta) ?? 0,
      tasks: numberValue(comparison.common_task_count) ?? undefined,
      metric: preferredKey,
    }];
  });
}

function sourceKind(experiment: Experiment, model?: ModelDescriptor): SourceKind {
  const results = experiment.results;
  const summary = asRecord(results?.summary);
  const values = [
    results?.synthetic,
    summary?.synthetic,
    experiment.synthetic,
    model?.synthetic,
  ];
  if (values.some((value) => value === true)) return "synthetic";
  if (values.some((value) => value === false)) return "real";
  const names = [results?.result_kind, model?.result_kind, model?.id, model?.provider]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  if (/synthetic|fixture|scripted|test|mock/.test(names)) return "synthetic";
  if (/ollama|openai|anthropic|gemini|local/.test(names)) return "real";
  return "unknown";
}

function formatDate(value?: string | null, withTime = true) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  }).format(date);
}

function experimentDuration(experiment: Experiment) {
  if (!experiment.started_at) return "—";
  const start = new Date(experiment.started_at).getTime();
  const end = experiment.completed_at ? new Date(experiment.completed_at).getTime() : Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "—";
  const seconds = Math.max(0, Math.floor((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function progressPercent(experiment: Experiment) {
  if (experiment.status === "completed" || experiment.status === "failed") return 100;
  if (typeof experiment.progress === "number") {
    return Math.max(0, Math.min(100, experiment.progress <= 1 ? experiment.progress * 100 : experiment.progress));
  }
  const progress = asRecord(experiment.progress);
  const explicit = numberValue(progress?.percent);
  if (explicit !== null) return Math.max(0, Math.min(100, explicit <= 1 ? explicit * 100 : explicit));
  const complete = numberValue(progress?.completed);
  const total = numberValue(progress?.total);
  if (complete !== null && total) return Math.max(0, Math.min(100, (complete / total) * 100));
  return experiment.status === "running" ? 42 : 8;
}

function taskTitle(task: TaskResult, index: number) {
  if (task.name) return task.name;
  if (typeof task.task === "string") return task.task;
  const record = asRecord(task.task);
  const title = record?.name ?? record?.title ?? record?.prompt;
  return typeof title === "string" ? title : `Task ${index + 1}`;
}

function taskId(task: TaskResult, index: number) {
  return String(task.task_id ?? task.id ?? task.run_id ?? `task-${index + 1}`);
}

function taskPassed(task: TaskResult) {
  if (typeof task.success === "boolean") return task.success;
  const accuracy = numberValue(task.task_accuracy ?? asRecord(task.metrics)?.task_accuracy);
  return accuracy !== null ? accuracy >= 0.5 : false;
}

function taskEvents(task: TaskResult): TraceEvent[] {
  if (Array.isArray(task.trace)) return task.trace;
  const trace = asRecord(task.trace);
  if (Array.isArray(trace?.events)) return trace.events as TraceEvent[];
  const events: TraceEvent[] = [];
  const plan = task.plan;
  if (plan) events.push({ stage: "planner", status: "succeeded", message: "Execution plan created", output: plan });
  const toolResults = task.tool_results;
  if (Array.isArray(toolResults)) {
    toolResults.forEach((result, index) => {
      const record = asRecord(result);
      events.push({
        stage: "tool",
        status: record?.success === false ? "failed" : "succeeded",
        message: String(record?.tool_name ?? `Tool call ${index + 1}`),
        output: result,
        duration_ms: numberValue(record?.duration_ms) ?? undefined,
      });
    });
  }
  if (task.verification) {
    const verification = asRecord(task.verification);
    events.push({
      stage: "verifier",
      status: verification?.passed === false ? "failed" : "succeeded",
      message: "Response verification completed",
      output: task.verification,
    });
  }
  const retryCount = numberValue(task.retries ?? asRecord(task.metrics)?.retries ?? task.retry_count) ?? 0;
  if (retryCount > 0) {
    events.push({
      stage: "retry",
      status: "retrying",
      message: `${retryCount} bounded ${retryCount === 1 ? "retry" : "retries"} executed`,
      attempt: retryCount,
    });
  }
  if (events.length === 0 && task.diagnostics) {
    events.push({ stage: "evaluation", status: taskPassed(task) ? "succeeded" : "failed", message: "Task diagnostics", output: task.diagnostics });
  }
  return events;
}

function storageLabel(health: Health | null) {
  if (!health?.storage) return "Unknown";
  if (typeof health.storage === "string") return humanize(health.storage);
  return humanize(String(health.storage.backend ?? health.storage.status ?? "Available"));
}

function provenanceLabel(value: string | Record<string, unknown>) {
  if (typeof value === "string") return value;
  const label = value.name ?? value.source ?? value.kind;
  return typeof label === "string" ? label : "Versioned benchmark fixture";
}

function Logo() {
  return (
    <div className="logo" aria-label="SmallForge Agent Reliability Lab">
      <div className="logo-mark" aria-hidden="true"><FlaskConical size={18} /></div>
      <div><strong>SmallForge</strong><span>AGENT RELIABILITY LAB</span></div>
    </div>
  );
}

function StatusBadge({ status }: { status: Experiment["status"] }) {
  const Icon = status === "completed" ? CheckCircle2 : status === "failed" ? XCircle : status === "running" ? LoaderCircle : Clock3;
  return <span className={`status-badge ${status}`}><Icon size={13} className={status === "running" ? "spin" : ""} />{humanize(status)}</span>;
}

function SourceBadge({ kind }: { kind: SourceKind }) {
  const label = kind === "synthetic" ? "Synthetic fixture" : kind === "real" ? "Real model run" : "Source unverified";
  return <span className={`source-badge ${kind}`}><CircleDot size={12} />{label}</span>;
}

function EmptyState({ icon, title, detail, action }: { icon: ReactNode; title: string; detail: string; action?: ReactNode }) {
  return <div className="empty-state">{icon}<h3>{title}</h3><p>{detail}</p>{action}</div>;
}

function App() {
  const [view, setView] = useState<View>("overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkDescriptor[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const loadDashboard = useCallback(async (quiet = false, signal?: AbortSignal) => {
    if (quiet) setRefreshing(true);
    else setLoading(true);
    setError("");
    try {
      const [healthResult, modelsResult, benchmarksResult, experimentsResult] = await Promise.allSettled([
        api.health(signal),
        api.models(signal),
        api.benchmarks(signal),
        api.experiments(signal),
      ]);
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (modelsResult.status === "fulfilled") setModels(modelsResult.value);
      if (benchmarksResult.status === "fulfilled") setBenchmarks(benchmarksResult.value);
      if (experimentsResult.status === "fulfilled") setExperiments(experimentsResult.value);
      const failure = [healthResult, modelsResult, benchmarksResult, experimentsResult]
        .find((result) => result.status === "rejected");
      if (failure?.status === "rejected" && !signal?.aborted) {
        setError(failure.reason instanceof Error ? failure.reason.message : "Some dashboard data could not be loaded.");
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadDashboard(false, controller.signal);
    return () => controller.abort();
  }, [loadDashboard]);

  useEffect(() => {
    if (!selected || !["pending", "running"].includes(selected.status)) return;
    let requesting = false;
    const poll = window.setInterval(async () => {
      if (requesting) return;
      requesting = true;
      try {
        const updated = await api.experiment(selected.id);
        setSelected(updated);
        setExperiments((current) => {
          const exists = current.some((item) => item.id === updated.id);
          return exists
            ? current.map((item) => item.id === updated.id ? updated : item)
            : [updated, ...current];
        });
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Experiment status could not be refreshed.");
      } finally {
        requesting = false;
      }
    }, 2_000);
    return () => window.clearInterval(poll);
  }, [selected?.id, selected?.status]);

  function navigate(next: View) {
    setView(next);
    setMobileNav(false);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function openExperiment(experiment: Experiment) {
    setSelected(experiment);
    setView("experiment");
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
    try {
      const detail = await api.experiment(experiment.id);
      setSelected(detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment details could not be loaded.");
    }
  }

  async function createExperiment(payload: ExperimentRequest) {
    setError("");
    const created = await api.createExperiment(payload);
    setSelected(created);
    setExperiments((current) => [created, ...current.filter((item) => item.id !== created.id)]);
    setView("experiment");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const activeCount = experiments.filter((item) => item.status === "pending" || item.status === "running").length;

  return (
    <div className="app-frame">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <div className="sidebar-head"><Logo /><button className="mobile-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18} /></button></div>
        <nav aria-label="Primary navigation">
          <button className={view === "overview" ? "active" : ""} onClick={() => navigate("overview")}><Gauge size={17} /><span>Overview</span></button>
          <button className={view === "configure" ? "active" : ""} onClick={() => navigate("configure")}><Beaker size={17} /><span>New experiment</span></button>
          <button className={view === "runs" || view === "experiment" ? "active" : ""} onClick={() => navigate("runs")}><History size={17} /><span>Prior runs</span>{activeCount > 0 && <strong>{activeCount}</strong>}</button>
        </nav>
        <div className="sidebar-foot">
          <div className="service-state"><span className={`live-dot ${health?.status === "ok" ? "online" : ""}`} /><div><strong>{health?.status === "ok" ? "Lab online" : health ? humanize(health.status) : "API unavailable"}</strong><span>{health?.version ? `Core ${health.version}` : "Awaiting service status"}</span></div></div>
          <p><ShieldCheck size={13} /> Evidence-first evaluation. No hidden scoring.</p>
        </div>
      </aside>

      {mobileNav && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}

      <div className="content-shell">
        <header className="topbar">
          <button className="menu-button" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>Reliability Lab</span><ChevronRight size={13} /><strong>{view === "configure" ? "New experiment" : view === "runs" ? "Prior runs" : view === "experiment" ? selected?.name ?? "Experiment" : "Overview"}</strong></div>
          <div className="topbar-actions">
            {activeCount > 0 && <span className="active-runs"><Activity size={14} /> {activeCount} active</span>}
            <button className="icon-action" onClick={() => void loadDashboard(true)} disabled={refreshing} aria-label="Refresh dashboard"><RefreshCw size={16} className={refreshing ? "spin" : ""} /></button>
            <button className="new-run-button" onClick={() => navigate("configure")}><Play size={14} /> New run</button>
          </div>
        </header>

        <main id="main-content" tabIndex={-1}>
          {error && <div className="global-error" role="alert"><TriangleAlert size={17} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={15} /></button></div>}
          {loading ? (
            <LoadingScreen />
          ) : view === "overview" ? (
            <Overview
              health={health}
              models={models}
              benchmarks={benchmarks}
              experiments={experiments}
              onConfigure={() => navigate("configure")}
              onOpen={openExperiment}
            />
          ) : view === "configure" ? (
            <ConfigureExperiment
              models={models}
              benchmarks={benchmarks}
              onCancel={() => navigate("overview")}
              onCreate={createExperiment}
            />
          ) : view === "runs" ? (
            <RunsPage experiments={experiments} models={models} onOpen={openExperiment} onConfigure={() => navigate("configure")} />
          ) : selected ? (
            <ExperimentPage experiment={selected} models={models} benchmarks={benchmarks} onBack={() => navigate("runs")} />
          ) : (
            <EmptyState icon={<History size={28} />} title="No experiment selected" detail="Choose a prior run or configure a new reliability experiment." action={<button className="primary-button" onClick={() => navigate("configure")}>Configure experiment</button>} />
          )}
        </main>
      </div>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="loading-screen" role="status" aria-live="polite">
      <div className="loading-orbit"><FlaskConical size={23} /></div>
      <strong>Connecting to the reliability lab</strong>
      <span>Loading models, benchmarks, and experiment history…</span>
    </div>
  );
}

function Overview({ health, models, benchmarks, experiments, onConfigure, onOpen }: {
  health: Health | null;
  models: ModelDescriptor[];
  benchmarks: BenchmarkDescriptor[];
  experiments: Experiment[];
  onConfigure: () => void;
  onOpen: (experiment: Experiment) => void;
}) {
  const completed = experiments.filter((item) => item.status === "completed");
  const active = experiments.filter((item) => item.status === "pending" || item.status === "running");
  const latestRows = normalizeComparisons(completed[0]?.results);
  const leadingDelta = latestRows.find((row) => row.key.includes("accuracy")) ?? latestRows[0];
  const providers = new Set(models.map((item) => item.provider));

  return (
    <div className="page overview-page">
      <section className="page-heading overview-heading">
        <div><span className="eyebrow"><Sparkles size={14} /> CONTROLLED AGENT EVALUATION</span><h1>Reliability you can<br /><em>inspect, not assume.</em></h1><p>Run paired experiments, isolate the contribution of each harness control, and inspect every planner, tool, verifier, and retry decision.</p></div>
        <button className="primary-button hero-action" onClick={onConfigure}><Play size={16} /> Configure experiment <ArrowRight size={16} /></button>
      </section>

      <section className="system-cards" aria-label="Lab status">
        <div className="system-card"><div className="system-icon"><Server size={18} /></div><div><span>API STATUS</span><strong>{health ? humanize(health.status) : "Unavailable"}</strong><small>{health?.version ? `Version ${health.version}` : "No version reported"}</small></div><span className={`health-indicator ${health?.status === "ok" ? "healthy" : ""}`} /></div>
        <div className="system-card"><div className="system-icon"><Bot size={18} /></div><div><span>MODEL CATALOG</span><strong>{models.filter((model) => model.available).length} available</strong><small>{providers.size} configured {providers.size === 1 ? "provider" : "providers"}</small></div></div>
        <div className="system-card"><div className="system-icon"><Layers3 size={18} /></div><div><span>BENCHMARKS</span><strong>{benchmarks.length} registered</strong><small>{benchmarks.reduce((sum, item) => sum + item.task_count, 0)} versioned tasks</small></div></div>
        <div className="system-card"><div className="system-icon"><Database size={18} /></div><div><span>RUN REGISTRY</span><strong>{storageLabel(health)}</strong><small>{completed.length} completed · {active.length} active</small></div></div>
      </section>

      <div className="overview-grid">
        <section className="panel latest-panel">
          <div className="panel-heading"><div><span>LATEST SIGNAL</span><h2>Harness impact</h2></div>{completed[0] && <button className="text-button" onClick={() => onOpen(completed[0])}>Open report <ArrowRight size={14} /></button>}</div>
          {completed[0] && leadingDelta ? (
            <div className="impact-content">
              <SourceBadge kind={sourceKind(completed[0], models.find((model) => model.id === completed[0].request.model_id))} />
              <div className="impact-number"><strong>{formatDelta(leadingDelta)}</strong><span>{leadingDelta.label.toLowerCase()} vs direct baseline</span></div>
              <div className="comparison-mini"><div><span>Baseline</span><strong>{formatMetric(leadingDelta.baseline, leadingDelta.key, leadingDelta.unit)}</strong></div><div className="connector"><ArrowRight size={16} /></div><div><span>Harness</span><strong>{formatMetric(leadingDelta.harness, leadingDelta.key, leadingDelta.unit)}</strong></div></div>
              <p>From <strong>{completed[0].name ?? completed[0].id}</strong>, completed {formatDate(completed[0].completed_at)}. Open the report to verify task-level evidence and confidence intervals.</p>
            </div>
          ) : (
            <EmptyState icon={<Gauge size={24} />} title="No measured delta yet" detail="Complete a paired baseline and harness run to establish the first reliability signal." action={<button className="secondary-button" onClick={onConfigure}>Create first run</button>} />
          )}
        </section>

        <section className="panel principles-panel">
          <div className="panel-heading"><div><span>EXPERIMENT CONTRACT</span><h2>What SmallForge records</h2></div><ShieldCheck size={18} /></div>
          <div className="principle-list">
            <div><span>01</span><div><strong>Paired execution</strong><p>Baseline and harness variants share the same benchmark tasks.</p></div></div>
            <div><span>02</span><div><strong>Versioned provenance</strong><p>Model, benchmark, fixture status, and execution settings remain attached.</p></div></div>
            <div><span>03</span><div><strong>Inspectable traces</strong><p>Planner, tool, validation, verifier, and retry events remain reviewable.</p></div></div>
          </div>
        </section>
      </div>

      <section className="panel recent-panel">
        <div className="panel-heading"><div><span>RUN REGISTRY</span><h2>Recent experiments</h2></div><span className="panel-count">{experiments.length} total</span></div>
        <ExperimentTable experiments={experiments.slice(0, 6)} models={models} onOpen={onOpen} emptyAction={onConfigure} />
      </section>
    </div>
  );
}

function ConfigureExperiment({ models, benchmarks, onCancel, onCreate }: {
  models: ModelDescriptor[];
  benchmarks: BenchmarkDescriptor[];
  onCancel: () => void;
  onCreate: (payload: ExperimentRequest) => Promise<void>;
}) {
  const providers = useMemo(() => [...new Set(models.map((model) => model.provider))], [models]);
  const [provider, setProvider] = useState(providers[0] ?? "");
  const filteredModels = models.filter((model) => !provider || model.provider === provider);
  const [modelId, setModelId] = useState("");
  const [benchmarkId, setBenchmarkId] = useState("");
  const [modes, setModes] = useState<string[]>(["baseline", "full"]);
  const [name, setName] = useState("");
  const [allTasks, setAllTasks] = useState(false);
  const [maxTasks, setMaxTasks] = useState(20);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  useEffect(() => {
    if (!provider && providers[0]) setProvider(providers[0]);
  }, [provider, providers]);

  useEffect(() => {
    if (!filteredModels.some((model) => model.id === modelId)) {
      setModelId(filteredModels.find((model) => model.available)?.id ?? filteredModels[0]?.id ?? "");
    }
  }, [filteredModels, modelId]);

  useEffect(() => {
    if (!benchmarkId && benchmarks[0]) {
      setBenchmarkId(benchmarks[0].id);
      setMaxTasks(Math.min(20, benchmarks[0].task_count || 20));
    }
  }, [benchmarkId, benchmarks]);

  const model = models.find((item) => item.id === modelId);
  const benchmark = benchmarks.find((item) => item.id === benchmarkId);
  const synthetic = Boolean(
    (model && (/synthetic|fixture|scripted|test|mock/i.test(`${model.id} ${model.provider} ${model.result_kind ?? ""}`) || model.synthetic === true)) ||
    (benchmark && /synthetic|fixture|test|mock/i.test(provenanceLabel(benchmark.provenance))),
  );

  function toggleMode(mode: string) {
    setModes((current) => current.includes(mode) ? current.filter((item) => item !== mode) : [...current, mode]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setFormError("");
    if (!modelId || !benchmarkId || modes.length === 0) {
      setFormError("Select an available model, benchmark, and at least one execution mode.");
      return;
    }
    if (model && !model.available) {
      setFormError("The selected model is currently unavailable.");
      return;
    }
    setSubmitting(true);
    try {
      await onCreate({
        ...(name.trim() ? { name: name.trim() } : {}),
        model_id: modelId,
        benchmark_id: benchmarkId,
        modes,
        max_tasks: allTasks ? null : maxTasks,
      });
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "The experiment could not be created.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page configure-page">
      <section className="page-heading compact-heading"><div><span className="eyebrow"><Beaker size={14} /> EXPERIMENT BLUEPRINT</span><h1>Configure a controlled run.</h1><p>Lock model, benchmark, and execution modes before any task begins. The resulting record is immutable and comparable.</p></div></section>
      <form className="config-layout" onSubmit={submit}>
        <div className="config-main">
          {formError && <div className="inline-error" role="alert"><TriangleAlert size={16} />{formError}</div>}
          <section className="panel config-section">
            <div className="section-index">01</div><div className="section-body"><div className="section-title"><div><span>RUN IDENTITY</span><h2>Name and model target</h2></div><Bot size={18} /></div>
              <label className="field"><span>Experiment name <small>Optional</small></span><input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Planner retry boundary · Aug 15" maxLength={100} /></label>
              <div className="field-grid">
                <label className="field"><span>Provider</span><select value={provider} onChange={(event) => setProvider(event.target.value)}>{providers.length ? providers.map((item) => <option key={item} value={item}>{humanize(item)}</option>) : <option value="">No providers registered</option>}</select></label>
                <label className="field"><span>Model</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}>{filteredModels.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.name}{item.available ? "" : " · unavailable"}</option>)}</select></label>
              </div>
              {model && <div className="selection-note"><div><span className={`availability-dot ${model.available ? "available" : ""}`} /><strong>{model.available ? "Ready for evaluation" : "Provider unavailable"}</strong></div><span>{model.provider} · {model.id}</span></div>}
            </div>
          </section>

          <section className="panel config-section">
            <div className="section-index">02</div><div className="section-body"><div className="section-title"><div><span>BENCHMARK CONTRACT</span><h2>Task suite and sample size</h2></div><ListChecks size={18} /></div>
              <label className="field"><span>Benchmark</span><select value={benchmarkId} onChange={(event) => { const id = event.target.value; setBenchmarkId(id); const next = benchmarks.find((item) => item.id === id); if (next) setMaxTasks(Math.min(20, next.task_count || 20)); }}>{benchmarks.length ? benchmarks.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.task_count} tasks</option>) : <option value="">No benchmarks registered</option>}</select></label>
              {benchmark && <div className="benchmark-preview"><p>{benchmark.description}</p><div>{benchmark.categories.map((category) => <span key={category}>{humanize(category)}</span>)}</div><small><Database size={12} /> {provenanceLabel(benchmark.provenance)}</small></div>}
              <div className="task-limit-row"><label className="check-control"><input type="checkbox" checked={allTasks} onChange={(event) => setAllTasks(event.target.checked)} /><span><Check size={12} /></span><div><strong>Use all benchmark tasks</strong><small>Run the full registered suite</small></div></label><label className={`compact-field ${allTasks ? "disabled" : ""}`}><span>Maximum tasks</span><input type="number" min={1} max={Math.min(200, benchmark?.task_count || 200)} value={maxTasks} disabled={allTasks} onChange={(event) => setMaxTasks(Math.min(200, Math.max(1, Number(event.target.value))))} /></label></div>
            </div>
          </section>

          <section className="panel config-section">
            <div className="section-index">03</div><div className="section-body"><div className="section-title"><div><span>EXECUTION MODES</span><h2>Choose comparison variants</h2></div><Route size={18} /></div>
              <div className="mode-grid">{MODE_OPTIONS.map((option) => { const checked = modes.includes(option.id); return <label className={`mode-option ${checked ? "selected" : ""}`} key={option.id}><input type="checkbox" checked={checked} onChange={() => toggleMode(option.id)} /><span className="mode-check">{checked && <Check size={13} />}</span><div><strong>{option.title}</strong><p>{option.detail}</p></div></label>; })}</div>
              {modes.length < 2 && <p className="comparison-warning"><TriangleAlert size={13} /> Select both modes to calculate baseline-versus-harness deltas.</p>}
            </div>
          </section>
        </div>

        <aside className="launch-panel panel">
          <div className="launch-head"><Zap size={18} /><div><span>READY CHECK</span><h2>Run manifest</h2></div></div>
          <dl><div><dt>Provider</dt><dd>{provider ? humanize(provider) : "—"}</dd></div><div><dt>Model</dt><dd>{model?.name ?? "—"}</dd></div><div><dt>Benchmark</dt><dd>{benchmark?.name ?? "—"}</dd></div><div><dt>Tasks</dt><dd>{allTasks ? `${benchmark?.task_count ?? "All"}` : maxTasks}</dd></div><div><dt>Variants</dt><dd>{modes.length}</dd></div></dl>
          <div className={`provenance-callout ${synthetic ? "synthetic" : "real"}`}><CircleDot size={15} /><div><strong>{synthetic ? "Synthetic fixture run" : "Real model run"}</strong><p>{synthetic ? "Results validate the evaluation pipeline; they are not evidence of production model quality." : "Results will be attributed to the configured model and benchmark provenance."}</p></div></div>
          <div className="launch-actions"><button type="button" className="secondary-button" onClick={onCancel}>Cancel</button><button className="primary-button" disabled={submitting || !modelId || !benchmarkId}>{submitting ? <><LoaderCircle size={16} className="spin" /> Starting…</> : <><Play size={15} /> Start experiment</>}</button></div>
          <p className="immutable-note"><ShieldCheck size={12} /> Configuration is stored with the run for reproducibility.</p>
        </aside>
      </form>
    </div>
  );
}

function RunsPage({ experiments, models, onOpen, onConfigure }: {
  experiments: Experiment[];
  models: ModelDescriptor[];
  onOpen: (experiment: Experiment) => void;
  onConfigure: () => void;
}) {
  const [filter, setFilter] = useState<"all" | Experiment["status"]>("all");
  const filtered = filter === "all" ? experiments : experiments.filter((item) => item.status === filter);
  const counts = { completed: 0, running: 0, pending: 0, failed: 0 };
  experiments.forEach((item) => { counts[item.status] += 1; });
  return (
    <div className="page runs-page">
      <section className="page-heading compact-heading split-heading"><div><span className="eyebrow"><History size={14} /> EXPERIMENT REGISTRY</span><h1>Prior runs.</h1><p>Revisit immutable configurations, metrics, ablations, and task-level execution traces.</p></div><button className="primary-button" onClick={onConfigure}><Play size={15} /> New experiment</button></section>
      <section className="run-summary-grid">{(["completed", "running", "pending", "failed"] as const).map((status) => <button key={status} onClick={() => setFilter(filter === status ? "all" : status)} className={filter === status ? "selected" : ""}><span>{humanize(status)}</span><strong>{counts[status]}</strong></button>)}</section>
      <section className="panel registry-panel"><div className="panel-heading"><div><span>FILTER · {filter.toUpperCase()}</span><h2>{filtered.length} {filtered.length === 1 ? "experiment" : "experiments"}</h2></div>{filter !== "all" && <button className="text-button" onClick={() => setFilter("all")}>Clear filter <X size={13} /></button>}</div><ExperimentTable experiments={filtered} models={models} onOpen={onOpen} emptyAction={onConfigure} expanded /></section>
    </div>
  );
}

function ExperimentTable({ experiments, models, onOpen, emptyAction, expanded = false }: {
  experiments: Experiment[];
  models: ModelDescriptor[];
  onOpen: (experiment: Experiment) => void;
  emptyAction: () => void;
  expanded?: boolean;
}) {
  if (!experiments.length) return <EmptyState icon={<History size={23} />} title="No runs found" detail="Configure an experiment to create the first traceable result." action={<button className="secondary-button" onClick={emptyAction}>Configure experiment</button>} />;
  return (
    <div className="table-scroll"><table className="runs-table"><thead><tr><th>Experiment</th><th>Source</th><th>Status</th>{expanded && <th>Modes</th>}<th>Created</th><th><span className="sr-only">Open</span></th></tr></thead><tbody>{experiments.map((experiment) => { const model = models.find((item) => item.id === experiment.request.model_id); return <tr key={experiment.id} onClick={() => onOpen(experiment)}><td><strong>{experiment.name ?? experiment.id}</strong><span>{model?.name ?? experiment.request.model_id} · {experiment.request.benchmark_id}</span></td><td><SourceBadge kind={sourceKind(experiment, model)} /></td><td><StatusBadge status={experiment.status} /></td>{expanded && <td><span className="mode-text">{experiment.request.modes.map(humanize).join(" + ")}</span></td>}<td><span className="date-text">{formatDate(experiment.created_at)}</span></td><td><button className="row-action" onClick={(event) => { event.stopPropagation(); onOpen(experiment); }} aria-label={`Open ${experiment.name ?? experiment.id}`}><ArrowRight size={15} /></button></td></tr>; })}</tbody></table></div>
  );
}

function ExperimentPage({ experiment, models, benchmarks, onBack }: {
  experiment: Experiment;
  models: ModelDescriptor[];
  benchmarks: BenchmarkDescriptor[];
  onBack: () => void;
}) {
  const model = models.find((item) => item.id === experiment.request.model_id);
  const benchmark = benchmarks.find((item) => item.id === experiment.request.benchmark_id);
  if (experiment.status === "pending" || experiment.status === "running") {
    return <RunningExperiment experiment={experiment} model={model} benchmark={benchmark} onBack={onBack} />;
  }
  if (experiment.status === "failed") {
    return <FailedExperiment experiment={experiment} model={model} onBack={onBack} />;
  }
  return <CompletedExperiment experiment={experiment} model={model} benchmark={benchmark} onBack={onBack} />;
}

function RunningExperiment({ experiment, model, benchmark, onBack }: {
  experiment: Experiment;
  model?: ModelDescriptor;
  benchmark?: BenchmarkDescriptor;
  onBack: () => void;
}) {
  const progress = progressPercent(experiment);
  const stages = [
    { label: "Manifest locked", threshold: 4 },
    { label: "Baseline execution", threshold: 18 },
    { label: "Harness execution", threshold: 48 },
    { label: "Metric aggregation", threshold: 82 },
  ];
  return (
    <div className="page experiment-page">
      <button className="back-button" onClick={onBack}><ArrowLeft size={15} /> Prior runs</button>
      <section className="run-hero running-hero"><div><div className="run-kicker"><StatusBadge status={experiment.status} /><SourceBadge kind={sourceKind(experiment, model)} /></div><h1>{experiment.name ?? `Experiment ${experiment.id}`}</h1><p>Run <code>{experiment.id}</code> · started {formatDate(experiment.started_at ?? experiment.created_at)}</p></div><div className="running-orbit" aria-hidden="true"><Activity size={29} /></div></section>
      <section className="panel progress-panel" aria-live="polite"><div className="progress-heading"><div><span>EXECUTION PROGRESS</span><h2>{experiment.status === "pending" ? "Waiting for an evaluation worker" : "Evaluating paired variants"}</h2></div><strong>{Math.round(progress)}%</strong></div><div className={`run-progress ${experiment.progress ? "" : "indeterminate"}`} role="progressbar" aria-label="Experiment progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: `${progress}%` }} /></div><div className="stage-list">{stages.map((stage, index) => { const complete = progress >= stage.threshold; const active = complete && (index === stages.length - 1 || progress < stages[index + 1].threshold); return <div className={`${complete ? "complete" : ""} ${active ? "active" : ""}`} key={stage.label}><span>{complete ? <Check size={12} /> : index + 1}</span><strong>{stage.label}</strong></div>; })}</div><p className="poll-note"><RefreshCw size={12} className="spin-slow" /> Status refreshes automatically every two seconds. You can leave this view without stopping the run.</p></section>
      <div className="run-context-grid"><section className="panel context-panel"><div className="panel-heading"><div><span>IMMUTABLE MANIFEST</span><h2>Configuration</h2></div><Code2 size={17} /></div><dl><div><dt>Model</dt><dd>{model?.name ?? experiment.request.model_id}</dd></div><div><dt>Provider</dt><dd>{model?.provider ?? "—"}</dd></div><div><dt>Benchmark</dt><dd>{benchmark?.name ?? experiment.request.benchmark_id}</dd></div><div><dt>Modes</dt><dd>{experiment.request.modes.map(humanize).join(" + ")}</dd></div><div><dt>Task cap</dt><dd>{experiment.request.max_tasks ?? "All tasks"}</dd></div></dl></section><section className="panel context-panel"><div className="panel-heading"><div><span>WHAT HAPPENS NEXT</span><h2>Measured output</h2></div><Gauge size={17} /></div><ul className="output-list"><li><CheckCircle2 size={14} /> Baseline-versus-harness metric deltas</li><li><CheckCircle2 size={14} /> Confidence and sample-size context</li><li><CheckCircle2 size={14} /> Component ablation comparisons</li><li><CheckCircle2 size={14} /> Planner, tool, verifier, and retry traces</li></ul></section></div>
    </div>
  );
}

function FailedExperiment({ experiment, model, onBack }: { experiment: Experiment; model?: ModelDescriptor; onBack: () => void }) {
  return (
    <div className="page experiment-page"><button className="back-button" onClick={onBack}><ArrowLeft size={15} /> Prior runs</button><section className="run-hero"><div><div className="run-kicker"><StatusBadge status="failed" /><SourceBadge kind={sourceKind(experiment, model)} /></div><h1>{experiment.name ?? experiment.id}</h1><p>Run <code>{experiment.id}</code> · ended {formatDate(experiment.completed_at)}</p></div></section><section className="panel failure-panel"><XCircle size={27} /><h2>Experiment did not complete</h2><p>{experiment.error ?? "The evaluation worker reported an unspecified failure."}</p><div><strong>Configuration was preserved</strong><span>Inspect the service logs using experiment ID <code>{experiment.id}</code>, then create a new run after resolving the failure.</span></div></section></div>
  );
}

function CompletedExperiment({ experiment, model, benchmark, onBack }: {
  experiment: Experiment;
  model?: ModelDescriptor;
  benchmark?: BenchmarkDescriptor;
  onBack: () => void;
}) {
  const comparisons = normalizeComparisons(experiment.results);
  const ablations = normalizeAblations(experiment.results);
  const tasks = Array.isArray(experiment.results?.task_results) ? experiment.results.task_results : [];
  const [selectedTask, setSelectedTask] = useState(0);
  const kind = sourceKind(experiment, model);
  const successes = tasks.filter(taskPassed).length;
  return (
    <div className="page experiment-page">
      <button className="back-button" onClick={onBack}><ArrowLeft size={15} /> Prior runs</button>
      <section className="run-hero"><div><div className="run-kicker"><StatusBadge status="completed" /><SourceBadge kind={kind} /></div><h1>{experiment.name ?? `Experiment ${experiment.id}`}</h1><p>{model?.name ?? experiment.request.model_id} · {benchmark?.name ?? experiment.request.benchmark_id} · {tasks.length || experiment.request.max_tasks || "All"} task results</p></div><div className="run-meta"><span>DURATION<strong>{experimentDuration(experiment)}</strong></span><span>COMPLETED<strong>{formatDate(experiment.completed_at)}</strong></span></div></section>

      <section className={`result-provenance ${kind}`}><CircleDot size={16} /><div><strong>{kind === "synthetic" ? "Synthetic fixture results" : kind === "real" ? "Real model results" : "Result provenance is unverified"}</strong><p>{kind === "synthetic" ? "Use this run to validate pipeline mechanics and evaluation contracts. Do not present these numbers as real model performance." : kind === "real" ? "Metrics are attributed to the configured model and benchmark. Review sample size and trace evidence before drawing conclusions." : "The API did not provide enough provenance to classify this run. Treat all metrics as non-production evidence."}</p></div></section>

      <section className="results-section"><div className="content-heading"><div><span>PAIRED COMPARISON</span><h2>Baseline vs reliability harness</h2></div><p>Positive coloring reflects the metric direction, not absolute production readiness.</p></div>{comparisons.length ? <div className="metric-grid">{comparisons.slice(0, 6).map((row) => <MetricCard key={row.key} row={row} />)}</div> : <EmptyState icon={<Gauge size={24} />} title="No comparison metrics returned" detail="This completed run does not contain baseline-versus-harness metric pairs." />}</section>

      <AblationPanel ablations={ablations} />

      <section className="results-section"><div className="content-heading"><div><span>TASK EVIDENCE</span><h2>Trace inspector</h2></div>{tasks.length > 0 && <p>{successes}/{tasks.length} task runs passed their configured checks.</p>}</div>{tasks.length ? <TraceInspector tasks={tasks} selected={selectedTask} onSelect={setSelectedTask} /> : <EmptyState icon={<SquareTerminal size={24} />} title="No task traces returned" detail="The experiment result contains aggregate data only. Enable trace persistence in the evaluation worker to inspect decisions." />}</section>
    </div>
  );
}

function MetricCard({ row }: { row: ComparisonRow }) {
  const positive = isPositiveDelta(row);
  const neutral = row.delta === 0;
  const DeltaIcon = neutral ? ArrowRight : positive ? ArrowUpRight : ArrowDownRight;
  const max = Math.max(Math.abs(row.baseline), Math.abs(row.harness), 0.0001);
  return (
    <article className="metric-card"><div className="metric-top"><span>{row.label}</span><span className={`delta-pill ${neutral ? "neutral" : positive ? "positive" : "negative"}`}><DeltaIcon size={12} />{formatDelta(row)}</span></div><div className="metric-values"><div><span>BASELINE</span><strong>{formatMetric(row.baseline, row.key, row.unit)}</strong></div><div><span>HARNESS</span><strong>{formatMetric(row.harness, row.key, row.unit)}</strong></div></div><div className="metric-bars" aria-label={`${row.label}: baseline ${formatMetric(row.baseline, row.key, row.unit)}, harness ${formatMetric(row.harness, row.key, row.unit)}`}><div><span>BL</span><i><b style={{ width: `${Math.max(2, Math.abs(row.baseline) / max * 100)}%` }} /></i></div><div><span>HR</span><i><b className="harness" style={{ width: `${Math.max(2, Math.abs(row.harness) / max * 100)}%` }} /></i></div></div></article>
  );
}

function AblationPanel({ ablations }: { ablations: NormalizedAblation[] }) {
  const max = Math.max(...ablations.map((item) => Math.abs(item.score)), 1);
  return (
    <section className="results-section ablation-section"><div className="content-heading"><div><span>COMPONENT ABLATION</span><h2>Which controls create the lift?</h2></div><p>Compare harness variants with one reliability component removed.</p></div><div className="panel ablation-panel">{ablations.length ? <><div className="ablation-chart" aria-label="Ablation score chart">{ablations.map((item) => <div className="ablation-bar" key={item.id}><span>{item.label}</span><div><i style={{ width: `${Math.max(2, Math.abs(item.score) / max * 100)}%` }} /></div><strong>{formatMetric(item.score, item.metric)}</strong></div>)}</div><div className="table-scroll"><table className="ablation-table"><thead><tr><th>Variant</th><th>Primary metric</th><th>Score</th><th>Delta</th><th>Common tasks</th></tr></thead><tbody>{ablations.map((item) => <tr key={item.id}><td><strong>{item.label}</strong></td><td>{metricLabel(item.metric)}</td><td>{formatMetric(item.score, item.metric)}</td><td className={item.delta < 0 ? "negative-text" : item.delta > 0 ? "positive-text" : ""}>{item.delta > 0 ? "+" : ""}{formatMetric(item.delta, item.metric)}</td><td>{item.tasks ?? "—"}</td></tr>)}</tbody></table></div></> : <EmptyState icon={<Wrench size={23} />} title="No ablation result in this run" detail="Run component-removal variants to isolate planner, retry, and verifier contribution." />}</div></section>
  );
}

function TraceInspector({ tasks, selected, onSelect }: { tasks: TaskResult[]; selected: number; onSelect: (index: number) => void }) {
  const task = tasks[Math.min(selected, tasks.length - 1)];
  const events = taskEvents(task);
  const passed = taskPassed(task);
  const metrics = asRecord(task.metrics);
  const verification = asRecord(task.verification);
  const answer = typeof task.answer === "string" ? task.answer : String(asRecord(task.answer)?.answer ?? "");
  return (
    <div className="trace-inspector panel">
      <div className="task-list" role="tablist" aria-label="Task results">{tasks.map((item, index) => { const success = taskPassed(item); return <button role="tab" aria-selected={selected === index} className={selected === index ? "selected" : ""} key={`${taskId(item, index)}-${item.mode ?? "run"}`} onClick={() => onSelect(index)}><span className={success ? "passed" : "failed"}>{success ? <Check size={12} /> : <X size={12} />}</span><div><strong>{taskId(item, index)}</strong><p>{taskTitle(item, index)}</p></div><small>{humanize(String(item.mode ?? "run"))}</small></button>; })}</div>
      <div className="trace-detail" role="tabpanel">
        <div className="trace-head"><div><span>{taskId(task, selected)} · {humanize(String(task.mode ?? "run"))}</span><h3>{taskTitle(task, selected)}</h3></div><span className={`task-outcome ${passed ? "passed" : "failed"}`}>{passed ? <CheckCircle2 size={14} /> : <XCircle size={14} />}{passed ? "Passed" : "Failed"}</span></div>
        {metrics && <div className="trace-metrics">{Object.entries(metrics).slice(0, 5).map(([key, value]) => <div key={key}><span>{metricLabel(key)}</span><strong>{typeof value === "number" ? formatMetric(value, key) : String(value)}</strong></div>)}</div>}
        <div className="trace-timeline">{events.length ? events.map((event, index) => <TraceEventRow key={event.id ?? `${event.stage}-${event.step ?? event.sequence ?? index}`} event={event} index={index} />) : <p className="trace-empty">No event-level trace was attached to this task result.</p>}</div>
        {(answer || verification || task.error) && <div className="trace-output">{answer && <div><span>FINAL ANSWER</span><p>{answer}</p></div>}{verification && <div><span>VERIFIER DECISION</span><p>{String(verification.reason ?? (Array.isArray(verification.reasons) ? verification.reasons.join(" · ") : verification.passed ? "Passed configured checks." : "Did not pass configured checks."))}</p></div>}{task.error && <div className="trace-error"><span>ERROR</span><p>{task.error}</p></div>}</div>}
      </div>
    </div>
  );
}

function TraceEventRow({ event, index }: { event: TraceEvent; index: number }) {
  const stage = String(event.stage ?? event.type ?? "event");
  const status = String(event.status ?? "succeeded");
  const details = event.output ?? event.details ?? event.metadata ?? event.input;
  const icon = stage.includes("plan") ? <Route size={14} /> : stage.includes("tool") ? <Wrench size={14} /> : stage.includes("verif") ? <ShieldCheck size={14} /> : stage.includes("retry") ? <RefreshCw size={14} /> : <Activity size={14} />;
  return (
    <div className={`trace-event ${status}`}><div className="event-rail"><span>{icon}</span><i /></div><div className="event-body"><div><span>{String(event.step ?? event.sequence ?? index + 1).padStart(2, "0")} · {humanize(stage)}</span>{typeof event.duration_ms === "number" && <small>{event.duration_ms.toFixed(0)}ms</small>}</div><strong>{event.title ?? event.message ?? `${humanize(stage)} ${humanize(status)}`}</strong>{details !== undefined && <details><summary>Inspect payload <ChevronRight size={13} /></summary><pre>{JSON.stringify(details, null, 2)}</pre></details>}</div><span className={`event-status ${status}`}>{humanize(status)}</span></div>
  );
}

export default App;
