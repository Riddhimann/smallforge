export type ExperimentStatus = "pending" | "running" | "completed" | "failed";

export type Health = {
  status: "ok" | "degraded" | "error" | string;
  version?: string;
  active_experiments?: number;
  storage?: string | { status?: string; backend?: string; [key: string]: unknown };
  [key: string]: unknown;
};

export type ModelDescriptor = {
  id: string;
  name: string;
  provider: string;
  available: boolean;
  details?: Record<string, unknown> | string | null;
  result_kind?: string;
  synthetic?: boolean;
};

export type BenchmarkDescriptor = {
  id: string;
  name: string;
  description: string;
  task_count: number;
  categories: string[];
  provenance: string | Record<string, unknown>;
};

export type ExperimentRequest = {
  name?: string;
  model_id: string;
  benchmark_id: string;
  modes: string[];
  max_tasks?: number | null;
};

export type MetricComparison = {
  key?: string;
  metric?: string;
  name?: string;
  label?: string;
  unit?: string;
  direction?: "higher" | "lower" | string;
  baseline?: number;
  harness?: number;
  delta?: number;
  delta_pct?: number;
  baseline_value?: number;
  harness_value?: number;
  [key: string]: unknown;
};

export type TraceEvent = {
  id?: string;
  step?: number;
  sequence?: number;
  stage?: string;
  type?: string;
  status?: string;
  title?: string;
  message?: string;
  input?: unknown;
  output?: unknown;
  duration_ms?: number;
  attempt?: number;
  tool?: string;
  details?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
};

export type Verification = {
  passed?: boolean;
  score?: number;
  reason?: string;
  checks?: Array<{ name?: string; passed?: boolean; detail?: string }>;
  [key: string]: unknown;
};

export type TaskResult = {
  id?: string;
  run_id?: string;
  task_id?: string;
  name?: string;
  task?: string | Record<string, unknown>;
  mode?: "baseline" | "harness" | string;
  success?: boolean;
  answer?: string | { answer?: string; citations?: string[]; confidence?: number; [key: string]: unknown } | null;
  model?: string;
  plan?: Record<string, unknown> | null;
  tool_results?: Array<Record<string, unknown>>;
  evidence?: unknown[];
  verification?: Verification | string;
  metrics?: Record<string, number | string | boolean | null>;
  trace?: TraceEvent[] | { events?: TraceEvent[]; [key: string]: unknown };
  retries?: number;
  retry_count?: number;
  diagnostics?: unknown;
  validation_issues?: unknown[];
  error?: string | null;
  [key: string]: unknown;
};

export type AblationResult = {
  id?: string;
  mode?: string;
  name?: string;
  label?: string;
  success_rate?: number;
  score?: number;
  delta?: number;
  tasks?: number;
  [key: string]: unknown;
};

export type ExperimentResults = {
  summary?: Record<string, unknown>;
  comparisons?: MetricComparison[] | Record<string, unknown>;
  ablations?: AblationResult[];
  ablation?: AblationResult[] | Record<string, unknown>;
  task_results?: TaskResult[];
  provenance?: string | Record<string, unknown>;
  result_kind?: string;
  synthetic?: boolean;
  [key: string]: unknown;
};

export type Experiment = {
  id: string;
  status: ExperimentStatus;
  request: ExperimentRequest;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  results?: ExperimentResults | null;
  progress?: number | { completed?: number; total?: number; percent?: number };
  name?: string;
  [key: string]: unknown;
};

export type ItemEnvelope<T> = { items: T[] };
