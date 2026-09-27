import type {
  BenchmarkDescriptor,
  Experiment,
  ExperimentRequest,
  Health,
  ItemEnvelope,
  ModelDescriptor,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly requestId?: string;

  constructor(message: string, status: number, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 20_000): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const abortFromCaller = () => controller.abort();
  init?.signal?.addEventListener("abort", abortFromCaller, { once: true });

  try {
    const response = await fetch(path, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });

    if (!response.ok) {
      const requestId = response.headers.get("x-request-id") ?? undefined;
      let message = `Request failed with status ${response.status}.`;
      try {
        const body = (await response.json()) as { detail?: unknown; message?: unknown };
        const detail = body.detail ?? body.message;
        if (typeof detail === "string") message = detail;
        else if (detail) message = JSON.stringify(detail);
      } catch {
        // Preserve the useful HTTP fallback for empty or non-JSON responses.
      }
      throw new ApiError(message, response.status, requestId);
    }

    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The reliability service did not respond in time.");
    }
    throw new Error("Could not reach the SmallForge API. Check that the service is running.", {
      cause: error,
    });
  } finally {
    window.clearTimeout(timeout);
    init?.signal?.removeEventListener("abort", abortFromCaller);
  }
}

function items<T>(payload: ItemEnvelope<T> | T[]): T[] {
  return Array.isArray(payload) ? payload : payload.items ?? [];
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<Health>("/api/health", { signal }),

  models: async (signal?: AbortSignal) =>
    items(await request<ItemEnvelope<ModelDescriptor> | ModelDescriptor[]>("/api/models", { signal })),

  benchmarks: async (signal?: AbortSignal) =>
    items(
      await request<ItemEnvelope<BenchmarkDescriptor> | BenchmarkDescriptor[]>(
        "/api/benchmarks",
        { signal },
      ),
    ),

  experiments: async (signal?: AbortSignal) =>
    items(await request<ItemEnvelope<Experiment> | Experiment[]>("/api/experiments", { signal })),

  createExperiment: (payload: ExperimentRequest, signal?: AbortSignal) =>
    request<Experiment>(
      "/api/experiments",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      },
      30_000,
    ),

  experiment: (id: string, signal?: AbortSignal) =>
    request<Experiment>(`/api/experiments/${encodeURIComponent(id)}`, { signal }),
};
