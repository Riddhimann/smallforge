# SmallForge

SmallForge is an evaluation-first agent harness that answers a concrete question:

> Can planning, structured output, bounded tools, retrieval, verification, and retries make the
> **same small language model** more reliable—and what do those controls cost?

It is a working local application: Ollama and OpenAI-compatible model adapters, a typed agent state
machine, safe calculator/read-only SQL/BM25 retrieval tools, a versioned benchmark, paired bootstrap
comparisons, durable SQLite experiments, FastAPI, CLI, Docker, CI, and a React reliability dashboard.

## Measured output

A checked-in real-model run used local `qwen2.5:3b` (3.1B, Q4_K_M) on the 20-task synthetic v1
benchmark:

| Metric | Direct baseline | Full harness | Change |
|---|---:|---:|---:|
| Task accuracy | 30% | 60% | **+30 percentage points** |
| Schema validity | 75% | 100% | +25 pp |
| Groundedness | 25% | 75% | +50 pp |
| Mean latency | 1.51 s | 12.94 s | **+11.43 s** |

This is an observed result, not a production claim. The paired 95% interval for accuracy change was
-5 to +60 pp, the dataset is synthetic and small, and arithmetic accuracy regressed from 75% to 0%.
The report makes that failure visible: grounded tool output can still originate from a semantically
wrong plan.

Read the [run manifest](results/published/qwen2.5-3b-v1-2026-08-15/RUN_MANIFEST.md),
[paired comparison](results/published/qwen2.5-3b-v1-2026-08-15/comparison.md), or raw
[comparison JSON](results/published/qwen2.5-3b-v1-2026-08-15/comparison.json).

## Quick start on Windows

Prerequisites: Python 3.11+, Node.js 20+, and [Ollama](https://ollama.com/).

```powershell
cd SmallForge
python -m pip install -e ".[dev]"
ollama pull qwen2.5:3b
python -m smallforge.cli doctor

cd frontend
npm install
npm run build
cd ..

python -m smallforge.cli serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Configure `baseline + full`, start an
experiment, and inspect metric deltas and individual planner/tool/verifier traces. The API also
publishes interactive documentation at `/docs`.

The 3B Qwen artifact is under the Qwen Research License. Review its terms before non-research use;
the [Ollama model page](https://ollama.com/library/qwen2.5%3A3b-instruct) documents the model and
license. You can select any installed generative Ollama model in the dashboard.

## Run from the CLI

```powershell
python -m smallforge.cli run `
  --name "Qwen 3B paired run" `
  --model "ollama:qwen2.5:3b" `
  --benchmark v1 `
  --modes baseline,full
```

Reports are written as JSON and Markdown under `results/`. Temperature is zero and the paired seed
is 42. Provider seeds improve repeatability but do not guarantee bitwise reproducibility across
provider/runtime versions.

## Execution modes

| Mode | Controls |
|---|---|
| `baseline` | One direct model call; no external resources or tools |
| `structured` | Provider-native JSON schema plus local validation |
| `tools` | Planning and non-retrieval task tools, without corrective retries |
| `retrieval` | Task-local evidence retrieval and citation-aware synthesis |
| `verified` | All advertised tools plus deterministic verification, no correction retry |
| `full` | Planning, task tools, retrieval, typed synthesis, verification, and bounded retries |

The default comparison is `baseline,full`. Additional modes are component profiles; select them to
create an ablation chart. Expected answers stay inside the evaluator and are never added to model
prompts. Extraction output schemas are task contracts, so those schemas are supplied without their
expected values.

## Architecture

```text
React dashboard / CLI
          |
       FastAPI -------- SQLite experiment registry
          |
  versioned benchmark
          |
          +------ direct baseline
          |
          +------ plan -> policy validation -> tools -> typed answer -> verify -> retry
                                      |
                                observable trace
          |
     task scoring -> paired confidence intervals -> JSON/Markdown reports
```

Important boundaries:

- `models/`: async Ollama and OpenAI-compatible adapters; scripted responses exist only in tests.
- `harness/`: explicit plan/execute/synthesize/verify/retry state machine and safe observable events.
- `tools/`: AST calculator (never `eval`), query-only bounded SQLite, BM25 retrieval, document
  extraction context, and claim-evidence inspection.
- `evaluation/`: framework-neutral task/result contracts, scoring, confidence intervals, and reports.
- `experiments/`: background execution, progress, artifacts, and SQLite persistence.
- `api/` + `frontend/`: experiment control and result inspection; neither calculates scores.

See [architecture details](docs/architecture.md), [evaluation methodology](docs/evaluation-methodology.md),
and the [Google ADK/LangChain mapping](docs/framework-mapping.md).

## Benchmark and metrics

The bundled v1 benchmark contains 20 fictional tasks across arithmetic, SQL, retrieval, structured
extraction, unanswerable questions, and claim verification. It measures task accuracy, schema
validity, tool selection/arguments, groundedness, citation precision/recall, abstention,
verification, retries, and latency. Rates have confidence intervals; variant deltas use paired
bootstrap intervals on common task IDs.

Use this suite for regression and mechanism testing. Before a domain claim, build a larger,
independently reviewed dataset sampled from the intended workflow and add repeated stochastic runs,
human agreement, prompt-injection tests, cost, p95 latency, and failure-rate monitoring.

## Tests and quality checks

```powershell
pytest
ruff check src tests
cd frontend
npm run build
```

The current implementation passes 51 Python tests and the production frontend build. CI runs Python
lint/tests and the TypeScript/Vite build. Docker Compose connects the containerized API/dashboard to
Ollama on the host through `host.docker.internal`.

## Configuration

Copy `.env.example` or set environment variables in your shell:

- `SMALLFORGE_OLLAMA_URL`
- `SMALLFORGE_DEFAULT_MODEL`
- `SMALLFORGE_OPENAI_BASE_URL`
- `SMALLFORGE_OPENAI_API_KEY`
- `SMALLFORGE_OPENAI_MODEL`
- `SMALLFORGE_DATA_DIR`
- `SMALLFORGE_RESULTS_DIR`
- `SMALLFORGE_MAX_CONCURRENT_EXPERIMENTS`

Secrets are read from the environment and are not stored in experiment records.

## Citi interview preparation

The project demonstrates Python, SQL, RAG-style retrieval, GenAI/agent design, FastAPI, prompt and
schema engineering, validation, experimental comparison, visualization, and communicating business
trade-offs. The optional Google ADK bridge is installed with:

```powershell
python -m pip install -e ".[adk]"
```

It exposes SmallForge's bounded tools as ADK function tools but was not used for the published
result. That provenance distinction is intentional. Use the [interview guide](docs/interview-guide.md)
for the two-minute explanation, design trade-offs, likely follow-ups, and honest resume wording.
