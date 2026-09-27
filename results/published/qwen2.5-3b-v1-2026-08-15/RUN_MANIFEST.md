# Published real-model run

This folder is a frozen, reproducible result from SmallForge. It is checked in so reviewers can
inspect measured output without accepting a screenshot or running a local model.

## Run configuration

| Field | Value |
|---|---|
| Date | 2026-08-15 |
| Experiment ID | `2df56e94fa22` |
| Inference provenance | Real local model |
| Provider | Ollama 0.24.0 |
| Model | `qwen2.5:3b` |
| Model size / quantization | 3.1B parameters / Q4_K_M |
| Benchmark | `smallforge-core-synthetic` v1.0.0 |
| Tasks | 20 (synthetic, fictional data) |
| Variants | direct baseline, full harness |
| Sampling | temperature 0, seed 42 |

The model artifact is distributed under the Qwen Research License, not Apache 2.0. This run is a
local research evaluation; commercial users must review the model terms independently.

The direct baseline received each instruction without external resource/tool access. The full
harness received only the task-local resources exposed through allowlisted tools. The comparison
therefore measures the combined value of planning, resource access, structured output, tools,
verification, and bounded retries—not an intrinsic change to the model weights.

## Measured result

| Metric | Baseline | Full harness | Delta |
|---|---:|---:|---:|
| Task accuracy | 30.0% | 60.0% | **+30.0 pp** |
| Schema validity | 75.0% | 100.0% | +25.0 pp |
| Groundedness | 25.0% | 75.0% | +50.0 pp |
| Citation recall | 0.0% | 83.3% | +83.3 pp |
| Tool selection accuracy | 0.0% | 81.2% | +81.2 pp |
| Tool argument accuracy | 0.0% | 55.0% | +55.0 pp |
| Mean latency | 1.51 s | 12.94 s | **+11.43 s** |
| Mean retries | 0.00 | 0.80 | +0.80 |

The paired bootstrap 95% interval for the task-accuracy delta was **-5 to +60 percentage points**.
It crosses zero because the benchmark has only 20 tasks, so the observed +30 pp should be reported
as a result of this run, not a proven population-wide improvement.

Category accuracy exposes where the aggregate came from:

| Category | Baseline | Full harness |
|---|---:|---:|
| Arithmetic | 75% | 0% |
| Read-only SQL | 0% | 50% |
| Retrieval | 0% | 100% |
| Structured extraction | 0% | 100% |
| Safe abstention | 100% | 50% |
| Claim verification | 50% | 50% |

The arithmetic regression is material: tool-produced evidence can be perfectly cited yet originate
from a semantically wrong plan. SmallForge retains that failure because grounding is not the same as
correctness. A next experiment should add independent plan verification and a larger, reviewed test
set before claiming production readiness.

## Files

- `baseline.json` / `baseline.md`: direct-run aggregate and task scores.
- `full.json` / `full.md`: harness aggregate and task scores.
- `comparison.json` / `comparison.md`: paired metric deltas and confidence intervals.

