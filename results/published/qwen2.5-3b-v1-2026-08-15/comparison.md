# SmallForge ablation comparison

> **Scope:** Results apply only to the named benchmark version. A confidence interval
> crossing zero does not demonstrate a reliable improvement on this suite.

- Benchmark: `smallforge-core-synthetic` `1.0.0`
- Baseline: `baseline`

## full versus baseline

Paired common tasks: `20`

| Metric | Baseline | Candidate | Delta | Delta CI | Direction |
|---|---:|---:|---:|---:|---|
| `task_accuracy` | 30.0% | 60.0% | 30.0% | -5.0% – 60.0% | higher is better |
| `schema_validity` | 75.0% | 100.0% | 25.0% | 0.0% – 75.0% | higher is better |
| `tool_selection_accuracy` | 0.0% | 81.2% | 81.2% | 67.8% – 92.0% | higher is better |
| `tool_argument_accuracy` | 0.0% | 55.0% | 55.0% | 32.5% – 75.0% | higher is better |
| `groundedness` | 25.0% | 75.0% | 50.0% | -12.5% – 100.0% | higher is better |
| `citation_precision` | 25.0% | 75.0% | 50.0% | 0.0% – 87.5% | higher is better |
| `citation_recall` | 0.0% | 83.3% | 83.3% | 50.0% – 100.0% | higher is better |
| `abstention_accuracy` | 100.0% | 50.0% | -50.0% | -100.0% – 0.0% | higher is better |
| `verification_accuracy` | 75.0% | 50.0% | -25.0% | -75.0% – 0.0% | higher is better |
| `retry_count` | 0.00 | 0.80 | 0.80 | 0.50 – 1.10 | lower is better |
| `latency_ms` | 1,511.8 ms | 12,944.4 ms | 11,432.5 ms | 8,955.6 ms – 13,986.5 ms | lower is better |

## Notes

- Same model, seed, benchmark tasks, and host were used for paired variants.
- The benchmark data are synthetic; inference was performed by the configured real model.
