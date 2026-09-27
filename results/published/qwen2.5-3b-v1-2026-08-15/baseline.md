# SmallForge evaluation: baseline

> **Scope:** This benchmark uses synthetic task data. Scores measure this versioned
> suite only and do not establish real-world, production, safety, or model quality.

## Run information

- Run ID: `2df56e94fa22-baseline`
- System: `ollama:qwen2.5:3b`
- Variant: `baseline`
- Benchmark: `smallforge-core-synthetic` `1.0.0`
- Created (UTC): `2026-08-15T04:57:13+00:00`
- Evaluated tasks: `20`

## Aggregate metrics

| Metric | Estimate | Confidence interval | n | Method | Direction |
|---|---:|---:|---:|---|---|
| `task_accuracy` | 30.0% | 14.5% – 51.9% | 20 | Wilson score | higher is better |
| `schema_validity` | 75.0% | 30.1% – 95.4% | 4 | Wilson score | higher is better |
| `tool_selection_accuracy` | 0.0% | 0.0% – 0.0% | 20 | bootstrap percentile | higher is better |
| `tool_argument_accuracy` | 0.0% | 0.0% – 0.0% | 20 | bootstrap percentile | higher is better |
| `groundedness` | 25.0% | 0.0% – 62.5% | 8 | bootstrap percentile | higher is better |
| `citation_precision` | 25.0% | 0.0% – 62.5% | 8 | bootstrap percentile | higher is better |
| `citation_recall` | 0.0% | 0.0% – 0.0% | 6 | bootstrap percentile | higher is better |
| `abstention_accuracy` | 100.0% | 34.2% – 100.0% | 2 | Wilson score | higher is better |
| `verification_accuracy` | 75.0% | 30.1% – 95.4% | 4 | Wilson score | higher is better |
| `retry_count` | 0.00 | 0.00 – 0.00 | 20 | bootstrap percentile | lower is better |
| `latency_ms` | 1,511.8 ms | 958.8 ms – 2,114.4 ms | 20 | bootstrap percentile | lower is better |

## Accuracy by category

| Category | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| arithmetic | 3 | 4 | 75.0% |
| read_only_sql | 0 | 4 | 0.0% |
| retrieval | 0 | 4 | 0.0% |
| structured_extraction | 0 | 4 | 0.0% |
| unanswerable | 2 | 2 | 100.0% |
| verification | 1 | 2 | 50.0% |

## Task traces

| Task | Category | Accuracy | Schema | Tool | Args | Grounded | Retries | Latency | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `arith_001` | arithmetic | 100% | — | 0% | 0% | — | 0 | 1,430.4 ms |  |
| `arith_002` | arithmetic | 100% | — | 0% | 0% | — | 0 | 3,400.4 ms |  |
| `arith_003` | arithmetic | 100% | — | 0% | 0% | — | 0 | 3,143.2 ms |  |
| `arith_004` | arithmetic | 0% | — | 0% | 0% | — | 0 | 3,947.8 ms |  |
| `sql_001` | read_only_sql | 0% | — | 0% | 0% | — | 0 | 3,177.0 ms |  |
| `sql_002` | read_only_sql | 0% | — | 0% | 0% | — | 0 | 1,807.8 ms |  |
| `sql_003` | read_only_sql | 0% | — | 0% | 0% | — | 0 | 2,381.0 ms |  |
| `sql_004` | read_only_sql | 0% | — | 0% | 0% | — | 0 | 4,066.9 ms |  |
| `ret_001` | retrieval | 0% | — | 0% | 0% | 0% | 0 | 683.2 ms |  |
| `ret_002` | retrieval | 0% | — | 0% | 0% | 0% | 0 | 532.5 ms |  |
| `ret_003` | retrieval | 0% | — | 0% | 0% | 0% | 0 | 518.5 ms |  |
| `ret_004` | retrieval | 0% | — | 0% | 0% | 0% | 0 | 912.3 ms |  |
| `ext_001` | structured_extraction | 0% | 100% | 0% | 0% | — | 0 | 783.5 ms |  |
| `ext_002` | structured_extraction | 0% | 100% | 0% | 0% | — | 0 | 920.9 ms |  |
| `ext_003` | structured_extraction | 0% | 100% | 0% | 0% | — | 0 | 836.4 ms |  |
| `ext_004` | structured_extraction | 0% | 0% | 0% | 0% | — | 0 | 686.7 ms |  |
| `unans_001` | unanswerable | 100% | — | 0% | 0% | 100% | 0 | 195.5 ms |  |
| `unans_002` | unanswerable | 100% | — | 0% | 0% | 100% | 0 | 295.6 ms |  |
| `verify_001` | verification | 0% | — | 0% | 0% | 0% | 0 | 371.0 ms |  |
| `verify_002` | verification | 100% | — | 0% | 0% | 0% | 0 | 146.2 ms |  |

## Diagnostics

- `ret_001`: Missing required evidence IDs: POL-01
- `ret_002`: Missing required evidence IDs: SLA-A
- `ret_003`: Missing required evidence IDs: NIM-END
- `ret_004`: Missing required evidence IDs: FIN-Q2, FIN-Q3
- `ext_004`: $.due_date: expected string
- `verify_001`: Missing required evidence IDs: CEDAR-1
- `verify_002`: Missing required evidence IDs: ATLAS-A, ATLAS-B

## Metric interpretation

- `task_accuracy`: deterministic match against the task's declared answer rule.
- `schema_validity`: validation against the task's dependency-free JSON Schema subset.
- `tool_selection_accuracy`: multiset F1 over expected and emitted tool names.
- `tool_argument_accuracy`: expected argument/rule matches after same-name call pairing.
- `groundedness`: citation provenance and claim-to-evidence attachment, not semantic entailment.
- `retry_count` and `latency_ms`: observable execution cost; lower is better.
