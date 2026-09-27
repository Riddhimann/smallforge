# SmallForge evaluation: full

> **Scope:** This benchmark uses synthetic task data. Scores measure this versioned
> suite only and do not establish real-world, production, safety, or model quality.

## Run information

- Run ID: `2df56e94fa22-full`
- System: `ollama:qwen2.5:3b`
- Variant: `full`
- Benchmark: `smallforge-core-synthetic` `1.0.0`
- Created (UTC): `2026-08-15T05:01:32+00:00`
- Evaluated tasks: `20`

## Aggregate metrics

| Metric | Estimate | Confidence interval | n | Method | Direction |
|---|---:|---:|---:|---|---|
| `task_accuracy` | 60.0% | 38.7% – 78.1% | 20 | Wilson score | higher is better |
| `schema_validity` | 100.0% | 51.0% – 100.0% | 4 | Wilson score | higher is better |
| `tool_selection_accuracy` | 81.2% | 68.3% – 92.0% | 20 | bootstrap percentile | higher is better |
| `tool_argument_accuracy` | 55.0% | 35.0% – 75.0% | 20 | bootstrap percentile | higher is better |
| `groundedness` | 75.0% | 37.5% – 100.0% | 8 | bootstrap percentile | higher is better |
| `citation_precision` | 75.0% | 37.5% – 100.0% | 8 | bootstrap percentile | higher is better |
| `citation_recall` | 83.3% | 50.0% – 100.0% | 6 | bootstrap percentile | higher is better |
| `abstention_accuracy` | 50.0% | 9.5% – 90.5% | 2 | Wilson score | higher is better |
| `verification_accuracy` | 50.0% | 15.0% – 85.0% | 4 | Wilson score | higher is better |
| `retry_count` | 0.80 | 0.50 – 1.10 | 20 | bootstrap percentile | lower is better |
| `latency_ms` | 12,944.4 ms | 10,798.9 ms – 15,312.1 ms | 20 | bootstrap percentile | lower is better |

## Accuracy by category

| Category | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| arithmetic | 0 | 4 | 0.0% |
| read_only_sql | 2 | 4 | 50.0% |
| retrieval | 4 | 4 | 100.0% |
| structured_extraction | 4 | 4 | 100.0% |
| unanswerable | 1 | 2 | 50.0% |
| verification | 1 | 2 | 50.0% |

## Task traces

| Task | Category | Accuracy | Schema | Tool | Args | Grounded | Retries | Latency | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `arith_001` | arithmetic | 0% | — | 100% | 0% | — | 0 | 4,525.8 ms |  |
| `arith_002` | arithmetic | 0% | — | 100% | 100% | — | 2 | 8,085.8 ms | Answer did not pass validation and verification within the retry budget. |
| `arith_003` | arithmetic | 0% | — | 100% | 0% | — | 2 | 8,468.9 ms | Answer did not pass validation and verification within the retry budget. |
| `arith_004` | arithmetic | 0% | — | 100% | 0% | — | 2 | 9,621.3 ms | Answer did not pass validation and verification within the retry budget. |
| `sql_001` | read_only_sql | 100% | — | 100% | 100% | — | 1 | 6,938.2 ms |  |
| `sql_002` | read_only_sql | 0% | — | 100% | 0% | — | 1 | 10,100.8 ms |  |
| `sql_003` | read_only_sql | 0% | — | 67% | 0% | — | 1 | 11,906.2 ms |  |
| `sql_004` | read_only_sql | 100% | — | 67% | 0% | — | 1 | 14,919.1 ms |  |
| `ret_001` | retrieval | 100% | — | 100% | 100% | 100% | 0 | 13,045.0 ms |  |
| `ret_002` | retrieval | 100% | — | 67% | 100% | 100% | 0 | 12,481.6 ms |  |
| `ret_003` | retrieval | 100% | — | 67% | 100% | 100% | 0 | 11,699.4 ms |  |
| `ret_004` | retrieval | 100% | — | 40% | 0% | 0% | 2 | 19,176.9 ms | Answer did not pass validation and verification within the retry budget. |
| `ext_001` | structured_extraction | 100% | 100% | 100% | 100% | — | 1 | 13,961.1 ms |  |
| `ext_002` | structured_extraction | 100% | 100% | 100% | 100% | — | 1 | 14,000.5 ms |  |
| `ext_003` | structured_extraction | 100% | 100% | 67% | 100% | — | 1 | 23,049.6 ms |  |
| `ext_004` | structured_extraction | 100% | 100% | 100% | 100% | — | 1 | 18,217.8 ms |  |
| `unans_001` | unanswerable | 100% | — | 100% | 100% | 0% | 0 | 10,831.1 ms |  |
| `unans_002` | unanswerable | 0% | — | 0% | 0% | 100% | 0 | 9,693.7 ms | Unable to produce a valid bounded plan. |
| `verify_001` | verification | 0% | — | 100% | 50% | 100% | 0 | 12,901.4 ms |  |
| `verify_002` | verification | 100% | — | 50% | 50% | 100% | 0 | 25,263.2 ms |  |

## Diagnostics

- `arith_002`: Candidate error: Answer did not pass validation and verification within the retry budget.
- `arith_003`: Candidate error: Answer did not pass validation and verification within the retry budget.
- `arith_004`: Candidate error: Answer did not pass validation and verification within the retry budget.
- `ret_004`: Unknown or disallowed evidence IDs: calculator:09ab4d10466a, calculator:56824d16dcb6; Missing required evidence IDs: FIN-Q2, FIN-Q3; Candidate error: Answer did not pass validation and verification within the retry budget.
- `unans_001`: Expected no citations, but candidate cited: NORTH-PROD
- `unans_002`: Candidate error: Unable to produce a valid bounded plan.

## Metric interpretation

- `task_accuracy`: deterministic match against the task's declared answer rule.
- `schema_validity`: validation against the task's dependency-free JSON Schema subset.
- `tool_selection_accuracy`: multiset F1 over expected and emitted tool names.
- `tool_argument_accuracy`: expected argument/rule matches after same-name call pairing.
- `groundedness`: citation provenance and claim-to-evidence attachment, not semantic entailment.
- `retry_count` and `latency_ms`: observable execution cost; lower is better.
