# SmallForge benchmarks

SmallForge benchmarks are immutable, versioned evaluation inputs. The default
`v1` suite contains **synthetic, fictional task data** designed to test harness
behaviour; it contains no customer data and is not a claim about production model
quality.

## Candidate output contract

Each candidate receives `BenchmarkTask.to_payload()`. Expected labels are omitted.
Return a mapping with these fields:

```json
{
  "task_id": "arith_001",
  "answer": 1090,
  "structured_output": null,
  "tool_calls": [{"name": "calculator", "arguments": {"expression": "1250 * 0.872"}}],
  "evidence_ids": [],
  "claims": [],
  "abstained": false,
  "verification": null,
  "retry_count": 0,
  "latency_ms": 83.4,
  "error": null,
  "metadata": {}
}
```

The evaluator accepts `citations` as an alias for `evidence_ids`, `retries` for
`retry_count`, and `elapsed_ms` for `latency_ms`.

## Metrics

- Task answer accuracy
- Structured-output schema validity
- Tool-selection multiset F1
- Tool-argument validity
- Citation provenance groundedness, precision, and recall
- Correct abstention and verification
- Retry count and wall latency

Confidence intervals are Wilson intervals for binary rates and seeded percentile
bootstrap intervals for continuous metrics. Ablations use paired bootstrap
candidate-minus-baseline intervals on common task IDs.

Groundedness is intentionally narrow: it checks whether citations are attached and
allowed by the benchmark labels. It does **not** claim that arbitrary prose is
semantically entailed. Verification accuracy is scored separately.

## Version policy

Never edit a released suite after publishing results. Create `v2` and document the
change. Reports must always include the benchmark name and semantic version.

Run the CLI after installing the package:

```bash
python -m smallforge.evaluation list
python -m smallforge.evaluation score \
  --benchmark v1 \
  --predictions predictions.jsonl \
  --system-name qwen-small \
  --variant standalone \
  --output-dir artifacts/evaluations
```

Use `--synthetic-fixture-results` only for demos/tests whose outputs did not come
from a model. Reports then display an explicit warning.
