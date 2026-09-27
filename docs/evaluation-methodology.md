# Evaluation methodology

## Question being tested

> For one fixed small language model and one fixed benchmark version, how do structured output,
> bounded tools, retrieval, verification, and retry policies change observable reliability and
> runtime cost?

The design deliberately avoids comparing different base models and then crediting the harness for
the difference. A baseline and candidate are paired by task ID.

The direct baseline receives the user instruction but has no external resource or tool access. The
harness variants receive task-local resources only through their configured capability boundary.
This intentionally measures the value of retrieval/tool access as part of the harness; it is not a
claim that prompting alone changed a model's intrinsic knowledge.

## Benchmark

The bundled `v1` suite is a compact synthetic benchmark covering deterministic arithmetic, SQL
over supplied tables, retrieval over supplied documents, structured extraction, abstention, and
verification. Expected labels are held by the evaluator and are never included in the candidate
payload.

Synthetic tasks are useful for regression testing because their answers and required tool calls
are unambiguous. They do **not** estimate production accuracy on arbitrary enterprise work. Add a
representative, independently reviewed benchmark before making domain claims.

## Metrics

- Task accuracy: category-specific exact, numeric-tolerance, or structured matching.
- Schema validity: whether requested structured output validates.
- Tool selection and argument accuracy: whether the expected capability was invoked correctly.
- Groundedness and citation precision/recall: whether claims cite expected supplied evidence.
- Abstention accuracy: whether unanswerable prompts are refused as expected.
- Verification accuracy and recovery: whether failures are detected and corrected.
- Latency and retry count: the operational trade-off of reliability mechanisms.

Aggregate rates include confidence intervals. Variant deltas use paired task observations, because
the same tasks appear in baseline and candidate runs. On only twenty tasks the intervals will be
wide; that uncertainty is a result, not an inconvenience to hide.

## Provenance rules

Each report keeps two independent labels:

1. `inference_provenance`: `real_model` or `scripted_fixture`.
2. `benchmark_provenance`: `synthetic` or `curated_real_world`.

The checked-in scripted fixture exists only to make CI deterministic. It must never be presented
as evidence that a language model improved. Resume claims should use a committed report produced
by a real provider and name the model, quantization, benchmark version, task count, seed, and date.

## Reproducibility checklist

- Pin the benchmark version and source revision.
- Record the exact provider model identifier and quantization.
- Use temperature zero and a fixed seed where the provider supports it.
- Run baseline and variants in the same environment.
- Preserve raw per-task observations, not only aggregate scores.
- Report failed provider calls and invalid outputs in the denominator.
- Repeat runs when estimating stochastic variance; a seed does not guarantee cross-platform bitwise
  reproducibility.
