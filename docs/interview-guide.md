# SmallForge interview guide

## A concise project explanation

“SmallForge tests whether an explicit agent harness can make a small local language model more
reliable. I run the same model on the same versioned tasks in a direct baseline and in controlled
ablations. The harness adds schema-constrained planning, allowlisted tools, retrieval, evidence-
aware synthesis, verification, and bounded retries. I persist per-step traces and compare paired
accuracy, tool correctness, grounding, latency, and recovery—not just a hand-picked demo.”

## The design choices you should be able to defend

### Why a custom harness instead of starting with LangChain?

The experiment needs each mechanism to be independently observable and removable. A small custom
state machine makes the ablation boundary, retry count, provider calls, validation failures, and
tool policy explicit. A framework adapter can be added later; the evaluation contract remains
framework-neutral. This is not an argument that application teams should always build their own
orchestrator.

### Why structured output is not enough

A JSON schema can guarantee shape, not truth. `{"revenue": 900}` may be syntactically valid and
factually unsupported. SmallForge therefore scores schema validity separately from task accuracy
and evidence support.

### Why tools help a small model

The model is used for language-level routing and synthesis while deterministic systems perform
operations they are better suited for: arithmetic, SQL execution, and document lookup. The tool
result becomes an observation; it does not automatically become a trusted final answer. Argument
validation and verification remain necessary.

### Why verification can fail

A model-based verifier may share the generator's blind spots, add latency, or confidently approve
an error. SmallForge keeps deterministic checks where possible, records verification separately,
and bounds retries. For a high-stakes system, add rule-based invariants, independent models, human
review, and calibrated thresholds.

### Why paired evaluation?

Task difficulty varies. Subtracting the baseline score from the candidate score on each identical
task controls that source of variance better than comparing unrelated samples. Paired bootstrap
intervals then estimate uncertainty in the mean delta without assuming normality.

## Likely interview questions

**What is the biggest limitation?** The bundled benchmark is small and synthetic. It validates the
mechanism and catches regressions, but it does not prove enterprise generalization.

**How would you scale it?** Put experiment jobs on a durable queue, store normalized traces and
artifacts in object storage plus Postgres, add authenticated tenants and quotas, batch provider
requests, and export OpenTelemetry spans. Keep benchmark labels in a separate evaluation service.

**How do you prevent prompt injection?** Tools are capability-scoped and task-local, retrieved text
is treated as untrusted data, generated calls are schema-validated, and SQL is read-only and
resource-bounded. A production version would add content provenance, instruction/data delimiters,
injection classifiers, egress controls, and adversarial evals.

**What would you monitor?** Task-family success, invalid-schema rate, tool error rate, verifier
reject/recovery rate, unsupported-claim rate, retries, tokens, p50/p95 latency, provider failures,
and changes relative to a pinned baseline.

**When would you use a larger model instead?** When the small model plus harness has worse total
cost or latency at the required quality threshold, when failures are not detectable, or when tasks
depend on broad reasoning that tools cannot reliably decompose.

## Honest resume wording

The checked-in Qwen run supports this two-line version:

> Built an evaluation-first agent harness for a local 3.1B-parameter LLM with schema-constrained
> planning, allowlisted calculator/read-only SQL/BM25 retrieval tools, evidence verification, and
> bounded retries.
>
> Observed task accuracy improve from 30% to 60% on a versioned 20-task synthetic benchmark, while
> reporting the paired 95% interval (-5 to +60 percentage points), latency increase (1.51 s to
> 12.94 s), and arithmetic regression instead of presenting the result as production-ready.

Keep “synthetic benchmark” in the project description or methodology, and update every number if
you publish a later run.
