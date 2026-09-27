# SmallForge architecture

SmallForge is an evaluation workbench, not a general autonomous-agent framework. Its unit of
analysis is a paired experiment: the same model attempts the same versioned task once directly
and once with an explicit harness. The observable difference is attributed to the configured
harness components, subject to the limitations in the experiment report.

```text
React workbench / CLI
          |
          v
 FastAPI experiment API ---- SQLite experiment registry
          |
          v
  Versioned benchmark loader
          |
          +---------------- baseline -----------------+
          |                                            |
          +--> plan --> validate --> tools --> answer --> verify --> bounded retry
                                      |          |
                                      + evidence + citations
          |
          v
 Per-task scoring --> paired comparison + confidence intervals --> JSON / Markdown report
```

## Runtime boundaries

- `models/` is a provider boundary. Ollama and OpenAI-compatible endpoints implement one async
  request/response contract, so evaluation logic is provider-neutral.
- `harness/` owns orchestration. It exposes stage events but never logs hidden chain-of-thought.
- `tools/` is an allowlisted capability boundary. Arguments are validated before execution.
- `evaluation/` does not import the harness. It scores normalized candidate observations, which
  allows historical results to remain comparable while orchestration evolves.
- `experiments/` persists immutable requests and completed result payloads in SQLite.
- `api/` and `frontend/` are presentation and control surfaces; they do not calculate scores.

## Agent state machine

```text
START
  |
  v
PLAN -- invalid schema --> REPAIR PLAN -- retry budget exhausted --> FAIL CLOSED
  |
  v
EXECUTE ALLOWLISTED TOOLS -- invalid args/tool error --> observation
  |
  v
SYNTHESIZE STRUCTURED ANSWER
  |
  v
VERIFY -- pass --> COMPLETE
  |
  +-- fail + retry budget --> REVISE --> VERIFY
  |
  +-- fail + no budget --> RETURN FLAGGED RESULT
```

Retries are bounded. A verifier failure is retained in the result rather than hidden, and every
provider/tool attempt contributes to latency and usage metrics.

## Security decisions

- Calculator expressions are interpreted through an allowlisted AST; Python `eval` is never used.
- SQL executes against experiment-local in-memory tables, permits only one `SELECT`/`WITH`
  statement, uses SQLite authorizer and VM-step limits, and caps returned rows.
- Retrieval is restricted to the documents bundled with the current task.
- Model-generated tool names and arguments must match registered tools and Pydantic schemas.
- Provider keys come from environment variables and are never stored in experiment records.

These controls reduce risk; they are not a substitute for tenant isolation, egress policy,
secret scanning, authentication, rate limiting, or a production sandbox.

