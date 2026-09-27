# Google ADK and LangChain mapping

SmallForge's measured runtime is intentionally framework-independent so each reliability mechanism
can be removed and evaluated. That makes it straightforward to explain how the same design maps to
frameworks named in the Citi role.

| SmallForge concept | Google ADK analogue | LangChain/LangGraph analogue |
|---|---|---|
| `ModelAdapter` | model used by `Agent` / `LlmAgent` | chat-model interface |
| Pydantic tool input | Python function tool signature | `StructuredTool` / `@tool` schema |
| `AgentHarness` stages | agent + callbacks/workflow agents | `StateGraph` nodes and edges |
| Bounded retry | `LoopAgent` or explicit callback policy | conditional edge with retry counter |
| Observable `RunTrace` | ADK events/evaluation trajectory | LangSmith/LangGraph state events |
| Versioned evaluation suite | ADK eval set and criteria | LangSmith dataset/evaluator |
| FastAPI experiment API | ADK API server or custom service | custom FastAPI deployment |

The optional [`google_adk.py`](../src/smallforge/integrations/google_adk.py) bridge exposes the safe
calculator, task-local retrieval, and read-only SQL implementations as plain async Python functions
inside an ADK `Agent`. It is not used to produce the published result; keeping that provenance
separate avoids implying that an unexecuted framework adapter was benchmarked.

Install it only when needed:

```bash
pip install -e ".[adk]"
```

Google describes ADK as a code-first, model- and deployment-agnostic toolkit with function tools,
agent composition, evaluation, and deployment support. Its stable Python package is `google-adk`.
Useful primary references:

- [Google ADK Python repository](https://github.com/google/adk-python)
- [Google ADK documentation](https://google.github.io/adk-docs/)
- [Google agents-cli evaluation workflow](https://google.github.io/agents-cli/guide/evaluation/)

## Interview position

Do not say “I used ADK to obtain the 30-point result.” Say:

> I built the evaluation state machine explicitly so I could ablate planning, tools, retrieval,
> verification, and retries. I then added a thin Google ADK bridge showing how the bounded tools map
> to ADK function tools. In a delivery project I would choose ADK or LangGraph when their session,
> deployment, and observability ecosystem reduces operational work; I would retain the independent
> benchmark contract to avoid framework lock-in.

