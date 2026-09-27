"""Sequential executor for validated bounded plans."""

from __future__ import annotations

from smallforge.schemas import EventStatus, ExecutionPlan, RunTrace, ToolCall, ToolResult
from smallforge.tools import ToolRegistry


class ToolExecutor:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    async def execute(self, plan: ExecutionPlan, trace: RunTrace) -> list[ToolResult]:
        results: list[ToolResult] = []
        for step in plan.steps:
            trace.add(
                "tool_execution",
                EventStatus.STARTED,
                f"Executing {step.tool} for step {step.id}.",
                details={"step_id": step.id, "tool": step.tool},
            )
            result = await self.tools.execute(
                ToolCall(step_id=step.id, name=step.tool, arguments=step.arguments)
            )
            results.append(result)
            trace.add(
                "tool_execution",
                EventStatus.SUCCEEDED if result.success else EventStatus.FAILED,
                (
                    f"Tool {step.tool} completed."
                    if result.success
                    else f"Tool {step.tool} failed with {result.error_code}."
                ),
                duration_ms=result.duration_ms,
                details={
                    "step_id": step.id,
                    "tool": step.tool,
                    "success": result.success,
                    "error_code": result.error_code,
                    "evidence_ids": [item.source_id for item in result.evidence],
                },
            )
        return results
