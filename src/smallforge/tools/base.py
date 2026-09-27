"""Typed tool contracts and a failure-isolating registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from smallforge.schemas import Evidence, ToolCall, ToolDefinition, ToolResult


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class ToolPayload(BaseModel):
    output: dict[str, Any]
    evidence: list[Evidence] = Field(default_factory=list)


class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @abstractmethod
    async def run(self, arguments: BaseModel) -> ToolPayload:
        """Execute with arguments already validated against input_model."""


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition for name in sorted(self._tools)]

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        # Constructing the definition validates the public tool name/schema now.
        _ = tool.definition
        self._tools[tool.name] = tool

    async def execute(self, call: ToolCall) -> ToolResult:
        started = perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                step_id=call.step_id,
                tool_name=call.name,
                success=False,
                error_code="unknown_tool",
                error_message=f"Tool {call.name!r} is not registered.",
                duration_ms=(perf_counter() - started) * 1_000,
            )

        try:
            validated = tool.input_model.model_validate(call.arguments)
            payload = await tool.run(validated)
            return ToolResult(
                step_id=call.step_id,
                tool_name=call.name,
                success=True,
                output=payload.output,
                evidence=payload.evidence,
                duration_ms=(perf_counter() - started) * 1_000,
            )
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_url=False, include_input=False)
            )
            return ToolResult(
                step_id=call.step_id,
                tool_name=call.name,
                success=False,
                error_code="invalid_arguments",
                error_message=messages[:1_000],
                duration_ms=(perf_counter() - started) * 1_000,
            )
        except ToolExecutionError as exc:
            return ToolResult(
                step_id=call.step_id,
                tool_name=call.name,
                success=False,
                error_code=exc.code,
                error_message=exc.safe_message[:1_000],
                duration_ms=(perf_counter() - started) * 1_000,
            )
        except Exception:
            # Do not leak driver paths, SQL internals or secrets into model context.
            return ToolResult(
                step_id=call.step_id,
                tool_name=call.name,
                success=False,
                error_code="internal_tool_error",
                error_message="The tool failed unexpectedly.",
                duration_ms=(perf_counter() - started) * 1_000,
            )
