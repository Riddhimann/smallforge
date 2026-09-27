"""A resource-bounded arithmetic evaluator; never calls Python eval()."""

from __future__ import annotations

import ast
import hashlib
import math
import operator
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from smallforge.schemas import Evidence
from smallforge.tools.base import BaseTool, ToolExecutionError, ToolPayload


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=512)


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate bounded arithmetic deterministically. Supports +, -, *, /, //, %, "
        "**, parentheses, pi/e and selected numeric functions."
    )
    input_model = CalculatorInput

    _binary: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _functions: dict[str, Callable[..., float]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
    }
    _constants = {"pi": math.pi, "e": math.e}

    def __init__(self, *, max_nodes: int = 80, max_abs_value: float = 1e100) -> None:
        self.max_nodes = max_nodes
        self.max_abs_value = max_abs_value

    async def run(self, arguments: BaseModel) -> ToolPayload:
        parsed = CalculatorInput.model_validate(arguments)
        try:
            tree = ast.parse(parsed.expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ToolExecutionError(
                "invalid_expression", "Expression is not valid arithmetic."
            ) from exc
        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise ToolExecutionError("expression_too_complex", "Expression exceeds the node limit.")
        try:
            result = self._evaluate(tree.body)
        except ToolExecutionError:
            raise
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ToolExecutionError(
                "arithmetic_error", "Expression cannot be evaluated safely."
            ) from exc
        result = self._bounded(result)
        if isinstance(result, float) and result.is_integer() and abs(result) < 2**53:
            serializable: int | float = int(result)
        else:
            serializable = result
        digest = hashlib.sha256(parsed.expression.encode("utf-8")).hexdigest()[:12]
        content = f"{parsed.expression} = {serializable}"
        return ToolPayload(
            output={"expression": parsed.expression, "result": serializable},
            evidence=[
                Evidence(
                    source_id=f"calculator:{digest}",
                    content=content,
                    score=1.0,
                    metadata={"kind": "deterministic_calculation"},
                )
            ],
        )

    def _evaluate(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolExecutionError(
                    "unsupported_expression", "Only numeric literals are allowed."
                )
            return self._bounded(node.value)
        if isinstance(node, ast.Name) and node.id in self._constants:
            return self._constants[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._bounded(self._unary[type(node.op)](self._evaluate(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e12):
                raise ToolExecutionError("unsafe_power", "Exponent or base exceeds the safe limit.")
            return self._bounded(self._binary[type(node.op)](left, right))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._functions
            and not node.keywords
            and 1 <= len(node.args) <= 8
        ):
            values = [self._evaluate(argument) for argument in node.args]
            return self._bounded(self._functions[node.func.id](*values))
        raise ToolExecutionError(
            "unsupported_expression",
            "Expression contains an operation that is not allowlisted.",
        )

    def _bounded(self, value: int | float) -> float:
        if isinstance(value, bool) or isinstance(value, complex):
            raise ToolExecutionError("non_real_result", "Result must be a real number.")
        numeric = float(value)
        if not math.isfinite(numeric) or abs(numeric) > self.max_abs_value:
            raise ToolExecutionError(
                "result_out_of_range", "Result exceeds the safe numeric range."
            )
        return numeric
