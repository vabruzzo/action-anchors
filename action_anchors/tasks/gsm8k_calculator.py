import ast
import json
import operator
from pathlib import Path

from datasets import load_dataset

from action_anchors.tasks.base import BaseTask, TaskExample

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Use this for any arithmetic calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '(16 - 3) * 4 + 2'",
                }
            },
            "required": ["expression"],
        },
    },
}

# Supported operators for safe eval
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float | int:
    """Safely evaluate a mathematical expression using AST parsing.

    Only supports arithmetic operators: + - * / // % **
    Raises ValueError for anything else (function calls, attribute access, etc.).
    """
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _SAFE_OPS[op_type](operand)
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class GSM8KCalculatorTask(BaseTask):
    def __init__(self, config: dict, data_path: str = "action_anchors/data/gsm8k_subset.json"):
        self.config = config
        self.data_path = Path(data_path)
        self._examples: list[TaskExample] | None = None

    def get_system_prompt(self) -> str:
        return self.config["tasks"]["gsm8k"]["system_prompt"].strip()

    def get_tools(self) -> list[dict]:
        return [CALCULATOR_TOOL]

    def get_examples(self) -> list[TaskExample]:
        if self._examples is None:
            with open(self.data_path) as f:
                data = json.load(f)
            self._examples = [
                TaskExample(
                    id=item["id"],
                    question=item["question"],
                    ground_truth=item["answer"],
                    metadata={"answer_str": item["answer_str"]},
                )
                for item in data
            ]
        return self._examples

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name != "calculator":
            return f"Error: unknown tool '{tool_name}'"
        expression = arguments.get("expression", "")
        try:
            result = safe_eval(expression)
            # Return clean integer if result is a whole number
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


def create_gsm8k_subset(
    n: int = 100,
    output_path: str = "action_anchors/data/gsm8k_subset.json",
    seed: int = 42,
):
    """Download GSM8K from HuggingFace and create a curated subset."""
    ds = load_dataset("openai/gsm8k", "main", split="test")

    subset = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        # Extract numeric answer from "#### {answer}" format
        answer_str = ex["answer"].split("####")[-1].strip().replace(",", "")
        try:
            answer = int(answer_str) if "." not in answer_str else float(answer_str)
        except ValueError:
            continue
        subset.append(
            {
                "id": f"gsm8k_{i:03d}",
                "question": ex["question"],
                "answer": answer,
                "answer_str": answer_str,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Saved {len(subset)} GSM8K problems to {output_path}")
    return subset


if __name__ == "__main__":
    create_gsm8k_subset()
