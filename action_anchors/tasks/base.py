from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskExample:
    id: str
    question: str
    ground_truth: Any
    metadata: dict = field(default_factory=dict)


class BaseTask(ABC):
    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return list of tool specs in OpenAI function-calling format."""
        ...

    @abstractmethod
    def get_examples(self) -> list[TaskExample]:
        """Return the curated dataset for this task."""
        ...

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result string."""
        ...
