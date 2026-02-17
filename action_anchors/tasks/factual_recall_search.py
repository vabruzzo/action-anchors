import json
from pathlib import Path

from action_anchors.tasks.base import BaseTask, TaskExample

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or factual information. Returns a short snippet with the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


class FactualRecallSearchTask(BaseTask):
    def __init__(
        self,
        config: dict,
        data_path: str = "action_anchors/data/factual_recall.json",
    ):
        self.config = config
        self.data_path = Path(data_path)
        self._examples: list[TaskExample] | None = None
        # Map from example id to ground truth, set during collection
        self._ground_truth_map: dict[str, str] = {}

    def get_system_prompt(self) -> str:
        return self.config["tasks"]["factual_recall"]["system_prompt"].strip()

    def get_tools(self) -> list[dict]:
        return [SEARCH_TOOL]

    def get_examples(self) -> list[TaskExample]:
        if self._examples is None:
            with open(self.data_path) as f:
                data = json.load(f)
            self._examples = []
            for item in data:
                example = TaskExample(
                    id=item["id"],
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    metadata={
                        "difficulty": item.get("difficulty", "medium"),
                        "expected_tool_use": item.get("expected_tool_use", True),
                        "category": item.get("category", "unknown"),
                    },
                )
                self._examples.append(example)
                self._ground_truth_map[item["id"]] = item["ground_truth"]
        return self._examples

    def set_current_example(self, example_id: str) -> None:
        """Set the current example so execute_tool can return the right ground truth."""
        self._current_example_id = example_id

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Return a canned search response with the ground truth answer.

        This keeps the experiment controlled — the tool response is deterministic,
        so any variation in behavior comes from the CoT, not from noisy search results.
        """
        if tool_name != "web_search":
            return f"Error: unknown tool '{tool_name}'"

        query = arguments.get("query", "")
        ground_truth = self._ground_truth_map.get(
            getattr(self, "_current_example_id", ""),
            "Information not available.",
        )
        return f'Search results for "{query}": According to Wikipedia, {ground_truth}.'
