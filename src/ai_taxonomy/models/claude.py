import json
import anthropic
from .. import config
from ..taxonomy.schema import ClassificationResult
from .base import BaseClassifier

_SYSTEM = """You are a taxonomy classifier for AI tools.
You will be given the name and description of an AI tool.
Classify it using the taxonomy below. Return ONLY valid JSON matching the schema.

Rules:
- Choose the single most accurate L3 node.
- If genuinely ambiguous between two L2s, set ambiguous=true and fill alternative_l2.
- Confidence reflects how well the description matches the chosen category (not your model certainty).
- reasoning must be one sentence, max 20 words."""

_SCHEMA = ClassificationResult.model_json_schema()


class ClaudeClassifier(BaseClassifier):
    model_name = config.CLAUDE_MODEL

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def classify(self, tool_id: str, tool_name: str, tool_description: str, taxonomy_json: str) -> ClassificationResult:
        system = f"{_SYSTEM}\n\nTaxonomy:\n{taxonomy_json}\n\nSchema:\n{json.dumps(_SCHEMA, indent=2)}"
        user = f"Name: {tool_name}\nDescription: {tool_description}"

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text
        data = json.loads(raw)
        data["tool_id"] = tool_id
        data["tool_name"] = tool_name
        return ClassificationResult(**data)
