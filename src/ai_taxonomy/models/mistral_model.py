import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from .. import config
from ..taxonomy.schema import ClassificationResult
from .base import BaseClassifier

_SYSTEM = """You are a taxonomy classifier for AI tools.
Classify the given tool using the provided taxonomy. Return ONLY valid JSON matching the schema.

Rules:
- Choose the single most accurate L3 node.
- If genuinely ambiguous between two L2s, set ambiguous=true and fill alternative_l2.
- Confidence reflects how well the description matches the chosen category.
- reasoning must be one sentence, max 20 words."""

_SCHEMA = ClassificationResult.model_json_schema()


class MistralClassifier(BaseClassifier):
    model_name = config.MISTRAL_MODEL

    def __init__(self):
        self._client = MistralClient(api_key=config.MISTRAL_API_KEY)

    def classify(self, tool_id: str, tool_name: str, tool_description: str, taxonomy_json: str) -> ClassificationResult:
        system = f"{_SYSTEM}\n\nTaxonomy:\n{taxonomy_json}\n\nSchema:\n{json.dumps(_SCHEMA, indent=2)}"
        user = f"Name: {tool_name}\nDescription: {tool_description}"

        response = self._client.chat(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        data["tool_id"] = tool_id
        data["tool_name"] = tool_name
        return ClassificationResult(**data)
