import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .. import config
from ..taxonomy.schema import ClassificationResult
from .base import BaseClassifier

_PROMPT_TEMPLATE = """You are a taxonomy classifier for AI tools.
Classify the given tool using the taxonomy below. Return ONLY valid JSON matching the schema.

Rules:
- Choose the single most accurate L3 node.
- If genuinely ambiguous between two L2s, set ambiguous=true and fill alternative_l2.
- Confidence reflects how well the description matches the chosen category.
- reasoning must be one sentence, max 20 words.

Taxonomy:
{taxonomy_json}

Schema:
{schema_json}

Name: {tool_name}
Description: {tool_description}"""

_SCHEMA = ClassificationResult.model_json_schema()


class GeminiClassifier(BaseClassifier):
    model_name = config.GEMINI_MODEL

    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    def classify(self, tool_id: str, tool_name: str, tool_description: str, taxonomy_json: str) -> ClassificationResult:
        prompt = _PROMPT_TEMPLATE.format(
            taxonomy_json=taxonomy_json,
            schema_json=json.dumps(_SCHEMA, indent=2),
            tool_name=tool_name,
            tool_description=tool_description,
        )
        response = self._model.generate_content(prompt)
        data = json.loads(response.text)
        data["tool_id"] = tool_id
        data["tool_name"] = tool_name
        return ClassificationResult(**data)
