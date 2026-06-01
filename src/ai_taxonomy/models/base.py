from abc import ABC, abstractmethod
from ..taxonomy.schema import ClassificationResult


class BaseClassifier(ABC):
    model_name: str = ""

    @abstractmethod
    def classify(self, tool_id: str, tool_name: str, tool_description: str, taxonomy_json: str) -> ClassificationResult:
        """Classify a single tool. Returns a validated ClassificationResult."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
