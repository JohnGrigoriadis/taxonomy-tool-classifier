from pydantic import BaseModel, Field
from typing import Optional


class TaxonomyNode(BaseModel):
    id: str
    label: str
    level: int


class ClassificationResult(BaseModel):
    tool_id: str
    tool_name: str
    l1_id: str
    l1_label: str
    l2_id: str
    l2_label: str
    l3_id: str
    l3_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One sentence justifying the classification.")
    ambiguous: bool = Field(description="True if the tool plausibly fits multiple L2 categories.")
    alternative_l2: Optional[str] = Field(default=None)
