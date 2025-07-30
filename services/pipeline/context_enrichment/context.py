from pydantic import BaseModel, Field
from typing import List

class Context(BaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="List of entities directly shown in the snippet.",
    )
    prepend: str = Field(
        default="",
        description="Minimal context to place BEFORE the snippet (imports, parent entity signatures, docstrings, critical comments).",
    )
    postpend: str = Field(
        default="",
        description="Minimal context to place AFTER the snippet, only if critical to understanding.",
    )