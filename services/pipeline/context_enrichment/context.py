from pydantic import BaseModel, Field
from typing import List

class Context(BaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="List of main entities (functions, methods, classes, types) directly shown in the snippet.",
    )
    imports: List[str] = Field(
        default_factory=list,
        description="List of imports explicitly used in the snippet.",
    )
    local_dependencies: List[str] = Field(
        default_factory=list,
        description="List of attributes, methods, or internal functions directly referenced but not defined in the snippet itself.",
    )
    prepend: str = Field(
        default="",
        description="Minimal context to place BEFORE the snippet (imports, parent entity signatures, docstrings, critical comments).",
    )
    postpend: str = Field(
        default="",
        description="Minimal context to place AFTER the snippet, only if critical to understanding.",
    )