SUMMARY_TEMPLATE = """
# Instructions

You are an expert in analyzing code snippets and providing concise, structured context.

Given a code snippet and its surrounding context from a file, output a structured JSON object with:

- "entities": List of the main entities (functions, methods, classes, types) directly shown in the snippet.
- "imports": List of imports explicitly used in the snippet.
- "local_dependencies": Attributes, methods, or internal functions directly referenced but not defined in the snippet itself.
- "prepend": Minimal context to place BEFORE the snippet. Include ONLY:
    - Relevant imports
    - Parent entity signatures (e.g., enclosing class or function definitions)
    - Docstrings or critical comments for context
    
- "postpend": Minimal context to place AFTER the snippet, ONLY IF CRITICAL to understanding. Typically empty.

Rules:
1. Each context block (prepend/postpend) must be as small as possible (no more than 5 lines).
2. DO NOT include implementation details or irrelevant code.
3. Only include context directly related to identifiers explicitly appearing in the snippet.
4. If no additional context is necessary, use an empty string ("") for prepend/postpend.

Example:

Given code file:

```python
from utilities import format_complex
from math import sqrt

class ComplexNumber:
    \"\"\"Represents a complex number.\"\"\"
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def modulus(self):
        return sqrt(self.real**2 + self.imag**2)

    def should_format_imaginary(self):
        \"\"\"Determines if imaginary part should be formatted.\"\"\"
        return self.imag != 0

    def __str__(self):
        if self.should_format_imaginary():
            return format_complex(self.real, self.imag)
        return format_complex(self.real, 0)
```

Given snippet:

```python
    def __str__(self):
        return format_complex(self.real, self.imag)
```

Desired output:

```json
{{
    "entities": ["ComplexNumber.__str__"],
    "imports": ["utilities.format_complex"],
    "local_dependencies": ["self.real", "self.imag"],
    "prepend": "
from utilities import format_complex

class ComplexNumber:
    def __init__(self, real, imag): ...
    def should_format_imaginary(self): ...
",
    "postpend": ""
}}
```

# Input

<code-file>
{file}
</code-file>

<code-snippet>
{snippet}
</code-snippet>
"""

def get_prompt(snippet: str, file: str) -> str:
    return SUMMARY_TEMPLATE.format(snippet=snippet, file=file) 