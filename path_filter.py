import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class FilterType(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class Filter:
    regex: str
    filter_type: FilterType


class PathFilter:
    """Pre‑compiles regex filters and decides whether a path should be kept."""

    def __init__(self, filters: List[Filter]) -> None:
        self.include = [re.compile(f.regex) for f in filters if f.filter_type == FilterType.INCLUDE]
        self.exclude = [re.compile(f.regex) for f in filters if f.filter_type == FilterType.EXCLUDE]

    def match(self, path: str) -> bool:
        """Check if the path matches the include and exclude filters, returning True if it passes."""
        if any(p.search(path) for p in self.exclude):
            return False
        if self.include and not any(p.search(path) for p in self.include):
            return False
        return True