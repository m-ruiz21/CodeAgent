import re
from dataclasses import dataclass
from enum import Enum
from typing import List

class FilterType(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"

class FilteredObjectType(Enum):
    FILE = "file"
    DIRECTORY = "directory"

@dataclass
class FileFilter:
    regex: str
    filter_type: FilterType

@dataclass 
class DirectoryFilter:
    regex: str
    filter_type: FilterType

class PathFilter:
    """Pre‑compiles regex filters and decides whether a path should be kept."""

    def __init__(self, file_filters: List[FileFilter], path_filters: List[DirectoryFilter]) -> None:
        self.include_file_patterns = [re.compile(f.regex) for f in file_filters if f.filter_type == FilterType.INCLUDE]
        self.exclude_file_patterns = [re.compile(f.regex) for f in file_filters if f.filter_type == FilterType.EXCLUDE]
        self.include_directory_patterns = [re.compile(d.regex) for d in path_filters if d.filter_type == FilterType.INCLUDE]
        self.exclude_directory_patterns = [re.compile(d.regex) for d in path_filters if d.filter_type == FilterType.EXCLUDE]

    def match(self, path: str, type: FilteredObjectType) -> bool:
        """Check if the path matches the include and exclude filters, returning True if it passes."""
        
        if type == FilteredObjectType.FILE:
            return self._match_file(path)
        elif type == FilteredObjectType.DIRECTORY:
            return self._match_directory(path)

    def _match_file(self, path: str) -> bool:
        """Check if the file path matches the include and exclude filters."""
        if not self.include_file_patterns and not self.exclude_file_patterns:
            return True
        
        return any(
            pf.search(path) for pf in self.include_file_patterns
        ) and not any(
            pf.search(path) for pf in self.exclude_file_patterns
        )
    
    def _match_directory(self, path: str) -> bool:
        """Check if the directory path matches the include and exclude filters."""
        if not self.include_directory_patterns and not self.exclude_directory_patterns:
            return True

        return any(
            pf.search(path) for pf in self.include_directory_patterns
        ) and not any(
            pf.search(path) for pf in self.exclude_directory_patterns
        )
