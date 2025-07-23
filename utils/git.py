from __future__ import annotations

import fnmatch
import os
import subprocess
from typing import Dict, List, Tuple

from llama_index.core import Document
from registry import CodeSplitterRegistry 

def _parse_gitignore(path: str) -> List[str]:
    patterns: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    except FileNotFoundError:
        pass
    return patterns


def _should_ignore(fp: str, patterns: List[str], base: str) -> bool:
    rel = os.path.relpath(fp, base)
    if rel.startswith(".git/") or "/.git/" in rel or rel == ".git":
        return True

    for pat in patterns:
        if pat.endswith("/"):                    # directory pattern
            if rel.startswith(pat) or f"/{pat}" in rel:
                return True
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat):
            return True
    return False


def get_head_commit(repo_path: str) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path)
        .decode()
        .strip()
    )


def git_pull(repo_path: str) -> None:
    subprocess.run(["git", "pull", "--ff-only"], cwd=repo_path, check=True)


def get_changed_files(repo_path: str, old: str, new: str) -> List[Tuple[str, str]]:
    """
    Return list of (status, absolute_path) since `old` -> `new`.
    Status codes: A, M, D, R...
    """
    diff = subprocess.check_output(
        ["git", "diff", "--name-status", old, new], cwd=repo_path
    ).decode()

    return [
        (line.split("\t")[0], os.path.join(repo_path, line.split("\t")[1]))
        for line in diff.strip().splitlines()
    ]
