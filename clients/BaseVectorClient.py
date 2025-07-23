"""
VectorDbClientFactory — produce a VectorDbClient that keeps its index in-sync
with the underlying Git repo.
"""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from utils.git import get_changed_files, get_head_commit, git_pull, _parse_gitignore, _should_ignore
from registry import CodeSplitterRegistry

load_dotenv()

Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_MODEL"),
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15",
)

class VectorDbClient:
    """
    Thin wrapper over VectorStoreIndex that knows how to:
      • git pull
      • detect changed files
      • incrementally update vectors
    """

    def __init__(self, repo_path: str, index: VectorStoreIndex) -> None:
        self.repo_path = repo_path
        self.splitter_registry = CodeSplitterRegistry()
        self.index = index
        self.last_commit = get_head_commit(repo_path)


    def _split_file(self, fp: str) -> List[str]:
        with open(fp, "r", encoding="utf-8") as fh:
            code = fh.read()
        splitter = self.splitter_registry.get_splitter(fp)
        return splitter.split_text(code)

    
    def update_index(self) -> None:
        """Pull & patch the vector index if the repo advanced."""
        git_pull(self.repo_path)
        new_commit = get_head_commit(self.repo_path)
        if new_commit == self.last_commit:
            return 

        gitignore = _parse_gitignore(os.path.join(self.repo_path, ".gitignore"))
        for status, path in get_changed_files(self.repo_path, self.last_commit, new_commit):
            if _should_ignore(path, gitignore, self.repo_path):
                continue

            if path in self.file_doc_ids:
                for doc_id in self.file_doc_ids[path]:
                    self.index.delete(doc_id)
                self.file_doc_ids.pop(path, None)

            if status == "D": continue # skip deleted files
            
            try:
                chunks = self._split_file(path)
            except ValueError:
                continue

            new_ids: List[str] = []
            for idx, chunk in enumerate(chunks):
                doc_id = f"{path}::chunk{idx}"
                self.index.insert(
                    Document(
                        id_=doc_id,
                        text=chunk,
                        metadata={"file_path": path, "chunk_index": idx},
                    )
                )
                new_ids.append(doc_id)
            self.file_doc_ids[path] = new_ids

        self.last_commit = new_commit