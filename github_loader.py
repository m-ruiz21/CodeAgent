import asyncio
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.github.repository.github_client import GithubClient
from path_filter import Filter, PathFilter
from registry import CodeSplitterRegistry
from repo_walker import RepoFile, RepoWalker

class GithubReader(BaseReader):
    """
    Loads a GitHub repo, applies include/exclude filters,
    splits code into chunks, and returns a list of Documents.
    """

    def __init__(
        self,
        token: str,
        url: str,
        branch: str,
        parse: bool = True,
        show_progress: bool = True,
        filters: Optional[List[Filter]] = None,
    ) -> None:
        self.url = url
        self.branch = branch
        self.client = GithubClient(github_token=token)
        self.parse = parse
        self.show_progress = show_progress
        self.filters = filters or []
        self.splitter_registry = CodeSplitterRegistry()


    def load_data(self) -> List[Document]:
        owner, repo = self._parse_repo_url(self.url)
        print(f"Loading {owner}/{repo}@{self.branch}")

        repo_files = asyncio.run(
            RepoWalker(self.client, owner, repo, self.branch, self.show_progress).scrape(
                PathFilter(self.filters)
            )
        )
        docs = self._files_to_docs(repo_files)

        print(f"Loaded {len(docs)} code chunks from {owner}/{repo}@{self.branch}")
        return docs


    @staticmethod
    def _parse_repo_url(repo_url: str) -> Tuple[str, str]:
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        parts = urlparse(repo_url).path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("Repository URL must be https://github.com/{owner}/{repo}")
        return parts[0], parts[1]

    
    def _files_to_docs(self, repo_files: List[RepoFile]) -> List[Document]:
        """Split each supported file into chunks and wrap them in Documents."""
        bar = tqdm(
            total=len(repo_files),
            desc="Splitting files",
            unit="file",
            disable=not self.show_progress,
        )

        docs: List[Document] = []
        for rf in repo_files:
            bar.set_description(f"Splitting {rf.path}")
            bar.update(1)

            if not self.splitter_registry.is_supported(rf.path):
                continue

            splitter = self.splitter_registry.get_splitter(rf.path)
            chunks = splitter.split_text(rf.content)
            docs.extend(
                Document(text=chunk, metadata={"file_path": rf.path, "chunk_index": i})
                for i, chunk in enumerate(chunks)
            )
            bar.set_postfix({"chunks": len(chunks)})

        bar.close()
        return docs
