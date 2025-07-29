import asyncio
from typing import List, Optional, Tuple
from urllib.parse import urlparse
import redis
from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.github.repository.github_client import GithubClient

from services.github.utils.path_filter import PathFilter, FileFilter, DirectoryFilter
from services.github.utils.repo_walker import RepoFile, RepoWalker
from services.pipeline.code_splitter.registry import CodeSplitterRegistry

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
        file_filters: Optional[List[FileFilter]] = None,
        directory_filters: Optional[List[DirectoryFilter]] = None,
    ) -> None:
        self.url = url
        self.branch = branch
        self.client = GithubClient(github_token=token)
        self.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.parse = parse
        self.show_progress = show_progress
        self.file_filters = file_filters or []
        self.directory_filters = directory_filters or []
        self.splitter_registry = CodeSplitterRegistry()


    def load_data(self) -> List[Document]:
        owner, repo = self._parse_repo_url(self.url)
        print(f"Loading {owner}/{repo}@{self.branch}")

        repo_files = asyncio.run(
            RepoWalker(self.client, owner, repo, self.branch, self.show_progress).scrape(
                PathFilter(self.file_filters, self.directory_filters)
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
            desc=f"Creating Documents from {len(repo_files)} repo files",
            unit="file",
            disable=not self.show_progress,
        )

        docs: List[Document] = []
        for rf in repo_files:
            bar.update(1)
            
            if not self.parse:
                docs.append(
                    Document(
                        text=rf.content,
                        doc_id=rf.path,
                        metadata={"file_path": rf.path}
                    )
                )
                continue
            
            if not self.splitter_registry.is_supported(rf.path):
                continue

            splitter = self.splitter_registry.get_splitter(rf.path)
            chunks = splitter.split_text(rf.content)
            docs.extend(
                Document(text=chunk, doc_id=f'{rf.path}::chunk-{i}', metadata={"file_path": rf.path, "chunk_index": i})
                for i, chunk in enumerate(chunks)
            )
            bar.set_postfix({"chunks": len(chunks)})

        bar.close()
        return docs