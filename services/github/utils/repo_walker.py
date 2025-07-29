import base64
import binascii
from dataclasses import dataclass
from typing import List, Optional

from tqdm import tqdm

from llama_index.readers.github.repository.github_client import GithubClient

from services.github.utils.path_filter import PathFilter

@dataclass
class RepoFile:
    path: str
    content: str


class RepoWalker:
    """Recursively walks a GitHub tree and yields RepoFile objects."""

    def __init__(
        self,
        client: GithubClient,
        owner: str,
        repo: str,
        branch: str,
        show_progress: bool,
    ) -> None:
        self.client = client
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.show_progress = show_progress

    async def scrape(self, path_filter: PathFilter) -> List[RepoFile]:
        """Scrape the repository, returning a list of RepoFile objects."""
        bar = tqdm(
            desc="Scanning repo",
            unit="obj",
            disable=not self.show_progress,
        )
        root_sha = (
            await self.client.get_branch(self.owner, self.repo, self.branch)
        ).commit.commit.tree.sha

        collected: List[RepoFile] = []
        await self._walk(root_sha, path_filter, bar, collected)
        bar.close()
        return collected

    async def _walk(
        self,
        sha: str,
        path_filter: PathFilter,
        bar: tqdm,
        bucket: List[RepoFile],
        prefix: str = "",
    ) -> None:
        """Recursively walk the tree starting from the given SHA."""
        tree = await self.client.get_tree(self.owner, self.repo, sha)
        for obj in tree.tree:
            full_path = f"{prefix}{obj.path}" if prefix else obj.path
            bar.set_description(f'processing path: {full_path}')

            if obj.type == "blob":
                bar.update(1)
                if not path_filter.match(full_path): continue

                text = await self._decode_blob(obj.sha)
                if text: bucket.append(RepoFile(full_path, text))
            else:
                await self._walk(obj.sha, path_filter, bar, bucket, full_path + "/")

    async def _decode_blob(self, sha: str) -> Optional[str]:
        """Decode a blob's content from base64 to UTF-8."""
        blob = await self.client.get_blob(self.owner, self.repo, sha)
        if not blob or blob.encoding != "base64" or blob.content is None:
            return None
        try:
            raw = base64.b64decode(blob.content)
            return raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
