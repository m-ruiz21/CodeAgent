from __future__ import annotations
import asyncio, base64, binascii
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable, TypeVar

import httpx
from tqdm import tqdm

from llama_index.readers.github.repository.github_client import GithubClient, GitBranchResponseModel, GitBlobResponseModel, GitTreeResponseModel
from services.github.utils.path_filter import FilteredObjectType, PathFilter

T = TypeVar("T")

@dataclass
class RepoFile:
    path: str
    content: str


class RepoWalker:
    """Recursively walks a GitHub tree and yields RepoFile objects"""

    def __init__(
        self,
        client: GithubClient,
        owner: str,
        repo: str,
        branch: str,
        show_progress: bool = True,
        max_workers: int = 3,  
    ) -> None:
        self.client = client
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.show_progress = show_progress
        self._max_workers = max_workers

    async def scrape(self, path_filter: PathFilter) -> List[RepoFile]:
        """
        Traverse the repo and return all matching files.
        
        :param path_filter: PathFilter to apply on files and directories.
        :return: List of RepoFile objects that match the filter.
        :raises httpx.HTTPStatusError: If the GitHub API request fails.
        """

        bar = tqdm(desc="Scanning repo", unit="obj", disable=not self.show_progress)

        branch = await retries_wrapper(
            lambda: self.request(
                GitBranchResponseModel,
                "getBranch",
                "GET",
                owner=self.owner,
                repo=self.repo,
                branch=self.branch,
                timeout=15,
            ),
            retries=3,
            desc=f"getBranch(owner={self.owner}, repo={self.repo}, branch={self.branch})",
        )

        root_sha = branch.commit.commit.tree.sha

        bucket: List[RepoFile] = []
        semaphore = asyncio.Semaphore(self._max_workers)

        await self._walk(root_sha, path_filter, bar, bucket, semaphore)
        bar.close()
        return bucket

    async def _walk(
        self,
        sha: str,
        path_filter: PathFilter,
        bar: tqdm,
        bucket: List[RepoFile],
        sem: asyncio.Semaphore,
        prefix: str = "",
    ) -> None:
        """
        Depth‑first traversal with concurrency and retry support.
        
        :param sha: The SHA of the current tree.
        :param path_filter: PathFilter to apply on files and directories.
        :param bar: Progress bar for tracking progress.
        :param bucket: List to collect RepoFile objects.
        :param sem: Semaphore for controlling concurrency.
        :param prefix: Path prefix for nested directories.
        
        :raises httpx.HTTPStatusError: If the GitHub API request fails.
        """
        async with sem:
            tree = await retries_wrapper(
                lambda: self.request(
                    GitTreeResponseModel,
                    "getTree",
                    "GET",
                    owner=self.owner,
                    repo=self.repo,
                    tree_sha=sha,
                    timeout=15
                ),
                retries=3,
                desc=f"get_tree(owner={self.owner}, repo={self.repo}, sha={sha})",
            )

        if tree is None:
            return

        tasks: List[asyncio.Task] = []
        for obj in tree.tree:
            full_path = f"{prefix}{obj.path}" if prefix else obj.path

            if obj.type == "blob":
                if not path_filter.match(full_path, FilteredObjectType.FILE): continue
                
                tasks.append(
                    asyncio.create_task(
                        self._handle_blob(obj.sha, full_path, bar, bucket, sem)
                    )
                )

            elif obj.type == "tree": 
                if not path_filter.match(full_path, FilteredObjectType.DIRECTORY): continue
                
                tasks.append(
                    asyncio.create_task(
                        self._walk(
                            obj.sha,
                            path_filter,
                            bar,
                            bucket,
                            sem,
                            prefix=full_path + "/",
                        )
                    )
                )
            else: 
                continue

        if tasks:
            await asyncio.gather(*tasks)

    async def _handle_blob(
        self,
        sha: str,
        path: str,
        bar: tqdm,
        bucket: List[RepoFile],
        sem: asyncio.Semaphore,
    ) -> None:
        """
        Fetch & decode a single blob under concurrency control and retries.
        
        :param sha: The SHA of the blob.
        :param path: The path of the blob.
        :param bar: Progress bar for tracking progress.
        :param bucket: List to collect RepoFile objects.
        :param sem: Semaphore for controlling concurrency.

        :raises httpx.HTTPStatusError: If the GitHub API request fails.
        """
        
        async with sem:
            bar.update(1)
            text = await self._decode_blob(sha)

        if text:
            bucket.append(RepoFile(path, text))
            bar.set_description(f"Scanning repo, ingested {len(bucket)} files")

    async def _decode_blob(self, sha: str) -> Optional[str]:
        """
        Decode a blob's content from base64 to UTF‑8 (with retries).
        
        :param sha: The SHA of the blob to decode.
        :return: The decoded content as a string, or None if decoding fails.

        :raises httpx.HTTPStatusError: If the GitHub API request fails.
        """
        blob = await retries_wrapper(
            lambda: self.request(
                    GitBlobResponseModel,
                    "getBlob",
                    "GET",
                    owner=self.owner,
                    repo=self.repo,
                    file_sha=sha,
                    timeout=30,
                ),
            retries=3,
            desc=f"get_blob(owner={self.owner}, repo={self.repo}, sha={sha})",
        )

        if not blob or blob.encoding != "base64" or blob.content is None:
            return None
        try:
            raw = base64.b64decode(blob.content)
            return raw.decode("utf-8", errors="replace")
        except (binascii.Error, UnicodeDecodeError):
            return None
        
    async def request(
        self,
        response_model: T,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        timeout: Optional[int] = 5,
        retries: int = 0,
        **kwargs: Any,
    ) -> T:
        res: httpx.Response = await self.client.request(
            endpoint,
            method,
            headers,
            timeout,
            retries,
            **kwargs
        )

        if res.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Request failed with status {res.status_code}: {res.text}",
                request=res.request,
                response=res
            )

        return response_model.from_json(res.text) 


async def retries_wrapper(
    fun: Callable[[], Awaitable],
    retries: int,
    desc: str,
):
    """
    Retries a function call with exponential backoff
    
    :param fun: The function to call.
    :param retries: Number of retries.
    :param desc: Description for logging.

    :return: The result of the function call, or None if all retries fail.
    """
    delay = 0.0
    for attempt in range(retries):
        try:
            return await fun()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                reset = int(e.response.headers.get("x-rateLimit-reset", time.time()+60))
                delay = max(reset - time.time(), 30)
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                print("Awake to resume operation")
        
        except Exception as e:
            print(f"Operation '{desc}' failed on attempt {attempt + 1}: {type(e).__name__}: {e}")
            delay = 2 ** attempt
        finally:
            if attempt > retries - 2:
                print(f"Operation '{desc}' failed after {retries} retries")
                raise 
            await asyncio.sleep(delay)