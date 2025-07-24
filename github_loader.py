import os
import asyncio
import base64
import binascii
import re
from urllib.parse import urlparse
from typing import List, Optional
from llama_index.readers.github.repository.github_client import GithubClient
from llama_index.core import VectorStoreIndex, Document
from registry import CodeSplitterRegistry


class GithubLoader:
    """
    A class that loads GitHub repositories, splits code into chunks,
    creates a vector index, and persists it locally.
    """
    
    def __init__(self, github_token: str, persist_base_dir: str = "repositories"):
        """
        Initialize the GithubLoader.
        
        Args:
            github_token: GitHub API token for authentication
            persist_base_dir: Base directory for persisting indexes
        """
        self.github_token = github_token
        self.persist_base_dir = persist_base_dir
        self.client = GithubClient(github_token=github_token)
        self.splitter_registry = CodeSplitterRegistry()
    
    def load_repo(self, repo_url: str, branch: str, file_regex: Optional[str] = None) -> VectorStoreIndex:
        """
        Load a GitHub repository and create a queryable vector index.
        
        Args:
            repo_url: GitHub repository URL (https://github.com/{owner}/{repo})
            file_regex: Optional regex pattern to filter files
            
        Returns:
            VectorStoreIndex: A queryable vector index of the repository
        """
        # Parse repository information
        owner, repo = self._parse_repo_spec(repo_url)
        print(f"Loading repository: {owner}/{repo} (branch: {branch})")
        
        # Fetch and process repository files
        documents = self._fetch_and_process_repo(owner, repo, branch, file_regex)
        print(f"Loaded {len(documents)} code chunks from '{owner}/{repo}' (branch {branch})")
        
        # Create vector index
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print("Built VectorStoreIndex over all code chunks")
        
        # Persist the index
        self._persist_index(index, repo)
        
        return index

    def _parse_repo_spec(self, repo_url: str) -> tuple[str, str]:
        """Parse a GitHub repo URL of the form https://github.com/{owner}/{repo}"""
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        parsed = urlparse(repo_url)
        parts = parsed.path.strip("/").split("/")

        # Expect ['{owner}', '{repo}']
        if len(parts) < 2:
            raise ValueError(
                f"Repository URL must be in the form https://github.com/{{owner}}/{{repo}}"
            )

        owner, repo = parts[0], parts[1]
        return owner, repo

    def _fetch_and_process_repo(self, owner: str, repo: str, branch: str, file_regex: Optional[str]) -> List[Document]:
        """
        Fetch files from GitHub repo, split them into chunks, and return Document list.
        """
        documents: List[Document] = []
        
        async def fetch_repo_files():
            # Compile the regex pattern for file filtering (if provided)
            regex_pattern = None
            if file_regex:
                try:
                    regex_pattern = re.compile(file_regex)
                    print(f"Using regex filter: {file_regex}")
                except re.error as e:
                    print(f"Invalid regex pattern '{file_regex}': {e}")
                    return []
            else:
                print("No regex filter provided - processing all supported files")

            # Get tree SHA of specified branch
            branch_info = await self.client.get_branch(owner, repo, branch=branch)
            root_tree_sha = branch_info.commit.commit.tree.sha
            
            files_content: List[tuple[str, str]] = []

            # Recursive fetch helper that traverses the entire repository
            async def recurse(tree_sha: str, base_path: str = ""):
                tree = await self.client.get_tree(owner, repo, tree_sha)
                for obj in tree.tree:
                    full_path = f"{base_path}{obj.path}" if base_path else obj.path
                    
                    if obj.type == "blob":
                        # Check if this file path matches the regex pattern (if provided)
                        # If no regex is provided, include all files
                        if regex_pattern is None or regex_pattern.match(full_path):
                            blob = await self.client.get_blob(owner, repo, obj.sha)
                            if not blob or blob.content is None:
                                continue
                            # Decode base64
                            if blob.encoding != "base64":
                                continue
                            try:
                                data = base64.b64decode(blob.content)
                            except binascii.Error:
                                print(f"Could not decode {full_path}; skipping.")
                                continue
                            # Decode text
                            try:
                                text = data.decode("utf-8")
                            except UnicodeDecodeError:
                                print(f"Skipping non-text file {full_path}.")
                                continue
                            files_content.append((full_path, text))
                            print(f"Found file: {full_path}")
                    elif obj.type == "tree":
                        # Recursively traverse subdirectories
                        await recurse(obj.sha, full_path + "/")

            # Start traversal from the root
            await recurse(root_tree_sha)
            
            if not files_content:
                if file_regex:
                    print(f"No files found matching pattern '{file_regex}' in repository.")
                else:
                    print("No supported files found in repository.")
            else:
                if file_regex:
                    print(f"Found {len(files_content)} files matching pattern '{file_regex}'.")
                else:
                    print(f"Found {len(files_content)} supported files in repository.")
                
            return files_content

        # Run async fetch
        repo_files = asyncio.run(fetch_repo_files())
        
        # Split and wrap files into documents
        for file_path, text in repo_files:
            if not self.splitter_registry.is_supported(file_path):
                continue
            print(f"Splitting file: {file_path}")
            try:
                splitter = self.splitter_registry.get_splitter(file_path)
                chunks = splitter.split_text(text)
                documents.extend(self._create_documents(chunks, file_path))
            except ValueError as e:
                print(f"Error splitting {file_path}: {e}")
                continue
        
        return documents
    
    def _create_documents(self, chunks: List[str], file_path: str) -> List[Document]:
        """Convert code chunks into embedded Document objects."""
        documents: List[Document] = []
        for idx, chunk in enumerate(chunks):
            metadata = {"file_path": file_path, "chunk_index": idx}
            documents.append(Document(text=chunk, metadata=metadata))
        print(f"Embedded {len(documents)} chunks from {file_path}")
        return documents
    
    def _persist_index(self, index: VectorStoreIndex, repo_name: str) -> None:
        """Persist the vector index to local storage."""
        persist_dir = os.path.join(self.persist_base_dir, repo_name)
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"Persisted vector store to '{persist_dir}'")
