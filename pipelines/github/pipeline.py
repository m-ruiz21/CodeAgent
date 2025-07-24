import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from llama_index.core import Settings
from typing import Optional
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

from pipelines.IPipeline import IPipeline
from github_loader import GithubLoader

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

load_dotenv()

# Get Azure OpenAI credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
github_key = os.getenv("GITHUB_KEY")

Settings.llm = AzureOpenAI(
    model="o3-mini",
    deployment_name="o3-mini",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview",
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2023-05-15",
)

class GitHubPipeline(IPipeline):
    """
    GitHub pipeline that loads repositories and creates queryable vector indexes.
    Implements the IPipeline interface using the GithubLoader functionality.
    """
    
    def __init__(
        self,  
        repo_url: str, 
        branch: str,
        file_regex: Optional[str] = None,
        persist_base_dir: str = "stores"
    ):
        """
        Initialize the GitHub pipeline.
        
        Args:
            github_token: GitHub API token for authentication
            repo_url: GitHub repository URL (https://github.com/{owner}/{repo}/tree/{branch})
            file_regex: Optional regex pattern to filter files
            persist_base_dir: Base directory for persisting indexes
        """
        github_token=os.getenv("GITHUB_KEY")
        self.github_token = github_token
        self.repo_url = repo_url
        self.branch = branch
        self.file_regex = file_regex
        self.persist_base_dir = persist_base_dir
        self.loader = GithubLoader(github_token=github_token, persist_base_dir=persist_base_dir)
        self._index: Optional[VectorStoreIndex] = None
    
    def load(self) -> VectorStoreIndex:
        """
        Load data from the GitHub repository with optional file filtering.
        
        Returns:
            VectorStoreIndex: A queryable vector index of the repository
        """
        print(f"Loading GitHub repository: {self.repo_url}")
        if self.file_regex:
            print(f"Using file filter: {self.file_regex}")
        
        # Use the GithubLoader to load and process the repository
        self._index = self.loader.load_repo(self.repo_url, self.branch, self.file_regex)
        
        return self._index
    
    def poll(self):
        """
        Poll for updates in the data source and update the index if necessary.
        
        Note: This method is currently unimplemented as requested.
        Future implementation could check for new commits and update the index accordingly.
        """
        pass
    
    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """Get the current vector index if available."""
        return self._index
    

# example with https://github.com/m-ruiz21/pound/tree/master 
# Global instance of the GitHub pipeline
github_pipeline = GitHubPipeline(
    repo_url="https://github.com/m-ruiz21/pound",
    branch="master",
    file_regex=None
)

github_pipeline.load()