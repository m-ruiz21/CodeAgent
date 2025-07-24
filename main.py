import os
import argparse
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from github_loader import GithubLoader

# Load environment variables
load_dotenv()

def configure_llama_models():
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

    return github_key

def main():
    parser = argparse.ArgumentParser(
        description="Index code from a GitHub repo into a LlamaIndex VectorStore with configurable file filtering."
    )
    parser.add_argument(
        "url",
        help="GitHub repository URL (https://github.com/{owner}/{repo}/tree/{branch})"
    )
    parser.add_argument(
        "--file-regex",
        default=None,
        help="Optional regex pattern to filter files. If not provided, all supported files will be processed."
    )
    args = parser.parse_args()
    
    # Configure Azure OpenAI models
    github_key = configure_llama_models()
    
    # Create GitHub loader and load repository
    loader = GithubLoader(github_token=str(github_key))
    index = loader.load_repo(args.url, args.file_regex)

    # Example query
    query = (
        "Give a basic rundown of how the scrolling logic works"
    )
    response = index.as_query_engine().query(query)
    print(f"Query response: {response}")

if __name__ == "__main__":
    main()
