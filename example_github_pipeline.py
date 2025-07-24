#!/usr/bin/env python3
"""
Example script demonstrating how to use the GitHub pipeline.
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from pipelines.registry import pipeline_registry

# Load environment variables
load_dotenv()

def configure_llama_models():
    """Configure Azure OpenAI models for LlamaIndex."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
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

def main():
    """Example usage of the GitHub pipeline."""
    # Configure models
    configure_llama_models()
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_KEY")
    if not github_token:
        raise ValueError("GITHUB_KEY environment variable is required")
    
    # Example repository URL - replace with your desired repo
    repo_url = "https://github.com/owner/repo/tree/main"
    
    # Get the GitHub pipeline class from registry
    GitHubPipeline = pipeline_registry.get_pipeline("github")
    
    # Create pipeline instance
    pipeline = GitHubPipeline(
        github_token=github_token,
        repo_url=repo_url,
        file_regex=r".*\.(py|js|ts|jsx|tsx)$",  # Only process certain file types
        persist_base_dir="./pipeline_indexes"
    )
    
    # Load the repository
    print("Loading repository...")
    index = pipeline.load()
    
    # Example query
    query = "How does the main functionality work?"
    print(f"\nQuerying: {query}")
    response = pipeline.query(query)
    print(f"Response: {response}")
    
    # The poll() method is currently unimplemented as requested
    print("\nPolling for updates...")
    pipeline.poll()  # This will do nothing for now
    print("Poll completed (no-op)")

if __name__ == "__main__":
    main()
