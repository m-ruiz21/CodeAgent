import os
import argparse
import pickle
from typing import Dict, Iterator, List, TypeVar, Sequence
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from redisvl.schema import IndexSchema
from pipeline import run_pipeline
from services.github.utils.path_filter import DirectoryFilter, FileFilter, FilterType
from services.pipeline.context_enrichment.context_enricher import ContextEnricher
from services.pipeline.code_splitter.code_splitter import CodeSplitter
from services.github.github_loader import GithubReader
from services.pipeline.code_splitter.registry import CodeSplitterRegistry
from services.cache.doc_service import DocService, set_doc_service
from services.pipeline.solution_adder import SolutionAdder

load_dotenv()
github_key = os.getenv("GITHUB_KEY")

T = TypeVar("T")
def batched(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield fixed‑size chunks from seq (last chunk may be smaller)."""
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])

def configure_llama_models() -> Dict[str, LLM|BaseEmbedding]:
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

    return {
        'o3-mini': Settings.llm,
        'text-embedding-ada-002': Settings.embed_model,
        'gpt-4.1-mini': AzureOpenAI(
            model="gpt-4.1-mini",
            deployment_name="gpt-4.1-mini",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-12-01-preview",
        ),
    }

SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

In your summary, make sure to:
1. Mention the main entities and their roles in the code. 
2. Provide a concise summary, under 100 words, that captures the essence of the code snippet and its relationship to the surrounding context.

Summary: """


def main():
    parser = argparse.ArgumentParser(
        description="Index code from a GitHub repo into a LlamaIndex VectorStore with configurable file filtering."
    )
    parser.add_argument(
        "url",
        help="GitHub repository URL (https://github.com/{owner}/{repo}/tree/{branch})"
    )

    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name to fetch (default: main)"
    )

    parser.add_argument(
        "--file-regex",
        default=None,
        help="Optional regex pattern to filter files. If not provided, all supported files will be processed."
    )
    args = parser.parse_args()

    models = configure_llama_models()

    schema = IndexSchema.from_dict({
        "index": {"name": "redis_vector_store", "prefix": "doc"},  # Index name and key prefix in Redis
        "fields": [
            {"type": "tag", "name": "id"},       # Unique node ID tag
            {"type": "tag", "name": "doc_id"},   # Document ID tag (e.g. file path or name)
            {"type": "text", "name": "text"},    # Text content (for fallback text search)
            {
                "type": "vector", "name": "vector",
                "attrs": {
                    "dims": 1536,               # Dimension of Ada-002 embeddings
                    "algorithm": "hnsw",        # Use HNSW indexing for vectors
                    "distance_metric": "cosine"
                }
            }
        ]
    })

    vector_store = RedisVectorStore(schema=schema, redis_url="redis://localhost:6379")

    run_pipeline(
        url=args.url,
        branch=args.branch,
        language_model=models['gpt-4.1-mini'],
        embed_model=models['text-embedding-ada-002'],
        vector_store=vector_store,
        github_key=github_key
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        show_progress=True,
    )

    response = index.as_query_engine().query("What are the steps and parameters needed to deploy the 1password connector?")
    print("Query response:", response)

if __name__ == "__main__":
    main()