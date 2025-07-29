import os
import argparse
from typing import Dict
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
from llama_index.core.extractors import (
    SummaryExtractor
)

from redisvl.schema import IndexSchema

from services.github.utils.path_filter import DirectoryFilter, FileFilter, FilterType
from services.pipeline.context_enrichment.context_enricher import ContextEnricher
from services.pipeline.code_splitter.code_splitter import CodeSplitter
from services.github.github_loader import GithubReader
from services.pipeline.code_splitter.registry import CodeSplitterRegistry
from services.cache.doc_service import DocService, set_doc_service

load_dotenv()
github_key = os.getenv("GITHUB_KEY")

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
    embed_model = models['text-embedding-ada-002']
    pipeline_model = models['gpt-4.1-mini']

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

    cache = IngestionCache(
            cache=RedisKVStore.from_host_and_port("localhost", 6379),
            collection="redis_cache"
        )
    
    vector_store.delete_index()
    vector_store.create_index()
    cache.clear()

    docs = GithubReader(token=str(github_key), url=args.url, branch=args.branch, parse=False,
                        file_filters=[
                            FileFilter(regex=r"^Solutions/[^/]+/Data Connectors/.*", filter_type=FilterType.INCLUDE),
                        ],
                        directory_filters=[
                            DirectoryFilter(regex=r"^Solutions", filter_type=FilterType.INCLUDE),
                            DirectoryFilter(regex=r"^Solutions/[^/]+/(?!Data Connectors).*", filter_type=FilterType.EXCLUDE),
                        ]).load_data()

    print("loaded", len(docs), "docs")

    set_doc_service(DocService(docs)) # 'cache' the documents for later use by our ContextEnricher

    splitter_registry = CodeSplitterRegistry(chunk_lines=100, max_chars=3000)

    pipeline = IngestionPipeline(
        transformations=[CodeSplitter(splitter_registry=splitter_registry), ContextEnricher(llm=pipeline_model), SummaryExtractor(llm=pipeline_model, prompt_template=SUMMARY_EXTRACT_TEMPLATE), embed_model],
        docstore=RedisDocumentStore.from_host_and_port("localhost", 6379, namespace="document_store"),
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE
    )

    print(f"Loaded {len(docs)} documents from GitHub repository.")
    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"Ingested {len(nodes)} nodes.")

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        show_progress=True,
    )

    response = index.as_chat_engine().chat("What are the main structs defined in the code and what are their responsibilities? Count them for me, list them with the struct name, its fields, and a brief description of its purpose.")
    print("Query response:", response)

if __name__ == "__main__":
    main()