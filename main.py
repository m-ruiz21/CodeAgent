import os
import argparse
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from redisvl.schema import IndexSchema

from github_loader import GithubReader

load_dotenv()

def configure_llama_models() -> str | None:
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
    
    github_key = configure_llama_models()

    embed_model = Settings.embed_model

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
                    "distance_metric": "cosine" # Cosine similarity for nearest neighbor search
                }
            }
        ]
    })

    
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        docstore=RedisDocumentStore.from_host_and_port("localhost", 6379, namespace="document_store"),
        vector_store=RedisVectorStore(schema=schema, redis_url="redis://localhost:6379"),
        cache=IngestionCache(
            cache=RedisKVStore.from_host_and_port("localhost", 6379),
            collection="redis_cache"
        ),
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE
    )
    
    loader = GithubReader(token=str(github_key), url=args.url, branch=args.branch)
    documents = loader.load_data()

    nodes = pipeline.run(documents=documents)
    print(f"Ingested {len(nodes)} nodes.")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        parse_code=True
    )

    # Example query
    query = (
        "Give a basic rundown of how the scrolling logic works"
    )
    response = index.as_query_engine().query(query)
    print(f"Query response: {response}")

if __name__ == "__main__":
    main()
