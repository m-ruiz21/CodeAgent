# MCP server for querying
from typing import List, Optional
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from llama_index.core import Document
import os
from dotenv import load_dotenv

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from server.llm import SamplingLLM

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import QueryBundle

load_dotenv()
server = FastMCP("Github Repo Expert")

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

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2023-05-15",
)

class SearchFilters(BaseModel):
    """Filters for searching the vector store."""
    solution: Optional[str] = Field(
        None, description="Filter results by solution name or identifier."
    )
    file_path: Optional[str] = Field(
        None, description="Filter results by file path."
    )

@server.tool
def query(query: str, ctx: Context) -> int:
    """Ask natural language query about the repository"""
    llm = SamplingLLM(mcp_context=ctx)
    vector_store = RedisVectorStore(schema=schema, redis_url="redis://localhost:6379")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        show_progress=True,
    )

    response = index.as_query_engine(llm=llm).query(query)
    print(response)
    return response.response

@server.tool
def search(query: str, filters: List[MetadataFilters], ctx: Context) -> List[Document]:
    """Search the vector store"""
    llm = SamplingLLM(mcp_context=ctx)
    query_bundle = QueryBundle(query)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        show_progress=True,
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        filters=filters,
    )

    retrieved_nodes = retriever.retrieve(query)
    reranker = LLMRerank(
        llm=llm,
        choice_batch_size=5,
        top_n=3,
    )
    
    rereanked_nodes = reranker.postprocess_nodes(
        retrieved_nodes,
        query_bundle
    )

    return rerereanked_nodes


@server.tool
def fetch(path: str) -> str:
    """Get complete / raw document content by its path / document ID"""
    pass