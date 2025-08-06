# MCP server for querying
from typing import List, Literal 
from fastmcp import Context, FastMCP

from github import Github
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.readers.github.repository.github_client import GithubClient
server = FastMCP("Azure Sentinel Connector Query Server")

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
github_key = os.getenv("GITHUB_KEY")

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2023-05-15",
)

Settings.llm = AzureOpenAI(
    model="o3-mini",
    deployment_name="o3-mini",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview",
)

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

class RetrievalResult(BaseModel):
    id: str
    title: str
    text: str
    url: str 
    metadata: dict = {}

@server.tool
async def search(
        query: str = Field(description="natural language search query, helps if you include solution name and relevant entities")
) -> List[RetrievalResult]:
    """Retrieve relevant Sentinel Connector code snippets based on the query."""
    query_bundle = QueryBundle(query)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=6,
    )

    retrieved_nodes = retriever.retrieve(query)
    reranker = LLMRerank(
        choice_batch_size=3,
        top_n=3
    )

    rereanked_nodes = await reranker.apostprocess_nodes(
        retrieved_nodes,
        query_bundle
    )

    def to_res(node: NodeWithScore) -> RetrievalResult:
        file_path = node.node.metadata.get("file_path", "")
        solution_name = node.node.metadata.get("solution_name", "")
        summary = node.node.metadata.get("section_summary", "No summary available")

        node_content = """\
        # Metadata
        File Path: {file_path}
        Solution Name: {solution_name}
        Summary: \n {summary}
        
        # Node Content
        {node_content}
        """.format(
            file_path=file_path,
            solution_name=solution_name,
            summary=summary,
            node_content=node.node.get_content()
        )

        url = "https://github.com/Azure/Azure-Sentinel/blob/master/{file_path}".format(
            file_path=file_path
        )
        title = f"{file_path}"

        return RetrievalResult(
            id=node.node_id,
            title=title,
            text=node_content,
            url=url
        )

    return [to_res(node) for node in rereanked_nodes]

@server.tool
def fetch(path: str = Field(description="The path of the document to retrieve")) -> RetrievalResult:
    """Retrieve the full contents of a sentinel connector search result document or item by id / path."""
    # get document using llama index github client
    gh = Github(github_key)
    try:
        repository = gh.get_repo("Azure/Azure-Sentinel")
        file_content = repository.get_contents(path)
        text = file_content.decoded_content.decode()

        return RetrievalResult(
            id=path,
            title=path,
            text=text,
            url=f"https://github.com/Azure/Azure-Sentinel/blob/master/{path}",
            metadata={"solution": path.split('/')[1] if '/' in path else "Unknown"}
        )
    
    except Exception as e:
        return RetrievalResult(
            id=path,
            title=path,
            text=f"Error fetching document: {e}",
            url=f"https://github.com/Azure/Azure-Sentinel/blob/master/{path}",
            metadata={"solution": path.split('/')[1] if '/' in path else "Unknown"}
        )


if __name__ == "__main__":
    server.run(transport="http")