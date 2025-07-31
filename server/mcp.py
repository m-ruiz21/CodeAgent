# MCP server for querying
from fastmcp import FastMCP

server = FastMCP("Github Repo Expert")

@server.tool
def query(query: str) -> int:
    """Ask natural language query about the repository"""
    pass

@server.tool 
def retrieve(query: str, filters) -> List[Document]:
    """Retrieve documents by directly querying the vector store"""
    pass

@server.tool
def get_document(path: str) -> str:
    """Get complete / raw document content by its path / document ID"""
    pass