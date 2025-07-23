from typing import List
import os
import argparse
import fnmatch
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from registry import CodeSplitterRegistry

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment variables
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

def parse_gitignore(gitignore_path: str) -> List[str]:
    """Parse .gitignore file and return a list of patterns."""
    patterns = []
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # skip empty lines and comments
                if line and not line.startswith('#'):
                    patterns.append(line)
    except FileNotFoundError:
        pass  
    return patterns


def should_ignore_file(file_path: str, gitignore_patterns: List[str], base_path: str) -> bool:
    """Check if a file should be ignored based on .gitignore patterns and .git files."""
    rel_path = os.path.relpath(file_path, base_path)

    # git files won't be in the .gitignore, so we handle them separately
    if rel_path.startswith('.git/') or rel_path == '.git' or '/.git/' in rel_path:
        return True
    
    if not gitignore_patterns:
        return False
    
    for pattern in gitignore_patterns:
        # handle directory patterns
        if pattern.endswith('/'):
            if rel_path.startswith(pattern) or ('/' + pattern) in rel_path:
                return True
        # handle file patterns
        elif fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True
        # handle patterns with path separators
        elif '/' in pattern and fnmatch.fnmatch(rel_path, pattern):
            return True
    
    return False


def split_file(path: str, registry: CodeSplitterRegistry) -> List[str]:
    """Split a code file into chunks using the appropriate CodeSplitter."""
    # Check if the file extension is supported before attempting to read
    if not registry.is_supported(path):
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        print('Splitting file:', path)
        code = f.read()

    try:
        splitter = registry.get_splitter(path)
        return splitter.split_text(code)
    except ValueError as e:
        print(f"Error getting splitter for {path}: {e}")
        return []


def load_documents(chunks: list[str], file_path: str) -> list[Document]:
    """Convert code chunks into embedded Document objects."""
    documents = []
    for idx, chunk in enumerate(chunks):
        metadata = {
            "file_path": file_path,
            "chunk_index": idx
        }
        documents.append(Document(text=chunk, metadata=metadata))
    
    print(f"Embedded {len(documents)} chunks from {file_path}")

    return documents


def load_and_split(folder_path: str) -> list[Document]:
    """
    Traverse 'folder_path', split each file into AST-aware chunks,
    and return a list of Document objects. Respects .gitignore if present.
    """
    documents = []
    splitter_registry = CodeSplitterRegistry()
    
    # Check for .gitignore in the folder
    gitignore_path = os.path.join(folder_path, '.gitignore')
    gitignore_patterns = parse_gitignore(gitignore_path)
    
    if gitignore_patterns:
        print(f"Found .gitignore with {len(gitignore_patterns)} patterns")

    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            
            # Skip files that match .gitignore patterns
            if should_ignore_file(full_path, gitignore_patterns, folder_path):
                continue
                
            chunks = split_file(full_path, splitter_registry)
            
            documents.extend(load_documents(chunks, full_path))

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Index all Python code files in a folder into a LlamaIndex VectorStore."
    )
    
    parser.add_argument("folder", help="Path to the folder containing code files")
    parser.add_argument(
        "--persist_dir",
        help="Directory to persist the vector store (optional)",
        default=None,
    )

    args = parser.parse_args()

    docs = load_and_split(args.folder)
    print(f"Loaded {len(docs)} code chunks from '{args.folder}'")

    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    print("Built VectorStoreIndex over all code chunks")

    if args.persist_dir:
        index.storage_context.persist(persist_dir=args.persist_dir)
        print(f"Persisted vector store to '{args.persist_dir}'")
    
    # run search query over the index
    query = "How does the logging for the ArmorBlox Connector work? What url is it referencing and how does it conduct polling? Make sure to reference and include the code in your response with the file and line number you found the code snippets in."
    response = index.as_query_engine().query(query)
    print(f"Query response: {response}")

if __name__ == "__main__":
    main()