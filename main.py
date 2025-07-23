import os
import argparse

from llama_index.core import VectorStoreIndex, Document

from registry import CodeSplitterRegistry

def split_text(path: str, registry: CodeSplitterRegistry):
    """
    Split the code in 'path' into chunks using the appropriate CodeSplitter.
    """
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        splitter = registry.get_splitter(path)
        return splitter.split_text(code)
    except ValueError as e:
        print(f"Error getting splitter for {path}: {e}")
        return []


def load_and_split(folder_path: str):
    """
    Traverse 'folder_path', split each .py file into AST-aware chunks,
    and return a list of Document objects.
    """
    documents = []
    splitter_registry = CodeSplitterRegistry()

    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(dirpath, fname)
            chunks = split_text(full_path, splitter_registry)
            for idx, chunk in enumerate(chunks):
                metadata = {
                    "file_path": full_path,
                    "chunk_index": idx
                }
                documents.append(Document(text=chunk, metadata=metadata))
    
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

    # Load and split all code files
    docs = load_and_split(args.folder)
    print(f"Loaded {len(docs)} code chunks from '{args.folder}'")

    # Build the vector store index
    index = VectorStoreIndex.from_documents(docs)
    print("Built VectorStoreIndex over all code chunks")

    # Persist to disk if requested
    if args.persist_dir:
        index.storage_context.persist(persist_dir=args.persist_dir)
        print(f"Persisted vector store to '{args.persist_dir}'")

 if __name__ == "__main__":
    main()
