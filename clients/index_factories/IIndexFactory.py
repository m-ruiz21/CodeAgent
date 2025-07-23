from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document, VectorStoreIndex

class IIndexFactory(ABC):
    """
    Contract every backend factory must satisfy.
    Call‐site:
        index = IndexRegistry.get("azure-search").create_index(docs)
    """

    @abstractmethod
    def create_index(self, docs: List[Document]) -> VectorStoreIndex: 
        pass