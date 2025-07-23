"""
Index-factory primitives.
─────────────────────────
Any new backend (Chroma, Qdrant, Pinecone…) just implements IIndexFactory
and registers itself once with IndexRegistry.register("pinecone", MyFactory()).
"""
from __future__ import annotations
from typing import Dict

from .IIndexFactory import IIndexFactory

class IndexRegistry:
    """Global mapping name → factory instance."""

    _index_factories: Dict[str, IIndexFactory] = {}

    @classmethod
    def register(cls, name: str, factory: IIndexFactory) -> None:
        cls._index_factories[name] = factory

    @classmethod
    def get(cls, name: str) -> IIndexFactory:
        try:
            return cls._index_factories[name]
        except KeyError as err:
            raise ValueError(f"No factory registered under '{name}'") from err
