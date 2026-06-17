"""Local vector store backends."""

from local_context_engine.vector_store.base_store import BaseVectorStore, VectorSearchResult
from local_context_engine.vector_store.faiss_store import FAISSVectorStore
from local_context_engine.vector_store.factory import VectorStoreFactory

__all__ = ["BaseVectorStore", "VectorSearchResult", "FAISSVectorStore", "VectorStoreFactory"]
