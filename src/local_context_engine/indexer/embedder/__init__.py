"""Local embedding engine — no cloud APIs required."""

from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder
from local_context_engine.indexer.embedder.factory import EmbedderFactory
from local_context_engine.indexer.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)

__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder", "EmbedderFactory"]
