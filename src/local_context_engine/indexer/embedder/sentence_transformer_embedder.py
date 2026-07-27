"""
Sentence Transformers embedding engine.

Supports all models compatible with the ``sentence-transformers`` library:
  - BAAI/bge-small-en-v1.5    (384-dim, fast)
  - BAAI/bge-base-en-v1.5     (768-dim, balanced)
  - nomic-ai/nomic-embed-text-v1.5 (768-dim, best quality, requires trust_remote_code)
  - all-MiniLM-L6-v2          (384-dim, general purpose)

BGE models require a query prefix for asymmetric retrieval:
  Query: "Represent this sentence for searching relevant passages: <text>"
  Document: (no prefix)

Nomic models use:
  Query:    "search_query: <text>"
  Document: "search_document: <text>"

Models are cached locally after first download (HuggingFace cache).
All inference runs on the local device — no cloud calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from local_context_engine.core.exceptions import ModelNotAvailableError
from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

# Model-specific prefixes for asymmetric embedding
_MODEL_PREFIXES: dict[str, dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
    "BAAI/bge-base-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
    "BAAI/bge-large-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "query": "search_query: ",
        "document": "search_document: ",
    },
}

# Models that need trust_remote_code=True
_TRUST_REMOTE_CODE_MODELS = {
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v1",
}


def resolve_device(device: str) -> str:
    """
    Resolve ``"auto"`` to a concrete torch device.

    Deliberately imports torch only here (not at config-load time) so that
    lightweight commands and MCP startup do not pay the torch/CUDA RSS cost.
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding model powered by ``sentence-transformers``.

    Lazy-loads the model on first use to avoid startup overhead.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        backend: str = "torch",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._backend = backend
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._local_files_only = local_files_only
        self._normalize = normalize_embeddings
        self._batch_size = batch_size

        # Resolve prefixes: explicit args > model-specific defaults > empty
        model_defaults = _MODEL_PREFIXES.get(model_name, {})
        self._query_prefix = query_prefix or model_defaults.get("query", "")
        self._document_prefix = document_prefix or model_defaults.get("document", "")

        self._model = None  # Lazy load
        self._dimension: int | None = None

        logger.info(
            "SentenceTransformerEmbedder configured: model=%s device=%s backend=%s",
            model_name,
            device,
            backend,
        )

    def _load_model(self) -> None:
        """Load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._device = resolve_device(self._device)
            kwargs: dict = {
                "device": self._device,
            }
            if self._backend != "torch":
                kwargs["backend"] = self._backend
            if self._backend == "onnx":
                # sentence-transformers otherwise selects the first provider
                # reported by ONNX Runtime. GPU installations commonly report
                # TensorRT first even when its native DLLs are unavailable.
                try:
                    import onnxruntime as ort

                    available = ort.get_available_providers()
                    preferred = (
                        "CUDAExecutionProvider"
                        if self._device == "cuda"
                        else "CPUExecutionProvider"
                    )
                    provider = preferred if preferred in available else "CPUExecutionProvider"
                    kwargs["model_kwargs"] = {"provider": provider}
                    logger.info("Selected ONNX execution provider: %s", provider)
                except ImportError:
                    # SentenceTransformer will emit the actionable missing
                    # dependency error when it attempts to load the backend.
                    pass
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            if self._local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1":
                kwargs["local_files_only"] = True
            if self._model_name in _TRUST_REMOTE_CODE_MODELS:
                kwargs["trust_remote_code"] = True

            logger.info(
                "Loading embedding model: %s (device=%s, backend=%s)",
                self._model_name, self._device, self._backend,
            )
            self._model = SentenceTransformer(self._model_name, **kwargs)
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self._dimension = get_dim()
            logger.info(
                "Model loaded: %s | dimension=%d", self._model_name, self._dimension
            )
        except Exception as exc:
            raise ModelNotAvailableError(self._model_name, str(exc)) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of document strings."""
        self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed = [f"{self._document_prefix}{t}" for t in texts]
        embeddings = self._model.encode(  # type: ignore[union-attr]
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self._batch_size,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string with the query prefix."""
        self._load_model()
        prefixed = f"{self._query_prefix}{text}"
        embedding = self._model.encode(  # type: ignore[union-attr]
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def embed_documents_batched(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Embed documents in batches using the model's native batching."""
        self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        effective_batch = batch_size or self._batch_size
        prefixed = [f"{self._document_prefix}{t}" for t in texts]

        embeddings = self._model.encode(  # type: ignore[union-attr]
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=effective_batch,
        )
        return embeddings.astype(np.float32)

    @classmethod
    def from_config(cls, config: "EmbeddingConfig") -> "SentenceTransformerEmbedder":  # type: ignore[name-defined]
        from local_context_engine.core.config import EmbeddingConfig  # noqa: PLC0415

        assert isinstance(config, EmbeddingConfig)
        return cls(
            model_name=config.model,
            device=config.device,
            backend=config.backend,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            normalize_embeddings=config.normalize_embeddings,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            batch_size=config.batch_size,
        )
