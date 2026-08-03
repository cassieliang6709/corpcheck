"""
Text embedder
==============
Generates 384-dimensional sentence embeddings using the
all-MiniLM-L6-v2 model from sentence-transformers.

Embeddings are L2-normalised so cosine similarity equals dot product,
which is required for pgvector's ``<=>`` operator.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from corpcheck.ingestion.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Wrapper around sentence-transformers ``SentenceTransformer`` with
    batched encoding and normalisation.

    Parameters
    ----------
    model_name:
        HuggingFace / sentence-transformers model identifier.
    batch_size:
        Number of texts to encode per forward pass.
    normalize:
        If True (default), L2-normalise every embedding vector.
    device:
        Torch device string ('cpu', 'cuda', 'mps', etc.).
        ``None`` lets sentence-transformers auto-detect.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded (dim=%d)", EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # Core encode method
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a sequence of strings into embedding vectors.

        Parameters
        ----------
        texts:
            Strings to embed.  Empty strings produce zero-vectors.
        show_progress_bar:
            Display tqdm progress bar during encoding (useful for large batches).

        Returns
        -------
        np.ndarray
            Float32 array of shape (len(texts), EMBEDDING_DIM).
            Rows are L2-normalised if ``self.normalize`` is True.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # Replace empty strings with a single space so the model doesn't crash
        safe_texts = [t if t.strip() else " " for t in texts]

        embeddings: np.ndarray = self._model.encode(
            safe_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        # Ensure float32 (some backends return float16 or float64)
        embeddings = embeddings.astype(np.float32)

        logger.debug(
            "Encoded %d texts -> shape %s (normalised=%s)",
            len(texts),
            embeddings.shape,
            self.normalize,
        )
        return embeddings

    # ------------------------------------------------------------------
    # Convenience: encode with explicit batching + progress
    # ------------------------------------------------------------------

    def encode_large(
        self,
        texts: Sequence[str],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode a potentially large list of texts in configurable batches
        with progress logging.  Identical to ``encode`` but always shows a
        progress bar and logs batch progress to the logger.

        Parameters
        ----------
        texts:
            Strings to embed.

        Returns
        -------
        np.ndarray
            Float32 array of shape (len(texts), EMBEDDING_DIM).
        """
        total = len(texts)
        if total == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        result_parts: list[np.ndarray] = []
        for batch_start in range(0, total, self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_embeddings = self.encode(batch, show_progress_bar=False)
            result_parts.append(batch_embeddings)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Embedded batch %d-%d / %d",
                    batch_start,
                    min(batch_start + self.batch_size, total),
                    total,
                )

        return np.vstack(result_parts)

    # ------------------------------------------------------------------
    # Single-text helper
    # ------------------------------------------------------------------

    def encode_one(self, text: str) -> np.ndarray:
        """
        Embed a single string and return a 1D float32 array of length
        EMBEDDING_DIM.
        """
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1D embedding vectors.
        If embeddings are already L2-normalised this is equivalent to
        the dot product.
        """
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def top_k_similar(
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Return the indices and cosine-similarity scores of the *k* most
        similar embeddings in *corpus_embeddings* to *query_embedding*.

        Assumes both arrays are already L2-normalised (dot product = cosine).

        Parameters
        ----------
        query_embedding:
            Shape (D,) query vector.
        corpus_embeddings:
            Shape (N, D) matrix.
        k:
            Number of top results to return.

        Returns
        -------
        list[tuple[int, float]]
            Sorted (descending score) list of (index, score) pairs.
        """
        q = query_embedding.astype(np.float32)
        C = corpus_embeddings.astype(np.float32)
        scores = C @ q  # shape (N,) – dot product for normalised vectors
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# Lazy global embedder (initialised on first use)
_global_embedder: Embedder | None = None


def get_embedder(model_name: str = EMBEDDING_MODEL) -> Embedder:
    """Return a module-level cached :class:`Embedder` instance."""
    global _global_embedder
    if _global_embedder is None or _global_embedder.model_name != model_name:
        _global_embedder = Embedder(model_name=model_name)
    return _global_embedder


def embed_texts(
    texts: Sequence[str],
    normalize: bool = True,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of strings using the global cached embedder.

    Parameters
    ----------
    texts:
        Strings to embed.
    normalize:
        Whether to L2-normalise the output vectors.
    batch_size:
        Batch size for encoding.

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), 384).
    """
    embedder = get_embedder()
    embedder.batch_size = batch_size
    embedder.normalize = normalize
    return embedder.encode_large(texts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        "Apple reported record revenue of $394 billion in fiscal year 2022.",
        "The Federal Reserve raised interest rates by 75 basis points in June.",
        "Risk factors include macroeconomic uncertainty and supply chain disruptions.",
    ]
    emb = embed_texts(sample)
    print(f"Embeddings shape: {emb.shape}")
    for i, text in enumerate(sample):
        print(f"  [{i}] {text[:60]}... | norm={np.linalg.norm(emb[i]):.4f}")

    # Similarity demo
    sim_01 = Embedder.cosine_similarity(emb[0], emb[1])
    sim_02 = Embedder.cosine_similarity(emb[0], emb[2])
    print(f"Similarity(0,1): {sim_01:.4f}")
    print(f"Similarity(0,2): {sim_02:.4f}")
