"""Embedding service — thin wrapper over the LLM backend with batching."""
from __future__ import annotations

import numpy as np

from .config import Config
from .llm import LLM

EMBED_BATCH = 512


class Embedder:
    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg = cfg
        self.llm = llm

    def embed_one(self, text: str) -> np.ndarray:
        vecs = self.llm.embed([text])
        return np.array(vecs[0], dtype="float32")

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.embedding_dim), dtype="float32")
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out.extend(self.llm.embed(texts[i:i + EMBED_BATCH]))
        return np.array(out, dtype="float32")
