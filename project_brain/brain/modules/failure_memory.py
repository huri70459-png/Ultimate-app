"""Failure Memory — never retry a failed approach."""
from __future__ import annotations

import uuid

import numpy as np

from ..config import Config
from ..embeddings import Embedder
from ..models import FailureRecord
from ..storage import Storage, VectorIndex


class FailureMemory:
    def __init__(
        self, cfg: Config, storage: Storage,
        index: VectorIndex, embedder: Embedder,
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.index = index
        self.embedder = embedder

    def record(
        self, task_request: str, attempt_summary: str, failure_reason: str,
    ) -> FailureRecord:
        # The searchable vector encodes the *request* (that's what future
        # tasks are matched against); the full text stays in SQLite for
        # display. Embedding request+attempt+reason together would dilute
        # similarity below useful thresholds.
        vec = self.embedder.embed_one(task_request)
        rec = FailureRecord(
            id=uuid.uuid4().hex[:12],
            task_request=task_request,
            attempt_summary=attempt_summary,
            failure_reason=failure_reason,
            embedding=vec.tolist(),
        )
        self.storage.save_failure(rec)
        self.index.add([f"fail:{rec.id}"], np.asarray(vec, dtype="float32").reshape(1, -1))
        return rec

    def lookup(self, query: str, k: int = 3,
               threshold: float = 0.80) -> list[tuple[FailureRecord, float]]:
        qvec = self.embedder.embed_one(query)
        hits = self.index.search(qvec, k=k * 4)
        records = {f"fail:{r.id}": r for r in self.storage.all_failures()}
        out: list[tuple[FailureRecord, float]] = []
        for id_, score in hits:
            r = records.get(id_)
            if r and score >= threshold:
                out.append((r, score))
                if len(out) >= k:
                    break
        return out

    def warning_block(self, query: str) -> str:
        hits = self.lookup(query)
        if not hits:
            return ""
        lines = ["⚠️  SIMILAR PAST FAILURES — DO NOT REPEAT THESE APPROACHES:"]
        for rec, score in hits:
            lines.append(
                f"  • (sim={score:.2f}) Request: {rec.task_request[:120]}\n"
                f"    Attempt: {rec.attempt_summary[:200]}\n"
                f"    Failed because: {rec.failure_reason[:200]}"
            )
        lines.append("  → Choose a materially different approach and justify it.")
        return "\n".join(lines)

    def all(self) -> list[FailureRecord]:
        return self.storage.all_failures()
