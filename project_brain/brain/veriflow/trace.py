"""Trace — append-only execution log (replay & debug)."""
from __future__ import annotations

from ..models import TraceEntry
from ..storage import Storage


class Tracer:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def log(self, task_id: str, stage: str, payload: dict) -> None:
        self.storage.append_trace(TraceEntry(
            task_id=task_id, stage=stage, payload=payload,
        ))

    def replay(self, task_id: str) -> list[TraceEntry]:
        return self.storage.read_traces(task_id)
