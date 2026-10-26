"""Reality Graph — the persistent model of the project."""
from __future__ import annotations

import uuid

from ..config import Config
from ..models import Decision, ProjectMeta
from ..storage import Storage


class RealityGraph:
    def __init__(self, cfg: Config, storage: Storage) -> None:
        self.cfg = cfg
        self.storage = storage

    def init_project(self, name: str, stack: str, goal: str = "") -> ProjectMeta:
        meta = ProjectMeta(name=name, stack=stack, goal=goal)
        self.storage.save_project(meta)
        return meta

    def get(self) -> ProjectMeta | None:
        return self.storage.load_project()

    def add_rule(self, rule: str) -> None:
        self.storage.add_rule(rule)

    def rules(self) -> list[str]:
        return self.storage.all_rules()

    def add_decision(self, title: str, rationale: str,
                     files: list[str] | None = None) -> Decision:
        d = Decision(
            id=uuid.uuid4().hex[:12],
            title=title, rationale=rationale, files=files or [],
        )
        self.storage.save_decision(d)
        return d

    def search_decisions(self, q: str) -> list[Decision]:
        # LIKE match on the whole query is too strict; match on any
        # significant word so 'auth login bug' finds the 'JWT for auth' ADR.
        words = [w for w in q.lower().split() if len(w) >= 4]
        found: dict[str, Decision] = {}
        for w in words or [q]:
            for d in self.storage.search_decisions(w):
                found[d.id] = d
        return sorted(found.values(), key=lambda d: d.created_at, reverse=True)[:10]

    def all_decisions(self) -> list[Decision]:
        return self.storage.all_decisions()

    def as_context_block(self) -> str:
        meta = self.get()
        if not meta:
            return "(no project metadata yet — run `brain init`)"
        rules = "\n".join(f"- {r}" for r in self.rules()) or "(none)"
        return (
            f"PROJECT: {meta.name}\n"
            f"STACK: {meta.stack}\n"
            f"GOAL: {meta.goal or '(not set)'}\n"
            f"CRITICAL RULES:\n{rules}"
        )
