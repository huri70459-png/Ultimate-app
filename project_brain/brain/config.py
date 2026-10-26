"""Configuration and path management."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BRAIN_DIR_NAME = ".brain"


@dataclass
class Config:
    project_root: Path
    brain_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    index_path: Path = field(init=False)
    trace_path: Path = field(init=False)

    openai_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    max_tokens_per_task: int = 20_000
    max_retries: int = 2
    daily_budget_usd: float = 10.0

    # "auto" | "openai" | "offline"
    #   auto    -> openai when OPENAI_API_KEY is set, otherwise offline
    #   openai  -> always use the OpenAI API (key required)
    #   offline -> deterministic local backend (no network, no key, no cost).
    #              Full pipeline still runs: hashed embeddings provide real
    #              lexical similarity and a rule-based engine plays the LLM
    #              roles (intent / governor / engineer / critic).
    llm_backend: str = "auto"

    def __post_init__(self) -> None:
        self.brain_dir = self.project_root / BRAIN_DIR_NAME
        self.db_path = self.brain_dir / "brain.sqlite"
        self.index_path = self.brain_dir / "vectors.npz"
        self.trace_path = self.brain_dir / "traces.jsonl"
        self.brain_dir.mkdir(parents=True, exist_ok=True)

        # environment overrides
        self.llm_backend = os.environ.get("BRAIN_LLM", self.llm_backend).lower()
        self.openai_model = os.environ.get("BRAIN_MODEL", self.openai_model)
        self.critic_model = os.environ.get("BRAIN_CRITIC_MODEL", self.critic_model)
        self.embedding_model = os.environ.get("BRAIN_EMBED_MODEL", self.embedding_model)
        if os.environ.get("BRAIN_DAILY_BUDGET"):
            self.daily_budget_usd = float(os.environ["BRAIN_DAILY_BUDGET"])

    @property
    def offline(self) -> bool:
        """True when the deterministic local backend should be used."""
        if self.llm_backend == "offline":
            return True
        if self.llm_backend == "openai":
            return False
        return not bool(os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def discover(cls, start: Path | None = None) -> "Config":
        """Walk up from `start` to find an existing .brain dir, else use cwd."""
        start = start or Path.cwd()
        cur = start.resolve()
        while True:
            candidate = cur / BRAIN_DIR_NAME
            if candidate.is_dir():
                return cls(project_root=cur)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        return cls(project_root=start.resolve())


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either export it, or run in offline mode "
            "with BRAIN_LLM=offline (deterministic local backend, no API cost)."
        )
    return key
