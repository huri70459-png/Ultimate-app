"""Shared fixtures: build a fully-wired offline Brain in a temp project."""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.config import Config
from brain.embeddings import Embedder
from brain.llm import LLM
from brain.modules.change_governor import ChangeGovernor
from brain.modules.context_engine import ContextEngine
from brain.modules.failure_memory import FailureMemory
from brain.modules.token_governor import TokenGovernor
from brain.storage import Storage, VectorIndex
from brain.veriflow.pipeline import Pipeline
from brain.modules.reality_graph import RealityGraph


def build_parts(root: Path, **cfg_overrides) -> dict:
    cfg = Config(project_root=root, llm_backend="offline", **cfg_overrides)
    storage = Storage(cfg)
    index = VectorIndex(cfg)
    llm = LLM(cfg)
    embedder = Embedder(cfg, llm)
    reality = RealityGraph(cfg, storage)
    context = ContextEngine(cfg, storage, index, embedder)
    failures = FailureMemory(cfg, storage, index, embedder)
    governor = TokenGovernor(cfg, storage)
    change_gov = ChangeGovernor(cfg, llm)
    pipeline = Pipeline(cfg, storage, llm, reality, context, failures,
                        governor, change_gov)
    return {
        "cfg": cfg, "storage": storage, "index": index, "llm": llm,
        "embedder": embedder, "reality": reality, "context": context,
        "failures": failures, "governor": governor, "change_gov": change_gov,
        "pipeline": pipeline,
    }


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """A tiny but realistic codebase for grounding to bite into."""
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "@app.get('/users')\n"
        "def list_users():\n"
        "    return get_users()\n",
        encoding="utf-8",
    )
    (root / "app" / "users.py").write_text(
        "def get_users():\n    return db_query('SELECT * FROM users')\n\n"
        "def export_users_csv():\n    raise NotImplementedError\n",
        encoding="utf-8",
    )
    (root / "app" / "auth.py").write_text(
        "def login(username, password):\n    return verify_token(username)\n",
        encoding="utf-8",
    )
    return root
