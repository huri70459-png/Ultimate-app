"""Core data models — all persistent state flows through these."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def today() -> str:
    """UTC date string, e.g. '2026-10-26'."""
    return datetime.now(timezone.utc).date().isoformat()


class TaskKind(str, Enum):
    CODE_CHANGE = "code_change"
    QUESTION = "question"
    DEBUG = "debug"
    EXPLAIN = "explain"
    PLAN = "plan"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"


class ProjectMeta(BaseModel):
    name: str
    stack: str
    goal: str = ""
    critical_rules: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class Task(BaseModel):
    id: str
    request: str
    kind: TaskKind
    created_at: str = Field(default_factory=_now)


class Claim(BaseModel):
    text: str
    kind: str  # "file_exists" | "symbol_exists" | "schema_match" | "factual" | "path_in_output"
    target: str  # path / symbol / schema spec / free text
    verdict: Verdict = Verdict.UNVERIFIED
    evidence: str = ""


class ExecutionResult(BaseModel):
    id: str = ""
    task_id: str
    output: str
    citations: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    verdict: Verdict = Verdict.UNVERIFIED
    retries: int = 0
    quality_score: float = 0.0
    created_at: str = Field(default_factory=_now)


class FailureRecord(BaseModel):
    id: str
    task_request: str
    attempt_summary: str
    failure_reason: str
    embedding: list[float] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class TraceEntry(BaseModel):
    task_id: str
    stage: str
    payload: dict[str, Any]
    created_at: str = Field(default_factory=_now)


class Decision(BaseModel):
    id: str
    title: str
    rationale: str
    files: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class FileDigest(BaseModel):
    path: str
    summary: str
    symbols: list[str] = Field(default_factory=list)
    content_hash: str = ""
    embedding: list[float] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_now)
