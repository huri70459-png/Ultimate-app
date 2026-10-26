"""Intent classifier — what kind of task is this?"""
from __future__ import annotations

import json
import re

from ..llm import LLM
from ..models import TaskKind


class IntentResult:
    def __init__(self, kind: TaskKind, entities: list[str],
                 summary: str, tokens_in: int = 0, tokens_out: int = 0,
                 cost_usd: float = 0.0) -> None:
        self.kind = kind
        self.entities = entities
        self.summary = summary
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd


class IntentClassifier:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def classify(self, request: str) -> IntentResult:
        sys = (
            "Classify the user request. Respond with JSON: "
            '{"kind": "code_change"|"question"|"debug"|"explain"|"plan", '
            '"entities": [str], "summary": str}'
        )
        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": request},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        try:
            text = resp.content.strip()
            text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
            data = json.loads(text)
            kind = TaskKind(data.get("kind", "question"))
            return IntentResult(
                kind=kind,
                entities=[str(e) for e in data.get("entities", [])],
                summary=str(data.get("summary", request[:120])),
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cost_usd=resp.cost_usd,
            )
        except (json.JSONDecodeError, ValueError):
            return IntentResult(
                TaskKind.QUESTION, [], request[:120],
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cost_usd=resp.cost_usd,
            )
