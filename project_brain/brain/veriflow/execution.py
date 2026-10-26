"""Execution — call the LLM with budget and grounding."""
from __future__ import annotations

import re

from ..config import Config
from ..llm import LLM, LLMResponse
from ..models import Claim

CITATION_RE = re.compile(r"\[\[source:([^\]]+)\]\]")


class ExecutorOutput:
    def __init__(
        self,
        response: LLMResponse,
        claims: list[Claim],
        citations: list[str],
        raw_output: str,
    ) -> None:
        self.response = response
        self.claims = claims
        self.citations = citations
        self.raw_output = raw_output


class Executor:
    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg = cfg
        self.llm = llm

    def run(
        self,
        request: str,
        grounding_section: str,
        extra_instructions: str = "",
    ) -> ExecutorOutput:
        sys = (
            "You are a careful software engineer working inside a real codebase.\n"
            "RULES:\n"
            "1. Cite every file you reference as [[source:path/to/file]].\n"
            "2. For every factual claim, emit a line 'CLAIM:<kind>:<target>:<text>'.\n"
            "   Kinds: file_exists, symbol_exists, schema_match, factual.\n"
            "   Example: CLAIM:file_exists:app/main.py:The FastAPI entrypoint exists\n"
            "3. If you are unsure, say so — do not guess.\n"
            "4. Respect all CRITICAL RULES.\n"
            "5. Be concise and concrete.\n"
        )
        if extra_instructions:
            sys += "\n" + extra_instructions
        user = f"{grounding_section}\n\n=== USER REQUEST ===\n{request}"

        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=min(4096, self.cfg.max_tokens_per_task),
        )

        citations = sorted({m.group(1) for m in CITATION_RE.finditer(resp.content)})
        claims = self._extract_claims(resp.content)

        return ExecutorOutput(
            response=resp,
            claims=claims,
            citations=citations,
            raw_output=resp.content,
        )

    def _extract_claims(self, text: str) -> list[Claim]:
        claims: list[Claim] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("CLAIM:"):
                continue
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            _, kind, target, claim_text = parts
            claims.append(Claim(
                text=claim_text.strip(),
                kind=kind.strip(),
                target=target.strip(),
            ))
        return claims
