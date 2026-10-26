"""Change Governor — reject scope creep before execution."""
from __future__ import annotations

import json
import re

from ..config import Config
from ..llm import LLM


class ScopeDecision:
    def __init__(self, approved: bool, reason: str,
                 scope_files: list[str]) -> None:
        self.approved = approved
        self.reason = reason
        self.scope_files = scope_files


class ChangeGovernor:
    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg = cfg
        self.llm = llm

    def evaluate(self, request: str, rules: list[str]) -> ScopeDecision:
        rules_block = "\n".join(f"- {r}" for r in rules) if rules else "(none)"
        sys = (
            "You are a strict scope governor. Given a user request, decide:\n"
            "1. Is the request well-scoped? (reject vague 'fix everything' asks)\n"
            "2. What files are plausibly in scope?\n"
            "3. Does it violate any critical rules?\n"
            "Respond with JSON: "
            '{"approved": bool, "reason": str, "scope_files": [str]}'
        )
        user = (
            f"CRITICAL RULES:\n{rules_block}\n\n"
            f"USER REQUEST: {request}\n\n"
            "Respond with JSON only."
        )
        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            model=self.cfg.critic_model,
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(_strip_code_fences(resp.content))
            return ScopeDecision(
                approved=bool(data.get("approved", True)),
                reason=str(data.get("reason", "")),
                scope_files=[str(f) for f in data.get("scope_files", [])],
            )
        except json.JSONDecodeError:
            # fail closed for code changes: unparseable governance = not approved
            return ScopeDecision(
                False,
                "Governor response was not valid JSON — failing closed. "
                "Re-run or inspect the model output.",
                [],
            )


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()

