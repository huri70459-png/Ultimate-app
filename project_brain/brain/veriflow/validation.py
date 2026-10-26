"""Validation — claim verification + heuristic quality + self-critic."""
from __future__ import annotations

import json
import re

from ..llm import LLM
from ..models import Claim, Verdict
from ..modules.ground_truth import GroundTruth
from ..modules.quality_engine import QualityEngine, QualityReport


class ValidationResult:
    def __init__(
        self,
        verdict: Verdict,
        critic_reason: str,
        claims: list[Claim],
        quality_score: float,
        quality_issues: list[str],
        critic_cost_usd: float = 0.0,
        critic_tokens_in: int = 0,
        critic_tokens_out: int = 0,
    ) -> None:
        self.verdict = verdict
        self.critic_reason = critic_reason
        self.claims = claims
        self.quality_score = quality_score
        self.quality_issues = quality_issues
        self.critic_cost_usd = critic_cost_usd
        self.critic_tokens_in = critic_tokens_in
        self.critic_tokens_out = critic_tokens_out


class Validator:
    def __init__(
        self, llm: LLM, ground_truth: GroundTruth, quality: QualityEngine,
        critic_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.ground_truth = ground_truth
        self.quality = quality
        self.critic_model = critic_model

    def validate(
        self, request: str, output: str, claims: list[Claim],
    ) -> ValidationResult:
        # 1. verify factual claims against the codebase (ground truth)
        verified_claims = self.ground_truth.verify(claims)
        failed = [c for c in verified_claims if c.verdict == Verdict.FAIL]

        # 2. heuristic quality (instant, free)
        qr: QualityReport = self.quality.score(output, request)

        # 3. self-critic LLM call — only spent when cheap checks already
        #    look bad, keeping costs down
        critic_reason = ""
        critic_verdict = Verdict.PASS
        critic_cost = critic_tin = critic_tout = 0
        if failed or qr.score < 0.6:
            (critic_verdict, critic_reason,
             critic_cost, critic_tin, critic_tout) = self._critic(
                request, output, verified_claims, qr)

        # overall verdict
        if failed:
            verdict = Verdict.FAIL
        elif qr.score < 0.5:
            verdict = Verdict.FAIL
        elif critic_verdict == Verdict.FAIL:
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.PASS

        return ValidationResult(
            verdict=verdict,
            critic_reason=critic_reason,
            claims=verified_claims,
            quality_score=qr.score,
            quality_issues=qr.issues,
            critic_cost_usd=critic_cost,
            critic_tokens_in=critic_tin,
            critic_tokens_out=critic_tout,
        )

    def _critic(
        self, request: str, output: str,
        claims: list[Claim], qr: QualityReport,
    ) -> tuple[Verdict, str, float, int, int]:
        claim_block = "\n".join(
            f"- [{c.verdict.value}] {c.text} (target={c.target}) "
            f"evidence={c.evidence}"
            for c in claims
        ) or "(no claims)"
        sys = (
            "You are a strict code reviewer. Decide if the output safely and "
            "correctly addresses the request. A claim marked [fail] was "
            "contradicted by ground truth and must fail the review. "
            "Respond with JSON: "
            '{"verdict": "pass"|"fail", "reason": str}'
        )
        user = (
            f"REQUEST: {request}\n\n"
            f"OUTPUT:\n{output[:3000]}\n\n"
            f"CLAIMS:\n{claim_block}\n\n"
            f"QUALITY: score={qr.score:.2f} issues={qr.issues}"
        )
        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            model=self.critic_model,
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        try:
            text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "",
                          resp.content.strip()).strip()
            data = json.loads(text)
            v = Verdict.FAIL if data.get("verdict") == "fail" else Verdict.PASS
            return (v, str(data.get("reason", "")),
                    resp.cost_usd, resp.tokens_in, resp.tokens_out)
        except json.JSONDecodeError:
            return (Verdict.UNVERIFIED, "critic parse failure",
                    resp.cost_usd, resp.tokens_in, resp.tokens_out)
