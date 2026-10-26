"""VeriFlow pipeline — the runtime kernel.

Intent -> (Change Governor) -> Ground -> Execute -> Validate -> Trace,
with budget checks before every LLM spend and failure-memory capture on
final failure.
"""
from __future__ import annotations

import uuid

from ..config import Config
from ..llm import LLM
from ..models import (
    ExecutionResult as ExecModel,
)
from ..models import (
    Task,
    TaskKind,
    Verdict,
)
from ..modules.change_governor import ChangeGovernor
from ..modules.context_engine import ContextEngine
from ..modules.failure_memory import FailureMemory
from ..modules.ground_truth import GroundTruth
from ..modules.quality_engine import QualityEngine
from ..modules.reality_graph import RealityGraph
from ..modules.token_governor import BudgetExceeded, TokenGovernor
from .execution import Executor
from .grounding import Grounding
from .intent import IntentClassifier
from .trace import Tracer
from .validation import Validator


class PipelineResult:
    def __init__(self, execution: ExecModel, passed: bool,
                 failure_note: str = "") -> None:
        self.execution = execution
        self.passed = passed
        self.failure_note = failure_note


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        storage,
        llm: LLM,
        reality: RealityGraph,
        context: ContextEngine,
        failures: FailureMemory,
        governor: TokenGovernor,
        change_gov: ChangeGovernor,
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.llm = llm
        self.reality = reality
        self.context = context
        self.failures = failures
        self.governor = governor
        self.change_gov = change_gov

        self.intent = IntentClassifier(llm)
        self.grounding = Grounding(reality, context, failures)
        self.executor = Executor(cfg, llm)
        self.validator = Validator(llm, GroundTruth(cfg), QualityEngine(),
                                   critic_model=cfg.critic_model)
        self.tracer = Tracer(storage)

    def run(self, request: str) -> PipelineResult:
        task_id = uuid.uuid4().hex[:12]
        self.tracer.log(task_id, "start", {"request": request})

        # accumulated spend across every LLM call of this task
        tot_in = tot_out = 0
        tot_cost = 0.0
        tot_latency = 0

        # 0. budget check
        try:
            self.governor.check()
        except BudgetExceeded as e:
            return self._abort(task_id, request, str(e),
                               tot_in, tot_out, tot_cost)

        # 1. intent
        intent = self.intent.classify(request)
        tot_in += intent.tokens_in
        tot_out += intent.tokens_out
        tot_cost += intent.cost_usd
        task = Task(id=task_id, request=request, kind=intent.kind)
        self.storage.save_task(task)
        self.tracer.log(task_id, "intent", {
            "kind": intent.kind.value,
            "entities": intent.entities,
            "summary": intent.summary,
        })

        # 2. change governor (for code changes AND debug tasks, which usually
        #    end in edits) — scope creep dies here, before a single execution
        #    token is spent
        if intent.kind in (TaskKind.CODE_CHANGE, TaskKind.DEBUG):
            scope = self.change_gov.evaluate(request, self.reality.rules())
            self.tracer.log(task_id, "scope", {
                "approved": scope.approved,
                "reason": scope.reason,
                "files": scope.scope_files,
            })
            if not scope.approved:
                return self._abort(task_id, request,
                                   f"Scope rejected: {scope.reason}",
                                   tot_in, tot_out, tot_cost)

        # 3. grounding
        bundle = self.grounding.build(request)
        grounding_section = bundle.as_prompt_section()
        self.tracer.log(task_id, "grounding", {
            "has_failure_warning": bool(bundle.failure_warning),
            "failure_warning": bundle.failure_warning[:500],
            "decisions": len(bundle.decisions),
            "context_chars": len(bundle.file_context),
        })

        # 4. execute + validate with bounded retry
        retries = 0
        extra = ""
        last_exec: ExecModel | None = None
        while True:
            try:
                self.governor.check()
            except BudgetExceeded as e:
                if last_exec is not None:
                    self.storage.save_execution(last_exec)
                return self._abort(task_id, request,
                                   f"Budget exhausted mid-task: {e}",
                                   tot_in, tot_out, tot_cost)

            self.tracer.log(task_id, "execute", {"attempt": retries + 1})
            exec_result = self.executor.run(
                request, grounding_section, extra_instructions=extra,
            )
            validation = self.validator.validate(
                request, exec_result.raw_output, exec_result.claims,
            )

            tot_in += exec_result.response.tokens_in + validation.critic_tokens_in
            tot_out += exec_result.response.tokens_out + validation.critic_tokens_out
            tot_cost += exec_result.response.cost_usd + validation.critic_cost_usd
            tot_latency += exec_result.response.latency_ms

            last_exec = ExecModel(
                task_id=task_id,
                output=exec_result.raw_output,
                citations=exec_result.citations,
                claims=validation.claims,
                tokens_in=tot_in,
                tokens_out=tot_out,
                cost_usd=tot_cost,
                latency_ms=tot_latency,
                verdict=validation.verdict,
                retries=retries,
                quality_score=validation.quality_score,
            )
            self.tracer.log(task_id, "validate", {
                "attempt": retries + 1,
                "verdict": validation.verdict.value,
                "quality": round(validation.quality_score, 3),
                "quality_issues": validation.quality_issues,
                "critic": validation.critic_reason[:300],
                "claims": [
                    {"kind": c.kind, "target": c.target,
                     "verdict": c.verdict.value, "evidence": c.evidence[:200]}
                    for c in validation.claims
                ],
            })

            if validation.verdict == Verdict.PASS:
                break
            if retries >= self.cfg.max_retries:
                break
            retries += 1
            extra = (
                "PREVIOUS ATTEMPT FAILED. Fix these issues:\n"
                f"- Critic: {validation.critic_reason}\n"
                f"- Quality issues: {validation.quality_issues}\n"
                f"- Failed claims: "
                f"{[c.text for c in validation.claims if c.verdict == Verdict.FAIL]}\n"
                "Do not repeat the failed approach."
            )

        # 5. persist + trace
        assert last_exec is not None
        self.storage.save_execution(last_exec)
        self.tracer.log(task_id, "done", {
            "verdict": last_exec.verdict.value,
            "retries": last_exec.retries,
            "quality": last_exec.quality_score,
            "cost": last_exec.cost_usd,
            "tokens": last_exec.tokens_in + last_exec.tokens_out,
        })

        if last_exec.verdict != Verdict.PASS:
            # Failure Memory: this exact pain must never be retried blind
            self.failures.record(
                task_request=request,
                attempt_summary=last_exec.output[:400],
                failure_reason=(
                    f"verdict={last_exec.verdict.value} "
                    f"quality={last_exec.quality_score:.2f} "
                    f"retries={last_exec.retries}"
                ),
            )
            return PipelineResult(
                last_exec, passed=False,
                failure_note=f"verdict={last_exec.verdict.value}",
            )

        return PipelineResult(last_exec, passed=True)

    def _abort(self, task_id: str, request: str, reason: str,
               tokens_in: int = 0, tokens_out: int = 0,
               cost: float = 0.0) -> PipelineResult:
        e = ExecModel(
            task_id=task_id, output=f"[ABORTED] {reason}",
            verdict=Verdict.FAIL,
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
        )
        try:
            self.storage.save_task(Task(id=task_id, request=request,
                                        kind=TaskKind.QUESTION))
        except Exception:
            pass  # task may already exist
        self.storage.save_execution(e)
        self.tracer.log(task_id, "aborted", {"reason": reason})
        return PipelineResult(e, passed=False, failure_note=reason)
