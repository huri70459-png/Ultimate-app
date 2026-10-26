"""Grounding — assemble context from all Brain modules."""
from __future__ import annotations

from ..modules.context_engine import ContextEngine
from ..modules.failure_memory import FailureMemory
from ..modules.reality_graph import RealityGraph


class GroundingBundle:
    def __init__(
        self,
        project_block: str,
        file_context: str,
        failure_warning: str,
        decisions: list,
    ) -> None:
        self.project_block = project_block
        self.file_context = file_context
        self.failure_warning = failure_warning
        self.decisions = decisions

    def as_prompt_section(self) -> str:
        parts = [
            "=== PROJECT REALITY ===",
            self.project_block,
            "",
            "=== RELEVANT FILES ===",
            self.file_context,
        ]
        if self.decisions:
            parts.append("")
            parts.append("=== RELATED DECISIONS ===")
            for d in self.decisions:
                parts.append(f"- {d.title}: {d.rationale}")
        if self.failure_warning:
            parts.append("")
            parts.append(self.failure_warning)
        return "\n".join(parts)


class Grounding:
    def __init__(
        self,
        reality: RealityGraph,
        context: ContextEngine,
        failures: FailureMemory,
    ) -> None:
        self.reality = reality
        self.context = context
        self.failures = failures

    def build(self, request: str) -> GroundingBundle:
        return GroundingBundle(
            project_block=self.reality.as_context_block(),
            file_context=self.context.build_context(request, k=8),
            failure_warning=self.failures.warning_block(request),
            decisions=self.reality.search_decisions(request),
        )
