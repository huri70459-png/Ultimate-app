"""Quality Engine — heuristic scoring of LLM output. Instant, no LLM call."""
from __future__ import annotations

import re

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "have",
    "been", "will", "would", "should", "could", "please", "about",
}


class QualityReport:
    def __init__(self, score: float, issues: list[str]) -> None:
        self.score = score
        self.issues = issues


class QualityEngine:
    def score(self, output: str, request: str) -> QualityReport:
        issues: list[str] = []
        penalties = 0.0

        if len(output.strip()) < 30:
            issues.append("output suspiciously short")
            penalties += 0.3

        todo_count = len(re.findall(r"\bTODO\b|\bFIXME\b|\bXXX\b", output))
        if todo_count > 0:
            issues.append(f"contains {todo_count} TODO/FIXME markers")
            penalties += 0.1 * min(todo_count, 3)

        if re.search(r"I('|\s)(don't|do not)\s+know", output, re.I):
            issues.append("contains 'I don't know' hedge")
            penalties += 0.2

        lines = output.splitlines()
        long_lines = sum(1 for ln in lines if len(ln) > 200)
        if long_lines > 3:
            issues.append(f"{long_lines} lines exceed 200 chars")
            penalties += 0.05 * min(long_lines, 6)

        keywords = [w for w in re.findall(r"[A-Za-z0-9_]{5,}", request.lower())
                    if w not in STOPWORDS][:5]
        if keywords and not any(w in output.lower() for w in keywords):
            issues.append("output may not address the request")
            penalties += 0.15

        if output.count("```") % 2 != 0:
            issues.append("unbalanced code fences")
            penalties += 0.1

        placeholders = len(re.findall(
            r"\bplaceholder\b|\bexample\.com\b|\blorem ipsum\b|\.\.\.\s*rest",
            output, re.I))
        if placeholders > 0:
            issues.append(f"contains {placeholders} placeholder-ish patterns")
            penalties += 0.1 * min(placeholders, 2)

        score = max(0.0, min(1.0, 1.0 - penalties))
        return QualityReport(score=score, issues=issues)
