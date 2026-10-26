"""LLM abstraction — OpenAI backend + deterministic offline backend.

The offline backend keeps the *entire* VeriFlow pipeline functional with no
API key, no network, and zero cost. It plays every LLM role the pipeline
needs (intent classifier, scope governor, engineer, critic) with transparent
rule-based logic, and produces hashed bag-of-words embeddings that give real
lexical cosine similarity. It exists for demos, tests, and air-gapped use —
production runs on the OpenAI backend.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import numpy as np

from .config import Config, get_openai_key


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


# Pricing per 1M tokens (approx, update as needed)
_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
}


class LLM:
    """Unified facade. `backend` is 'openai' or 'offline'."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backend = "offline" if cfg.offline else "openai"
        self._client = None  # lazily created on first OpenAI call

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        if self.backend == "offline":
            return self._offline_complete(messages)
        return self._openai_complete(messages, model, temperature,
                                     max_tokens, response_format)

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if self.backend == "offline":
            return [hash_embed(t, self.cfg.embedding_dim).tolist() for t in texts]
        model = model or self.cfg.embedding_model
        resp = self._openai().embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    # ------------------------------------------------------------------ #
    # OpenAI backend
    # ------------------------------------------------------------------ #
    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=get_openai_key())
        return self._client

    def _openai_complete(
        self,
        messages: list[dict],
        model: str | None,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        model = model or self.cfg.openai_model
        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.perf_counter()
        resp = self._openai().chat.completions.create(**kwargs)
        latency = int((time.perf_counter() - t0) * 1000)

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        tin = usage.prompt_tokens if usage else 0
        tout = usage.completion_tokens if usage else 0
        return LLMResponse(
            content=content,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=self._cost(model, tin, tout),
            latency_ms=latency,
        )

    @staticmethod
    def _cost(model: str, tin: int, tout: int) -> float:
        pin, pout = _PRICING.get(model, (0.0, 0.0))
        return (tin * pin + tout * pout) / 1_000_000

    # ------------------------------------------------------------------ #
    # Offline backend — deterministic rule-based engine
    # ------------------------------------------------------------------ #
    def _offline_complete(self, messages: list[dict]) -> LLMResponse:
        t0 = time.perf_counter()
        system = next((m["content"] for m in messages
                       if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")

        if "Classify the user request" in system:
            content = json.dumps(_offline_intent(user))
        elif "scope governor" in system:
            content = json.dumps(_offline_scope(user))
        elif "strict code reviewer" in system:
            content = json.dumps(_offline_critic(user))
        else:
            content = _offline_engineer(system, user)

        tin = max(1, (len(system) + len(user)) // 4)
        tout = max(1, len(content) // 4)
        return LLMResponse(
            content=content,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=0.0,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


# ---------------------------------------------------------------------- #
# Offline helpers
# ---------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "with", "without", "all", "any", "must", "should", "never", "not",
    "do", "does", "don't", "cannot", "can", "will", "would", "please",
    "you", "your", "we", "our", "they", "them", "from", "into", "onto",
    "at", "by", "as", "if", "then", "than", "when", "while", "each",
    "every", "some", "only", "just", "very", "more", "most", "other",
    "such", "no", "yes", "there", "here", "what", "which", "who", "whom",
    "how", "why", "where", "about", "above", "below", "under", "over",
    "again", "further", "once", "too", "s", "t", "m", "re", "ve", "ll",
    "d", "explicit", "explicitly", "require", "requires", "required",
    "approval", "approved", "changes", "change", "modify", "modified",
    "modification", "use", "used", "using", "make", "makes", "made",
    "need", "needs", "needed", "want", "wants", "wanted", "get", "gets",
    "keep", "keeps", "new", "old", "one", "two", "first", "second",
    "verify", "verified", "verification", "screenshot", "screenshots",
}

_PATH_RE = re.compile(r"[\w./-]+/\w[\w./-]*|\w+\.(?:py|js|ts|tsx|jsx|json|sql|md|toml|yaml|yml|html|css|sh|txt)\b")
_WORD_RE = re.compile(r"[a-z0-9_]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower())
            if len(w) >= 4 and w not in _STOPWORDS}


def _request_from_prompt(user: str) -> str:
    marker = "=== USER REQUEST ==="
    if marker in user:
        return user.split(marker, 1)[1].strip()
    m = re.search(r"USER REQUEST:\s*(.+?)(?:\n\n|$)", user, re.S)
    return m.group(1).strip() if m else user.strip()


def _offline_intent(request: str) -> dict:
    text = request.lower()
    if any(w in text for w in ("explain", "walk me through", "describe")):
        kind = "explain"
    elif any(w in text for w in ("plan", "design the", "architect", "roadmap",
                                 "break down", "breakdown")):
        kind = "plan"
    elif any(w in text for w in ("fix", "bug", "error", "broken", "crash",
                                 "failing", "fails", "debug", "traceback",
                                 "exception")):
        kind = "debug"
    elif any(w in text for w in ("add", "implement", "create", "build",
                                 "change", "update", "remove", "delete",
                                 "refactor", "rename", "migrate", "write")):
        kind = "code_change"
    else:
        kind = "question"
    paths = list(dict.fromkeys(_PATH_RE.findall(request)))
    lead_verbs = {"explain", "add", "fix", "change", "update", "remove",
                  "create", "implement", "rewrite", "refactor", "debug",
                  "plan", "design", "write", "build", "migrate", "rename",
                  "delete", "what", "why", "how", "where", "when", "which"}
    entities = paths + [w for w in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", request)
                        if w not in paths and w.lower() not in lead_verbs][:8]
    summary = request.strip().splitlines()[0][:120] if request.strip() else ""
    return {"kind": kind, "entities": entities, "summary": summary}


def _offline_scope(user: str) -> dict:
    request = _request_from_prompt(user)
    text = request.lower()

    rules_block = ""
    m = re.search(r"CRITICAL RULES:\n(.*?)\n\n", user, re.S)
    if m:
        rules_block = m.group(1)
    rules = [ln.lstrip("- ").strip() for ln in rules_block.splitlines()
             if ln.strip() and ln.strip() != "(none)"]

    # 1. vague requests are rejected
    vague = (
        "fix everything" in text or "rewrite everything" in text
        or "make it better" in text or "improve everything" in text
        or "do whatever" in text or text.strip() in ("", "fix", "update", "change")
        or len(_WORD_RE.findall(text)) < 3
    )
    if vague:
        return {"approved": False,
                "reason": "Request is too vague to scope safely. "
                          "Specify concrete files/endpoints and the desired change.",
                "scope_files": []}

    # 2. critical-rule protection: a rule's protected nouns appearing in a
    #    change request means the governor blocks before execution.
    req_words = _content_words(request)
    for rule in rules:
        protected = _content_words(rule)
        overlap = protected & req_words
        if overlap:
            return {"approved": False,
                    "reason": f"Violates critical rule: '{rule}' "
                              f"(request touches: {', '.join(sorted(overlap))})",
                    "scope_files": []}

    scope_files = list(dict.fromkeys(_PATH_RE.findall(request)))
    return {"approved": True,
            "reason": "Request is specific, bounded, and violates no critical rules.",
            "scope_files": scope_files}


def _offline_critic(user: str) -> dict:
    failed = re.findall(r"^- \[fail\] (.+)$", user, re.M)
    qm = re.search(r"score=([0-9.]+)", user)
    quality = float(qm.group(1)) if qm else 1.0
    if failed:
        return {"verdict": "fail",
                "reason": f"{len(failed)} claim(s) contradicted by ground truth: "
                          + "; ".join(t.strip()[:80] for t in failed[:3])}
    if quality < 0.5:
        return {"verdict": "fail",
                "reason": f"Quality score {quality:.2f} below acceptance floor 0.5."}
    return {"verdict": "pass",
            "reason": "Output addresses the request; all verifiable claims "
                      "hold; quality acceptable."}


def _offline_engineer(system: str, user: str) -> str:
    """Produce a grounded engineer-style answer from the context actually
    present in the prompt. Only cites files that grounding really provided,
    so downstream GroundTruth verification passes honestly."""
    request = _request_from_prompt(user)
    intent = _offline_intent(request)["kind"]

    files = re.findall(r"^--- (\S+) \(score=", user, re.M)
    # prefer files the user explicitly named in the request (path, filename,
    # or stem: "users endpoint" should surface app/users.py)
    req_lower = request.lower()

    def _mentioned(f: str) -> bool:
        base = f.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        return (f in request or base in request
                or (len(stem) >= 4 and stem.lower() in req_lower))

    named = [f for f in files if _mentioned(f)]
    cited = (named + [f for f in files if f not in named])[:3]
    decisions = re.findall(r"^- ([^:\n]+): (.+)$", user, re.M)
    failure_warning = "SIMILAR PAST FAILURES" in user
    project = ""
    pm = re.search(r"=== PROJECT REALITY ===\n(.*?)\n\n", user, re.S)
    if pm:
        project = pm.group(1).strip()
    proj_name = ""
    nm = re.search(r"PROJECT:\s*(\S+)", project)
    if nm:
        proj_name = nm.group(1)

    lines: list[str] = []
    topic = request.strip().splitlines()[0] if request.strip() else "the request"
    lines.append(f"Task: {topic}")
    if proj_name:
        lines.append(f"Project: {proj_name}.")
    if failure_warning:
        lines.append("Note: past similar attempts failed; this approach avoids "
                     "the recorded failure patterns and stays minimal.")

    if intent == "code_change":
        lines.append("")
        lines.append("Plan (minimal, in-scope change):")
        if cited:
            for i, f in enumerate(cited, 1):
                lines.append(f"{i}. Update [[source:{f}]] — adjust the affected "
                             f"functions only, keep public signatures stable.")
        else:
            lines.append("1. Locate the target module via the code map before editing.")
        lines.append(f"{len(cited) + 1}. Add/extend tests covering the new behavior.")
        if cited:
            lines.append("")
            lines.append("Sketch of the change:")
            lines.append("```python")
            lines.append(f"# in {cited[0]}")
            lines.append("def handle_request(payload):")
            lines.append("    validate(payload)")
            lines.append("    return build_response(payload)")
            lines.append("```")
    elif intent in ("explain", "question"):
        lines.append("")
        lines.append("Answer grounded in the indexed codebase:")
        if cited:
            for f in cited:
                lines.append(f"- [[source:{f}]] participates in this flow; "
                             f"see its symbols in the context above.")
        else:
            lines.append("- The index returned no directly relevant files; "
                         "run `brain index` and retry for a grounded answer.")
        if decisions:
            lines.append("")
            lines.append("Relevant recorded decisions:")
            for title, why in decisions[:3]:
                lines.append(f"- {title.strip()}: {why.strip()}")
    elif intent == "debug":
        lines.append("")
        lines.append("Debugging steps (evidence-first):")
        lines.append("1. Reproduce with the smallest failing input and capture logs.")
        if cited:
            lines.append(f"2. Inspect [[source:{cited[0]}]] at the failing call site.")
        lines.append("3. Form one hypothesis, verify it against evidence, then fix.")
    else:  # plan
        lines.append("")
        lines.append("Implementation plan:")
        lines.append("1. Confirm scope and affected modules against the code map.")
        if cited:
            for i, f in enumerate(cited, 2):
                lines.append(f"{i}. Modify [[source:{f}]] with tests updated in the same step.")
        lines.append(f"{len(cited) + 2}. Verify with the project's test command and record evidence.")

    # verifiable claims about files grounding actually surfaced
    for f in cited:
        lines.append(f"CLAIM:file_exists:{f}:Referenced file {f} exists in the project")

    lines.append("")
    lines.append("Confidence: medium-high (grounded in indexed files above); "
                 "unverified assumptions are flagged rather than guessed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Hashed bag-of-words embeddings (offline lexical similarity)
# ---------------------------------------------------------------------- #
def hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic hashing-trick embedding.

    Each token maps (via SHA-1) to one signed dimension; the vector is the
    token sum, L2-normalized. Texts sharing vocabulary get genuinely high
    cosine similarity, which is all FailureMemory and ContextEngine need in
    offline mode.
    """
    vec = np.zeros(dim, dtype=np.float64)
    tokens = _WORD_RE.findall(text.lower())
    for tok in tokens:
        h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign
    # token bigrams sharpen phrase-level similarity
    for a, b in zip(tokens, tokens[1:]):
        h = int(hashlib.sha1(f"{a}_{b}".encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += 0.5 * sign
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec.astype("float32")
