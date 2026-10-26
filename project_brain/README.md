# 🧠 Project Brain

**The AI Operating System for building software** — a persistent reality, memory,
governance, and verification layer that sits *above* AI coding agents (Cursor,
Claude Code, Codex, Aider, Gemini CLI, Cline, Roo…) and turns unreliable LLM
calls into a reliable, traceable engineering pipeline.

> **Killer feature: never let the AI forget what is true.**
> Hallucinations, execution drift, context loss, no grounding, repeated
> mistakes, scope creep, fake verification, low quality, cost explosions —
> all come from one root cause: *the AI has no persistent model of reality.*
> Project Brain gives every project one.

The runtime kernel is **VeriFlow**: every task flows through
`Intent → Ground → Execute → Validate → Trace` with verification gates,
budget enforcement, and full traceability.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PROJECT BRAIN                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    PERSISTENCE LAYER                      │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐   │  │
│  │  │ SQLite   │  │ Vector index │  │ Trace log (JSONL) │   │  │
│  │  │ (state)  │  │ (FAISS/np)   │  │ (replay & debug)  │   │  │
│  │  └──────────┘  └──────────────┘  └───────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ▲                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    BRAIN MODULES                          │  │
│  │  RealityGraph · ContextEngine · FailureMemory             │  │
│  │  GroundTruth  · TokenGovernor · ChangeGovernor            │  │
│  │  QualityEngine                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ▲                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              VERIFLOW RUNTIME KERNEL                      │  │
│  │                                                           │  │
│  │  Intent ──► Ground ──► Execute ──► Validate ──► Trace     │  │
│  │    ▲          ▲          ▲          ▲           │         │  │
│  │    └──────────┴──────────┴──────────┴────► Dashboard      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ▲                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
         Cursor            Claude             Codex
        (future: MCP)     (future: CLI)     (future: API)
```

### Data flow for a single task

```
User Request
     │
     ▼
┌─────────────┐
│  0. BUDGET  │──► TokenGovernor: daily spend ≥ budget → ABORT
└──────┬──────┘
       ▼
┌─────────────┐
│  1. INTENT  │──► classify: code_change | question | debug | explain | plan
└──────┬──────┘
       ▼
┌─────────────┐
│   CHANGE    │──► code_change/debug: vague ask or critical-rule
│  GOVERNOR   │    violation → REJECT before any execution spend
└──────┬──────┘
       ▼
┌─────────────┐
│  2. GROUND  │──► RealityGraph (meta+rules) + ContextEngine (top-k files)
│             │──► FailureMemory warning ("96% similar past failure")
│             │──► related architectural decisions
└──────┬──────┘
       ▼
┌─────────────┐
│  3. EXECUTE │──► LLM with grounding; must emit [[source:path]] citations
│             │    and CLAIM:<kind>:<target>:<text> lines
└──────┬──────┘
       ▼
┌─────────────┐
│  4. VALIDATE│──► GroundTruth verifies every claim (file/symbol/schema)
│             │──► QualityEngine scores output (heuristics, free)
│             │──► self-critic LLM review (only when cheap checks look bad)
│             │──► FAIL → retry ≤ max_retries with failure feedback
└──────┬──────┘
       ▼
┌─────────────┐
│  5. TRACE   │──► append-only JSONL log of every stage (replayable)
│             │──► final failure → stored in FailureMemory forever
└──────┬──────┘
       ▼
   Final Output (citations + claim evidence + quality score + cost)
```

---

## 🧩 How each original problem is solved

| Problem | Module that kills it |
|---|---|
| **Hallucination** | `GroundTruth` — every claim verified against files, symbols, SQLite schemas; contradicted claims FAIL the run |
| **Token/cost explosion** | `TokenGovernor` — daily budget checked before *every* LLM call; runaway loops abort mid-task |
| **Execution drift** | `Trace` — every stage logged to JSONL; `brain trace <id>` replays any run |
| **Context loss** | `ContextEngine` — incremental semantic index; loads only relevant files/decisions per task |
| **No grounding** | `Grounding` — mandatory context bundle: project reality + top-k files + decisions + failure warnings |
| **Repeating failures** | `FailureMemory` — vector-similarity lookup warns "DO NOT REPEAT" before execution |
| **Scope creep** | `ChangeGovernor` — rejects vague asks and critical-rule violations *before* execution |
| **Low accuracy** | `Validator` — claim verification + self-critic + bounded retry with failure feedback |
| **Low quality adherence** | `QualityEngine` — instant heuristic rubric (TODOs, fences, hedging, relevance…) |

---

## 📁 File tree

```
project_brain/
├── README.md
├── pyproject.toml
├── brain/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point (brain …)
│   ├── config.py              # paths, settings, backend selection
│   ├── models.py              # Pydantic models (Task, Claim, ExecutionResult…)
│   ├── storage.py             # SQLite + VectorIndex (FAISS w/ numpy fallback)
│   ├── llm.py                 # OpenAI backend + deterministic offline backend
│   ├── embeddings.py          # embedding service (batched)
│   ├── dashboard.py           # rich terminal dashboard
│   ├── veriflow/              # runtime kernel
│   │   ├── pipeline.py        # orchestrator
│   │   ├── intent.py
│   │   ├── grounding.py
│   │   ├── execution.py
│   │   ├── validation.py
│   │   └── trace.py
│   └── modules/               # persistent brain modules
│       ├── reality_graph.py
│       ├── context_engine.py
│       ├── failure_memory.py
│       ├── ground_truth.py
│       ├── token_governor.py
│       ├── change_governor.py
│       └── quality_engine.py
└── tests/
    ├── conftest.py            # offline wiring fixture
    └── test_pipeline.py       # 22 tests incl. full e2e + CLI surface
```

---

## 🚀 Setup

```bash
cd project_brain
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[faiss,dev]"    # faiss extra is optional (numpy fallback built in)
```

### Choose a backend

| Mode | Env | Behavior |
|---|---|---|
| **OpenAI** | `export OPENAI_API_KEY=sk-…` | Real LLM calls (`gpt-4o-mini` default). Cost tracked per call in the token ledger. |
| **Offline** | `export BRAIN_LLM=offline` (or just no API key) | Deterministic local engine — full pipeline runs with **zero network and zero cost**. Embeddings use a hashing-trick bag-of-words model that provides real lexical similarity; the LLM roles (intent/governor/engineer/critic) are played by transparent rule-based logic. Great for demos, CI, and tests. |

Optional env overrides: `BRAIN_MODEL`, `BRAIN_CRITIC_MODEL`, `BRAIN_EMBED_MODEL`,
`BRAIN_DAILY_BUDGET` (USD).

---

## 🏃 Usage

All commands operate on the current project (a `.brain/` dir is discovered by
walking up from cwd, like `.git`).

```bash
cd /path/to/your/codebase

# 1. Give the project a persistent reality
brain init --name "RayWiki" --stack "Python/FastAPI/SQLite" --goal "AI knowledge platform"

# 2. Encode the laws of the project
brain rule "Never modify the auth layer without explicit approval"
brain rule "All API routes must return JSON-serializable dicts"

# 3. Record architectural decisions (ADRs)
brain decide "Use JWT for auth" "Simpler than sessions, stateless, works across services"

# 4. Index the codebase (incremental — only changed files re-embed)
brain index            # add --full to force re-embed everything

# 5. Run tasks through the VeriFlow pipeline
brain run "Add CSV export to the /users endpoint in app/users.py"
brain run "Rewrite the auth layer to use session cookies"   # → Scope rejected (rule)
brain run "fix everything"                                  # → Scope rejected (vague)

# 6. Observe
brain status                       # dashboard: health, spend, recent runs
brain failures                     # the failure memory
brain rules / brain decisions      # the project's laws and ADRs
brain trace <task_id>              # replay every stage of a run, with evidence
```

### What a passing run shows you

- The answer, with `[[source:…]]` citations to real files
- A **CLAIM VERIFICATION** table: every factual claim, its verdict, and the
  evidence (`path=app/users.py exists=True`)
- Token/cost/retry/quality accounting for the whole task

### What a failing run does

- Retries up to `max_retries` (default 2) with the critic's failure feedback
  injected into the prompt
- On final failure, stores the approach in **FailureMemory** — the next
  similar request gets a `⚠️ SIMILAR PAST FAILURES — DO NOT REPEAT` block
  injected into grounding before a single token is spent

---

## 🔌 Wiring (dependency injection)

```python
# brain/cli.py::_build() — everything is composed here
cfg      = Config.discover()          # finds .brain/ like git finds .git/
storage  = Storage(cfg)               # SQLite (state) + JSONL (traces)
index    = VectorIndex(cfg)           # cosine search: FAISS if installed, else numpy
llm      = LLM(cfg)                   # openai | offline (lazy — no key needed until call)
embedder = Embedder(cfg, llm)

# Brain modules (persistent state layer)
reality    = RealityGraph(cfg, storage)
context    = ContextEngine(cfg, storage, index, embedder)
failures   = FailureMemory(cfg, storage, index, embedder)
governor   = TokenGovernor(cfg, storage)
change_gov = ChangeGovernor(cfg, llm)

# VeriFlow kernel (runtime layer)
pipeline = Pipeline(cfg, storage, llm, reality, context,
                    failures, governor, change_gov)
```

Storage layout (inside `.brain/`):

| Artifact | Contents |
|---|---|
| `brain.sqlite` | project meta, tasks, executions, failures, decisions, file digests, token ledger, critical rules |
| `vectors.npz` | vector index (file digests `file:…`, failure records `fail:…`) |
| `traces.jsonl` | append-only per-stage execution log (replayable) |

---

## ✅ Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
# 22 passed — includes full offline pipeline e2e, hallucination → failure-memory
# flow, budget abort, scope rejection, incremental indexing, CLI surface
```

Everything runs on the offline backend: **no API key, no network, no cost.**

---

## 🧭 Blueprint → implementation notes

This is a faithful implementation of the Project Brain blueprint, with these
deliberate engineering fixes/additions:

1. **Vector index upsert + persistence** — the blueprint appended vectors on
   every re-index (duplicates) and had a `NameError` bug (`cfg` vs `self.cfg`)
   plus a fragile `.ids.json`. Replaced with an upsert dict persisted to a
   single `.npz`; FAISS is used when installed, numpy brute-force otherwise
   (identical cosine results at project scale).
2. **Incremental indexing** — files are embedded only when their content hash
   changes; deleted files are pruned. Re-running tasks no longer re-spends
   embedding tokens (which is the Token Governor's whole philosophy).
3. **Budget checked before every LLM call**, not just at task start — runaway
   retry loops are stopped mid-task.
4. **Cost accounting across retries** — tokens/cost accumulate over all
   attempts + critic calls instead of reporting only the last one.
5. **Change Governor fails closed** — an unparseable governor response now
   rejects (the blueprint approved blindly), and it gates `debug` tasks too.
6. **Offline backend** (`BRAIN_LLM=offline`) — deterministic, zero-cost engine
   so the entire system is testable and demoable without an API key.
7. **Ground-truth hardening** — path-traversal protection, read-only SQLite
   schema checks, table-name validation, per-batch scan caching.
8. **Traces rehydrated on the dashboard** — hallucination counts are real.

## 🔭 Extension points (from the blueprint roadmap)

| Want to add… | Where to plug in |
|---|---|
| **Visual Truth Engine** (Playwright screenshots) | `brain/modules/visual_truth.py`, called from `Validator.validate()` |
| **Architecture Guardian** (dependency graph, god classes) | `brain/modules/arch_guardian.py`, called from `ChangeGovernor.evaluate()` |
| **Multi-Agent Orchestrator** (architect/dev/tester/reviewer) | replace `Executor` with a role router |
| **MCP server** (so Cursor/Claude Code call Brain) | `brain/mcp_server.py` exposing `run`, `status`, `decide` |
| **Anthropic/Google backends** | extend `LLM` behind the same `complete()`/`embed()` interface |

---

> The coding agents will keep changing every year. **The project brain remains
> the permanent source of truth.**
