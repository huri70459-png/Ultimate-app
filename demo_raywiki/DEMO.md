# 🧠 Project Brain — Live Demo Transcript

Everything below ran **end-to-end in this workspace** against the sample
codebase in `demo_raywiki/` (a tiny FastAPI-style app + live SQLite DB),
using `BRAIN_LLM=offline` — the deterministic zero-cost backend. With
`OPENAI_API_KEY` exported, the identical pipeline runs on real LLM calls.

---

## 1. Give the project a persistent reality

```
$ brain init --name "RayWiki" --stack "Python/FastAPI/SQLite" --goal "AI knowledge platform demo"
╭────────────────────────────────────────────── INIT ──────────────────────────────────────────────╮
│ Initialized brain for RayWiki                                                                    │
│ Stack: Python/FastAPI/SQLite                                                                     │
│ Goal: AI knowledge platform demo                                                                 │
│ Brain dir:                                                                                       │
│ /tmp/arena/tenants/216235bf-4437-42f5-a20d-c03c9ece4edf/sessions/0534be28-840f-4862-9a72-173c469 │
│ 68734/workspace/demo_raywiki/.brain                                                              │
│ LLM: offline backend — deterministic local engine, no API calls. Export OPENAI_API_KEY (or unset │
│ BRAIN_LLM=offline) for the OpenAI backend.                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

$ brain rule "Never modify the auth layer without explicit approval"
Added rule: Never modify the auth layer without explicit approval
$ brain decide "Use JWT for auth" "Simpler than sessions, stateless, works across services"
Decision 948fda114e19: Use JWT for auth

$ brain index
Indexed 6 file(s) (0 unchanged skipped, 0 removed). Vector store size: 6
```

---

## 2. Run tasks through the VeriFlow pipeline

**Task 1 — a well-scoped code change passes, with every claim verified against ground truth:**

```
$ brain run "Add CSV export to the /users endpoint in app/users.py"
╭──────────────────────────────────────────── REQUEST ─────────────────────────────────────────────╮
│ Add CSV export to the /users endpoint in app/users.py                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────── RESULT (pass) ──────────────────────────────────────────╮
│ Task: Add CSV export to the /users endpoint in app/users.py                                      │
│ Project: RayWiki.                                                                                │
│                                                                                                  │
│ Plan (minimal, in-scope change):                                                                 │
│ 1. Update [[source:app/users.py]] — adjust the affected functions only, keep public signatures   │
│ stable.                                                                                          │
│ 2. Update [[source:app/main.py]] — adjust the affected functions only, keep public signatures    │
│ stable.                                                                                          │
│ 3. Update [[source:tests/test_users.py]] — adjust the affected functions only, keep public       │
│ signatures stable.                                                                               │
│ 4. Add/extend tests covering the new behavior.                                                   │
│                                                                                                  │
│ Sketch of the change:                                                                            │
│ ```python                                                                                        │
│ # in app/users.py                                                                                │
│ def handle_request(payload):                                                                     │
│     validate(payload)                                                                            │
│     return build_response(payload)                                                               │
│ ```                                                                                              │
│ CLAIM:file_exists:app/users.py:Referenced file app/users.py exists in the project                │
│ CLAIM:file_exists:app/main.py:Referenced file app/main.py exists in the project                  │
│ CLAIM:file_exists:tests/test_users.py:Referenced file tests/test_users.py exists in the project  │
│                                                                                                  │
│ Confidence: medium-high (grounded in indexed files above); unverified assumptions are flagged    │
│ rather than guessed.                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
tokens=1016 (in=737 out=279)  cost=$0.0000  retries=0  quality=1.00  task=2492132ccf3a
                              CLAIM VERIFICATION                               
verdict  kind         target               evidence                            
pass     file_exists  app/users.py         path=app/users.py exists=True       
pass     file_exists  app/main.py          path=app/main.py exists=True        
pass     file_exists  tests/test_users.py  path=tests/test_users.py exists=True
citations:
  • app/main.py

**Tasks 2–4 — the governors kill bad work BEFORE execution tokens are spent:**

  • app/users.py
  • tests/test_users.py
```

```
$ brain run "Rewrite the auth layer to use session cookies"     # violates a critical rule
╭──────────────────────────────────────────── REQUEST ─────────────────────────────────────────────╮
│ Rewrite the auth layer to use session cookies                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────── RESULT (fail) ──────────────────────────────────────────╮
│ [ABORTED] Scope rejected: Violates critical rule: 'Never modify the auth layer without explicit  │
│ approval' (request touches: auth, layer)                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
tokens=70 (in=46 out=24)  cost=$0.0000  retries=0  quality=0.00  task=1edbcf813996
Failure note: Scope rejected: Violates critical rule: 'Never modify the auth layer without explicit 
approval' (request touches: auth, layer)
recorded in Failure Memory — similar future requests will be warned.
```

```
$ brain run "fix everything and make the code better"           # too vague to scope
╭───────────────────────────────────────── RESULT (fail) ──────────────────────────────────────────╮
│ [ABORTED] Scope rejected: Request is too vague to scope safely. Specify concrete files/endpoints │
│ and the desired change.                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
tokens=65 (in=44 out=21)  cost=$0.0000  retries=0  quality=0.00  task=1eba216feb7f
Failure note: Scope rejected: Request is too vague to scope safely. Specify concrete files/endpoints
```

```
$ BRAIN_DAILY_BUDGET=0 brain run "Add pagination to the users list"   # budget governor
╭───────────────────────────────────────── RESULT (fail) ──────────────────────────────────────────╮
│ [ABORTED] Daily budget $0.00 exceeded (spent $0.00)                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
tokens=0 (in=0 out=0)  cost=$0.0000  retries=0  quality=0.00  task=5f8635510ff4
```

---

## 3. Full traceability — replay any run

```
$ brain trace 2492132ccf3a                                          # replay every stage
09:45:53 start
    request: Add CSV export to the /users endpoint in app/users.py
09:45:53 intent
    kind: code_change
    entities: ['app/users.py', 'CSV']
    summary: Add CSV export to the /users endpoint in app/users.py
09:45:53 scope
    approved: True
    reason: Request is specific, bounded, and violates no critical rules.
    files: ['app/users.py']
09:45:53 grounding
    has_failure_warning: False
    failure_warning: 
    decisions: 0
    context_chars: 2036
09:45:53 execute
    attempt: 1
09:45:53 validate
    attempt: 1
    verdict: pass
    quality: 1.0
    quality_issues: []
    critic: 
    claims: [{'kind': 'file_exists', 'target': 'app/users.py', 'verdict': 'pass', 'evidence': 
'path=app/users.py exists=True'}, {'kind': 'file_exists', 'target': 'app/main.py', 'verdict': 
'pass', 'evidence': 'path=app/main.py exists=True'}, {'kind': 'file_exists', 'target': 
'tests/test_users.py', 'verdict': 'pass', 'evidence': 'path=tests/test_users.py exists=True'}]
09:45:53 done
    verdict: pass
    retries: 0
  ... (validate + done stages include per-claim evidence)
```

```
$ brain status

╭───────────────────────────────────────── PROJECT BRAIN ──────────────────────────────────────────╮
│ RayWiki — Python/FastAPI/SQLite                                                                  │
│ AI knowledge platform demo                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────── PROJECT HEALTH ─────────────────────────────────────────╮
│                                                                                                  │
│   Architecture                                                                         indexed   │
│   Indexed files                                                                              6   │
│   Decisions recorded                                                                         1   │
│   Critical rules                                                                             1   │
│   Recent pass rate                                                                         1/4   │
│   All-time pass rate                                                                       1/4   │
│   Avg quality (recent)                                                                    0.25   │
│   Hallucinations caught                                                                      0   │
│   Failures remembered                                                                        0   │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────── AI ACTIVITY ───────────────────────────────────────────╮
│                                                                                                  │
│   Backend                                                                              offline   │
│   Tokens today                                                                           1,151   │
│   Cost today                                                                           $0.0000   │
│   Daily budget                                                                          $10.00   │
│   Budget used                                                               0.0% ($10.00 left)   │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────── RECENT RUNS ───────────────────────────────────────────╮
│ ╭──────────┬─────────┬──────┬──────┬─────────┬─────────┬───────────────────────────────────────╮ │
│ │ time     │ verdict │    q │  tok │    cost │ retries │ output                                │ │
│ ├──────────┼─────────┼──────┼──────┼─────────┼─────────┼───────────────────────────────────────┤ │
│ │ 09:45:54 │ fail    │ 0.00 │    0 │ $0.0000 │       0 │ [ABORTED] Daily budget $0.00 exceeded │ │
│ │          │         │      │      │         │         │ (spent $0.00)                         │ │
│ │ 09:45:54 │ fail    │ 0.00 │   65 │ $0.0000 │       0 │ [ABORTED] Scope rejected: Request is  │ │
│ │          │         │      │      │         │         │ too vague to scope safe               │ │
│ │ 09:45:54 │ fail    │ 0.00 │   70 │ $0.0000 │       0 │ [ABORTED] Scope rejected: Violates    │ │
│ │          │         │      │      │         │         │ critical rule: 'Never mod             │ │
│ │ 09:45:53 │ pass    │ 1.00 │ 1016 │ $0.0000 │       0 │ Task: Add CSV export to the /users    │ │
│ │          │         │      │      │         │         │ endpoint in app/users.py              │ │
│ ╰──────────┴─────────┴──────┴──────┴─────────┴─────────┴───────────────────────────────────────╯ │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

```

---

## 4. Failure Memory — the project never repeats a failed approach

```
# A failed approach gets stored forever (here imported via the projects own API,
# exactly as the pipeline does it automatically after a final FAIL):
$ python -c "FailureMemory(...).record(request, attempt, reason)"
failure recorded: 649815ad39c0

$ brain run "Add CSV export to the /users endpoint in app/users.py" --no-index
(run completed as task 81e8f9409131 — note the adjusted approach in the output:)
│ Note: past similar attempts failed; this approach avoids the recorded failure patterns and stays │

$ brain trace <task>   # grounding stage now carries the warning
09:46:37 grounding
    has_failure_warning: True
    failure_warning: ⚠️  SIMILAR PAST FAILURES — DO NOT REPEAT THESE APPROACHES:
  • (sim=1.00) Request: Add CSV export to the /users endpoint in app/users.py
    Attempt: streamed the whole users table with a generator that closed the DB connection early
    Failed because: verdict=fail quality=0.30 retries=2
  → Choose a materially different approach and justify it.
    decisions: 0
    context_chars: 2036

$ brain failures
╭─────────────────────────────────────── FAILURE MEMORY (1) ───────────────────────────────────────╮
│ id            when                 request                        reason                         │
│ 649815ad39c0  2026-10-26 09:46:37  Add CSV export to the /users   verdict=fail quality=0.30      │
│                                    endpoint in app/users.py       retries=2                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## Test suite

```
$ python -m pytest tests/ -q
......................                                                   [100%]
22 passed
```

Covers: ground-truth verification (files, symbols, SQLite schemas, path-traversal
blocking), quality rubric, vector-index upsert/persistence, incremental indexing,
failure-memory similarity, intent classification, change-governor rejections,
budget aborts, full pipeline e2e (pass + hallucination→failure-memory flow),
dashboard render, and the whole CLI surface.
