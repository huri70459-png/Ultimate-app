"""Smoke + integration tests for Project Brain.

Everything runs on the deterministic offline backend — no API key, no
network, no cost — but exercises the *real* pipeline end to end.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from brain.config import Config
from brain.llm import hash_embed
from brain.models import Claim, TaskKind, Verdict
from brain.modules.ground_truth import GroundTruth
from brain.modules.quality_engine import QualityEngine
from conftest import build_parts


# ---------------------------------------------------------------------- #
# Blueprint smoke tests
# ---------------------------------------------------------------------- #
def test_ground_truth_file_exists():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hello.py").write_text("def greet(): pass\n")
        cfg = Config(project_root=root, llm_backend="offline")
        gt = GroundTruth(cfg)
        c = Claim(text="hello.py exists", kind="file_exists", target="hello.py")
        assert gt.verify([c])[0].verdict == Verdict.PASS

        c2 = Claim(text="missing.py exists", kind="file_exists", target="missing.py")
        assert gt.verify([c2])[0].verdict == Verdict.FAIL


def test_ground_truth_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        root.mkdir()
        cfg = Config(project_root=root, llm_backend="offline")
        gt = GroundTruth(cfg)
        c = Claim(text="escape", kind="file_exists", target="../.ssh/id_rsa")
        out = gt.verify([c])[0]
        assert out.verdict == Verdict.FAIL


def test_ground_truth_symbol_and_schema(sample_project):
    cfg = Config(project_root=sample_project, llm_backend="offline")
    gt = GroundTruth(cfg)
    claims = [
        Claim(text="get_users exists", kind="symbol_exists", target="get_users"),
        Claim(text="nope missing", kind="symbol_exists", target="totally_missing_symbol"),
    ]
    out = gt.verify(claims)
    assert out[0].verdict == Verdict.PASS
    assert "users.py" in out[0].evidence
    assert out[1].verdict == Verdict.FAIL


def test_ground_truth_schema_match():
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = root / "app.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
        conn.commit()
        conn.close()
        cfg = Config(project_root=root, llm_backend="offline")
        gt = GroundTruth(cfg)
        ok = Claim(text="email col", kind="schema_match",
                   target="sqlite:app.sqlite table:users column:email")
        bad = Claim(text="phone col", kind="schema_match",
                    target="sqlite:app.sqlite table:users column:phone")
        out = gt.verify([ok, bad])
        assert out[0].verdict == Verdict.PASS
        assert out[1].verdict == Verdict.FAIL


def test_quality_engine_short_output():
    r = QualityEngine().score("hi", "please write a long explanation")
    assert r.score < 1.0
    assert any("short" in i for i in r.issues)


def test_quality_engine_clean_output():
    text = ("Here is a proper explanation of the login flow that addresses "
            "the request in detail.")
    r = QualityEngine().score(text, "explain the login flow")
    assert r.score >= 0.7


def test_quality_engine_penalties():
    text = "TODO: implement later\n```python\nx = 1\n"  # unbalanced fence
    r = QualityEngine().score(text, "write finished production code now")
    assert any("TODO" in i for i in r.issues)
    assert any("fence" in i for i in r.issues)
    assert r.score < 1.0


# ---------------------------------------------------------------------- #
# Storage + vector index
# ---------------------------------------------------------------------- #
def test_vector_index_upsert_and_search(sample_project):
    parts = build_parts(sample_project)
    idx = parts["index"]
    v1 = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    v2 = np.array([[0.0, 1.0, 0.0]], dtype="float32")
    idx.add(["a", "b"], np.vstack([v1, v2]))
    idx.add(["a"], np.array([[0.0, 0.0, 1.0]], dtype="float32"))  # upsert
    assert idx.size == 2
    hits = idx.search(np.array([0.0, 0.0, 1.0], dtype="float32"), k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    idx.remove(["a"])
    assert idx.size == 1

    # persistence round trip
    idx2 = type(idx)(parts["cfg"])
    assert idx2.size == 1
    assert idx2.search(np.array([0.0, 1.0, 0.0], dtype="float32"), k=1)[0][0] == "b"


def test_hash_embed_lexical_similarity():
    a = hash_embed("fix the login auth password reset flow", 512)
    b = hash_embed("auth login password flow broken", 512)
    c = hash_embed("csv export of quarterly revenue report", 512)
    sim_ab = float(a @ b)
    sim_ac = float(a @ c)
    assert sim_ab > 0.5
    assert sim_ac < sim_ab


def test_storage_roundtrip(sample_project):
    parts = build_parts(sample_project)
    st = parts["storage"]
    parts["reality"].init_project("T", "S", "G")
    parts["reality"].add_rule("never touch payments")
    d = parts["reality"].add_decision("Use JWT", "stateless")
    assert st.load_project().name == "T"
    assert st.all_rules() == ["never touch payments"]
    assert st.search_decisions("jwt")[0].id == d.id
    st.append_trace(__import__("brain.models", fromlist=["TraceEntry"]).TraceEntry(
        task_id="t1", stage="test", payload={"ok": True}))
    assert len(st.read_traces("t1")) == 1


# ---------------------------------------------------------------------- #
# Context engine + failure memory (offline embeddings)
# ---------------------------------------------------------------------- #
def test_context_engine_incremental_index(sample_project):
    parts = build_parts(sample_project)
    ctx = parts["context"]
    s1 = ctx.index_project()
    assert s1["indexed"] == 3
    s2 = ctx.index_project()
    assert s2["indexed"] == 0 and s2["skipped"] == 3   # no duplicate spend
    assert parts["index"].size == 3

    # edit one file -> only it re-embeds; delete one -> it is removed
    (sample_project / "app" / "users.py").write_text("# changed\ndef get_users(): pass\n")
    (sample_project / "app" / "auth.py").unlink()
    s3 = ctx.index_project()
    assert s3["indexed"] == 1 and s3["removed"] == 1
    assert parts["index"].size == 2

    hits = ctx.retrieve("list users endpoint export", k=2)
    assert hits and hits[0]["path"].startswith("app/")


def test_failure_memory_warns_on_similar_request(sample_project):
    parts = build_parts(sample_project)
    fm = parts["failures"]
    fm.record(
        task_request="add csv export to the users endpoint",
        attempt_summary="pandas to_csv inside the route handler",
        failure_reason="verdict=fail quality=0.30",
    )
    warn = fm.warning_block("add csv export to the users endpoint")
    assert "SIMILAR PAST FAILURES" in warn
    assert "pandas" in warn

    # unrelated request -> no warning
    assert fm.warning_block("explain how the database migrations work") == ""


# ---------------------------------------------------------------------- #
# VeriFlow kernel pieces
# ---------------------------------------------------------------------- #
def test_intent_classifier_offline(sample_project):
    parts = build_parts(sample_project)
    ic = parts["pipeline"].intent
    assert ic.classify("Add CSV export to the /users endpoint").kind == TaskKind.CODE_CHANGE
    assert ic.classify("Fix the crash in login").kind == TaskKind.DEBUG
    assert ic.classify("Explain how auth works").kind == TaskKind.EXPLAIN
    assert ic.classify("What does get_users return?").kind == TaskKind.QUESTION


def test_change_governor_blocks_rules_and_vagueness(sample_project):
    parts = build_parts(sample_project)
    gov = parts["change_gov"]
    parts["reality"].add_rule("Never modify the auth layer")

    ok = gov.evaluate("Add CSV export to app/users.py", parts["reality"].rules())
    assert ok.approved
    assert "app/users.py" in ok.scope_files

    bad_rule = gov.evaluate("Rewrite the auth layer to use sessions",
                            parts["reality"].rules())
    assert not bad_rule.approved
    assert "critical rule" in bad_rule.reason.lower()

    vague = gov.evaluate("fix everything", parts["reality"].rules())
    assert not vague.approved


def test_pipeline_rejects_vague_debug_request(sample_project):
    """'fix everything' classifies as debug — the governor must still gate it."""
    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI")
    parts["context"].index_project()
    result = parts["pipeline"].run("fix everything and make the code better")
    assert not result.passed
    assert "Scope rejected" in result.failure_note


# ---------------------------------------------------------------------- #
# Full pipeline, end to end
# ---------------------------------------------------------------------- #
def test_dashboard_renders(sample_project):
    from io import StringIO
    from rich.console import Console
    from brain.dashboard import Dashboard

    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI", "demo")
    parts["context"].index_project()
    parts["pipeline"].run("Add CSV export to the users endpoint")

    dash = Dashboard(parts["cfg"], parts["storage"], parts["index"])
    buf = StringIO()
    dash.console = Console(file=buf, width=100)
    dash.render()
    out = buf.getvalue()
    assert "PROJECT BRAIN" in out
    assert "PROJECT HEALTH" in out
    assert "AI ACTIVITY" in out
    assert "RECENT RUNS" in out
    assert "Demo" in out


def test_pipeline_e2e_pass(sample_project):
    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI", "demo app")
    parts["reality"].add_rule("Never modify the auth layer")
    parts["context"].index_project()

    result = parts["pipeline"].run("Add CSV export to the users endpoint")
    assert result.passed, result.failure_note
    e = result.execution
    assert e.verdict == Verdict.PASS
    assert e.citations, "output must cite grounded files"
    assert all((sample_project / c).exists() for c in e.citations)
    assert e.claims and all(c.verdict == Verdict.PASS for c in e.claims)
    assert e.quality_score >= 0.6

    # trace is replayable
    stages = [t.stage for t in parts["storage"].read_traces(e.task_id)]
    assert stages[0] == "start" and stages[-1] == "done"
    assert "intent" in stages and "grounding" in stages and "validate" in stages

    # ledger + dashboard data
    snap = parts["governor"].snapshot()
    assert snap["backend"] == "offline"
    assert parts["storage"].pass_rate() == (1, 1)


def test_pipeline_scope_rejected_records_no_failure(sample_project):
    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI")
    parts["reality"].add_rule("Never modify the auth layer")
    parts["context"].index_project()

    result = parts["pipeline"].run("Change the auth layer to use magic links")
    assert not result.passed
    assert "Scope rejected" in result.failure_note
    assert "[ABORTED]" in result.execution.output
    # governor aborts are not "failed approaches" — nothing to remember
    assert parts["failures"].all() == []


def test_pipeline_failure_feeds_failure_memory(sample_project, monkeypatch):
    """An executor that hallucinates a file must fail validation, and the
    failure must be remembered so the next similar task gets warned."""
    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI")
    parts["context"].index_project()

    def lying_run(request, grounding_section, extra_instructions=""):
        from brain.llm import LLMResponse
        from brain.veriflow.execution import ExecutorOutput
        content = (
            "I added the export_users feature by editing "
            "[[source:app/exporters.py]].\n"
            "CLAIM:file_exists:app/exporters.py:The exporter module exists\n"
            "CLAIM:symbol_exists:export_users:The export function exists\n"
        )
        resp = LLMResponse(content=content, tokens_in=10, tokens_out=10,
                           cost_usd=0.0, latency_ms=1)
        claims = parts["pipeline"].executor._extract_claims(content)
        return ExecutorOutput(resp, claims, ["app/exporters.py"], content)

    monkeypatch.setattr(parts["pipeline"].executor, "run", lying_run)
    result = parts["pipeline"].run("Add export_users to the reporting flow")
    assert not result.passed
    assert any(c.verdict == Verdict.FAIL for c in result.execution.claims)
    assert result.execution.retries == parts["cfg"].max_retries

    # the failure is now in memory and warned about on a similar request
    fails = parts["failures"].all()
    assert len(fails) == 1
    assert "export_users" in fails[0].task_request
    warn = parts["failures"].warning_block("Add export_users to the reporting flow")
    assert "SIMILAR PAST FAILURES" in warn


def test_pipeline_budget_exceeded_aborts(sample_project):
    parts = build_parts(sample_project, daily_budget_usd=0.0)
    parts["reality"].init_project("Demo", "Python/FastAPI")
    result = parts["pipeline"].run("Add CSV export to users")
    assert not result.passed
    assert "budget" in result.failure_note.lower()


def test_pipeline_warning_reaches_grounding_prompt(sample_project, monkeypatch):
    """When FailureMemory warns, the warning must be inside the prompt the
    executor actually receives."""
    parts = build_parts(sample_project)
    parts["reality"].init_project("Demo", "Python/FastAPI")
    parts["context"].index_project()
    parts["failures"].record(
        task_request="add csv export to the users endpoint",
        attempt_summary="streamed with a broken generator",
        failure_reason="verdict=fail quality=0.20",
    )

    captured = {}
    orig_run = parts["pipeline"].executor.run

    def spy(request, grounding_section, extra_instructions=""):
        captured["grounding"] = grounding_section
        return orig_run(request, grounding_section, extra_instructions)

    monkeypatch.setattr(parts["pipeline"].executor, "run", spy)
    result = parts["pipeline"].run("add csv export to the users endpoint")
    assert "SIMILAR PAST FAILURES" in captured["grounding"]
    assert "broken generator" in captured["grounding"]
    assert result.passed


# ---------------------------------------------------------------------- #
# CLI surface (catches wiring/render bugs like missing imports)
# ---------------------------------------------------------------------- #
def test_cli_end_to_end(sample_project, monkeypatch):
    from click.testing import CliRunner
    from brain.cli import cli

    monkeypatch.setenv("BRAIN_LLM", "offline")
    monkeypatch.chdir(sample_project)
    r = CliRunner()

    res = r.invoke(cli, ["init", "--name", "CliDemo", "--stack", "Python/FastAPI"])
    assert res.exit_code == 0, res.output

    res = r.invoke(cli, ["rule", "Never modify the auth layer"])
    assert res.exit_code == 0 and "Added rule" in res.output

    res = r.invoke(cli, ["decide", "Use JWT", "stateless and simple"])
    assert res.exit_code == 0 and "Decision" in res.output

    res = r.invoke(cli, ["index", "--full"])
    assert res.exit_code == 0 and "Indexed" in res.output

    res = r.invoke(cli, ["run", "Add CSV export to the users endpoint"])
    assert res.exit_code == 0
    assert "RESULT (pass)" in res.output
    assert "CLAIM VERIFICATION" in res.output

    # grab the task id from the run output for trace replay
    import re
    m = re.search(r"task=(\w+)", res.output)
    assert m
    task_id = m.group(1)

    res = r.invoke(cli, ["status"])
    assert res.exit_code == 0
    assert "PROJECT HEALTH" in res.output and "CliDemo" in res.output

    res = r.invoke(cli, ["trace", task_id])
    assert res.exit_code == 0
    for stage in ("start", "intent", "grounding", "execute", "validate", "done"):
        assert stage in res.output

    res = r.invoke(cli, ["failures"])
    assert res.exit_code == 0

    res = r.invoke(cli, ["rules"])
    assert res.exit_code == 0 and "auth layer" in res.output

    res = r.invoke(cli, ["decisions"])
    assert res.exit_code == 0 and "Use JWT" in res.output

    # governor rejection surfaces in the CLI too
    res = r.invoke(cli, ["run", "Rewrite the auth layer completely"])
    assert res.exit_code == 0 and "Scope rejected" in res.output
