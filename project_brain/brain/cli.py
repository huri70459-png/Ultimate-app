"""CLI entry point."""
from __future__ import annotations

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .dashboard import Dashboard
from .embeddings import Embedder
from .llm import LLM
from .modules.change_governor import ChangeGovernor
from .modules.context_engine import ContextEngine
from .modules.failure_memory import FailureMemory
from .modules.reality_graph import RealityGraph
from .modules.token_governor import TokenGovernor
from .storage import Storage, VectorIndex
from .veriflow.pipeline import Pipeline

console = Console()


def _build(cfg: Config) -> dict:
    """Compose the dependency graph (see README wiring diagram)."""
    storage = Storage(cfg)
    index = VectorIndex(cfg)
    llm = LLM(cfg)                       # lazy: no key needed until a call
    embedder = Embedder(cfg, llm)

    # Brain modules (persistent state layer)
    reality = RealityGraph(cfg, storage)
    context = ContextEngine(cfg, storage, index, embedder)
    failures = FailureMemory(cfg, storage, index, embedder)
    governor = TokenGovernor(cfg, storage)
    change_gov = ChangeGovernor(cfg, llm)

    # VeriFlow kernel (runtime layer)
    pipeline = Pipeline(
        cfg, storage, llm, reality, context, failures, governor, change_gov,
    )
    dashboard = Dashboard(cfg, storage, index)
    return {
        "cfg": cfg, "storage": storage, "index": index, "llm": llm,
        "embedder": embedder, "reality": reality, "context": context,
        "failures": failures, "governor": governor, "change_gov": change_gov,
        "pipeline": pipeline, "dashboard": dashboard,
    }


def _backend_note(llm: LLM) -> str:
    if llm.backend == "offline":
        return ("[yellow]offline backend[/yellow] — deterministic local engine, "
                "no API calls. Export OPENAI_API_KEY (or unset BRAIN_LLM=offline) "
                "for the OpenAI backend.")
    return f"[green]openai backend[/green] — model={llm.cfg.openai_model}"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def cli(ctx) -> None:
    """Project Brain — AI Operating System for building software."""
    ctx.ensure_object(dict)


@cli.command()
@click.option("--name", required=True, help="Project name")
@click.option("--stack", required=True,
              help="Tech stack, e.g. 'Python/FastAPI/React'")
@click.option("--goal", default="", help="Project goal")
def init(name: str, stack: str, goal: str) -> None:
    """Initialize a new project brain in the current directory."""
    cfg = Config.discover()
    parts = _build(cfg)
    meta = parts["reality"].init_project(name=name, stack=stack, goal=goal)
    console.print(Panel(
        f"Initialized brain for [bold]{meta.name}[/bold]\n"
        f"Stack: {meta.stack}\n"
        f"Goal: {meta.goal or '(not set)'}\n"
        f"Brain dir: {cfg.brain_dir}\n"
        f"LLM: {_backend_note(parts['llm'])}",
        title="INIT", border_style="green",
    ))


@cli.command(name="index")
@click.option("--full", is_flag=True,
              help="Re-embed every file even if unchanged")
def index_cmd(full: bool) -> None:
    """(Re)index project files into the vector store (incremental by default)."""
    cfg = Config.discover()
    parts = _build(cfg)
    stats = parts["context"].index_project(full=full)
    console.print(
        f"[green]Indexed {stats['indexed']} file(s)[/green] "
        f"({stats['skipped']} unchanged skipped, {stats['removed']} removed). "
        f"Vector store size: {parts['index'].size}"
    )


@cli.command()
@click.argument("request")
@click.option("--no-index", is_flag=True, help="Skip incremental re-indexing")
def run(request: str, no_index: bool) -> None:
    """Run a task through the VeriFlow pipeline."""
    cfg = Config.discover()
    parts = _build(cfg)
    if not no_index:
        stats = parts["context"].index_project()
        if stats["indexed"]:
            console.print(f"[dim]index: +{stats['indexed']} file(s)[/dim]")
    console.print(Panel(request, title="REQUEST", border_style="cyan"))
    result = parts["pipeline"].run(request)
    color = "green" if result.passed else "red"
    console.print(Panel(
        escape(result.execution.output),
        title=f"[{color}]RESULT ({result.execution.verdict.value})[/{color}]",
        border_style=color,
    ))
    e = result.execution
    console.print(
        f"tokens={e.tokens_in + e.tokens_out} "
        f"(in={e.tokens_in} out={e.tokens_out})  "
        f"cost=${e.cost_usd:.4f}  retries={e.retries}  "
        f"quality={e.quality_score:.2f}  task={e.task_id}"
    )
    if e.claims:
        claims_tbl = Table(title="CLAIM VERIFICATION", box=None, pad_edge=False)
        claims_tbl.add_column("verdict")
        claims_tbl.add_column("kind", style="dim")
        claims_tbl.add_column("target")
        claims_tbl.add_column("evidence", style="dim", max_width=60,
                              overflow="ellipsis")
        for c in e.claims:
            style = {"pass": "green", "fail": "red"}.get(c.verdict.value, "yellow")
            claims_tbl.add_row(f"[{style}]{c.verdict.value}[/{style}]",
                               c.kind, c.target, c.evidence)
        console.print(claims_tbl)
    if e.citations:
        console.print("[dim]citations:[/dim]")
        for c in e.citations:
            console.print(f"  • {c}")
    if not result.passed:
        console.print(f"[red]Failure note: {result.failure_note}[/red]")
        console.print("[dim]recorded in Failure Memory — similar future "
                      "requests will be warned.[/dim]")


@cli.command()
def status() -> None:
    """Show the project dashboard."""
    cfg = Config.discover()
    parts = _build(cfg)
    parts["dashboard"].render()


@cli.command()
@click.argument("rule")
def rule(rule: str) -> None:
    """Add a critical rule (e.g. 'Never modify auth layer')."""
    cfg = Config.discover()
    parts = _build(cfg)
    parts["reality"].add_rule(rule)
    console.print(f"[green]Added rule:[/green] {rule}")


@cli.command(name="rules")
def rules_cmd() -> None:
    """List critical rules."""
    cfg = Config.discover()
    parts = _build(cfg)
    rules = parts["reality"].rules()
    if not rules:
        console.print("[dim]no rules yet — add with `brain rule \"...\"`[/dim]")
        return
    for r in rules:
        console.print(f"  • {r}")


@cli.command()
@click.argument("title")
@click.argument("rationale")
def decide(title: str, rationale: str) -> None:
    """Record an architectural decision."""
    cfg = Config.discover()
    parts = _build(cfg)
    d = parts["reality"].add_decision(title=title, rationale=rationale, files=[])
    console.print(f"[green]Decision {d.id}:[/green] {title}")


@cli.command(name="decisions")
def decisions_cmd() -> None:
    """List recorded architectural decisions."""
    cfg = Config.discover()
    parts = _build(cfg)
    ds = parts["reality"].all_decisions()
    if not ds:
        console.print("[dim]no decisions yet — add with `brain decide "
                      "\"Title\" \"Rationale\"`[/dim]")
        return
    for d in ds:
        console.print(f"  [bold]{d.id}[/bold]  {d.title}")
        console.print(f"        [dim]{d.rationale}[/dim]")


@cli.command(name="failures")
def failures_cmd() -> None:
    """List the failure memory (approaches that must not be retried)."""
    cfg = Config.discover()
    parts = _build(cfg)
    fails = parts["failures"].all()
    if not fails:
        console.print("[green]failure memory is empty — nothing has "
                      "failed yet.[/green]")
        return
    tbl = Table(box=None, pad_edge=False)
    tbl.add_column("id", style="dim")
    tbl.add_column("when", style="dim")
    tbl.add_column("request", max_width=50, overflow="ellipsis")
    tbl.add_column("reason", style="red", max_width=40, overflow="ellipsis")
    for f in fails:
        tbl.add_row(f.id, f.created_at[:19].replace("T", " "),
                    f.task_request.replace("[", "\\["),
                    f.failure_reason.replace("[", "\\["))
    console.print(Panel(tbl, title=f"FAILURE MEMORY ({len(fails)})",
                        border_style="red"))


@cli.command()
@click.argument("task_id")
def trace(task_id: str) -> None:
    """Replay the full trace of a task (every pipeline stage)."""
    cfg = Config.discover()
    parts = _build(cfg)
    entries = parts["storage"].read_traces(task_id)
    if not entries:
        console.print(f"[yellow]no trace entries for task {task_id}[/yellow]")
        return
    for t in entries:
        console.print(f"[cyan]{t.created_at[11:19]}[/cyan] "
                      f"[bold]{t.stage}[/bold]")
        for k, v in t.payload.items():
            console.print(f"    [dim]{k}:[/dim] {v}")


if __name__ == "__main__":
    cli()
