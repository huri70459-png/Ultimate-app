"""Terminal dashboard — one screen project health."""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .modules.token_governor import TokenGovernor
from .storage import Storage, VectorIndex

VERDICT_STYLE = {"pass": "green", "fail": "red", "unverified": "yellow"}


class Dashboard:
    def __init__(self, cfg: Config, storage: Storage,
                 index: VectorIndex) -> None:
        self.cfg = cfg
        self.storage = storage
        self.index = index
        self.console = Console()
        self.governor = TokenGovernor(cfg, storage)

    def render(self) -> None:
        meta = self.storage.load_project()
        snap = self.governor.snapshot()
        recents = self.storage.recent_executions(10)

        # --- project health ---
        passed_all, total_all = self.storage.pass_rate()
        total = len(recents)
        passed = sum(1 for r in recents if r.verdict.value == "pass")
        avg_q = (sum(r.quality_score for r in recents) / total) if total else 0.0
        failed_claims = self.storage.unverified_claim_count()

        n_files = len([i for i in self.index.ids() if i.startswith("file:")])
        n_fails = len([i for i in self.index.ids() if i.startswith("fail:")])

        health = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
        health.add_column("metric", style="bold")
        health.add_column("value", justify="right")
        health.add_row("Architecture",
                       "[green]indexed[/green]" if n_files else "[yellow]not indexed[/yellow]")
        health.add_row("Indexed files", str(n_files))
        health.add_row("Decisions recorded", str(len(self.storage.all_decisions())))
        health.add_row("Critical rules", str(len(self.storage.all_rules())))
        health.add_row("Recent pass rate", f"{passed}/{total}")
        health.add_row("All-time pass rate", f"{passed_all}/{total_all}")
        health.add_row("Avg quality (recent)", f"{avg_q:.2f}")
        health.add_row("Hallucinations caught", str(failed_claims))
        health.add_row("Failures remembered", str(n_fails))

        # --- ai activity ---
        pct = snap["budget_used_pct"]
        bar_color = "green" if pct < 60 else ("yellow" if pct < 90 else "red")
        activity = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
        activity.add_column("metric", style="bold")
        activity.add_column("value", justify="right")
        activity.add_row("Backend", snap["backend"])
        activity.add_row("Tokens today", f"{snap['tokens_today']:,}")
        activity.add_row("Cost today", f"${snap['spent_today_usd']:.4f}")
        activity.add_row("Daily budget", f"${snap['budget_usd']:.2f}")
        activity.add_row("Budget used",
                         f"[{bar_color}]{pct:.1f}%[/{bar_color}] "
                         f"(${snap['budget_remaining_usd']:.2f} left)")

        # --- recent runs ---
        runs = Table(box=box.ROUNDED, expand=True)
        runs.add_column("time", style="dim")
        runs.add_column("verdict")
        runs.add_column("q", justify="right")
        runs.add_column("tok", justify="right")
        runs.add_column("cost", justify="right")
        runs.add_column("retries", justify="right")
        runs.add_column("output", overflow="ellipsis", max_width=60)
        for r in recents:
            style = VERDICT_STYLE.get(r.verdict.value, "white")
            preview = r.output.splitlines()[0][:60] if r.output else ""
            runs.add_row(
                r.created_at[11:19],
                f"[{style}]{r.verdict.value}[/{style}]",
                f"{r.quality_score:.2f}",
                str(r.tokens_in + r.tokens_out),
                f"${r.cost_usd:.4f}",
                str(r.retries),
                escape(preview),
            )
        if not recents:
            runs.add_row("—", "—", "—", "—", "—", "—",
                         "[dim]no runs yet — try `brain run \"...\"`[/dim]")

        project_name = meta.name if meta else "(uninitialized)"
        header = f"[bold]{project_name}[/bold]" + \
            (f" — {meta.stack}" if meta else "") + \
            (f"\n[dim]{meta.goal}[/dim]" if meta and meta.goal else "")

        self.console.print()
        self.console.print(Panel(header, title="PROJECT BRAIN",
                                 border_style="cyan"))
        self.console.print(Panel(health, title="PROJECT HEALTH",
                                 border_style="green"))
        self.console.print(Panel(activity, title="AI ACTIVITY",
                                 border_style="yellow"))
        self.console.print(Panel(runs, title="RECENT RUNS",
                                 border_style="magenta"))
        self.console.print()
