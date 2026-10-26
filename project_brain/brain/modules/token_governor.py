"""Token Governor — budget enforcement and spend snapshots."""
from __future__ import annotations

from ..config import Config
from ..storage import Storage


class BudgetExceeded(Exception):
    pass


class TokenGovernor:
    def __init__(self, cfg: Config, storage: Storage) -> None:
        self.cfg = cfg
        self.storage = storage

    def check(self) -> None:
        spent = self.storage.today_cost()
        if spent >= self.cfg.daily_budget_usd:
            raise BudgetExceeded(
                f"Daily budget ${self.cfg.daily_budget_usd:.2f} "
                f"exceeded (spent ${spent:.2f})"
            )

    def snapshot(self) -> dict:
        spent = self.storage.today_cost()
        budget = self.cfg.daily_budget_usd
        return {
            "spent_today_usd": spent,
            "budget_usd": budget,
            "budget_remaining_usd": max(0.0, budget - spent),
            "budget_used_pct": (spent / budget * 100.0) if budget > 0 else 0.0,
            "tokens_today": self.storage.today_tokens(),
            "backend": "offline" if self.cfg.offline else "openai",
        }
