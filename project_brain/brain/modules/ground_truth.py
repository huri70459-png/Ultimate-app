"""Ground Truth Engine — verify claims against the actual codebase.

Every claim an LLM makes gets a verdict backed by evidence:
  Claim -> Verification -> Evidence
No evidence: UNVERIFIED. Contradicted: FAIL. Confirmed: PASS.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from ..config import Config
from ..models import Claim, Verdict

SCAN_IGNORE_DIRS = {".brain", ".git", "node_modules", "__pycache__",
                    ".venv", "venv", "dist", "build"}
SCAN_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".sql"}


class GroundTruth:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._text_cache: dict[str, str] | None = None

    def verify(self, claims: list[Claim]) -> list[Claim]:
        # one codebase scan shared across all claims in this batch
        self._text_cache = None
        return [self._verify_one(c) for c in claims]

    def _verify_one(self, c: Claim) -> Claim:
        try:
            if c.kind == "file_exists":
                ok = self._safe_exists(c.target)
                c.verdict = Verdict.PASS if ok else Verdict.FAIL
                c.evidence = f"path={c.target} exists={ok}"
            elif c.kind == "symbol_exists":
                found_in = self._symbol_in_codebase(c.target)
                c.verdict = Verdict.PASS if found_in else Verdict.FAIL
                c.evidence = (f"symbol={c.target} found_in={found_in}"
                              if found_in else f"symbol={c.target} found=False")
            elif c.kind == "schema_match":
                ok, msg = self._check_schema(c.target)
                c.verdict = Verdict.PASS if ok else Verdict.FAIL
                c.evidence = msg
            elif c.kind == "path_in_output":
                ok = self._safe_exists(c.target)
                c.verdict = Verdict.PASS if ok else Verdict.FAIL
                c.evidence = f"output references {c.target}, exists={ok}"
            else:
                c.verdict = Verdict.UNVERIFIED
                c.evidence = f"no verifier for kind={c.kind}"
        except Exception as e:  # a verifier must never crash the pipeline
            c.verdict = Verdict.UNVERIFIED
            c.evidence = f"error: {e}"
        return c

    def _safe_exists(self, target: str) -> bool:
        p = (self.cfg.project_root / target).resolve()
        root = self.cfg.project_root.resolve()
        if not str(p).startswith(str(root)):
            return False  # block path traversal outside the project
        return p.exists()

    def _scan_texts(self) -> dict[str, str]:
        if self._text_cache is not None:
            return self._text_cache
        cache: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.cfg.project_root):
            dirnames[:] = [d for d in dirnames if d not in SCAN_IGNORE_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in SCAN_EXTS:
                    continue
                try:
                    cache[p.relative_to(self.cfg.project_root).as_posix()] = \
                        p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
        self._text_cache = cache
        return cache

    def _symbol_in_codebase(self, symbol: str) -> str | None:
        """Return the first file containing the symbol, or None."""
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
            return None
        pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
        for path, text in self._scan_texts().items():
            if pattern.search(text):
                return path
        return None

    def _check_schema(self, spec: str) -> tuple[bool, str]:
        """spec format: 'sqlite:path/to/db.sqlite table:users column:email'"""
        parts = dict(p.split(":", 1) for p in spec.split() if ":" in p)
        db = parts.get("sqlite")
        table = parts.get("table")
        column = parts.get("column")
        if not (db and table):
            return False, "schema spec must include sqlite: and table:"
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
            return False, f"invalid table name: {table}"
        db_path = (self.cfg.project_root / db).resolve()
        root = self.cfg.project_root.resolve()
        if not str(db_path).startswith(str(root)):
            return False, f"db path escapes project root: {db}"
        if not db_path.exists():
            return False, f"db not found: {db}"
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cur.fetchall()}
        except sqlite3.Error as e:
            return False, f"sqlite error: {e}"
        finally:
            conn.close()
        if not cols:
            return False, f"table={table} not found in {db}"
        if column:
            ok = column in cols
            return ok, (f"table={table} column={column} present={ok} "
                        f"cols={sorted(cols)}")
        return True, f"table={table} exists cols={sorted(cols)}"
