"""SQLite persistence + vector index (FAISS when available, numpy fallback)."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config
from .models import (
    Decision,
    ExecutionResult,
    FailureRecord,
    FileDigest,
    ProjectMeta,
    TraceEntry,
)
from .models import today as _today


class Storage:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._local = threading.local()
        self._init_db()

    # --- sqlite connection per thread ---
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.cfg.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        with self.tx() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS project_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    output TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    claims TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    verdict TEXT NOT NULL,
                    retries INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                CREATE TABLE IF NOT EXISTS failures (
                    id TEXT PRIMARY KEY,
                    task_request TEXT NOT NULL,
                    attempt_summary TEXT NOT NULL,
                    failure_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    files TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS file_digests (
                    path TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS token_ledger (
                    date TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    PRIMARY KEY (date, task_id)
                );
                CREATE TABLE IF NOT EXISTS critical_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule TEXT NOT NULL UNIQUE
                );
            """)

    # --- project meta ---
    def save_project(self, meta: ProjectMeta) -> None:
        blob = meta.model_dump_json()
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO project_meta(key, value) VALUES('meta', ?)",
                (blob,),
            )

    def load_project(self) -> ProjectMeta | None:
        row = self._conn().execute(
            "SELECT value FROM project_meta WHERE key='meta'"
        ).fetchone()
        return ProjectMeta.model_validate_json(row[0]) if row else None

    # --- tasks ---
    def save_task(self, t) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO tasks(id, request, kind, created_at) "
                "VALUES(?,?,?,?)",
                (t.id, t.request, t.kind.value, t.created_at),
            )

    # --- executions ---
    def save_execution(self, e: ExecutionResult) -> str:
        if not e.id:
            e.id = uuid.uuid4().hex[:12]
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO executions
                   (id, task_id, output, citations, claims, tokens_in, tokens_out,
                    cost_usd, latency_ms, verdict, retries, quality_score, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.id, e.task_id, e.output,
                    json.dumps(e.citations),
                    json.dumps([cl.model_dump(mode="json") for cl in e.claims]),
                    e.tokens_in, e.tokens_out, e.cost_usd,
                    e.latency_ms, e.verdict.value, e.retries,
                    e.quality_score, e.created_at,
                ),
            )
            day = e.created_at[:10]
            c.execute(
                """INSERT OR REPLACE INTO token_ledger
                   (date, task_id, tokens_in, tokens_out, cost_usd)
                   VALUES(?,?,?,?,?)""",
                (day, e.task_id, e.tokens_in, e.tokens_out, e.cost_usd),
            )
        return e.id

    def today_cost(self) -> float:
        row = self._conn().execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM token_ledger WHERE date=?",
            (_today(),),
        ).fetchone()
        return float(row[0])

    def today_tokens(self) -> int:
        row = self._conn().execute(
            "SELECT COALESCE(SUM(tokens_in + tokens_out),0) FROM token_ledger "
            "WHERE date=?",
            (_today(),),
        ).fetchone()
        return int(row[0])

    def recent_executions(self, n: int = 20) -> list[ExecutionResult]:
        rows = self._conn().execute(
            "SELECT * FROM executions ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (n,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(ExecutionResult(
                id=r["id"],
                task_id=r["task_id"],
                output=r["output"],
                citations=json.loads(r["citations"]),
                claims=[Claim_parse(cl) for cl in json.loads(r["claims"])],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                cost_usd=r["cost_usd"],
                latency_ms=r["latency_ms"],
                verdict=r["verdict"],
                retries=r["retries"],
                quality_score=r["quality_score"],
                created_at=r["created_at"],
            ))
        return out

    def pass_rate(self) -> tuple[int, int]:
        """(passed, total) over all executions."""
        row = self._conn().execute(
            "SELECT COALESCE(SUM(verdict='pass'),0), COUNT(*) FROM executions"
        ).fetchone()
        return int(row[0]), int(row[1])

    def unverified_claim_count(self) -> int:
        """Count of FAIL claims across all stored executions (hallucinations caught)."""
        rows = self._conn().execute("SELECT claims FROM executions").fetchall()
        n = 0
        for r in rows:
            for cl in json.loads(r["claims"]):
                if cl.get("verdict") == "fail":
                    n += 1
        return n

    # --- failures ---
    def save_failure(self, f: FailureRecord) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO failures
                   (id, task_request, attempt_summary, failure_reason, created_at)
                   VALUES(?,?,?,?,?)""",
                (f.id, f.task_request, f.attempt_summary,
                 f.failure_reason, f.created_at),
            )

    def all_failures(self) -> list[FailureRecord]:
        rows = self._conn().execute(
            "SELECT id, task_request, attempt_summary, failure_reason, created_at "
            "FROM failures ORDER BY created_at DESC"
        ).fetchall()
        return [FailureRecord(**dict(r)) for r in rows]

    # --- decisions ---
    def save_decision(self, d: Decision) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO decisions
                   (id, title, rationale, files, created_at)
                   VALUES(?,?,?,?,?)""",
                (d.id, d.title, d.rationale,
                 json.dumps(d.files), d.created_at),
            )

    def search_decisions(self, q: str) -> list[Decision]:
        rows = self._conn().execute(
            "SELECT * FROM decisions WHERE title LIKE ? OR rationale LIKE ? "
            "ORDER BY created_at DESC LIMIT 10",
            (f"%{q}%", f"%{q}%"),
        ).fetchall()
        return [
            Decision(
                id=r["id"], title=r["title"], rationale=r["rationale"],
                files=json.loads(r["files"]), created_at=r["created_at"],
            ) for r in rows
        ]

    def all_decisions(self) -> list[Decision]:
        rows = self._conn().execute(
            "SELECT * FROM decisions ORDER BY created_at DESC"
        ).fetchall()
        return [
            Decision(
                id=r["id"], title=r["title"], rationale=r["rationale"],
                files=json.loads(r["files"]), created_at=r["created_at"],
            ) for r in rows
        ]

    # --- file digests ---
    def save_file_digest(self, fd: FileDigest) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO file_digests
                   (path, summary, symbols, content_hash, updated_at)
                   VALUES(?,?,?,?,?)""",
                (fd.path, fd.summary, json.dumps(fd.symbols),
                 fd.content_hash, fd.updated_at),
            )

    def get_file_digest(self, path: str) -> FileDigest | None:
        r = self._conn().execute(
            "SELECT path, summary, symbols, content_hash, updated_at "
            "FROM file_digests WHERE path=?",
            (path,),
        ).fetchone()
        if not r:
            return None
        return FileDigest(
            path=r["path"], summary=r["summary"],
            symbols=json.loads(r["symbols"]),
            content_hash=r["content_hash"], updated_at=r["updated_at"],
        )

    def all_file_digests(self) -> list[FileDigest]:
        rows = self._conn().execute(
            "SELECT path, summary, symbols, content_hash, updated_at "
            "FROM file_digests"
        ).fetchall()
        return [
            FileDigest(
                path=r["path"], summary=r["summary"],
                symbols=json.loads(r["symbols"]),
                content_hash=r["content_hash"], updated_at=r["updated_at"],
            ) for r in rows
        ]

    def delete_file_digest(self, path: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM file_digests WHERE path=?", (path,))

    # --- critical rules ---
    def add_rule(self, rule: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO critical_rules(rule) VALUES(?)", (rule,)
            )

    def all_rules(self) -> list[str]:
        rows = self._conn().execute(
            "SELECT rule FROM critical_rules ORDER BY id"
        ).fetchall()
        return [r[0] for r in rows]

    # --- traces (append-only JSONL) ---
    def append_trace(self, t: TraceEntry) -> None:
        with self.cfg.trace_path.open("a", encoding="utf-8") as f:
            f.write(t.model_dump_json() + "\n")

    def read_traces(self, task_id: str | None = None) -> list[TraceEntry]:
        if not self.cfg.trace_path.exists():
            return []
        out = []
        for line in self.cfg.trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = TraceEntry.model_validate_json(line)
            if task_id is None or entry.task_id == task_id:
                out.append(entry)
        return out


def Claim_parse(data: dict):
    from .models import Claim
    return Claim(**data)


class VectorIndex:
    """Cosine-similarity index over named vectors.

    Vectors are stored in a dict (id -> unit vector) persisted to an .npz
    file. Search uses FAISS (IndexFlatIP on L2-normalized vectors) when
    installed, otherwise an equivalent numpy brute-force. Adding an existing
    id replaces its vector (upsert), so re-indexing never duplicates entries.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._vecs: dict[str, np.ndarray] = {}
        self._faiss = None
        self._faiss_ids: list[str] = []
        self._load()

    # --- persistence ---
    def _load(self) -> None:
        if self.cfg.index_path.exists():
            try:
                data = np.load(self.cfg.index_path, allow_pickle=False)
                ids = data["ids"]
                mat = data["vectors"]
                self._vecs = {str(i): mat[k].astype("float32")
                              for k, i in enumerate(ids)}
            except Exception:
                self._vecs = {}
        self._rebuild_faiss()

    def _save(self) -> None:
        ids = list(self._vecs.keys())
        if ids:
            mat = np.stack([self._vecs[i] for i in ids]).astype("float32")
        else:
            mat = np.zeros((0, self.cfg.embedding_dim), dtype="float32")
        # np.savez appends ".npz" when missing, so the tmp name must end in it
        tmp = self.cfg.index_path.with_name(self.cfg.index_path.stem + ".tmp.npz")
        np.savez(tmp, ids=np.array(ids, dtype=np.str_), vectors=mat)
        tmp.replace(self.cfg.index_path)

    def _rebuild_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            self._faiss = None
            return
        if not self._vecs:
            self._faiss = None
            self._faiss_ids = []
            return
        dims = {v.shape[0] for v in self._vecs.values()}
        if len(dims) != 1:
            self._faiss = None  # mixed dims -> numpy fallback
            self._faiss_ids = []
            return
        self._faiss_ids = list(self._vecs.keys())
        index = faiss.IndexFlatIP(dims.pop())
        mat = np.stack([self._normalize(self._vecs[i])
                        for i in self._faiss_ids]).astype("float32")
        index.add(mat)
        self._faiss = index

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32").reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    # --- api ---
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32").reshape(len(ids), -1)
        for i, vec in zip(ids, vectors):
            self._vecs[i] = self._normalize(vec)
        self._save()
        self._rebuild_faiss()

    def remove(self, ids: list[str]) -> int:
        doomed = [i for i in ids if i in self._vecs]
        for i in doomed:
            del self._vecs[i]
        if doomed:
            self._save()
            self._rebuild_faiss()
        return len(doomed)

    def search(self, q: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        qn = self._normalize(q).reshape(1, -1)
        k = min(k, len(self._vecs))
        if self._faiss is not None and self._faiss.ntotal > 0:
            scores, idxs = self._faiss.search(qn.astype("float32"), k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self._faiss_ids):
                    results.append((self._faiss_ids[idx], float(score)))
            return results
        ids = list(self._vecs.keys())
        mat = np.stack([self._vecs[i] for i in ids]).astype("float32")
        sims = mat @ qn[0]
        order = np.argsort(-sims)[:k]
        return [(ids[int(j)], float(sims[int(j)])) for j in order]

    def ids(self) -> list[str]:
        return list(self._vecs.keys())

    @property
    def size(self) -> int:
        return len(self._vecs)
