"""Context Engine — semantic retrieval over file digests + decisions.

Indexing is incremental: files are embedded only when their content hash
changes, so repeated runs never re-spend tokens on unchanged files and the
vector index never accumulates duplicates.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ..config import Config
from ..embeddings import Embedder
from ..models import FileDigest
from ..storage import Storage, VectorIndex

IGNORE_DIRS = {
    ".brain", ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".cache", ".idea", ".vscode", "site-packages",
}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".md", ".txt", ".html", ".css", ".scss", ".sql", ".toml", ".sh",
}
MAX_FILE_BYTES = 200_000


class ContextEngine:
    def __init__(
        self, cfg: Config, storage: Storage,
        index: VectorIndex, embedder: Embedder,
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.index = index
        self.embedder = embedder

    # --- indexing ---
    def index_project(self, full: bool = False) -> dict:
        """Walk the project, summarize changed text files, embed summaries.

        Returns stats: {'indexed': n_new_or_changed, 'skipped': n_unchanged,
                        'removed': n_deleted}.
        """
        root = self.cfg.project_root
        seen: set[str] = set()
        to_embed: list[FileDigest] = []
        skipped = 0

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size > MAX_FILE_BYTES or size == 0:
                    continue
                rel = p.relative_to(root).as_posix()
                seen.add(rel)
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                digest_hash = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
                prev = None if full else self.storage.get_file_digest(rel)
                if prev and prev.content_hash == digest_hash \
                        and self.index_has(rel):
                    skipped += 1
                    continue
                to_embed.append(FileDigest(
                    path=rel,
                    summary=self._summarize(rel, text),
                    symbols=self._extract_symbols(rel, text),
                    content_hash=digest_hash,
                ))

        # drop digests/vectors for files that disappeared
        removed = 0
        for fd in self.storage.all_file_digests():
            if fd.path not in seen:
                self.storage.delete_file_digest(fd.path)
                self.index.remove([f"file:{fd.path}"])
                removed += 1

        if to_embed:
            summaries = [
                f"{d.path}\n{d.summary}\nSymbols: {', '.join(d.symbols[:20])}"
                for d in to_embed
            ]
            vecs = self.embedder.embed_many(summaries)
            ids = [f"file:{d.path}" for d in to_embed]
            self.index.add(ids, vecs)
            for d in to_embed:
                self.storage.save_file_digest(d)

        return {"indexed": len(to_embed), "skipped": skipped, "removed": removed}

    def index_has(self, rel_path: str) -> bool:
        return f"file:{rel_path}" in set(self.index.ids())

    _PY_FUNC = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M)
    _PY_CLASS = re.compile(r"^\s*class\s+(\w+)", re.M)
    _JS_FUNC = re.compile(
        r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()", re.M)
    _JS_CLASS = re.compile(r"(?:class|interface)\s+(\w+)", re.M)

    def _summarize(self, path: str, text: str) -> str:
        """Cheap heuristic summary: first non-blank 12 lines."""
        head = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            head.append(s)
            if len(head) >= 12:
                break
        return " | ".join(head)[:800]

    def _extract_symbols(self, path: str, text: str) -> list[str]:
        symbols: list[str] = []
        if path.endswith(".py"):
            symbols.extend(self._PY_CLASS.findall(text))
            symbols.extend(self._PY_FUNC.findall(text))
        elif path.endswith((".js", ".ts", ".tsx", ".jsx")):
            symbols.extend(self._JS_CLASS.findall(text))
            for m in self._JS_FUNC.findall(text):
                symbols.append(m[0] or m[1])
        elif path.endswith(".sql"):
            symbols.extend(re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
                                      text, re.I))
        return list(dict.fromkeys(symbols))[:50]

    # --- retrieval ---
    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        qvec = self.embedder.embed_one(query)
        hits = self.index.search(qvec, k=k * 4)
        digests = {f"file:{d.path}": d for d in self.storage.all_file_digests()}
        results = []
        for id_, score in hits:
            d = digests.get(id_)
            if not d:
                continue
            results.append({
                "path": d.path,
                "summary": d.summary,
                "symbols": d.symbols,
                "score": score,
            })
            if len(results) >= k:
                break
        return results

    def build_context(self, query: str, k: int = 8) -> str:
        hits = self.retrieve(query, k=k)
        if not hits:
            return "(no relevant files found in index — run `brain index`)"
        parts = []
        for h in hits:
            syms = ", ".join(h["symbols"][:10]) or "—"
            parts.append(
                f"--- {h['path']} (score={h['score']:.2f}) ---\n"
                f"Symbols: {syms}\n"
                f"Summary: {h['summary']}"
            )
        return "\n\n".join(parts)
