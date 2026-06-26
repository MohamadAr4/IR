"""Disk-backed inverted index (requirement #3).

Backed by SQLite so it scales to the full corpora without holding the postings
in RAM. It is built by *streaming* the preprocessed docs, and serves as the
shared substrate for both the TF-IDF/VSM and BM25 scorers (they only differ in
the term-weighting formula, not in the index).

Schema
------
meta(key, value)                         -- N, avgdl, built_docs, ...
docs(rowid, doc_id, length, tfidf_norm, raw_text)
terms(term, df, idf)                     -- df = document frequency
postings(term, doc_rowid, tf)            -- one row per (term, doc)
"""
from __future__ import annotations

import math
import os
import sqlite3
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from ..config import DatasetSpec, ensure_index_dir
from ..data.corpus import Doc, iter_docs


@dataclass
class Posting:
    doc_rowid: int
    tf: int
    length: int
    tfidf_norm: float


def _db_path(spec: DatasetSpec) -> str:
    return os.path.join(spec.index_dir, "inverted_index.sqlite")


def _compress_text(text: str) -> Optional[bytes]:
    """zlib-compress the full document text for compact on-disk storage.
    The complete text is preserved (no truncation) — we just store it small."""
    if not text:
        return None
    return zlib.compress(text.encode("utf-8"), level=6)


def _decompress_text(value) -> str:
    """Inverse of :func:`_compress_text`. Tolerates legacy plain-TEXT rows
    (str) and empty/NULL values so old indexes still read."""
    if value is None:
        return ""
    if isinstance(value, str):          # legacy uncompressed rows
        return value
    try:
        return zlib.decompress(value).decode("utf-8")
    except (zlib.error, OSError):
        return ""


def _connect(path: str) -> sqlite3.Connection:
    # check_same_thread=False: the engine caches one read-only connection that
    # may be reused across threads (e.g. Streamlit reruns each interaction on a
    # different ScriptRunner thread). Safe here because query-time access is
    # read-only and serialised; index *builds* run in their own processes.
    conn = sqlite3.connect(path, check_same_thread=False)
    # pragmas tuned for bulk build + read throughput
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")  # ~200 MB page cache
    # Register LOG/SQRT explicitly: SQLite's math functions are only present
    # when compiled with SQLITE_ENABLE_MATH_FUNCTIONS, and we want natural-log
    # semantics that match the Python-side scorer exactly.
    conn.create_function("LOG", 1, lambda x: math.log(x) if x and x > 0 else 0.0)
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x and x > 0 else 0.0)
    return conn


class InvertedIndex:
    """Read/build interface over the SQLite inverted index for one dataset."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.path = _db_path(spec)
        self._conn: Optional[sqlite3.Connection] = None
        self._num_docs: Optional[int] = None
        self._avgdl: Optional[float] = None

    # -- connection -------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"No inverted index for '{self.spec.key}' at {self.path}. "
                    f"Build it first (scripts/build_indexes.py).")
            self._conn = _connect(self.path)
        return self._conn

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- build ------------------------------------------------------------
    def build(self, limit: Optional[int] = None, keep_raw: bool = True,
              log_every: int = 5000) -> dict:
        """Build the index by streaming the corpus. Idempotent: rebuilds from
        scratch. When ``keep_raw`` is set, the *complete* original document text
        is stored — zlib-compressed as a BLOB — so the UI can show the full,
        readable document straight from the database (set keep_raw=False to
        store nothing)."""
        ensure_index_dir(self.spec.key)
        if os.path.exists(self.path):
            self.close()
            for ext in ("", "-wal", "-shm"):
                try:
                    os.remove(self.path + ext)
                except OSError:
                    pass

        conn = _connect(self.path)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE docs(rowid INTEGER PRIMARY KEY,
                              doc_id TEXT, length INTEGER,
                              tfidf_norm REAL DEFAULT 0, raw_text BLOB);
            CREATE TABLE terms(term TEXT PRIMARY KEY, df INTEGER, idf REAL DEFAULT 0);
            CREATE TABLE postings(term TEXT, doc_rowid INTEGER, tf INTEGER);
            """
        )

        n_docs = 0
        total_len = 0
        doc_batch: list[tuple] = []
        post_batch: list[tuple] = []

        def flush():
            if doc_batch:
                cur.executemany(
                    "INSERT INTO docs(rowid, doc_id, length, raw_text) VALUES(?,?,?,?)",
                    doc_batch)
                doc_batch.clear()
            if post_batch:
                cur.executemany(
                    "INSERT INTO postings(term, doc_rowid, tf) VALUES(?,?,?)",
                    post_batch)
                post_batch.clear()

        for doc in iter_docs(self.spec, limit=limit):
            rowid = n_docs + 1
            tf = Counter(doc.tokens)
            length = len(doc.tokens)
            raw = _compress_text(doc.raw_text) if keep_raw else None
            doc_batch.append((rowid, doc.doc_id, length, raw))
            for term, c in tf.items():
                post_batch.append((term, rowid, c))

            n_docs += 1
            total_len += length
            if len(post_batch) >= 200_000:
                flush()
            if log_every and n_docs % log_every == 0:
                print(f"  [{self.spec.key}] indexed {n_docs} docs ...", flush=True)

        flush()
        conn.commit()

        # df per term from the postings (done in SQL, on disk)
        print(f"  [{self.spec.key}] computing document frequencies ...", flush=True)
        cur.execute(
            "INSERT INTO terms(term, df) SELECT term, COUNT(*) FROM postings GROUP BY term")
        conn.commit()

        # idf (smoothed, used by the VSM cosine weighting)
        idf_expr = "LOG((? * 1.0) / df)"  # base-e log; consistent on both sides
        cur.execute(f"UPDATE terms SET idf = {idf_expr}", (n_docs,))
        conn.commit()

        # indexes for fast lookup
        print(f"  [{self.spec.key}] creating B-tree indexes ...", flush=True)
        cur.execute("CREATE INDEX idx_postings_term ON postings(term)")
        cur.execute("CREATE INDEX idx_postings_doc ON postings(doc_rowid)")
        cur.execute("CREATE UNIQUE INDEX idx_docs_docid ON docs(doc_id)")
        conn.commit()

        # per-doc TF-IDF L2 norm (for cosine). A correlated SQL subquery per
        # doc is pathologically slow at corpus scale, so we stream the postings
        # once (naturally ascending by doc_rowid) and accumulate the norms in
        # Python — a single linear pass over the postings table.
        print(f"  [{self.spec.key}] computing TF-IDF document norms ...", flush=True)
        from collections import defaultdict
        sumsq: dict[int, float] = defaultdict(float)
        seen = 0
        for doc_rowid, tf, idf in cur.execute(
                "SELECT p.doc_rowid, p.tf, t.idf "
                "FROM postings p JOIN terms t ON t.term = p.term"):
            w = (1.0 + math.log(tf)) * idf
            sumsq[doc_rowid] += w * w
            seen += 1
            if log_every and seen % 2_000_000 == 0:
                print(f"    ...accumulated {seen} postings", flush=True)
        cur.executemany("UPDATE docs SET tfidf_norm = ? WHERE rowid = ?",
                        ((math.sqrt(s), rid) for rid, s in sumsq.items()))
        conn.commit()

        avgdl = (total_len / n_docs) if n_docs else 0.0
        meta = {"num_docs": n_docs, "avgdl": avgdl, "total_terms": total_len}
        cur.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
                        [(k, str(v)) for k, v in meta.items()])
        conn.commit()
        cur.execute("ANALYZE")
        conn.commit()
        conn.close()
        self._conn = None
        print(f"  [{self.spec.key}] inverted index done: {n_docs} docs, avgdl={avgdl:.1f}",
              flush=True)
        return meta

    # -- read accessors ---------------------------------------------------
    def _meta(self, key: str, cast):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return cast(row[0]) if row else None

    @property
    def num_docs(self) -> int:
        if self._num_docs is None:
            self._num_docs = self._meta("num_docs", int) or 0
        return self._num_docs

    @property
    def avgdl(self) -> float:
        if self._avgdl is None:
            self._avgdl = self._meta("avgdl", float) or 0.0
        return self._avgdl

    def df(self, term: str) -> int:
        row = self.conn.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
        return row[0] if row else 0

    def idf(self, term: str) -> float:
        row = self.conn.execute("SELECT idf FROM terms WHERE term=?", (term,)).fetchone()
        return row[0] if row else 0.0

    def postings(self, term: str) -> list[Posting]:
        rows = self.conn.execute(
            """SELECT p.doc_rowid, p.tf, d.length, d.tfidf_norm
               FROM postings p JOIN docs d ON d.rowid = p.doc_rowid
               WHERE p.term = ?""", (term,)).fetchall()
        return [Posting(*r) for r in rows]

    def doc_meta(self, term_count: bool = False):
        return self.num_docs, self.avgdl

    def get_docs(self, rowids: Iterable[int]) -> dict[int, dict]:
        rowids = list(rowids)
        if not rowids:
            return {}
        qmarks = ",".join("?" * len(rowids))
        rows = self.conn.execute(
            f"SELECT rowid, doc_id, raw_text, length FROM docs WHERE rowid IN ({qmarks})",
            rowids).fetchall()
        return {r[0]: {"doc_id": r[1], "raw_text": _decompress_text(r[2]), "length": r[3]}
                for r in rows}

    def get_docs_by_id(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Map ``doc_id -> raw_text`` for display (used by the embedding
        retrievers, which key on doc_id rather than rowid)."""
        doc_ids = list(doc_ids)
        if not doc_ids:
            return {}
        out: dict[str, str] = {}
        # chunk to stay under SQLite's variable limit
        for i in range(0, len(doc_ids), 500):
            chunk = doc_ids[i:i + 500]
            qmarks = ",".join("?" * len(chunk))
            for did, raw in self.conn.execute(
                    f"SELECT doc_id, raw_text FROM docs WHERE doc_id IN ({qmarks})", chunk):
                out[did] = _decompress_text(raw)
        return out

    def rowids_for(self, doc_ids: Iterable[str]) -> dict[str, int]:
        """Map ``doc_id -> rowid`` (used to turn an embedding-stage candidate
        set into a rowid filter for the lexical re-scorer)."""
        doc_ids = list(doc_ids)
        out: dict[str, int] = {}
        for i in range(0, len(doc_ids), 500):
            chunk = doc_ids[i:i + 500]
            qmarks = ",".join("?" * len(chunk))
            for did, rowid in self.conn.execute(
                    f"SELECT doc_id, rowid FROM docs WHERE doc_id IN ({qmarks})", chunk):
                out[did] = rowid
        return out

    def vocab_sample(self, prefix: str, limit: int = 10) -> list[str]:
        rows = self.conn.execute(
            "SELECT term FROM terms WHERE term LIKE ? ORDER BY df DESC LIMIT ?",
            (prefix + "%", limit)).fetchall()
        return [r[0] for r in rows]

    def all_terms_with_df(self, min_df: int = 1, limit: Optional[int] = None):
        q = "SELECT term, df FROM terms WHERE df >= ? ORDER BY df DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return self.conn.execute(q, (min_df,)).fetchall()
