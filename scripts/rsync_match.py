#!/usr/bin/env python3
"""
Rsync-style matcher + SQLite index (optimized for NVMe/Colab; still MD5).

Exports:
- build_source_index_sqlite(old_bytes, idx_path, block_size)
- build_source_index_sqlite_stream(old_stream, idx_path, block_size, batch_size=8192)
- iter_matches_and_unmatched(new_bytes, idx_path, block_size)
- iter_matches_and_unmatched_stream_sqlcached(new_stream, idx_path, block_size, cache_size=65536)

Yields regions that fully cover NEW:
  COPY: ("copy", new_start, new_len, old_start)
  LIT : ("lit",  new_start, new_len)

Notes:
- Strong checksum is MD5 **digest bytes** (not hex) for faster compare.
- Streaming variant emits **single-block** COPY (length = B); LIT spans are variable.
- LRU cache stores weak→[(strong_digest_bytes, old_offset), ...] lists to reduce per-byte SQL.
"""

from __future__ import annotations
import hashlib
import io
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Optional


# =======================
# Rolling checksum (rsync)
# =======================

@dataclass
class RollingChecksum:
    a: int
    b: int
    n: int  # window length

    @staticmethod
    def from_block(block: bytes) -> "RollingChecksum":
        a = 0
        b = 0
        n = len(block)
        for i, xv in enumerate(block):
            a += xv
            b += (n - i) * xv
        a &= 0xFFFF_FFFF
        b &= 0xFFFF_FFFF
        return RollingChecksum(a, b, n)

    def value(self) -> int:
        # Classic rsync weak checksum packing: (a & 0xffff) | (b << 16)
        return ((self.b & 0xFFFF) << 16) | (self.a & 0xFFFF)

    def roll(self, out_byte: int, in_byte: int) -> None:
        a = (self.a - out_byte + in_byte) & 0xFFFF_FFFF
        b = (self.b - self.n * out_byte + a) & 0xFFFF_FFFF
        self.a = a
        self.b = b


# ===================
# SQLite index schema
# ===================

def _init_db(conn: sqlite3.Connection) -> None:
    """
    Initialize a per-old-file index DB with performance-friendly PRAGMAs.
    These DBs are disposable; we favor throughput over durability.
    """
    cur = conn.cursor()
    # Pragmas tuned for fast, throwaway indexing/scanning (NVMe-friendly; fine on HDD too)
    cur.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-1048576;      -- ~1 GiB page cache if RAM allows (negative = KB)
        PRAGMA page_size=32768;          -- larger pages reduce I/O
        PRAGMA mmap_size=536870912;      -- 512 MiB mmap window if available
        PRAGMA locking_mode=EXCLUSIVE;   -- fewer fsync/lock transitions
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            val TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS blocks(
            weak   INTEGER,
            strong BLOB,   -- MD5 digest bytes
            off    INTEGER
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_weak ON blocks(weak);")
    conn.commit()


# ===============================
# Build index (in-memory or stream)
# ===============================

def build_source_index_sqlite(old_bytes: bytes, idx_path: str, block_size: int) -> None:
    """
    Index OLD as fixed-size blocks of B. Store (weak, MD5_digest, offset) in SQLite.
    """
    conn = sqlite3.connect(idx_path)
    try:
        _init_db(conn)
        cur = conn.cursor()
        # Reset
        cur.execute("DELETE FROM blocks;")
        cur.execute("DELETE FROM meta;")
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('block_size', ?);", (str(block_size),))
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('old_len', ?);", (str(len(old_bytes)),))

        B = block_size
        off = 0
        batch: List[Tuple[int, bytes, int]] = []
        while off < len(old_bytes):
            blk = old_bytes[off: off + B]
            if len(blk) < B:
                blk = blk + bytes(B - len(blk))
            rc = RollingChecksum.from_block(blk)
            weak = rc.value()
            strong = hashlib.md5(blk).digest()
            batch.append((weak, strong, off))
            off += B
            if len(batch) >= 8192:
                cur.executemany("INSERT INTO blocks(weak,strong,off) VALUES(?,?,?)", batch)
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO blocks(weak,strong,off) VALUES(?,?,?)", batch)
        conn.commit()
    finally:
        conn.close()


def build_source_index_sqlite_stream(old_stream: io.BufferedReader, idx_path: str, block_size: int, batch_size: int = 8192) -> None:
    """
    Stream OLD, indexing fixed-size B blocks without loading the whole file.
    """
    conn = sqlite3.connect(idx_path)
    try:
        _init_db(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM blocks;")
        cur.execute("DELETE FROM meta;")
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('block_size', ?);", (str(block_size),))

        B = block_size
        read = old_stream.read
        total = 0
        batch: List[Tuple[int, bytes, int]] = []

        while True:
            blk = read(B)
            if not blk:
                break
            if len(blk) < B:
                blk = blk + bytes(B - len(blk))
            rc = RollingChecksum.from_block(blk)
            weak = rc.value()
            strong = hashlib.md5(blk).digest()
            batch.append((weak, strong, total))
            total += B
            if len(batch) >= batch_size:
                cur.executemany("INSERT INTO blocks(weak,strong,off) VALUES(?,?,?)", batch)
                batch.clear()

        if batch:
            cur.executemany("INSERT INTO blocks(weak,strong,off) VALUES(?,?,?)", batch)
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('old_len', ?);", (str(total),))
        conn.commit()
    finally:
        conn.close()


# ================
# Meta / utilities
# ================

def _load_block_size(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT val FROM meta WHERE key='block_size'").fetchone()
    if not row:
        raise ValueError("Index missing block_size meta")
    return int(row[0])


class _LRU:
    """
    Simple LRU for weak -> candidate list.
    Stores: weak:int -> List[(strong_digest:bytes, off:int)]
    """
    __slots__ = ("cap", "dict", "order")

    def __init__(self, capacity: int):
        self.cap = max(1, capacity)
        self.dict: Dict[int, List[Tuple[bytes, int]]] = {}
        self.order: List[int] = []  # most-recent at end

    def get(self, k: int) -> Optional[List[Tuple[bytes, int]]]:
        v = self.dict.get(k)
        if v is not None:
            # move to MRU
            try:
                i = self.order.index(k)
                self.order.pop(i)
            except ValueError:
                pass
            self.order.append(k)
        return v

    def put(self, k: int, v: List[Tuple[bytes, int]]) -> None:
        if k in self.dict:
            self.dict[k] = v
            try:
                i = self.order.index(k)
                self.order.pop(i)
            except ValueError:
                pass
            self.order.append(k)
            return
        # new insert
        self.dict[k] = v
        self.order.append(k)
        if len(self.dict) > self.cap:
            # evict LRU
            oldk = self.order.pop(0)
            self.dict.pop(oldk, None)


def _cands_for_weak_cached(conn: sqlite3.Connection, lru: _LRU, weak: int) -> List[Tuple[bytes, int]]:
    hit = lru.get(weak)
    if hit is not None:
        return hit
    cur = conn.cursor()
    rows = cur.execute("SELECT strong, off FROM blocks WHERE weak=?", (weak,)).fetchall()
    cands = [(bytes(r[0]), int(r[1])) for r in rows] if rows else []
    lru.put(weak, cands)
    return cands


# ==========================
# Matchers (in-memory / LRU)
# ==========================

def iter_matches_and_unmatched(new_bytes: bytes, idx_path: str, block_size: int) -> Iterator[Tuple]:
    """
    In-memory NEW scan; good for small buffers (kept for completeness).
    Emits ("copy", nstart, nlen, ostart) and ("lit", nstart, nlen).
    """
    conn = sqlite3.connect(idx_path)
    try:
        B_idx = _load_block_size(conn)
        if B_idx != block_size:
            raise ValueError(f"Index built with block_size={B_idx}, but passed block_size={block_size}")
        B = B_idx
        n = len(new_bytes)
        if n == 0:
            return

        if n < B:
            yield ("lit", 0, n)
            return

        window = bytearray(new_bytes[0:B])
        rc = RollingChecksum.from_block(window)
        pos = 0
        out_lit_start: Optional[int] = None

        # load all cands into a dict (for small cases)
        idx_map: Dict[int, List[Tuple[bytes, int]]] = {}
        cur = conn.cursor()
        for weak, strong, off in cur.execute("SELECT weak, strong, off FROM blocks"):
            lst = idx_map.get(weak)
            if lst is None:
                idx_map[weak] = [(bytes(strong), int(off))]
            else:
                lst.append((bytes(strong), int(off)))

        while pos <= n - B:
            weak = rc.value()
            cands = idx_map.get(weak, [])
            matched = False
            if cands:
                strong = hashlib.md5(window).digest()
                for s_bytes, old_off in cands:
                    if s_bytes == strong:
                        if out_lit_start is not None:
                            yield ("lit", out_lit_start, pos - out_lit_start)
                            out_lit_start = None
                        yield ("copy", pos, B, old_off)  # single-block copy
                        pos += B
                        if pos <= n - B:
                            window[:] = new_bytes[pos:pos+B]
                            rc = RollingChecksum.from_block(window)
                        matched = True
                        break
            if matched:
                continue

            if out_lit_start is None:
                out_lit_start = pos
            if pos + B < n:
                out_b = window[0]
                in_b = new_bytes[pos + B]
                rc.roll(out_b, in_b)
                window[:-1] = window[1:]
                window[-1] = in_b
                pos += 1
            else:
                pos += 1
                break

        if pos < n:
            if out_lit_start is None:
                out_lit_start = pos
            yield ("lit", out_lit_start, n - out_lit_start)
        else:
            if out_lit_start is not None and out_lit_start < pos:
                yield ("lit", out_lit_start, pos - out_lit_start)
    finally:
        conn.close()


def iter_matches_and_unmatched_stream_sqlcached(
    new_stream: io.BufferedReader,
    idx_path: str,
    block_size: int,
    cache_size: int = 65536
) -> Iterator[Tuple]:
    """
    Streaming NEW scan with an adjustable LRU cache for weak→candidates.
    Emits as it scans (no big lists in RAM).

    COPYs are single-block (length=B) to keep streaming simple:
      ("copy", new_start, B, old_start)

    LITERAL spans are coalesced automatically by the logic below:
      ("lit", new_start, new_len)
    """
    conn = sqlite3.connect(idx_path)
    try:
        B_idx = _load_block_size(conn)
        if B_idx != block_size:
            raise ValueError(f"Index built with block_size={B_idx}, but passed block_size={block_size}")
        B = block_size

        lru = _LRU(capacity=max(1024, cache_size))

        read = new_stream.read
        pos = 0
        out_lit_start: Optional[int] = None

        # Bootstrap first window
        window = bytearray()
        first = read(B)
        if not first:
            return
        window.extend(first)

        if len(window) < B:
            # whole NEW < B → single literal
            yield ("lit", 0, len(window))
            return

        rc = RollingChecksum.from_block(window)

        while True:
            weak = rc.value()
            cands = _cands_for_weak_cached(conn, lru, weak)
            matched = False
            if cands:
                strong = hashlib.md5(window).digest()
                for s_bytes, old_off in cands:
                    if s_bytes == strong:
                        # flush any literal before the copy
                        if out_lit_start is not None:
                            yield ("lit", out_lit_start, pos - out_lit_start)
                            out_lit_start = None
                        # emit a single-block copy
                        yield ("copy", pos, B, old_off)
                        # advance by B (block jump); refill the window
                        nxt = read(B)
                        pos += B
                        if not nxt:
                            matched = True
                            break
                        if len(nxt) < B:
                            window[:] = nxt + bytes(B - len(nxt))
                        else:
                            window[:] = nxt
                        rc = RollingChecksum.from_block(window)
                        matched = True
                        break

            if matched:
                # if EOF reached during matched path, exit
                if len(window) == 0:
                    break
                continue

            # No match → grow literal and slide by 1 byte
            if out_lit_start is None:
                out_lit_start = pos

            nxt1 = read(1)
            if not nxt1:
                pos += 1
                break
            out_b = window[0]
            in_b = nxt1[0]
            rc.roll(out_b, in_b)
            window[:-1] = window[1:]
            window[-1] = in_b
            pos += 1

        # Tail literal
        if out_lit_start is not None and pos > out_lit_start:
            yield ("lit", out_lit_start, pos - out_lit_start)
    finally:
        conn.close()
