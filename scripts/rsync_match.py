#!/usr/bin/env python3
"""
Minimal rsync-style matcher with streaming:

Build OLD index (non-overlapping blocks, size B):
  - build_source_index_sqlite(old_bytes, idx_path, block_size)
  - build_source_index_sqlite_stream(old_stream, idx_path, block_size)

Scan NEW to emit regions that fully cover NEW:
  - iter_matches_and_unmatched(new_bytes, idx_path, block_size)          # in-memory NEW
  - iter_matches_and_unmatched_stream_sqlcached(new_stream, idx, B, ...) # NEW stream, SQLite+LRU cache
  - iter_matches_and_unmatched_stream_with_map(new_stream, B, idx_map)   # NEW stream, in-RAM idx_map

Optional helpers:
  - build_index_map_from_stream(old_stream, block_size)  # small/medium OLD → in-RAM weak→[(md5, off)]
  - register_old_bytes_for_index(idx_path, old_bytes)    # enables greedy extend for in-memory NEW scans

Weak checksum: Tridgell rolling (a,b); Strong checksum: MD5 (digest bytes).
COPY:   ("copy", new_start, new_len, old_start)
LITERAL:("lit",  new_start, new_len)
"""
from __future__ import annotations
import hashlib
import io
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

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
        # Classic rsync: (a & 0xffff) | (b << 16)
        return ((self.b & 0xFFFF) << 16) | (self.a & 0xFFFF)

    def roll(self, out_byte: int, in_byte: int) -> None:
        # Slide window by one: remove out_byte, add in_byte.
        a = (self.a - out_byte + in_byte) & 0xFFFF_FFFF
        b = (self.b - self.n * out_byte + a) & 0xFFFF_FFFF
        self.a = a
        self.b = b

# ============================
# SQLite schema & index builds
# ============================
def _init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Bulk-friendly pragmas; flip to safer settings if you persist across runs.
    cur.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-1048576;
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
            strong BLOB,
            off    INTEGER
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_weak ON blocks(weak);")
    conn.commit()

def build_source_index_sqlite(old_bytes: bytes, idx_path: str, block_size: int) -> None:
    """
    Build OLD index from in-memory bytes in non-overlapping blocks of size `block_size`.
    Stores weak (rolling checksum), strong (MD5 digest), and block offset in SQLite.
    """
    conn = sqlite3.connect(idx_path)
    try:
        _init_db(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM blocks;")
        cur.execute("DELETE FROM meta;")
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('block_size', ?)", (str(block_size),))
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('old_len', ?)", (str(len(old_bytes)),))

        B = block_size
        off = 0
        batch = []
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
    Build OLD index from a stream, reading sequentially in non-overlapping blocks of size `block_size`.
    """
    conn = sqlite3.connect(idx_path)
    try:
        _init_db(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM blocks;")
        cur.execute("DELETE FROM meta;")
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('block_size', ?)", (str(block_size),))

        B = block_size
        total = 0
        batch = []
        read = old_stream.read

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
        cur.execute("INSERT OR REPLACE INTO meta(key,val) VALUES('old_len', ?)", (str(total),))
        conn.commit()
    finally:
        conn.close()

# =============================
# Helpers to read/use the index
# =============================
def _load_block_size(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    row = cur.execute("SELECT val FROM meta WHERE key='block_size'").fetchone()
    if not row:
        raise ValueError("Index missing block_size meta")
    return int(row[0])

def _load_index_map(conn: sqlite3.Connection) -> Dict[int, List[Tuple[bytes, int]]]:
    """
    Load full OLD index into RAM:
        weak:int -> [(strong_digest:bytes, off:int), ...]
    Use for small/medium files (fastest).
    """
    cur = conn.cursor()
    m: Dict[int, List[Tuple[bytes, int]]] = {}
    for weak, strong, off in cur.execute("SELECT weak, strong, off FROM blocks"):
        lst = m.get(weak)
        if lst is None:
            m[weak] = [(bytes(strong), int(off))]
        else:
            lst.append((bytes(strong), int(off)))
    return m

def build_index_map_from_stream(old_stream: io.BufferedReader, block_size: int) -> Dict[int, List[Tuple[bytes, int]]]:
    """
    Sequentially read OLD in non-overlapping blocks and build an in-RAM map:
        weak:int -> [(md5_digest:bytes, off:int), ...]
    Use only for small/medium members (large files with tiny B make this huge).
    """
    B = block_size
    idx: Dict[int, List[Tuple[bytes, int]]] = {}
    off = 0
    read = old_stream.read
    while True:
        blk = read(B)
        if not blk:
            break
        if len(blk) < B:
            blk = blk + bytes(B - len(blk))
        rc = RollingChecksum.from_block(blk)
        weak = rc.value()
        strong = hashlib.md5(blk).digest()
        lst = idx.get(weak)
        if lst is None:
            idx[weak] = [(strong, off)]
        else:
            lst.append((strong, off))
        off += B
    return idx

# ===========================
# Optional greedy extend hook
# ===========================
_GLOBAL_OLD_BYTES: Dict[str, bytes] = {}

def register_old_bytes_for_index(idx_path: str, old_bytes: bytes) -> None:
    """
    Supply old_bytes for the given index path to enable greedy multi-block extension
    in in-memory NEW scans (iter_matches_and_unmatched).
    """
    _GLOBAL_OLD_BYTES[idx_path] = old_bytes

def _try_extend_run(new_bytes: bytes, old_bytes: memoryview, new_pos: int, old_pos: int, B: int) -> int:
    """
    After a single-block match at (new_pos, old_pos), greedily extend by whole blocks.
    """
    total = B
    np = new_pos + B
    op = old_pos + B
    n = len(new_bytes)
    while np < n:
        nblk = new_bytes[np: np+B]
        if len(nblk) < B:
            break
        oblk = old_bytes[op: op+B].tobytes()
        if nblk != oblk:
            break
        total += B
        np += B
        op += B
    return total

# ===============================
# In-memory NEW matcher (fastest)
# ===============================
def iter_matches_and_unmatched(new_bytes: bytes, idx_path: str, block_size: int) -> Iterator[Tuple]:
    """
    Scan NEW (bytes) using SQLite index; loads weak→cands fully into RAM (fast).
    Yields ("copy", new_start, new_len, old_start) and ("lit", new_start, new_len).
    Greedy extend is enabled if register_old_bytes_for_index(...) was called.
    """
    conn = sqlite3.connect(idx_path)
    try:
        B_idx = _load_block_size(conn)
        if B_idx != block_size:
            raise ValueError(f"Index built with block_size={B_idx}, but caller passed block_size={block_size}")
        B = B_idx

        idx_map = _load_index_map(conn)
        n = len(new_bytes)
        if n == 0:
            return

        out_lit_start = None
        pos = 0

        if n < B:
            yield ("lit", 0, n)
            return

        window = bytearray(new_bytes[0:B])
        rc = RollingChecksum.from_block(window)

        old_mem = None
        obuf = _GLOBAL_OLD_BYTES.get(idx_path)
        if obuf is not None:
            old_mem = memoryview(obuf)

        def extend_len(npos: int, opos: int) -> int:
            if old_mem is None:
                return B
            return _try_extend_run(new_bytes, old_mem, npos, opos, B)

        while pos <= n - B:
            weak = rc.value()
            cands = idx_map.get(weak)
            matched = False
            if cands:
                strong = hashlib.md5(window).digest()
                for s_bytes, old_off in cands:
                    if s_bytes == strong:
                        if out_lit_start is not None:
                            yield ("lit", out_lit_start, pos - out_lit_start)
                            out_lit_start = None
                        mlen = extend_len(pos, old_off)
                        yield ("copy", pos, mlen, old_off)
                        pos += mlen
                        if pos <= n - B:
                            window[:] = new_bytes[pos:pos+B]
                            rc = RollingChecksum.from_block(window)
                        else:
                            break
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

# ===========================================
# Streaming NEW: SQLite-backed with LRU cache
# ===========================================
def _make_weak_lookup(conn: sqlite3.Connection, cache_size: int = 65536):
    cur = conn.cursor()
    @lru_cache(maxsize=cache_size)
    def lookup(weak: int) -> List[Tuple[bytes, int]]:
        rows = cur.execute("SELECT strong, off FROM blocks WHERE weak=?", (weak,)).fetchall()
        return [(bytes(r[0]), int(r[1])) for r in rows]
    return lookup

def iter_matches_and_unmatched_stream_sqlcached(new_stream: io.BufferedReader, idx_path: str, block_size: int, cache_size: int = 65536):
    """
    Scan NEW as a stream using SQLite + LRU cache for weak→candidates.
    Emits single-block COPYs and LITERAL regions. Low RAM; far fewer SQL hits.
    """
    conn = sqlite3.connect(idx_path)
    try:
        B_idx = _load_block_size(conn)
        if B_idx != block_size:
            raise ValueError(f"Index built with block_size={B_idx}, but caller passed block_size={block_size}")
        B = block_size

        candidates = _make_weak_lookup(conn, cache_size=cache_size)

        read = new_stream.read
        window = bytearray()
        chunk = read(B)
        if not chunk:
            return
        window.extend(chunk)
        pos = 0
        out_lit_start = None

        if len(window) < B:
            yield ("lit", 0, len(window))
            return

        rc = RollingChecksum.from_block(window)

        while True:
            weak = rc.value()
            cands = candidates(weak)
            matched = False
            if cands:
                strong = hashlib.md5(window).digest()
                for s_bytes, old_off in cands:
                    if s_bytes == strong:
                        if out_lit_start is not None:
                            yield ("lit", out_lit_start, pos - out_lit_start)
                            out_lit_start = None
                        yield ("copy", pos, B, old_off)
                        next_block = read(B)
                        pos += B
                        if not next_block:
                            matched = True
                            break
                        if len(next_block) < B:
                            window[:] = next_block + bytes(B - len(next_block))
                        else:
                            window[:] = next_block
                        rc = RollingChecksum.from_block(window)
                        matched = True
                        break

            if matched:
                continue

            if out_lit_start is None:
                out_lit_start = pos
            nxt = read(1)
            if not nxt:
                pos += 1
                break
            out_b = window[0]
            in_b = nxt[0]
            rc.roll(out_b, in_b)
            window[:-1] = window[1:]
            window[-1] = in_b
            pos += 1

        if out_lit_start is not None and pos > out_lit_start:
            yield ("lit", out_lit_start, pos - out_lit_start)
    finally:
        conn.close()

# =================================================
# Streaming NEW: in-RAM map (fastest; higher memory)
# =================================================
def iter_matches_and_unmatched_stream_with_map(new_stream: io.BufferedReader, block_size: int, idx_map: Dict[int, List[Tuple[bytes, int]]]):
    """
    Scan NEW as a stream using a pre-built in-RAM OLD index map:
        weak:int -> [(md5_digest:bytes, old_off:int), ...]
    Emits single-block COPYs and LITERAL regions. Fastest, but RAM grows with OLD size/B.
    """
    B = block_size
    read = new_stream.read
    window = bytearray()
    chunk = read(B)
    if not chunk:
        return
    window.extend(chunk)
    pos = 0
    out_lit_start = None

    if len(window) < B:
        yield ("lit", 0, len(window))
        return

    rc = RollingChecksum.from_block(window)

    def flush_lit(upto_excl: int):
        nonlocal out_lit_start
        if out_lit_start is not None and upto_excl > out_lit_start:
            yield ("lit", out_lit_start, upto_excl - out_lit_start)
        out_lit_start = None

    while True:
        weak = rc.value()
        cands = idx_map.get(weak)
        matched = False
        if cands:
            strong = hashlib.md5(window).digest()
            for s_bytes, old_off in cands:
                if s_bytes == strong:
                    if out_lit_start is not None:
                        yield ("lit", out_lit_start, pos - out_lit_start)
                        out_lit_start = None
                    yield ("copy", pos, B, old_off)
                    next_block = read(B)
                    pos += B
                    if not next_block:
                        matched = True
                        break
                    if len(next_block) < B:
                        window[:] = next_block + bytes(B - len(next_block))
                    else:
                        window[:] = next_block
                    rc = RollingChecksum.from_block(window)
                    matched = True
                    break

        if matched:
            continue

        if out_lit_start is None:
            out_lit_start = pos
        nxt = read(1)
        if not nxt:
            pos += 1
            break
        out_b = window[0]
        in_b = nxt[0]
        rc.roll(out_b, in_b)
        window[:-1] = window[1:]
        window[-1] = in_b
        pos += 1

    if out_lit_start is not None and pos > out_lit_start:
        yield ("lit", out_lit_start, pos - out_lit_start)
