# scripts/rsync_match.py
import sqlite3, hashlib, mmap, os
from contextlib import contextmanager

def md5(b: bytes) -> bytes:
    return hashlib.md5(b).digest()

class RollingChecksum:
    __slots__ = ("N","s1","s2")
    def __init__(self, block: bytes):
        N = len(block)
        s1 = 0; s2 = 0
        for i, b in enumerate(block, start=1):
            s1 += b
            s2 += (N - i + 1) * b
        self.N = N
        self.s1 = s1 & 0xffffffff
        self.s2 = s2 & 0xffffffff
    def roll(self, b_out: int, b_in: int):
        N = self.N
        s1 = (self.s1 - b_out + b_in) & 0xffffffff
        s2 = (self.s2 - N*b_out + s1) & 0xffffffff
        self.s1, self.s2 = s1, s2
    @property
    def weak(self) -> int:
        return ((self.s1 & 0xffff) << 16) | (self.s2 & 0xffff)

@contextmanager
def open_mmap(path):
    f = open(path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        yield mm
    finally:
        mm.close(); f.close()

def build_source_index_sqlite(src_path: str, block_size: int, db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("CREATE TABLE idx(weak INTEGER, pos INTEGER)")
    cur.execute("CREATE INDEX idx_weak ON idx(weak)")
    batch = []
    with open_mmap(src_path) as smm:
        n = len(smm)
        for pos in range(0, n - block_size + 1, block_size):
            blk = smm[pos:pos+block_size]
            w = RollingChecksum(blk).weak
            batch.append((w, pos))
            if len(batch) >= 100000:
                cur.executemany("INSERT INTO idx(weak,pos) VALUES(?,?)", batch)
                con.commit()
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO idx(weak,pos) VALUES(?,?)", batch)
            con.commit()
    cur.close(); con.close()

def iter_matches_and_unmatched(src_path: str, tgt_path: str, block_size: int, db_path: str):
    """
    Yields:
      ("unmatched", start, end)
      ("match", tgt_pos, src_pos, block_size)
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # 1) Load DISTINCT weak keys into RAM to avoid per-byte DB queries
    weak_keys = {row[0] for row in cur.execute("SELECT DISTINCT weak FROM idx")}
    with open_mmap(src_path) as smm, open_mmap(tgt_path) as tmm:
        n = len(tmm)
        if n < block_size:
            yield ("unmatched", 0, n)
            cur.close(); con.close()
            return

        i = 0
        window = bytearray(tmm[0:block_size])
        r = RollingChecksum(window)
        import hashlib
        win_md5 = hashlib.md5(bytes(window)).digest()

        last_emitted = 0
        while i <= n - block_size:
            weak = r.weak
            hit = False
            if weak in weak_keys:  # 2) Only query when possible match exists
                # Verify against candidate src positions
                for (src_pos,) in cur.execute("SELECT pos FROM idx WHERE weak=?", (weak,)):
                    if hashlib.md5(smm[src_pos:src_pos+block_size]).digest() == win_md5:
                        if last_emitted < i:
                            yield ("unmatched", last_emitted, i)
                        yield ("match", i, src_pos, block_size)
                        i += block_size
                        if i <= n - block_size:
                            window[:] = tmm[i:i+block_size]
                            r = RollingChecksum(window)
                            win_md5 = hashlib.md5(bytes(window)).digest()
                        last_emitted = i
                        hit = True
                        break

            if not hit:
                if i + block_size >= n:
                    break
                b_out = window[0]
                b_in  = tmm[i + block_size]
                r.roll(b_out, b_in)
                # roll the window and update md5 incrementally (simple recompute is fine)
                window.pop(0); window.append(b_in)
                win_md5 = hashlib.md5(bytes(window)).digest()
                i += 1

        if last_emitted < n:
            yield ("unmatched", last_emitted, n)
    cur.close(); con.close()

