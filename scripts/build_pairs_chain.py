#!/usr/bin/env python3
# Option A — Literal spans → tiled, changed-only pairs
import argparse, base64, json, os, tarfile, tempfile, uuid, time, sqlite3, hashlib
from typing import Iterator, Tuple, Optional, List
from multiprocessing import Pool, cpu_count
from bisect import bisect_right
from contextlib import nullcontext

from rsync_match import (
    build_source_index_sqlite_stream,
    iter_matches_and_unmatched_stream_sqlcached,  # streaming matcher with LRU
    RollingChecksum,
)

# =============== Utilities ===============

def _ensure_seekable(fobj):
    try:
        fobj.seek(0, os.SEEK_CUR)
        return fobj
    except Exception:
        tmp = tempfile.TemporaryFile()
        while True:
            buf = fobj.read(1 << 20)
            if not buf:
                break
            tmp.write(buf)
        tmp.seek(0)
        return tmp

def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"

# =============== Member matching (normalized) ===============

def _common_members(old_tar, new_tar):
    def norm(name: str) -> str:
        return name.split("/", 1)[1] if "/" in name else name

    old_map = {}
    for m in old_tar.getmembers():
        if m.isfile():
            old_map.setdefault(norm(m.name), (m.name, m.size))

    new_map = {}
    for m in new_tar.getmembers():
        if m.isfile():
            new_map.setdefault(norm(m.name), (m.name, m.size))

    common_norm = set(old_map.keys()) & set(new_map.keys())
    # (old_name, new_name, new_size)
    pairs = [(old_map[n][0], new_map[n][0], new_map[n][1]) for n in common_norm]
    # sort by NEW size asc so we get quick wins
    pairs.sort(key=lambda x: x[2])
    return pairs


# =============== OLD lookup (skip-if-present-anywhere) ===============

def make_old_block_lookup(idx_path: str, block_size: int, cache_cap: int = 8192):
    """
    Return lookup(new_blk[B]) -> old_off or None, using the per-member SQLite index.
    Pads to B before hashing (the index was built on padded B blocks).
    """
    conn = sqlite3.connect(idx_path, check_same_thread=False)
    cur = conn.cursor()

    from collections import OrderedDict
    lru = OrderedDict()

    def lookup(new_blk: bytes) -> Optional[int]:
        B = block_size
        blk = new_blk if len(new_blk) == B else (new_blk + bytes(B - len(new_blk)))
        weak = RollingChecksum.from_block(blk).value()
        strong = hashlib.md5(blk).digest()
        key = (weak, strong)
        hit = lru.get(key)
        if hit is not None:
            lru.move_to_end(key, last=True)
            return hit
        row = cur.execute("SELECT off FROM blocks WHERE weak=? AND strong=? LIMIT 1", (weak, strong)).fetchone()
        off = int(row[0]) if row else None
        lru[key] = off
        if len(lru) > cache_cap:
            lru.popitem(last=False)
        return off

    return lookup

# =============== Anchor helpers ===============

def build_anchor_maps(regions: List[Tuple]) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    From matcher regions, collect COPY anchors as (n_pos, o_pos) lists.
    Returns (copy_positions_in_new, anchors list).
    """
    anchors: List[Tuple[int,int]] = []
    for r in regions:
        if r[0] == "copy":
            _, nstart, nlen, ostart = r
            anchors.append((nstart, ostart))
    anchors.sort(key=lambda x: x[0])
    copy_positions = [a[0] for a in anchors]
    return copy_positions, anchors

def prev_anchor(n_pos: int, copy_positions: List[int], anchors: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    i = bisect_right(copy_positions, n_pos) - 1
    return anchors[i] if i >= 0 else None

def next_anchor(n_pos: int, copy_positions: List[int], anchors: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    i = bisect_right(copy_positions, n_pos)
    return anchors[i] if i < len(anchors) else None

def bi_anchor_map(n: int, pa: Optional[Tuple[int,int]], na: Optional[Tuple[int,int]]) -> Optional[int]:
    """
    Map NEW byte position n → OLD position using linear interpolation between nearest
    previous/next COPY anchors. Return None if not possible.
    """
    if not pa or not na:
        return None
    n0, o0 = pa; n1, o1 = na
    if n1 == n0:
        return None
    ratio = (n - n0) / float(n1 - n0)
    return int(round(o0 + ratio * (o1 - o0)))

# =============== Tiling emitter (Option A) ===============

def emit_changed_tiles(old_seek, new_seek,
                       regions: List[Tuple],
                       B: int, stride: int,
                       lookup_old_off) -> Iterator[Tuple[bytes, bytes]]:
    """
    Option A: iterate LITERAL spans, tile them into fixed B targets with given stride,
    skip tiles that exist anywhere in OLD (lookup hit), and map NEW→OLD with
    bi-anchored interpolation; fallback to same-offset if no anchors.

    Yields (old_chunk[B], new_chunk[B])
    """
    copy_positions, anchors = build_anchor_maps(regions)

    def read_exact(f, off: int, ln: int) -> bytes:
        if off < 0:
            pad = -off
            f.seek(0)
            data = f.read(max(0, ln - pad)) or b""
            if len(data) < ln - pad:
                data = data + bytes(ln - pad - len(data))
            return b"\x00" * pad + data
        f.seek(off)
        data = f.read(ln) or b""
        if len(data) < ln:
            data = data + bytes(ln - len(data))
        return data

    for r in regions:
        if r[0] != "lit":
            continue
        _, nstart, nlen = r
        end = nstart + nlen
        pos = nstart
        while pos < end:
            # NEW tile
            new_chunk = read_exact(new_seek, pos, B)

            # Skip if NEW tile exists anywhere in OLD (unchanged/shifted)
            if lookup_old_off(new_chunk) is not None:
                pos += stride
                continue

            # Map NEW position to OLD via anchors
            pa = prev_anchor(pos, copy_positions, anchors)
            na = next_anchor(pos, copy_positions, anchors)
            o_start = bi_anchor_map(pos, pa, na)
            if o_start is None:
                o_start = max(0, pos)  # same-offset fallback

            old_chunk = read_exact(old_seek, o_start, B)
            yield old_chunk, new_chunk

            pos += stride

# =============== Worker (per member) ===============

def _process_member(args):
    (old_tar_path, new_tar_path, old_member_name, new_member_name,
     B, stride, cache_size, tmp_dir) = args

    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        om = told.getmember(old_member_name)
        nm = tnew.getmember(new_member_name)
        if not om.isfile() or not nm.isfile() or nm.size == 0:
            return None

        of = told.extractfile(om)
        nf = tnew.extractfile(nm)
        if of is None or nf is None:
            return None

        of = _ensure_seekable(of)
        nf = _ensure_seekable(nf)

        # 1) Build OLD index (SQLite)
        idx_path = os.path.join(tmp_dir, f"idx_{uuid.uuid4().hex}.sqlite")
        of.seek(0)
        build_source_index_sqlite_stream(of, idx_path, block_size=B)
        of.seek(0)

        # 2) Stream NEW to get regions (COPY/LITERAL)
        nf.seek(0)
        regions = list(iter_matches_and_unmatched_stream_sqlcached(
            nf, idx_path, block_size=B, cache_size=cache_size
        ))

        # 3) Emit changed-only tiles
        lookup = make_old_block_lookup(idx_path, B)
        nf.seek(0); of.seek(0)

        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for old_chunk, new_chunk in emit_changed_tiles(
                of, nf, regions, B=B, stride=stride, lookup_old_off=lookup
            ):
                rec = {
                    "old": base64.b64encode(old_chunk).decode("ascii"),
                    "new": base64.b64encode(new_chunk).decode("ascii"),
                    "member_old": old_member_name,
                    "member_new": new_member_name,
                }
                fout.write(json.dumps(rec) + "\n")
                written += 1

        # Clean index early
        try: os.remove(idx_path)
        except Exception: pass

        return (out_path, written)

# =============== Main ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_size", type=int, default=4096, help="Tile size B (and rsync index block)")
    ap.add_argument("--stride", type=int, default=4096, help="Tile stride (B for no overlap, B//2 for 50% overlap)")
    ap.add_argument("--out", required=True, help="Output JSONL of {old,new} base64 pairs (B bytes each)")
    ap.add_argument("--chain", nargs=2, required=True, help="Exactly two tar paths: OLD.tar NEW.tar")
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--limit", type=int, default=0, help="(Debug) limit number of members processed")
    ap.add_argument("--progress_every", type=int, default=500, help="Print progress every N members")
    ap.add_argument("--cache_size", type=int, default=65536, help="LRU size for weak→candidates in streaming matcher")
    ap.add_argument("--pool_chunksize", type=int, default=64, help="chunksize for imap_unordered batching")
    ap.add_argument("--tqdm", action="store_true", help="Show a tqdm progress bar")
    args = ap.parse_args()

    B = args.block_size
    stride = args.stride if args.stride > 0 else B

    old_tar_path, new_tar_path = args.chain

    t0 = time.time()
    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        pairs = _common_members(told, tnew)

    if args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"[info] limiting to first {args.limit} members")

    total_members = len(pairs)
    print(f"Matched {total_members} common members after normalizing top-level dirs")

    tmp_dir = tempfile.mkdtemp(prefix="pairs_tmp_")
    print(f"[info] writing per-worker shards under: {tmp_dir}")

    tasks = [
        (old_tar_path, new_tar_path, old_name, new_name, B, stride, args.cache_size, tmp_dir)
        for (old_name, new_name) in pairs
    ]

    def _status(done: int, total: int):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        remain = total - done
        eta = remain / rate if rate > 0 else float("inf")
        pct = (done / total * 100.0) if total else 100.0
        eta_txt = "∞" if eta == float("inf") else _fmt_dur(eta)
        print(f"[progress] {done}/{total} ({pct:.1f}%) | elapsed={_fmt_dur(elapsed)} | eta={eta_txt} | ~{rate:.2f} members/s")

    total_pairs = 0
    progress = 0
    with Pool(processes=args.jobs) as pool:
        iterator = pool.imap_unordered(_process_member, tasks, chunksize=args.pool_chunksize)

        # pick a context manager: tqdm or no-op
        bar_cm = nullcontext()
        if args.tqdm:
            try:
                from tqdm import tqdm
                bar_cm = tqdm(total=total_members, unit="file", dynamic_ncols=True, desc="Members")
            except Exception:
                pass  # fallback to prints if tqdm unavailable

        # drive the iterator and update either tqdm or periodic prints
        if args.tqdm and isinstance(bar_cm, object) and hasattr(bar_cm, "update"):
            with bar_cm as pbar:
                for res in iterator:
                    progress += 1
                    if res is not None:
                        out_path, written = res
                        total_pairs += written
                    pbar.update(1)
        else:
            # original print-based progress
            for res in iterator:
                progress += 1
                if res is not None:
                    out_path, written = res
                    total_pairs += written
                if progress % args.progress_every == 0:
                    _status(progress, total_members)

   # _status(progress, total_members)
    print(f"[info] concatenating shards → {args.out}")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fout:
        for fname in os.listdir(tmp_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(tmp_dir, fname), "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    # Cleanup
    for fname in os.listdir(tmp_dir):
        try: os.remove(os.path.join(tmp_dir, fname))
        except Exception: pass
    try: os.rmdir(tmp_dir)
    except Exception: pass

    print(f"[done] Wrote {total_pairs} changed-only pairs (B={B}, stride={stride}, jobs={args.jobs}) → {args.out}")
    print(f"[time] total elapsed = {_fmt_dur(time.time() - t0)}")

if __name__ == "__main__":
    main()


