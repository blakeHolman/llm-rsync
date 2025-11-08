#!/usr/bin/env python3
# scripts/build_pairs_chain.py
# Option A — Literal spans → tiled, changed-only pairs (with valid-lengths for masking)

import argparse, base64, json, os, tarfile, tempfile, uuid, time, sqlite3, hashlib
from typing import Iterator, Tuple, Optional, List
from multiprocessing import Pool, cpu_count
from bisect import bisect_right
from contextlib import nullcontext

from rsync_match import (
    build_source_index_sqlite_stream,
    iter_matches_and_unmatched_stream_sqlcached,  # streaming matcher with LRU cache
    RollingChecksum,
)

# ===================== Utilities =====================

def _ensure_seekable(fobj):
    """Return a seekable file-like object; spill to a temp file if needed."""
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

def _read_exact_with_pad(f, off: int, ln: int) -> bytes:
    """
    Read [off, off+ln) from seekable 'f'. If off<0 or past EOF, pad with zeros to ln.
    """
    if off < 0:
        pad = -off
        f.seek(0)
        data = f.read(max(0, ln - pad)) or b""
        if len(data) < ln - pad:
            data = data + bytes((ln - pad) - len(data))
        return b"\x00" * pad + data
    f.seek(off)
    data = f.read(ln) or b""
    if len(data) < ln:
        data = data + bytes(ln - len(data))
    return data

# =========== Member matching (normalized names) ===========

def _common_members(old_tar: tarfile.TarFile, new_tar: tarfile.TarFile) -> List[Tuple[str, str, int]]:
    """
    Return list of (old_member_name, new_member_name, new_size), matching after
    stripping the top-level directory prefix (e.g., 'linux-6.1.154/').
    Sorted by NEW size ascending so we get quick completions first.
    """
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
    pairs = [(old_map[n][0], new_map[n][0], int(new_map[n][1])) for n in common_norm]
    pairs.sort(key=lambda x: x[2])  # NEW size asc
    return pairs

# ======== OLD lookup (skip-if-present-anywhere in OLD) ========

def make_old_block_lookup(idx_path: str, block_size: int, cache_cap: int = 8192):
    """
    Return lookup(new_blk[B]) -> old_off or None, using the per-member SQLite index.
    Pads to B before hashing (the index is built on padded B blocks).
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

# ===================== Anchor helpers =====================

def build_anchor_maps(regions: List[Tuple]) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    From matcher regions, collect COPY anchors as (n_pos, o_pos) along NEW.
    Returns (copy_positions_in_new, anchors list).
    """
    anchors: List[Tuple[int,int]] = []
    for r in regions:
        if r[0] == "copy":
            _, nstart, nlen, ostart = r
            anchors.append((int(nstart), int(ostart)))
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

# ============== Tiling emitter (Option A, with valid lengths) ==============

def emit_changed_tiles(old_seek,
                       new_seek,
                       regions: List[Tuple],
                       B: int,
                       stride: int,
                       lookup_old_off,
                       old_size: int,
                       new_size: int
                       ) -> Iterator[Tuple[bytes, bytes, int, int]]:
    """
    Iterate LITERAL spans, tile them into fixed B targets with 'stride',
    skip tiles that exist anywhere in OLD (lookup hit),
    map NEW→OLD via COPY anchors (bi-anchored) else same-offset fallback.

    Yields (old_chunk[B], new_chunk[B], old_valid, new_valid)
    """
    copy_positions, anchors = build_anchor_maps(regions)

    for r in regions:
        if r[0] != "lit":
            continue
        _, nstart, nlen = r
        nstart = int(nstart); nlen = int(nlen)
        end = nstart + nlen
        pos = nstart

        while pos < end:
            # NEW tile: real NEW bytes before padding
            take = min(B, end - pos)
            new_valid = take
            new_chunk = _read_exact_with_pad(new_seek, pos, B)

            # Skip if NEW tile exists anywhere in OLD (unchanged or shifted copy)
            if lookup_old_off(new_chunk) is not None:
                pos += stride
                continue

            # Map NEW position to OLD via anchors; fallback to same-offset
            pa = prev_anchor(pos, copy_positions, anchors)
            na = next_anchor(pos, copy_positions, anchors)
            o_start = bi_anchor_map(pos, pa, na)
            if o_start is None:
                o_start = max(0, pos)

            # OLD tile + valid length (cap by file size)
            if o_start >= old_size:
                old_valid = 0
            else:
                old_valid = min(B, old_size - o_start)
            old_chunk = _read_exact_with_pad(old_seek, o_start, B)

            yield old_chunk, new_chunk, int(old_valid), int(new_valid)
            pos += stride

# ===================== Worker (per member) =====================

def _process_member(args):
    (
        old_tar_path, new_tar_path,
        old_member_name, new_member_name,
        B, stride, cache_size, tmp_dir
    ) = args

    # print a tiny heartbeat for long members
    #print(f"[worker] starting: {new_member_name}", flush=True)

    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        om = told.getmember(old_member_name)
        nm = tnew.getmember(new_member_name)
        # Only process regular files with size > 0 in NEW
        if not om.isfile() or not nm.isfile() or nm.size == 0:
            print(f"[worker] skip: {new_member_name}", flush=True)
            return None

        old_size = int(om.size)
        new_size = int(nm.size)

        of = told.extractfile(om)
        nf = tnew.extractfile(nm)
        if of is None or nf is None:
            print(f"[worker] no streams: {new_member_name}", flush=True)
            return None

        of = _ensure_seekable(of)
        nf = _ensure_seekable(nf)

        # 1) Build OLD per-member index (SQLite)
        idx_path = os.path.join(tmp_dir, f"idx_{uuid.uuid4().hex}.sqlite")
        of.seek(0)
        build_source_index_sqlite_stream(of, idx_path, block_size=B)
        of.seek(0)

        # 2) Stream NEW to get COPY/LITERAL regions
        nf.seek(0)
        regions = list(
            iter_matches_and_unmatched_stream_sqlcached(
                nf, idx_path, block_size=B, cache_size=cache_size
            )
        )

        # 3) Emit changed-only tiles with valid lengths
        lookup = make_old_block_lookup(idx_path, B, cache_cap=8192)
        nf.seek(0); of.seek(0)

        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for old_chunk, new_chunk, old_valid, new_valid in emit_changed_tiles(
                of, nf, regions, B=B, stride=stride, lookup_old_off=lookup,
                old_size=old_size, new_size=new_size
            ):
                rec = {
                    "old": base64.b64encode(old_chunk).decode("ascii"),
                    "new": base64.b64encode(new_chunk).decode("ascii"),
                    "old_valid": int(old_valid),
                    "new_valid": int(new_valid),
                    "block": int(B),
                    "stride": int(stride),
                    "member_old": old_member_name,
                    "member_new": new_member_name,
                }
                fout.write(json.dumps(rec) + "\n")
                written += 1

        # Clean index early
        try:
            os.remove(idx_path)
        except Exception:
            pass

    #print(f"[worker] done: {new_member_name} (pairs={written})", flush=True)
    return (out_path, written)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Build OLD→NEW training pairs from rsync-style literal spans (Option A)")
    ap.add_argument("--block_size", type=int, default=4096, help="Tile size B (and rsync index block)")
    ap.add_argument("--stride", type=int, default=4096, help="Tile stride (B=no overlap, B//2=50% overlap)")
    ap.add_argument("--out", required=True, help="Output JSONL of {old,new,old_valid,new_valid,...} pairs")
    ap.add_argument("--chain", nargs=2, required=True, help="Exactly two tar paths: OLD.tar NEW.tar")
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()-1), help="Parallel workers")
    ap.add_argument("--limit", type=int, default=0, help="(Debug) limit number of members processed")
    ap.add_argument("--progress_every", type=int, default=500, help="Print progress every N members (when --tqdm not used)")
    ap.add_argument("--cache_size", type=int, default=65536, help="LRU size for weak→candidates in streaming matcher")
    ap.add_argument("--pool_chunksize", type=int, default=1, help="chunksize for imap_unordered batching (1 = most responsive)")
    ap.add_argument("--tqdm", action="store_true", help="Show a tqdm progress bar")
    args = ap.parse_args()

    B = int(args.block_size)
    stride = int(args.stride) if args.stride > 0 else B
    old_tar_path, new_tar_path = args.chain

    t0 = time.time()
    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        pairs = _common_members(told, tnew)  # (old_name, new_name, new_size)

    if args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"[info] limiting to first {args.limit} members")

    total_members = len(pairs)
    print(f"Matched {total_members} common members after normalizing top-level dirs")

    tmp_dir = tempfile.mkdtemp(prefix="pairs_tmp_")
    print(f"[info] writing per-worker shards under: {tmp_dir}")

    tasks = [
        (old_tar_path, new_tar_path, old_name, new_name, B, stride, args.cache_size, tmp_dir)
        for (old_name, new_name, _sz) in pairs
    ]

    # Progress helpers
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

        # Optional tqdm progress bar
        bar_cm = nullcontext()
        pbar = None
        if args.tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_members, unit="file", dynamic_ncols=True, desc="Members")
            except Exception:
                pbar = None

        if pbar is not None:
            with pbar:
                for res in iterator:
                    progress += 1
                    if res is not None:
                        out_path, written = res
                        total_pairs += written
                    pbar.update(1)
        else:
            for res in iterator:
                progress += 1
                if res is not None:
                    out_path, written = res
                    total_pairs += written
                if progress % args.progress_every == 0:
                    _status(progress, total_members)

    # Final status
    if pbar is None:
        _status(progress, total_members)
    print(f"[info] concatenating shards → {args.out}")

    # Ensure out dir
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Concatenate per-worker shards
    with open(args.out, "w", encoding="utf-8") as fout:
        for fname in os.listdir(tmp_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(tmp_dir, fname), "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    # Cleanup temp shards
    for fname in os.listdir(tmp_dir):
        try:
            os.remove(os.path.join(tmp_dir, fname))
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(f"[done] Wrote {total_pairs} changed-only pairs (B={B}, stride={stride}, jobs={args.jobs}) → {args.out}")
    print(f"[time] total elapsed = {_fmt_dur(time.time() - t0)}")

if __name__ == "__main__":
    main()

