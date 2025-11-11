#!/usr/bin/env python3
# scripts/build_pairs_chain.py
# Option A — Literal spans → tiled, changed-only pairs (with valid-lengths for masking)

import argparse, base64, json, os, tarfile, tempfile, uuid, time, sqlite3, hashlib
from typing import Iterator, Tuple, Optional, List
from bisect import bisect_right
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp

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

def _common_members(old_tar: tarfile.TarFile, new_tar: tarfile.TarFile):
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
    conn = sqlite3.connect(idx_path, timeout=60, check_same_thread=False)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=OFF;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
        cur.execute("PRAGMA busy_timeout=60000;")
    except Exception:
        pass

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

# ============== Single-pass tiling emitter (constant memory) ==============

def emit_changed_tiles_singlepass(old_seek,
                                  new_seek,
                                  idx_path: str,
                                  B: int,
                                  stride: int,
                                  cache_size: int,
                                  old_size: int,
                                  new_size: int) -> Iterator[Tuple[bytes, bytes, int, int]]:
    """
    Single-pass over the streaming matcher:
      - Track only the LAST copy anchor (n_prev, o_prev).
      - For each 'lit' span, tile it; for each tile, estimate OLD start as:
          o_start = (o_prev + (pos - n_prev)) if we have a prev anchor,
                    else same-offset fallback o_start = pos (clamped).
      - Skip tiles that exist anywhere in OLD via per-member index lookup.
    This uses O(1) memory regardless of member size.
    """
    lookup = make_old_block_lookup(idx_path, B, cache_cap=8192)

    n_prev = None  # last copy NEW offset
    o_prev = None  # last copy OLD offset

    new_seek.seek(0)
    for rec in iter_matches_and_unmatched_stream_sqlcached(new_seek, idx_path, block_size=B, cache_size=cache_size):
        kind = rec[0]

        if kind == "copy":
            # ("copy", nstart, nlen, ostart)
            _, nstart, nlen, ostart = rec
            n_prev = int(nstart)
            o_prev = int(ostart)
            continue

        if kind != "lit":
            continue

        # ("lit", nstart, nlen)
        _, nstart, nlen = rec
        nstart = int(nstart); nlen = int(nlen)
        end = nstart + nlen
        pos = nstart

        while pos < end:
            take = min(B, end - pos)
            new_valid = take
            new_chunk = _read_exact_with_pad(new_seek, pos, B)

            # Skip if NEW tile exists anywhere in OLD (unchanged or shifted copy)
            if lookup(new_chunk) is not None:
                pos += stride
                continue

            # Map NEW position to OLD using last anchor; else same-offset fallback
            if n_prev is not None and o_prev is not None:
                o_start = o_prev + (pos - n_prev)
            else:
                o_start = pos

            if o_start < 0:
                o_start = 0
            if o_start >= old_size:
                old_valid = 0
            else:
                old_valid = min(B, old_size - o_start)

            old_chunk = _read_exact_with_pad(old_seek, o_start, B)

            yield old_chunk, new_chunk, int(old_valid), int(new_valid)
            pos += stride

# ============== Lossless literal fallback for a member ==============

def emit_literal_member_pairs(old_seek, new_seek, B: int, stride: int,
                              old_size: int, new_size: int) -> Iterator[Tuple[bytes, bytes, int, int]]:
    """
    Emit literal tiles without any matching (lossless: preserves all bytes via valid lengths).
    Useful as a timeout/oom fallback.
    """
    pos = 0
    while pos < new_size:
        take = min(B, new_size - pos)
        new_chunk = _read_exact_with_pad(new_seek, pos, B)
        o_start = pos
        if o_start >= old_size:
            old_valid = 0
        else:
            old_valid = min(B, old_size - o_start)
        old_chunk = _read_exact_with_pad(old_seek, o_start, B)
        yield old_chunk, new_chunk, int(old_valid), int(take)
        pos += stride

# ===================== Worker (per member) =====================

def _process_member(args):
    (
        old_tar_path, new_tar_path,
        old_member_name, new_member_name,
        B, stride, cache_size, tmp_dir,
        member_timeout_s, big_literal_fallback, big_threshold
    ) = args

    t_start = time.time()

    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        om = told.getmember(old_member_name)
        nm = tnew.getmember(new_member_name)
        # Only process regular files with size > 0 in NEW
        if not om.isfile() or not nm.isfile() or nm.size == 0:
            return None

        old_size = int(om.size)
        new_size = int(nm.size)

        of = told.extractfile(om)
        nf = tnew.extractfile(nm)
        if of is None or nf is None:
            return None

        of = _ensure_seekable(of)
        nf = _ensure_seekable(nf)

        # Tiny-file fast path: avoid DB setup for very small members
        if new_size <= B:
            of.seek(0); nf.seek(0)
            old_chunk = _read_exact_with_pad(of, 0, B)
            new_chunk = _read_exact_with_pad(nf, 0, B)
            out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
            with open(out_path, "w", encoding="utf-8") as fout:
                rec = {
                    "old": base64.b64encode(old_chunk).decode("ascii"),
                    "new": base64.b64encode(new_chunk).decode("ascii"),
                    "old_valid": int(min(B, old_size)),
                    "new_valid": int(min(B, new_size)),
                    "block": int(B),
                    "stride": int(stride),
                    "member_old": old_member_name,
                    "member_new": new_member_name,
                }
                fout.write(json.dumps(rec) + "\n")
            return (out_path, 1)

        # Optional: if "big literal fallback" is enabled for very large members,
        # skip matching completely to avoid RAM spikes.
        if big_literal_fallback and new_size >= big_threshold:
            nf.seek(0); of.seek(0)
            out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
            written = 0
            with open(out_path, "w", encoding="utf-8") as fout:
                for old_chunk, new_chunk, old_valid, new_valid in emit_literal_member_pairs(
                    of, nf, B=B, stride=stride, old_size=old_size, new_size=new_size
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
            return (out_path, written)

        # Build OLD per-member index (SQLite)
        idx_path = os.path.join(tmp_dir, f"idx_{uuid.uuid4().hex}.sqlite")
        of.seek(0)
        build_source_index_sqlite_stream(of, idx_path, block_size=B)
        of.seek(0)

        # Emit changed-only tiles with valid lengths (single-pass; constant memory)
        nf.seek(0); of.seek(0)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
        written = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for old_chunk, new_chunk, old_valid, new_valid in emit_changed_tiles_singlepass(
                of, nf, idx_path, B=B, stride=stride, cache_size=cache_size,
                old_size=old_size, new_size=new_size
            ):
                # Per-member timeout: bail to literal tiling if taking too long
                if member_timeout_s > 0 and (time.time() - t_start) > member_timeout_s:
                    # Finish remaining bytes with lossless literal fallback
                    nf_pos = nf.seek(0, os.SEEK_CUR)  # best-effort; some fileobjs don't tell pos
                    nf.seek(nf_pos if isinstance(nf_pos, int) else 0)
                    for o2, n2, ov2, nv2 in emit_literal_member_pairs(
                        of, nf, B=B, stride=stride, old_size=old_size, new_size=new_size
                    ):
                        rec = {
                            "old": base64.b64encode(o2).decode("ascii"),
                            "new": base64.b64encode(n2).decode("ascii"),
                            "old_valid": int(ov2),
                            "new_valid": int(nv2),
                            "block": int(B),
                            "stride": int(stride),
                            "member_old": old_member_name,
                            "member_new": new_member_name,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        written += 1
                    break

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

    return (out_path, written)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Build OLD→NEW training pairs from rsync-style literal spans (Option A)")
    ap.add_argument("--block_size", type=int, default=8192, help="Tile size B (and rsync index block)")
    ap.add_argument("--stride", type=int, default=8192, help="Tile stride (B=no overlap, B//2=50% overlap)")
    ap.add_argument("--out", required=True, help="Output JSONL of {old,new,old_valid,new_valid,...} pairs")
    ap.add_argument("--chain", nargs=2, required=True, help="Exactly two tar paths: OLD.tar NEW.tar")
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count()-1), help="Parallel workers")
    ap.add_argument("--pool_chunksize", type=int, default=1, help="Task chunking (1 = best balance)")
    ap.add_argument("--cache_size", type=int, default=8192, help="LRU size for streaming matcher")
    ap.add_argument("--tqdm", action="store_true", help="Show a tqdm progress bar")
    ap.add_argument("--limit", type=int, default=0, help="(Debug) limit number of members")
    ap.add_argument("--member_timeout", type=int, default=180, help="Seconds before falling back to literal tiling for a member (0=off)")
    ap.add_argument("--big_literal_fallback", action="store_true", help="Process very large members in literal mode only")
    ap.add_argument("--big_threshold", type=int, default=128*1024*1024, help="Bytes; members >= threshold use literal fallback when enabled")
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
        (old_tar_path, new_tar_path, old_name, new_name, B, stride, args.cache_size, tmp_dir,
         int(args.member_timeout), bool(args.big_literal_fallback), int(args.big_threshold))
        for (old_name, new_name, _sz) in pairs
    ]

    # Run with ProcessPoolExecutor to support timeouts & better isolation
    total_pairs = 0
    shard_paths: List[str] = []
    done = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx, max_tasks_per_child=8) as ex:
        futs = [ex.submit(_process_member, t) for t in tasks]

        pbar = None
        if args.tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_members, unit="file", dynamic_ncols=True, desc="Members")
            except Exception:
                pbar = None

        for fut in as_completed(futs):
            try:
                res = fut.result(timeout=None)
            except TimeoutError:
                # Shouldn't happen with as_completed, but keep for safety
                res = None
            except Exception as e:
                # Fail-open: count progress, but no shard
                res = None

            done += 1
            if res is not None:
                out_path, written = res
                shard_paths.append(out_path)
                total_pairs += written

            if pbar is not None:
                pbar.update(1)
            else:
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    pct = (done / total_members * 100.0) if total_members else 100.0
                    print(f"[progress] {done}/{total_members} ({pct:.1f}%) | {rate:.2f} members/s | elapsed={_fmt_dur(elapsed)}")

    # Final status
    if pbar is not None:
        pbar.close()
    elapsed = time.time() - t0
    print(f"[info] concatenating {len(shard_paths)} shards → {args.out}")

    # Ensure out dir
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Concatenate per-worker shards
    with open(args.out, "w", encoding="utf-8") as fout:
        for shard in shard_paths:
            try:
                with open(shard, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
            except Exception as e:
                print(f"[warn] failed to read shard {shard}: {e}")

    # Cleanup temp shards
    for shard in shard_paths:
        try:
            os.remove(shard)
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(f"[done] Wrote {total_pairs} changed-only pairs (B={B}, stride={stride}, jobs={args.jobs}) → {args.out}")
    print(f"[time] total elapsed = {_fmt_dur(elapsed)}")

if __name__ == "__main__":
    main()


