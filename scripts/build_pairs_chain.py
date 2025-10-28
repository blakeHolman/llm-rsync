#!/usr/bin/env python3
# scripts/build_pairs_chain.py  (streaming, coalesced I/O, Colab-friendly)
import argparse, base64, json, os, tarfile, tempfile, uuid, time
from typing import Iterator, Tuple, Optional, List
from multiprocessing import Pool, cpu_count

from rsync_match import (
    build_source_index_sqlite_stream,
    iter_matches_and_unmatched_stream_sqlcached,  # streaming matcher with LRU
)

# =============== Utilities ===============

def _ensure_seekable(fobj):
    """
    Return a seekable file-like object. If fobj is not seekable (e.g., from .tar.xz),
    spill once to a temporary file and return that temp file (positioned at 0).
    """
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

def _read_old_window(old_seek, begin: int, length: int) -> bytes:
    """
    Read [begin, begin+length) from OLD, padding with zeros if out-of-bounds.
    Caller guarantees old_seek is seekable.
    """
    if begin < 0:
        pad_left = -begin
        old_seek.seek(0)
        data = b"\x00" * pad_left + (old_seek.read(max(0, length - pad_left)) or b"")
    else:
        old_seek.seek(begin)
        data = old_seek.read(length) or b""
    if len(data) < length:
        data = data + bytes(length - len(data))
    return data

def _fmt_dur(sec: float) -> str:
    """Format seconds as H:MM:SS."""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"

# =============== Member matching (normalized) ===============

def _common_members(old_tar: tarfile.TarFile, new_tar: tarfile.TarFile) -> List[Tuple[str, str]]:
    """
    Return list of (old_member_name, new_member_name) for files that match after
    stripping the top-level directory prefix (e.g., 'linux-6.1.154/').
    """
    def norm(name: str) -> str:
        return name.split("/", 1)[1] if "/" in name else name

    old_map = {}
    for m in old_tar.getmembers():
        if m.isfile():
            old_map.setdefault(norm(m.name), m.name)

    new_map = {}
    for m in new_tar.getmembers():
        if m.isfile():
            new_map.setdefault(norm(m.name), m.name)

    common_norm = sorted(set(old_map.keys()) & set(new_map.keys()))
    return [(old_map[n], new_map[n]) for n in common_norm]

# =============== Literal emission (coalesced) ===============

def _emit_literal_region(
    old_seek, new_seek,
    nstart: int, nlen: int,
    anchor_old_start: Optional[int],
    B: int, C: int
) -> Iterator[Tuple[bytes, bytes]]:
    """
    Emit training pairs for ONE literal region [nstart, nstart+nlen),
    using a single OLD span read (if anchor available) and a single NEW span read.
      - target NEW chunks are size B (pad final)
      - input OLD windows are (B + 2*C) centered relative to anchor
    """
    zeros = bytes(B + 2 * C)
    nbeg, nend = nstart, nstart + nlen
    nchunks = (nlen + B - 1) // B

    # Read NEW literal region once
    new_seek.seek(nbeg)
    new_buf = new_seek.read(nlen) or b""

    if anchor_old_start is None:
        # No COPY seen recently → zero-context windows
        for i in range(nchunks):
            s = i * B
            e = min(s + B, nlen)
            new_chunk = new_buf[s:e]
            if len(new_chunk) < B:
                new_chunk = new_chunk + bytes(B - len(new_chunk))
            yield zeros, new_chunk
        return

    # With an anchor, compute the contiguous OLD span needed for all chunks
    # center for chunk i at: anchor + ((nbeg + i*B) - nstart)
    o_center0 = anchor_old_start + (nbeg - nstart)
    o_beg0 = o_center0 - C
    o_center_last = anchor_old_start + (nbeg + (nchunks - 1) * B - nstart)
    o_beg_last = o_center_last - C

    o_span_beg = o_beg0
    o_span_len = (o_beg_last - o_beg0) + (B + 2 * C)

    old_span = _read_old_window(old_seek, o_span_beg, o_span_len)

    # Slice per B-chunk
    for i in range(nchunks):
        s = i * B
        e = min(s + B, nlen)
        new_chunk = new_buf[s:e]
        if len(new_chunk) < B:
            new_chunk = new_chunk + bytes(B - len(new_chunk))
        span_off = (o_beg0 + i * B) - o_span_beg
        old_win = old_span[span_off: span_off + (B + 2 * C)]
        yield old_win, new_chunk

# =============== Worker (streaming; low RAM) ===============

def _process_member(args):
    (
        old_tar_path, new_tar_path,
        old_member_name, new_member_name,
        B, C, tmp_dir,
        cache_size
    ) = args

    # Open fresh tar handles in the worker
    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        om = told.getmember(old_member_name)
        nm = tnew.getmember(new_member_name)

        # Only process regular files with size > 0 (NEW)
        if not om.isfile() or not nm.isfile() or nm.size == 0:
            return None

        # Extract file-like streams and ensure seekable
        of = told.extractfile(om)
        nf = tnew.extractfile(nm)
        if of is None or nf is None:
            return None

        of = _ensure_seekable(of)
        nf = _ensure_seekable(nf)

        # 1) Build OLD index (sequential)
        idx_path = os.path.join(tmp_dir, f"idx_{uuid.uuid4().hex}.sqlite")
        of.seek(0)
        build_source_index_sqlite_stream(of, idx_path, block_size=B)
        of.seek(0)

        # 2) Stream NEW regions; emit pairs on the fly (anchor = last COPY old_start)
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jsonl")
        written = 0
        nf.seek(0); of.seek(0)
        last_copy_old: Optional[int] = None

        with open(out_path, "w", encoding="utf-8") as fout:
            for reg in iter_matches_and_unmatched_stream_sqlcached(
                nf, idx_path, block_size=B, cache_size=cache_size
            ):
                if reg[0] == "copy":
                    # remember anchor for upcoming literal regions
                    _, nstart, nlen, ostart = reg
                    last_copy_old = ostart
                    continue

                # literal region
                _, nstart, nlen = reg
                for old_win, new_chunk in _emit_literal_region(
                    of, nf, nstart, nlen, last_copy_old, B, C
                ):
                    rec = {
                        "old": base64.b64encode(old_win).decode("ascii"),
                        "new": base64.b64encode(new_chunk).decode("ascii"),
                        "member_old": old_member_name,
                        "member_new": new_member_name,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    written += 1

        # Remove index to free space early (best-effort)
        try:
            os.remove(idx_path)
        except Exception:
            pass

    return (out_path, written)

# =============== Main orchestration ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_size", type=int, default=1024, help="B (bytes) for NEW literal chunks and OLD index blocks")
    ap.add_argument("--context", type=int, default=256, help="OLD-side context per side (bytes)")
    ap.add_argument("--out", required=True, help="Output JSONL of {old,new} base64 training pairs")
    ap.add_argument("--chain", nargs=2, required=True, help="Exactly two tar paths: OLD.tar NEW.tar")
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1), help="Parallel workers (processes)")
    ap.add_argument("--limit", type=int, default=0, help="(Debug) limit number of members processed")
    ap.add_argument("--progress_every", type=int, default=1000, help="Print progress every N members")
    ap.add_argument("--cache_size", type=int, default=65536, help="LRU size for weak→candidates in the streaming matcher")
    ap.add_argument("--pool_chunksize", type=int, default=64, help="chunksize for imap_unordered task batching")
    args = ap.parse_args()

    B, C = args.block_size, args.context
    old_tar_path, new_tar_path = args.chain

    t0 = time.time()

    # Enumerate common members once (normalized)
    with tarfile.open(old_tar_path, "r:*") as told, tarfile.open(new_tar_path, "r:*") as tnew:
        pairs = _common_members(told, tnew)

    if args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"[info] limiting to first {args.limit} members")

    total_members = len(pairs)
    print(f"Matched {total_members} common members after normalizing top-level dirs")

    # Parallel over members
    total_pairs = 0
    tmp_dir = tempfile.mkdtemp(prefix="pairs_tmp_")
    print(f"[info] writing per-worker shards under: {tmp_dir}")

    tasks = [
        (old_tar_path, new_tar_path, old_name, new_name, B, C, tmp_dir, args.cache_size)
        for (old_name, new_name) in pairs
    ]

    def _status(now_sec: float, done: int, total: int):
        elapsed = now_sec - t0
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        remaining = total - done
        eta = remaining / rate if rate > 0 else float("inf")
        pct = (done / total * 100.0) if total else 100.0
        eta_txt = "∞" if eta == float("inf") else _fmt_dur(eta)
        print(f"[progress] {done}/{total} ({pct:.1f}%) | elapsed={_fmt_dur(elapsed)} | eta={eta_txt} | ~{rate:.1f} members/s")

    progress = 0
    with Pool(processes=args.jobs) as pool:
        for res in pool.imap_unordered(_process_member, tasks, chunksize=args.pool_chunksize):
            progress += 1
            if res is not None:
                out_path, written = res
                total_pairs += written
            if progress % args.progress_every == 0:
                _status(time.time(), progress, total_members)

    _status(time.time(), progress, total_members)
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

    total_elapsed = time.time() - t0
    print(f"[done] Wrote {total_pairs} rsync-aware literal pairs → {args.out}  (B={B}, ctx={C}, jobs={args.jobs}, cache={args.cache_size}, chunksize={args.pool_chunksize})")
    print(f"[time] total elapsed = {_fmt_dur(total_elapsed)}")

if __name__ == "__main__":
    main()
