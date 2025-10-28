#!/usr/bin/env python3
# scripts/predict_and_eval.py
import argparse, tempfile
from pathlib import Path
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from rsync_match import (
    build_source_index_sqlite,
    register_old_bytes_for_index,
    iter_matches_and_unmatched,
)

def read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def build_index_if_needed(old_bytes: bytes, B: int) -> str:
    idx_path = Path(tempfile.gettempdir()) / f"rs_idx_eval_{hash((len(old_bytes), B))}.sqlite"
    if not idx_path.exists():
        build_source_index_sqlite(old_bytes, str(idx_path), block_size=B)
    # Register old bytes so the matcher can greedily extend COPY runs
    register_old_bytes_for_index(str(idx_path), old_bytes)
    return str(idx_path)

def nearest_copy_anchor(regions, lit_start: int, lit_end: int):
    best_gap = 1 << 60
    anchor = None
    for r in regions:
        if r[0] != "copy":
            continue
        _, nstart, nlen, ostart = r
        gap = min(abs(lit_start - (nstart + nlen)), abs(lit_end - nstart))
        if gap < best_gap:
            best_gap = gap
            anchor = ostart
    return anchor

def gen_bytes(model, tok, old_ctx: bytes, want_len: int, max_tokens: int, device: str) -> bytes:
    """
    Generate predicted bytes for a literal chunk using ByT5.
    Ensures the decoding is at least want_len bytes (padding if shorter).
    """
    # ByT5 is byte-aware; latin-1 preserves 0..255
    enc = tok([old_ctx.decode("latin-1")], return_tensors="pt",
              truncation=True, max_length=max_tokens).to(device)
    # Ensure we can at least generate want_len chars; cap to max_tokens
    gen_len = max(want_len, 1)
    gen_len = min(gen_len, max_tokens)
    out = model.generate(**enc, max_length=gen_len, do_sample=False)
    txt = tok.batch_decode(out, skip_special_tokens=True)[0]
    b = txt.encode("latin-1", errors="ignore")
    if len(b) < want_len:
        b += bytes(want_len - len(b))
    return b[:want_len]

def _compress(data: bytes, codec: str) -> int:
    if codec == "none":
        return len(data)
    if codec == "zstd":
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            # Fallback to zlib if zstd not available
            import zlib
            return len(zlib.compress(data))
        c = zstd.ZstdCompressor(level=3)
        return len(c.compress(data))
    # default: zlib
    import zlib
    return len(zlib.compress(data))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_path", required=True, help="e.g., data/linux-6.16.9.tar")
    ap.add_argument("--new_path", required=True, help="e.g., data/linux-6.17.tar")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--block_size", type=int, default=1024, help="B for literal chunking")
    ap.add_argument("--context", type=int, default=256, help="old-side context per side (bytes)")
    ap.add_argument("--gen_max_tokens", type=int, default=1600, help="cap on tokens when generating")
    ap.add_argument("--residual_codec", choices=["zlib", "zstd", "none"], default="zlib",
                    help="compression used on XOR residuals")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    B, C = args.block_size, args.context
    oldb = read_bytes(Path(args.old_path))
    newb = read_bytes(Path(args.new_path))

    # Build/attach index and enable greedy COPY extension
    idx = build_index_if_needed(oldb, B)
    regions = list(iter_matches_and_unmatched(newb, idx, block_size=B))

    tok = ByT5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device).eval()

    total_llm_bytes = 0
    total_rsync_literal = 0

    for r in regions:
        if r[0] != "lit":
            continue
        _, nstart, nlen = r  # literal region in new
        total_rsync_literal += nlen

        # Choose anchor from nearest copy region to guide context
        anchor = nearest_copy_anchor(regions, nstart, nstart + nlen)

        nbeg, nend = nstart, nstart + nlen
        while nbeg < nend:
            want = min(B, nend - nbeg)
            nchunk = newb[nbeg: nbeg + want]
            if len(nchunk) < want:
                nchunk += bytes(want - len(nchunk))

            # Build old context window (B + 2*C bytes)
            if anchor is not None:
                # align by relative offset into literal
                o_center = anchor + (nbeg - nstart)
                o_beg = max(0, o_center - C)
                o_end = min(len(oldb), o_beg + (B + 2*C))
                old_ctx = oldb[o_beg:o_end]
                if len(old_ctx) < (B + 2*C):
                    old_ctx += bytes(B + 2*C - len(old_ctx))
            else:
                old_ctx = bytes(B + 2*C)

            with torch.no_grad():
                pred = gen_bytes(model, tok, old_ctx, want_len=want,
                                 max_tokens=max(args.gen_max_tokens, want), device=args.device)

            # XOR residual over *literal* bytes only, then compress
            resid = bytes([a ^ b for a, b in zip(nchunk, pred)])
            total_llm_bytes += _compress(resid, args.residual_codec)

            nbeg += want

    total_new = len(newb)
    print("===== RSYNC-AWARE EVAL (6.16.9 → 6.17) =====")
    print(f"Block size (B): {B}   Context per side (C): {C}")
    print(f"Residual codec: {args.residual_codec}")
    print(f"Total new bytes: {total_new}")
    print(f"LLM compressed XOR residual over literals: {total_llm_bytes}  ({100*total_llm_bytes/total_new:.2f}% of new)")
    print(f'Basic rsync "literal residual" bytes:      {total_rsync_literal}  ({100*total_rsync_literal/total_new:.2f}% of new)')
    smart = min(total_llm_bytes, total_rsync_literal)
    print(f"SMART pick (min):                          {smart}  ({100*smart/total_new:.2f}% of new)")

if __name__ == "__main__":
    main()

