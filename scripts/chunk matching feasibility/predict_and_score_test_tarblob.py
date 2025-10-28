# scripts/predict_and_score_test_tarblob.py
import argparse, base64, io, gzip, os, math, hashlib
from pathlib import Path
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from rsync_match import build_source_index_sqlite, iter_matches_and_unmatched
from typing import Optional  # <-- add this

def read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def avg_logprob(model, tok, src_text: str, tgt_text: str, max_len=1024, device="cpu"):
    enc = tok([src_text], return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with tok.as_target_tokenizer():
        lab = tok([tgt_text], return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**enc, labels=lab["input_ids"])
        return -out.loss.item()

def gzip_size(b: bytes) -> int:
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", compresslevel=6) as gz:
        gz.write(b)
    return len(bio.getvalue())

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def bytes_for_checksum(bits: int) -> int:
    return (bits + 7) // 8

def extract_pred_b64(txt: str, require_prefix: bool) -> Optional[str]:  # <-- change return type
    """
    Extract the base64 after 'tgt=' up to first newline or EOS; return None if not found.
    """
    key = "tgt="
    idx = txt.find(key)
    if idx < 0:
        return None if require_prefix else txt.strip()
    rest = txt[idx + len(key):]
    # stop at first newline if present
    nl = rest.find("\n")
    if nl >= 0:
        rest = rest[:nl]
    return rest.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels_dir", default="data/kernels")
    ap.add_argument("--src_tar", default="linux-6.16.9.tar")
    ap.add_argument("--tgt_tar", default="linux-6.17.tar")
    ap.add_argument("--model_dir", default="work/byt5-ckpt")
    ap.add_argument("--block_size", type=int, default=300)
    ap.add_argument("--tau", type=float, default=-1.5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--best_of_gzip", action="store_true",
                    help="Count gzip(literal) for chunks where model not trusted or verify fails")

    # --- New flags for generate→verify ---
    ap.add_argument("--generate_verify", action="store_true",
                    help="After τ-accept, actually generate, verify by checksum, else fallback")
    ap.add_argument("--checksum_bits", type=int, default=16,
                    help="Bits you would transmit to let receiver verify (overhead)")
    ap.add_argument("--gen_max_new_tokens", type=int, default=600,
                    help="Cap on generated tokens for tgt=<base64...>")
    ap.add_argument("--decode_strategy", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=1,
                    help="Number of beams if --decode_strategy=beam")
    ap.add_argument("--verify_scope", choices=["accepted", "all"], default="accepted",
                    help="Generate/verify only τ-accepted chunks (recommended) or all unmatched")
    ap.add_argument("--strict_prefix", action="store_true",
                    help="Require generated text to contain 'tgt=' prefix")
    args = ap.parse_args()

    kd = Path(args.kernels_dir)
    src_p = kd / args.src_tar
    tgt_p = kd / args.tgt_tar
    total_new = os.path.getsize(tgt_p)

    # Ensure SQLite index exists for the source tar
    db_path = f"work/rsync_index_{args.src_tar}_{args.block_size}.sqlite"
    Path("work").mkdir(exist_ok=True)
    if not os.path.exists(db_path):
        print(f"Building source index for prediction: {src_p} (block={args.block_size}) → {db_path}")
        build_source_index_sqlite(str(src_p), args.block_size, db_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByT5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device).eval()

    matched_bytes = 0
    model_predicted = 0  # predicted (τ-accepted); may be verified if generate_verify is ON
    literal_residual = 0

    # New metrics for generate→verify
    accepted_chunks = 0
    gen_attempts = 0
    verify_success = 0
    model_predicted_verified = 0
    verify_overhead_bytes = 0
    fallback_literal_bytes = 0  # subset of literal_residual when verify fails
    checksum_bytes_per_chunk = bytes_for_checksum(args.checksum_bits)

    def payload_size(chunk: bytes) -> int:
        return gzip_size(chunk) if args.best_of_gzip else len(chunk)

    # Stream through matches & unmatched spans
    with open(tgt_p, "rb") as tf:
        for rec in iter_matches_and_unmatched(str(src_p), str(tgt_p), args.block_size, db_path):
            kind = rec[0]
            if kind == "match":
                _, _tpos, _spos, blen = rec
                matched_bytes += blen
            else:
                _, u0, u1 = rec
                # Evaluate unmatched region in block-sized slices
                i = u0
                while i < u1:
                    j = min(i + args.block_size, u1)
                    tf.seek(i)
                    tgt_chunk = tf.read(j - i)
                    tgt_b64 = base64.b64encode(tgt_chunk).decode("ascii")

                    src_text = (
                        f"pair={args.src_tar}->{args.tgt_tar}\n"
                        f"mode=tarblob_rsync\n"
                        f"offset={i}\n"
                        f"block={args.block_size}\n"
                        f"src_hint=NA\n"
                        f"src=NA\n"
                    )
                    tgt_text = f"tgt={tgt_b64}\n"
                    score = avg_logprob(model, tok, src_text, tgt_text,
                                        max_len=args.max_len, device=device)

                    accepted_by_tau = (score >= args.tau)
                    if accepted_by_tau:
                        accepted_chunks += 1

                    # --- Original behavior (no generate/verify) ---
                    if not args.generate_verify:
                        if accepted_by_tau:
                            model_predicted += len(tgt_chunk)
                        else:
                            literal_residual += payload_size(tgt_chunk)
                        i = j
                        continue

                    # --- Generate→verify path ---
                    # Scope control: generate only for accepted chunks unless 'all'
                    should_attempt = accepted_by_tau or (args.verify_scope == "all")
                    if should_attempt:
                        gen_attempts += 1
                        # Encode prompt only
                        enc = tok([src_text], return_tensors="pt", truncation=True,
                                  max_length=args.max_len).to(device)
                        gen_kwargs = dict(
                            max_new_tokens=args.gen_max_new_tokens,
                            do_sample=False,
                        )
                        if args.decode_strategy == "beam":
                            gen_kwargs.update(num_beams=max(1, args.num_beams), early_stopping=True)
                        with torch.no_grad():
                            gen_ids = model.generate(**enc, **gen_kwargs)
                        gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)

                        pred_b64 = extract_pred_b64(gen_text, require_prefix=args.strict_prefix)
                        ok = False
                        if pred_b64:
                            try:
                                pred_bytes = base64.b64decode(pred_b64, validate=False)
                                ok = (sha256(pred_bytes) == sha256(tgt_chunk))
                            except Exception:
                                ok = False

                        if ok:
                            verify_success += 1
                            model_predicted += len(tgt_chunk)  # logical predicted count
                            model_predicted_verified += len(tgt_chunk)
                            verify_overhead_bytes += checksum_bytes_per_chunk
                        else:
                            # fallback literal
                            lit = payload_size(tgt_chunk)
                            literal_residual += lit
                            fallback_literal_bytes += lit
                    else:
                        # not attempted (e.g., rejected by τ and scope='accepted')
                        literal_residual += payload_size(tgt_chunk)

                    i = j

    changed_bytes = total_new - matched_bytes

    # When generate_verify is ON, we count net residual = literal + checksum overhead
    net_residual_bytes = literal_residual + (verify_overhead_bytes if args.generate_verify else 0)
    residual_percent = (net_residual_bytes / total_new) if total_new else 0.0

    print("=== TEST RESULTS (tar-blob, rsync-style) ===")
    print(f"block_size:                         {args.block_size}")
    print(f"tau (avg logprob):                  {args.tau}")
    print(f"total_new_bytes:                    {total_new}")
    print(f"matched_bytes (copy):               {matched_bytes}")
    print(f"changed_bytes:                      {changed_bytes}")
    print(f"model_predicted_bytes (count):      {model_predicted}")
    if args.generate_verify:
        print("--- generate → verify ---")
        print(f"accepted_chunks_by_tau:             {accepted_chunks}")
        print(f"gen_attempts:                       {gen_attempts}")
        print(f"verify_successes:                   {verify_success}")
        sr = (verify_success / gen_attempts) if gen_attempts else 0.0
        ar = (accepted_chunks / (changed_bytes / max(1, args.block_size))) if changed_bytes else 0.0
        print(f"verify_success_rate:                {sr:.3f}")
        print(f"model_predicted_bytes_verified:     {model_predicted_verified}")
        print(f"verify_overhead_bytes (checksums):  {verify_overhead_bytes}  "
              f"(= {bytes_for_checksum(args.checksum_bits)} B * successes)")
        print(f"fallback_literal_bytes:             {fallback_literal_bytes}")
    print(f"literal_residual_bytes:             {literal_residual}")
    print(f"net_residual_bytes:                 {net_residual_bytes}")
    print(f"residual_percent:                   {100.0*residual_percent:.2f}%")
    if args.best_of_gzip:
        print("(best-of gzip ON)")
    if args.generate_verify:
        print(f"(generate_verify ON; checksum_bits={args.checksum_bits}, "
              f"strategy={args.decode_strategy}, beams={args.num_beams}, scope={args.verify_scope})")

if __name__ == "__main__":
    main()

