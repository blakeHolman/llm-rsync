# scripts/build_examples_tarblob_rsync.py
import argparse, base64, json, os
from pathlib import Path
from rsync_match import build_source_index_sqlite, iter_matches_and_unmatched

def write_examples(src_tar: Path, tgt_tar: Path, block_size: int, split: str, writer):
    # index on disk
    db_path = f"work/rsync_index_{src_tar.name}_{block_size}.sqlite"
    if not os.path.exists(db_path):
        Path("work").mkdir(exist_ok=True)
        print(f"Building source index: {src_tar} (block={block_size}) → {db_path}")
        build_source_index_sqlite(str(src_tar), block_size, db_path)

    total_new = os.path.getsize(tgt_tar)
    matched_bytes = 0
    changed_bytes = 0
    n_examples = 0

    for rec in iter_matches_and_unmatched(str(src_tar), str(tgt_tar), block_size, db_path):
        kind = rec[0]
        if kind == "match":
            _, _tpos, _spos, blen = rec
            matched_bytes += blen
        else:
            _, u0, u1 = rec
            changed_bytes += (u1 - u0)
            # emit in block-sized slices
            i = u0
            with open(tgt_tar, "rb") as f:
                while i < u1:
                    j = min(i + block_size, u1)
                    f.seek(i); chunk = f.read(j - i)
                    ex = {
                        "split": split,
                        "pair": f"{src_tar.name}->{tgt_tar.name}",
                        "offset": i,
                        "block_size": block_size,
                        "prompt": (
                            f"pair={src_tar.name}->{tgt_tar.name}\n"
                            f"mode=tarblob_rsync\n"
                            f"offset={i}\n"
                            f"block={block_size}\n"
                            f"src_hint=NA\n"
                            f"src=NA\n"
                        ),
                        "tgt_b64": base64.b64encode(chunk).decode("ascii"),
                        "total_new": total_new,
                    }
                    writer(ex)
                    n_examples += 1
                    i = j
    return total_new, matched_bytes, changed_bytes, n_examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels_dir", default="data/kernels")
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--block_size", type=int, default=300)
    args = ap.parse_args()

    kd = Path(args.kernels_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    
    train_pairs = [
        ("linux-5.15.193.tar", "linux-6.1.154.tar"),
        ("linux-6.1.154.tar", "linux-6.6.108.tar"),
        ("linux-6.6.108.tar", "linux-6.12.49.tar"),
        ("linux-6.12.49.tar", "linux-6.16.9.tar"),
    ]
    
    test_pair = ("linux-6.16.9.tar", "linux-6.17.tar")

    train_jsonl = out / "train.tarblob.jsonl"
    test_jsonl  = out / "test.tarblob.jsonl"
    
    with open(train_jsonl, "w", encoding="utf-8") as ft:
        def wtrain(ex): ft.write(json.dumps(ex) + "\n")
        for a, b in train_pairs:
            total, matched, changed, n = write_examples(kd/a, kd/b, args.block_size, "train", wtrain)
            print(f"{a}->{b}  total={total} matched={matched} changed={changed} examples={n}")
    
    with open(test_jsonl, "w", encoding="utf-8") as fz:
        def wtest(ex): fz.write(json.dumps(ex) + "\n")
        total, matched, changed, n = write_examples(kd/test_pair[0], kd/test_pair[1], args.block_size, "test", wtest)
        print(f"TEST {test_pair[0]}->{test_pair[1]}  total={total} matched={matched} changed={changed} examples={n}")

    print(f"Wrote {train_jsonl} and {test_jsonl}")

if __name__ == "__main__":
    main()
