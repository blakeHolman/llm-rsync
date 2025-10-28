# scripts/build_examples_multi_from_tar.py
import argparse, tarfile, hashlib, base64, json
from pathlib import Path

def open_plain_tar(path: Path):
    # Force plain .tar (we assume you've xz -d to .tar already)
    return tarfile.open(path, mode='r:')

def norm_name(mname: str) -> str:
    # Drop top-level linux-<ver>/ directory
    parts = Path(mname).parts
    return '/'.join(parts[1:]) if len(parts) > 1 else mname

def is_text_like(path: str) -> bool:
    # Keep to source/texty stuff (adjust as you like)
    p = path.lower()
    allow = ('.c','.h','.s','.kconfig','.dts','.dtsi','.txt','.md','.rst','.py','.sh')
    if any(p.endswith(x) for x in allow): return True
    base = Path(path).name
    return base in ('Makefile','Kbuild')

def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def chunk(b: bytes, sz: int):
    return [b[i:i+sz] for i in range(0, len(b), sz)]

def files_map_from_tar(tar_path: Path):
    out = {}
    with open_plain_tar(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile(): continue
            rel = norm_name(m.name)
            if not rel or not is_text_like(rel): continue
            fobj = tf.extractfile(m)
            out[rel] = fobj.read() if fobj else b""
    return out

def build_pair_examples(src_tar: Path, tgt_tar: Path, chunk_size: int, split: str, fout):
    src_map = files_map_from_tar(src_tar)
    tgt_map = files_map_from_tar(tgt_tar)
    count = 0
    for path, tgt_bytes in tgt_map.items():
        if path not in src_map:
            continue
        src_bytes = src_map[path]
        src_chunks = chunk(src_bytes, chunk_size)
        tgt_chunks = chunk(tgt_bytes, chunk_size)
        n = max(len(src_chunks), len(tgt_chunks))
        for i in range(n):
            s = src_chunks[i] if i < len(src_chunks) else b""
            t = tgt_chunks[i] if i < len(tgt_chunks) else b""
            if s == t:
                continue
            ex = {
                "split": split,
                "pair": f"{src_tar.name}->{tgt_tar.name}",
                "path": path,
                "chunk_index": i,
                "chunk_size": chunk_size,
                "src_sha1": sha1(s),
                "tgt_sha1": sha1(t),
                "src_b64": base64.b64encode(s).decode("ascii"),
                "tgt_b64": base64.b64encode(t).decode("ascii"),
                "ctx_prev_sha1": sha1(src_chunks[i-1]) if i-1>=0 and i-1<len(src_chunks) else "",
                "ctx_next_sha1": sha1(src_chunks[i+1]) if i+1<len(src_chunks) else "",
            }
            fout.write(json.dumps(ex) + "\n")
            count += 1
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_size", type=int, default=300)
    ap.add_argument("--kernels_dir", default="data/kernels")
    ap.add_argument("--out_jsonl", default="work/examples.jsonl")
    args = ap.parse_args()

    Path("work").mkdir(exist_ok=True)
    kd = Path(args.kernels_dir)

    train_pairs = [
        ("linux-5.15.193.tar", "linux-6.1.154.tar"),
        ("linux-6.1.154.tar", "linux-6.6.108.tar"),
        ("linux-6.6.108.tar", "linux-6.12.49.tar"),
        ("linux-6.12.49.tar", "linux-6.16.9.tar"),
    ]
    test_pair = ("linux-6.16.9.tar", "linux-6.17.tar")

    total_train = total_test = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for a,b in train_pairs:
            total_train += build_pair_examples(kd/a, kd/b, args.chunk_size, "train", fout)
        total_test += build_pair_examples(kd/test_pair[0], kd/test_pair[1], args.chunk_size, "test", fout)

    # split files
    import subprocess
    subprocess.run(f'jq -c \'select(.split=="train")\' {args.out_jsonl} > work/train.jsonl', shell=True, check=True)
    subprocess.run(f'jq -c \'select(.split=="test")\'  {args.out_jsonl} > work/test.jsonl',  shell=True, check=True)

    print(f"train examples: {total_train}, test examples: {total_test}")
    print("Wrote work/train.jsonl and work/test.jsonl")

if __name__ == "__main__":
    main()
