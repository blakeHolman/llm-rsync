# scripts/make_hf_dataset_train_only.py
import json, argparse
from pathlib import Path

def to_src(ex):
    return (
        f"path={ex['path']}\n"
        f"idx={ex['chunk_index']}\n"
        f"prev={ex['ctx_prev_sha1']}\n"
        f"next={ex['ctx_next_sha1']}\n"
        f"src={ex['src_b64']}\n"
    )
def to_tgt(ex): return f"tgt={ex['tgt_b64']}\n"

def convert(jsonl, out_src, out_tgt):
    with open(jsonl, 'r', encoding='utf-8') as fin, \
         open(out_src, 'w', encoding='utf-8') as fs, \
         open(out_tgt, 'w', encoding='utf-8') as ft:
        for line in fin:
            ex = json.loads(line)
            fs.write(to_src(ex))
            ft.write(to_tgt(ex))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="work/train.jsonl")
    ap.add_argument("--out_dir", default="work/hf_data")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    convert(args.train_jsonl, f"{args.out_dir}/train.src", f"{args.out_dir}/train.tgt")
    print("Wrote HF train text files.")
