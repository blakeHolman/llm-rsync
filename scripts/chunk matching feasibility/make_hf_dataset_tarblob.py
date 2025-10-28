# scripts/make_hf_dataset_tarblob.py
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="work/train.tarblob.jsonl")
    ap.add_argument("--out_dir", default="work/hf_data")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(args.train_jsonl, "r", encoding="utf-8") as f, \
         open(f"{args.out_dir}/train.src", "w", encoding="utf-8") as xs, \
         open(f"{args.out_dir}/train.tgt", "w", encoding="utf-8") as ys:
        for line in f:
            ex = json.loads(line)
            if ex.get("split") != "train": continue
            xs.write(ex["prompt"])
            ys.write(f"tgt={ex['tgt_b64']}\n")
    print("Wrote HF data at", args.out_dir)

if __name__ == "__main__":
    main()
