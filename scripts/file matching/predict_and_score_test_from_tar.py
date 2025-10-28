# scripts/predict_and_score_test_from_tar.py
import argparse, json, base64, tarfile
from pathlib import Path
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

def open_plain_tar(path: Path):
    return tarfile.open(path, mode='r:')

def norm_name(mname: str) -> str:
    parts = Path(mname).parts
    return '/'.join(parts[1:]) if len(parts)>1 else mname

def is_text_like(path: str) -> bool:
    p = path.lower()
    allow = ('.c','.h','.s','.kconfig','.dts','.dtsi','.txt','.md','.rst','.py','.sh')
    if any(p.endswith(x) for x in allow): return True
    base = Path(path).name
    return base in ('Makefile','Kbuild')

def load_map_from_tar(tar_path: Path):
    out = {}
    with open_plain_tar(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile(): continue
            rel = norm_name(m.name)
            if not rel or not is_text_like(rel): continue
            f = tf.extractfile(m)
            out[rel] = f.read() if f else b""
    return out

def avg_logprob(model, tok, src: str, tgt: str, max_len=2048):
    enc = tok([src], return_tensors="pt", truncation=True, max_length=max_len)
    with tok.as_target_tokenizer():
        lab = tok([tgt], return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(model.device) for k,v in enc.items()}
    labels = lab["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=labels)
        return -out.loss.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="work/byt5-ckpt")
    ap.add_argument("--test_jsonl", default="work/test.jsonl")
    ap.add_argument("--chunk_size", type=int, default=300)
    ap.add_argument("--tau", type=float, default=-1.5)
    ap.add_argument("--kernels_dir", default="data/kernels")
    ap.add_argument("--src_tar", default="linux-6.16.9.tar")
    ap.add_argument("--tgt_tar", default="linux-6.17.tar")
    args = ap.parse_args()

    kd = Path(args.kernels_dir)
    src_map = load_map_from_tar(kd/args.src_tar)
    tgt_map = load_map_from_tar(kd/args.tgt_tar)

    total_new_bytes = sum(len(b) for b in tgt_map.values())
    matched_bytes = 0
    model_predicted_bytes = 0
    literal_residual_bytes = 0

    # matched_bytes at chunk granularity
    for path, tgt in tgt_map.items():
        src = src_map.get(path, b"")
        for i in range(0, max(len(tgt), len(src)), args.chunk_size):
            t = tgt[i:i+args.chunk_size]
            s = src[i:i+args.chunk_size]
            if t == s:
                matched_bytes += len(t)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByT5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device).eval()

    # score changed examples
    with open(args.test_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            ex = json.loads(line)
            if ex["split"] != "test": continue
            src_text = (
                f"path={ex['path']}\n"
                f"idx={ex['chunk_index']}\n"
                f"prev={ex['ctx_prev_sha1']}\n"
                f"next={ex['ctx_next_sha1']}\n"
                f"src={ex['src_b64']}\n"
            )
            tgt_text = f"tgt={ex['tgt_b64']}\n"
            score = avg_logprob(model, tok, src_text, tgt_text)
            tgt_bytes = base64.b64decode(ex["tgt_b64"].encode("ascii"))
            if score >= args.tau:
                model_predicted_bytes += len(tgt_bytes)
            else:
                literal_residual_bytes += len(tgt_bytes)

    changed_bytes = total_new_bytes - matched_bytes
    residual_percent = (literal_residual_bytes / total_new_bytes) if total_new_bytes else 0.0

    print("=== TEST RESULTS (6.16.9.tar → 6.17.tar) ===")
    print(f"chunk_size:              {args.chunk_size}")
    print(f"tau (avg logprob):       {args.tau}")
    print(f"total_new_bytes:         {total_new_bytes}")
    print(f"matched_bytes (copy):    {matched_bytes}")
    print(f"changed_bytes:           {changed_bytes}")
    print(f"model_predicted_bytes:   {model_predicted_bytes}")
    print(f"literal_residual_bytes:  {literal_residual_bytes}")
    print(f"residual_percent:        {100.0*residual_percent:.2f}%")

if __name__ == "__main__":
    main()
