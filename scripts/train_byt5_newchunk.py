#!/usr/bin/env python3
# scripts/train_byt5_newchunk.py
import argparse, json, base64, torch, random, os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

class PairDS(Dataset):
    def __init__(self, path):
        self.items = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        return base64.b64decode(rec["old"]), base64.b64decode(rec["new"])

def collate(batch, tok, max_src_tokens, max_tgt_tokens):
    olds_b, news_b = zip(*batch)
    olds_txt = [b.decode("latin-1") for b in olds_b]
    news_txt = [b.decode("latin-1") for b in news_b]
    enc = tok(list(olds_txt), return_tensors="pt", padding=True, truncation=True, max_length=max_src_tokens)
    lab = tok(text_target=list(news_txt), return_tensors="pt", padding=True, truncation=True, max_length=max_tgt_tokens)
    enc["labels"] = lab["input_ids"]
    return enc

def set_seed(seed):
    if seed is None: return
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model_name", default="google/byt5-small")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_src_tokens", type=int, default=1600)
    ap.add_argument("--max_tgt_tokens", type=int, default=1200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    tok = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)

    tr = PairDS(args.train)
    va = PairDS(args.val)

    coll = lambda b: collate(b, tok, args.max_src_tokens, args.max_tgt_tokens)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                       num_workers=args.num_workers, pin_memory=(args.device=="cuda"))
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                       num_workers=args.num_workers, pin_memory=(args.device=="cuda"))

    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(dl_tr) // max(1, args.grad_accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    for ep in range(1, args.epochs + 1):
        model.train(); tr_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr, 1):
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
            tr_loss += out.loss.item()

        tr_loss /= max(1, len(dl_tr))
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
                va_loss += model(**batch).loss.item()
        va_loss /= max(1, len(dl_va))
        print(f"epoch {ep}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved →", args.out_dir)

if __name__ == "__main__":
    main()
