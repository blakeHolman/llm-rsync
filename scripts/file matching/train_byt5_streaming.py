# scripts/train_byt5_streaming.py
import os, math, argparse, itertools, random
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

try:
    from transformers import Adafactor
    _HAS_ADAFACTOR = True
except Exception:
    _HAS_ADAFACTOR = False
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def batched(iterable, n):
    """Yield lists of length n from an iterator (last may be shorter)."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def line_reader(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="work/hf_data")
    ap.add_argument("--out_dir", default="work/byt5-ckpt")
    ap.add_argument("--model_name", default="google/byt5-small")

    # Faster defaults
    ap.add_argument("--max_len", type=int, default=512)          # shorter sequences = big speedup
    ap.add_argument("--batch_size", type=int, default=1)         # tiny microbatch for low mem
    ap.add_argument("--grad_accum", type=int, default=256)       # effective batch = 256
    ap.add_argument("--epochs", type=int, default=1)             # one pass is usually enough

    # Optim & scheduling
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--lr", type=float, default=5e-4)            # used if adamw, ignored for adafactor(rel_step=True)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=500)

    # Early stopping / dataset thinning
    ap.add_argument("--max_updates", type=int, default=20000)    # hard stop; 0 = no cap
    ap.add_argument("--sample_prob", type=float, default=1.0)    # keep each line with this prob (0< p <=1)
    ap.add_argument("--max_lines", type=int, default=0)          # cap total kept examples; 0 = no cap

    # Mixed precision
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast if available")
    ap.add_argument("--fp16", action="store_true", help="Use float16 autocast if available")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Memory / speed savers
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        # Optional for newer PyTorch: improves matmul perf on Ampere+
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model.to(device)

    # ----- Optimizer & Scheduler -----
    use_adafactor = (args.optimizer == "adafactor") and _HAS_ADAFACTOR
    if use_adafactor:
        # Adafactor with relative step: no explicit LR & scheduler needed
        opt = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )
        sched = None
    else:
        opt = AdamW(model.parameters(), lr=args.lr)
        # We need an approximate step count for scheduler; base it on line count * sampling prob
        est_lines = sum(1 for _ in line_reader(f"{args.data_dir}/train.src"))
        if args.sample_prob < 1.0:
            est_lines = int(est_lines * args.sample_prob)
        if args.max_lines and est_lines > args.max_lines:
            est_lines = args.max_lines
        steps_per_epoch = max(1, math.ceil(est_lines / (args.batch_size * args.grad_accum)))
        total_steps = max(1, steps_per_epoch * args.epochs)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=min(args.warmup_steps, total_steps//10), num_training_steps=total_steps)

    # ----- Mixed precision setup -----
    use_bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and args.fp16 and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)  # scaler only for fp16

    def keep_example(idx_kept):
        """Return False to drop this example based on sample_prob / max_lines."""
        if args.sample_prob < 1.0 and random.random() > args.sample_prob:
            return False
        if args.max_lines and idx_kept >= args.max_lines:
            return False
        return True

    global_step = 0
    running_loss = 0.0

    model.train()
    kept_count = 0

    for ep in range(args.epochs):
        src_iter = line_reader(f"{args.data_dir}/train.src")
        tgt_iter = line_reader(f"{args.data_dir}/train.tgt")

        # Stream, but let sampling drop lines before batching
        def kept_pairs():
            nonlocal kept_count
            for s, t in zip(src_iter, tgt_iter):
                if keep_example(kept_count):
                    kept_count += 1
                    yield (s, t)

        micro = []
        for s, t in kept_pairs():
            micro.append((s, t))
            if len(micro) == args.batch_size:
                src_batch, tgt_batch = zip(*micro)
                micro = []

                # Tokenize
                enc = tok(list(src_batch), return_tensors="pt",
                          padding=True, truncation=True, max_length=args.max_len)
                with tok.as_target_tokenizer():
                    lab = tok(list(tgt_batch), return_tensors="pt",
                              padding=True, truncation=True, max_length=args.max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = lab["input_ids"].to(device)

                # Forward (mixed precision if set)
                autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
                if autocast_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        out = model(**enc, labels=labels)
                        loss = out.loss / args.grad_accum
                else:
                    out = model(**enc, labels=labels)
                    loss = out.loss / args.grad_accum

                # Backward
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += out.loss.item()

                # Optim step every grad_accum microbatches
                step_mod = (global_step + 1) % 1  # placeholder to keep logic similar
                if ( ( (global_step + 1) * args.batch_size ) % (args.batch_size * args.grad_accum) ) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if use_fp16:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    if sched is not None:
                        sched.step()
                    opt.zero_grad(set_to_none=True)

                # Update step counters for logging / early stop
                # Count "optimizer updates" rather than micro-batches:
                if ( ( (global_step + 1) * args.batch_size ) % (args.batch_size * args.grad_accum) ) == 0:
                    if args.max_updates and ((global_step + 1) // args.grad_accum) >= args.max_updates:
                        print(f"Reached max_updates={args.max_updates}, stopping early.")
                        model.save_pretrained(args.out_dir)
                        tok.save_pretrained(args.out_dir)
                        return

                # Log on micro-batch counts (cheap approximate)
                if (global_step + 1) % args.log_every == 0:
                    avg = running_loss / args.log_every
                    print(f"[epoch {ep+1}/{args.epochs}] micro_step {global_step+1}  loss={avg:.4f}  kept={kept_count}")
                    running_loss = 0.0

                global_step += 1

        # Save a checkpoint each epoch
        model.save_pretrained(os.path.join(args.out_dir, f"epoch{ep+1}"))
        tok.save_pretrained(os.path.join(args.out_dir, f"epoch{ep+1}"))

    # Final save
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Training complete. Saved to", args.out_dir)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # Make sampling deterministic if you like:
    random.seed(int(os.environ.get("SEED", "42")))
    main()

