# scripts/train_byt5_train_only.py
import argparse, datasets, torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

def load_lines(p):
    with open(p,'r',encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='work/hf_data')
    ap.add_argument('--model_name', default='google/byt5-small')
    ap.add_argument('--out_dir', default='work/byt5-ckpt')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max_len', type=int, default=2048)
    args = ap.parse_args()

    train_src = load_lines(f"{args.data_dir}/train.src")
    train_tgt = load_lines(f"{args.data_dir}/train.tgt")

    tok = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    enc = tok(train_src, padding=True, truncation=True, max_length=args.max_len, return_tensors=None)
    with tok.as_target_tokenizer():
        lab = tok(train_tgt, padding=True, truncation=True, max_length=args.max_len, return_tensors=None)

    train_ds = datasets.Dataset.from_dict({**enc, "labels": lab["input_ids"]})
    coll = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    tr_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=100,
        bf16=torch.cuda.is_available(),
        predict_with_generate=False
    )
    trainer = Seq2SeqTrainer(model=model, args=tr_args, train_dataset=train_ds, data_collator=coll, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()
