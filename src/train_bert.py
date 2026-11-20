"""Area C - Transformer: fine-tune BERTurk (dbmdz/bert-base-turkish-cased) for clean / offensive classification.

Needs a GPU (we ran it on Google Colab, T4).  Run:  python -m src.train_bert
The best epoch (by val macro-F1) is saved to models/berturk-offensive/
"""
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import MODELS_DIR, SEED
from .data import load_split
from .normalize import normalize
from .utils import macro_f1, tune_threshold

BERT_BASE = "dbmdz/bert-base-turkish-cased"
BERT_DIR = MODELS_DIR / "berturk-offensive"
MAX_LEN = 64       # tweets are short; a shorter max length also means lower latency
EPOCHS = 3
BATCH = 32
LR = 2e-5


def prep(texts):
    # light normalization; keep casing because BERTurk is a cased model
    return [normalize(t, lower=False) for t in texts]


@torch.no_grad()
def predict_proba(model, tok, texts, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(prep(texts[i:i + batch]), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train, val = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}
    ).to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    data = list(zip(prep(train["tweet"].tolist()), train["label"].tolist()))
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    use_amp = device == "cuda"                      # mixed precision on GPU = faster training
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, val_texts = -1.0, val["tweet"].tolist()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} | loss {running / step:.4f}")

        val_probs = predict_proba(model, tok, val_texts, device)
        f1 = macro_f1(val["label"], val_probs)
        print(f"epoch {epoch} | val macro-F1 = {f1:.4f} | {time.time() - t0:.0f}s")

        if f1 > best_f1:                            # keep only the best epoch
            best_f1 = f1
            threshold = tune_threshold(val["label"], val_probs)
            model.save_pretrained(BERT_DIR)
            tok.save_pretrained(BERT_DIR)
            (BERT_DIR / "threshold.json").write_text(
                json.dumps({"threshold": threshold, "val_macro_f1": round(f1, 4)}))
            print(f"  -> saved (threshold tuned on val = {threshold})")

    print(f"Best val macro-F1 = {best_f1:.4f} | model saved to {BERT_DIR}")


if __name__ == "__main__":
    main()