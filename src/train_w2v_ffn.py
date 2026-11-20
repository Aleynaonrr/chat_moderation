"""Area A - Neural network trained from scratch:
  1) our own word2vec (skip-gram) trained ONLY on the training messages
  2) message vector = [mean ; max] of its word vectors
  3) a feedforward network trained with SGD + backpropagation (dropout, class weights)

Run:  python -m src.train_w2v_ffn
"""
import time

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors, Word2Vec

from .config import MODELS_DIR, SEED
from .data import load_split
from .normalize import normalize_tokens
from .utils import macro_f1, tune_threshold

W2V_PATH = MODELS_DIR / "w2v.kv"
FFN_PATH = MODELS_DIR / "w2v_ffn.pt"
EMB_DIM = 100


class FFN(nn.Module):
    """Feedforward network: 200 -> 256 -> 64 -> 2"""

    def __init__(self, in_dim: int = 2 * EMB_DIM, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def message_vector(tokens: list[str], kv: KeyedVectors) -> np.ndarray:
    """[mean ; max] pooling of the word vectors; zeros if no known word."""
    vecs = [kv[w] for w in tokens if w in kv]
    if not vecs:
        return np.zeros(2 * kv.vector_size, dtype=np.float32)
    m = np.stack(vecs)
    return np.concatenate([m.mean(0), m.max(0)]).astype(np.float32)


def featurize(texts, kv) -> np.ndarray:
    return np.stack([message_vector(normalize_tokens(t), kv) for t in texts])


class W2VFFNModel:
    """Inference wrapper used by evaluate.py and the app."""

    def __init__(self):
        self.kv = KeyedVectors.load(str(W2V_PATH))
        ckpt = torch.load(FFN_PATH, map_location="cpu")
        self.net = FFN()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.threshold = ckpt["threshold"]

    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        x = torch.from_numpy(featurize(texts, self.kv))
        return torch.softmax(self.net(x), dim=1)[:, 1].numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")

    # ---- 1) word2vec on the training messages ----
    t0 = time.time()
    sentences = [normalize_tokens(t) for t in train["tweet"]]
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=5, min_count=2, sg=1,
                   epochs=15, workers=4, seed=SEED)
    w2v.wv.save(str(W2V_PATH))
    print(f"[word2vec] vocab = {len(w2v.wv):,} words | {time.time() - t0:.1f}s")

    # ---- 2) features ----
    x_tr = torch.from_numpy(featurize(train["tweet"], w2v.wv))
    y_tr = torch.tensor(train["label"].values)
    x_va = torch.from_numpy(featurize(val["tweet"], w2v.wv))

    # ---- 3) feedforward network, SGD + backpropagation ----
    net = FFN()
    pos_ratio = y_tr.float().mean().item()
    class_w = torch.tensor([1 / (1 - pos_ratio), 1 / pos_ratio]) / 2   # balance the two classes
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    best_f1, best_state, batch = -1.0, None, 64
    for epoch in range(1, 41):
        net.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_probs = torch.softmax(net(x_va), 1)[:, 1].numpy()
        f1 = macro_f1(val["label"], val_probs)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d} | train loss {total / len(perm):.4f} | val macro-F1 {f1:.4f}")

    # ---- 4) keep the best epoch (by val), tune threshold on val, save ----
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        val_probs = torch.softmax(net(x_va), 1)[:, 1].numpy()
    threshold = tune_threshold(val["label"], val_probs)
    torch.save({"state_dict": best_state, "threshold": threshold}, FFN_PATH)
    print(f"[w2v+FFN] best val macro-F1 @0.5 = {best_f1:.4f} | "
          f"@{threshold} = {macro_f1(val['label'], val_probs, threshold):.4f}")
    print(f"Saved: {W2V_PATH.name}, {FFN_PATH.name}")


if __name__ == "__main__":
    main()