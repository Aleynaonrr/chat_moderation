"""Shared helpers: metrics, threshold tuning, latency measurement, hardware info."""
import os
import platform
import time

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, probs, threshold: float = 0.5) -> float:
    preds = (np.asarray(probs) >= threshold).astype(int)
    return f1_score(y_true, preds, average="macro")


def tune_threshold(y_val, probs_val) -> float:
    """Pick the decision threshold that maximizes macro-F1 on the VALIDATION set.
    The test set is never used here."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = macro_f1(y_val, probs_val, t)
        if f > best_f:
            best_t, best_f = float(t), f
    return round(best_t, 2)


def measure_latency(predict_one, texts, n_runs: int = 200, warmup: int = 10) -> dict:
    """Latency of a single message (batch size 1) in ms, averaged over n_runs (>= 100)."""
    for t in texts[:warmup]:
        predict_one(t)
    times = []
    for i in range(n_runs):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - start) * 1000)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "runs": n_runs}


def hardware_info() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 1e9:.1f} GB RAM"
        cores = f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical cores"
    except ImportError:
        ram, cores = "", f"{os.cpu_count()} cores"
    try:
        import torch
        th = f"torch {torch.__version__}, {torch.get_num_threads()} threads"
    except ImportError:
        th = ""
    return f"{platform.system()} {platform.release()} | {cpu} | {cores} | {ram} | {th} | Python {platform.python_version()}"


def file_size_mb(*paths) -> float:
    return sum(os.path.getsize(p) for p in paths) / 1e6