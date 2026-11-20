"""Area C + D - BERTurk inference on CPU, in FP32 or with dynamic INT8 quantization.

Dynamic quantization: the weights of every nn.Linear layer are stored as 8-bit integers
(4x smaller than FP32) and activations are quantized on the fly -> smaller and faster on CPU.
Run (quick test):  python -m src.bert_model
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MODELS_DIR
from .normalize import normalize

BERT_DIR = MODELS_DIR / "berturk-offensive"
MAX_LEN = 64
LATENCY_BUDGET_MS = 50.0   # our budget: one message must be classified in <= 50 ms on CPU


def _pick_quant_engine():
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return None


class BertClassifier:
    def __init__(self, quantized: bool = True, model_dir: Path = BERT_DIR):
        self.quantized = quantized
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).float().eval()
        if quantized:
            _pick_quant_engine()
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.threshold = json.loads((Path(model_dir) / "threshold.json").read_text())["threshold"]

    @torch.inference_mode()
    def predict_proba(self, texts, batch: int = 64) -> np.ndarray:
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok([normalize(t, lower=False) for t in texts[i:i + batch]], truncation=True,
                           max_length=MAX_LEN, padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, 1])
        return torch.cat(out).numpy()

    def size_mb(self) -> float:
        """Size of the serialized weights (state_dict) in MB."""
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getbuffer().nbytes / 1e6


if __name__ == "__main__":
    msgs = ["bugün maç çok güzeldi", "sen tam bir s4l4ksın", "yarın görüşürüz kanka"]
    for q in (False, True):
        m = BertClassifier(quantized=q)
        name = "BERTurk INT8" if q else "BERTurk FP32"
        for t in msgs:                           # warm-up
            m.predict_proba([t])
        times = []
        for i in range(100):
            s = time.perf_counter()
            m.predict_proba([msgs[i % len(msgs)]])
            times.append((time.perf_counter() - s) * 1000)
        ms = float(np.mean(times))
        print(f"{name}: size = {m.size_mb():6.1f} MB | latency = {ms:6.1f} ms/msg "
              f"| budget {LATENCY_BUDGET_MS:.0f} ms -> {'OK' if ms <= LATENCY_BUDGET_MS else 'OVER'}")
        for t, p in zip(msgs, m.predict_proba(msgs)):
            print(f"    {t!r:28} P(offensive) = {p:.3f}")