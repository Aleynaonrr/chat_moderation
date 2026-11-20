"""Evaluates ALL models on the SAME held-out test set (official OffensEval-TR 2020 test, 3,528 tweets).

Reports: macro-F1 (main metric, classes are imbalanced), accuracy, F1 of the offensive class,
model size (MB) and CPU latency per message (batch size 1, mean over 200 runs).
Outputs:  results/results.csv, results/results.md, results/errors_bert_int8.csv
Run:      python -m src.evaluate
"""
import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .bert_model import LATENCY_BUDGET_MS, BertClassifier
from .config import MODELS_DIR, RESULTS_DIR
from .data import load_split
from .edit_distance import LexiconMatcher
from .train_w2v_ffn import W2VFFNModel
from .utils import file_size_mb, hardware_info, measure_latency

N_LATENCY_RUNS = 200


def load_models():
    """(name, technique area, precision, predict_proba(list[str]) -> probs, threshold, size in MB)"""
    lex = LexiconMatcher.load(MODELS_DIR / "lexicon.json")
    tfidf = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
    w2v = W2VFFNModel()
    bert32 = BertClassifier(quantized=False)
    bert8 = BertClassifier(quantized=True)
    return [
        ("Lexicon + edit distance", "B", "rules", lex.predict_proba, 0.5,
         file_size_mb(MODELS_DIR / "lexicon.json")),
        ("tf-idf + LogReg", "B", "FP64", lambda t: tfidf["pipeline"].predict_proba(list(t))[:, 1],
         tfidf["threshold"], file_size_mb(MODELS_DIR / "tfidf_lr.joblib")),
        ("word2vec + FFN", "A", "FP32", w2v.predict_proba, w2v.threshold,
         file_size_mb(*glob.glob(str(MODELS_DIR / "w2v.kv*")), MODELS_DIR / "w2v_ffn.pt")),
        ("BERTurk", "C", "FP32", bert32.predict_proba, bert32.threshold, bert32.size_mb()),
        ("BERTurk quantized", "D", "INT8 (dynamic)", bert8.predict_proba, bert8.threshold, bert8.size_mb()),
    ]


def main():
    test = load_split("test")
    texts, y = test["tweet"].tolist(), test["label"].values
    hw = hardware_info()
    print(f"Test set: {len(texts)} messages | OFF ratio = {y.mean():.3f}")
    print(f"Hardware: {hw}\n")

    rows = []
    for name, area, prec, predict, thr, size in load_models():
        probs = np.asarray(predict(texts))
        preds = (probs >= thr).astype(int)
        if name.startswith("Lexicon"):
            # measure cold lookups (no word cache from the batch run above)
            lex_words = LexiconMatcher.load(MODELS_DIR / "lexicon.json").lexicon
            lat = measure_latency(lambda t: LexiconMatcher(lex_words).predict_proba([t]),
                                  texts, n_runs=N_LATENCY_RUNS)
        else:
            lat = measure_latency(lambda t: predict([t]), texts, n_runs=N_LATENCY_RUNS)
        row = {
            "Model": name, "Area": area, "Precision": prec, "Threshold": thr,
            "Macro-F1": round(f1_score(y, preds, average="macro"), 4),
            "Accuracy": round(accuracy_score(y, preds), 4),
            "F1 (OFF)": round(f1_score(y, preds, pos_label=1), 4),
            "Size (MB)": round(size, 1),
            "Latency mean (ms)": round(lat["mean_ms"], 2),
            "Latency p95 (ms)": round(lat["p95_ms"], 2),
            f"<= {LATENCY_BUDGET_MS:.0f} ms": "yes" if lat["mean_ms"] <= LATENCY_BUDGET_MS else "no",
        }
        rows.append(row)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        print(f"{name:25s} macro-F1 {row['Macro-F1']:.4f} | {row['Size (MB)']:7.1f} MB | "
              f"{row['Latency mean (ms)']:7.2f} ms | TN {tn} FP {fp} FN {fn} TP {tp}")

        if name == "BERTurk quantized":   # failure examples for the presentation
            df = test.assign(prob_off=probs.round(3), pred=preds)
            fn_df = df[(df.label == 1) & (df.pred == 0)].sort_values("prob_off").head(15)
            fp_df = df[(df.label == 0) & (df.pred == 1)].sort_values("prob_off", ascending=False).head(15)
            pd.concat([fn_df.assign(error="false negative (missed offensive)"),
                       fp_df.assign(error="false positive (clean flagged)")]) \
              .to_csv(RESULTS_DIR / "errors_bert_int8.csv", index=False, encoding="utf-8-sig")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "results.csv", index=False, encoding="utf-8-sig")
    md = ["# Results on the held-out test set", "",
          f"Test set: official OffensEval-TR 2020 test split, {len(texts)} messages "
          f"({y.mean():.1%} offensive). Same test set for every model.", "",
          f"Latency: one message at a time (batch size 1), mean of {N_LATENCY_RUNS} runs on CPU.", "",
          f"Hardware: {hw}", "",
          "| " + " | ".join(res.columns) + " |",
          "|" + "---|" * len(res.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in res.values]
    (RESULTS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: results/results.csv, results/results.md, results/errors_bert_int8.csv")


if __name__ == "__main__":
    main()