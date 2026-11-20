"""Area B - Classical baselines:
  1) Edit-distance lexicon matcher (rule baseline)
  2) tf-idf (word 1-2 grams + char 2-5 grams) + Logistic Regression

Run:  python -m src.train_classic
"""
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODELS_DIR, SEED
from .data import load_split
from .edit_distance import LexiconMatcher, build_lexicon
from .normalize import norm_text
from .utils import macro_f1, tune_threshold

LEXICON_PATH = MODELS_DIR / "lexicon.json"
TFIDF_PATH = MODELS_DIR / "tfidf_lr.joblib"

def build_tfidf_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=norm_text, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, max_features=100_000)),
        ("char", TfidfVectorizer(preprocessor=norm_text, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True, max_features=200_000)),
    ])
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    return Pipeline([("tfidf", features), ("clf", clf)])


def main():
    train, val = load_split("train"), load_split("val")

    # ---- 1) Lexicon matcher ----
    t0 = time.time()
    lexicon = build_lexicon(train["tweet"], train["label"])
    matcher = LexiconMatcher(lexicon)
    matcher.save(LEXICON_PATH)
    f1 = macro_f1(val["label"], matcher.predict_proba(val["tweet"]))
    print(f"[lexicon] {len(lexicon)} words | val macro-F1 = {f1:.4f} | {time.time() - t0:.1f}s")

    # ---- 2) tf-idf + Logistic Regression ----
    t0 = time.time()
    pipe = build_tfidf_pipeline()
    pipe.fit(train["tweet"], train["label"])
    val_probs = pipe.predict_proba(val["tweet"])[:, 1]
    threshold = tune_threshold(val["label"], val_probs)   # chosen on VAL, never on test
    joblib.dump({"pipeline": pipe, "threshold": threshold}, TFIDF_PATH)
    print(f"[tfidf+LR] val macro-F1 @0.5 = {macro_f1(val['label'], val_probs):.4f} | "
          f"@{threshold} = {macro_f1(val['label'], val_probs, threshold):.4f} | {time.time() - t0:.1f}s")
    print(f"Saved: {LEXICON_PATH.name}, {TFIDF_PATH.name}")


if __name__ == "__main__":
    main()