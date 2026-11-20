"""Prepares everything the app needs on a fresh clone (runs only what is missing).

1) data      -> downloads OffensEval-TR 2020 (src/data.py)
2) lexicon + tf-idf, word2vec + FFN -> trained locally on CPU in about 1 minute
3) BERTurk   -> our fine-tuned model is downloaded from Hugging Face Hub
Run on its own:  python -m src.setup_models
"""
from .config import DATA_PROCESSED, MODELS_DIR

BERT_DIR = MODELS_DIR / "berturk-offensive"
HF_REPO_ID = "aleynaaonrr/berturk-offensive-tr"   # fine-tuned on Google Colab (T4), see src/train_bert.py


def ensure_models():
    if not all((DATA_PROCESSED / f"{s}.csv").exists() for s in ("train", "val", "test")):
        print("[setup] Downloading OffensEval-TR 2020 ...")
        from .data import prepare
        prepare()

    if not ((MODELS_DIR / "lexicon.json").exists() and (MODELS_DIR / "tfidf_lr.joblib").exists()):
        print("[setup] Training the lexicon matcher and the tf-idf baseline ...")
        from .train_classic import main as train_classic
        train_classic()

    if not ((MODELS_DIR / "w2v.kv").exists() and (MODELS_DIR / "w2v_ffn.pt").exists()):
        print("[setup] Training word2vec + feedforward network ...")
        from .train_w2v_ffn import main as train_w2v_ffn
        train_w2v_ffn()

    if not (BERT_DIR / "config.json").exists():
        print(f"[setup] Downloading the fine-tuned BERTurk from Hugging Face ({HF_REPO_ID}) ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO_ID, local_dir=str(BERT_DIR))


if __name__ == "__main__":
    ensure_models()
    print("[setup] All models are ready.")