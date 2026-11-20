"""Downloads the OffensEval-TR 2020 dataset and creates train / val / test splits.
Run:  python -m src.data

Source: Çöltekin (2020), A Corpus of Turkish Offensive Language on Social Media, LREC. CC-BY 2.0
"""
import io
import sys
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PROCESSED, DATA_RAW, SEED

ZIP_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
HF_REPO = "coltekin/offenseval2020_tr"
LABEL2ID = {"NOT": 0, "OFF": 1}


def _from_hf_parquet():
    """Option 1: download the auto-converted parquet files from Hugging Face Hub."""
    from huggingface_hub import HfApi, hf_hub_download
    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files(HF_REPO, repo_type="dataset", revision=rev)
             if f.endswith(".parquet")]
    out = {}
    for split in ("train", "test"):
        parts = [f for f in files if f"/{split}/" in f or f.split("/")[-1].startswith(split)]
        df = pd.concat([pd.read_parquet(hf_hub_download(HF_REPO, p, repo_type="dataset", revision=rev))
                        for p in parts], ignore_index=True)
        out[split] = pd.DataFrame({"tweet": df["tweet"], "label": df["subtask_a"].astype(int)})
    print("Downloaded from Hugging Face.")
    return out["train"], out["test"]


def _from_zip():
    """Option 2: download the original zip file from the dataset's website."""
    import requests
    r = requests.get(ZIP_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    z.extractall(DATA_RAW)
    read = lambda n: pd.read_csv(io.BytesIO(z.read(n)), sep="\t", quoting=3)
    tr = read(next(n for n in names if "train" in n.lower() and n.endswith(".tsv")))
    te = read(next(n for n in names if "test" in n.lower() and n.endswith(".tsv")))
    # Test labels are stored in a separate file -> join them by tweet id
    lab = pd.read_csv(io.BytesIO(z.read(next(n for n in names if "label" in n.lower()))),
                      header=None, names=["id", "subtask_a"], sep=None, engine="python")
    lab = lab[lab["subtask_a"].isin(LABEL2ID)].astype({"id": int})
    te = te.merge(lab, on="id")
    print("Downloaded from the original zip.")
    return (pd.DataFrame({"tweet": tr["tweet"], "label": tr["subtask_a"].map(LABEL2ID)}),
            pd.DataFrame({"tweet": te["tweet"], "label": te["subtask_a"].map(LABEL2ID)}))


def load_raw():
    """Try each download source in order until one works."""
    for fn in (_from_hf_parquet, _from_zip):
        try:
            return fn()
        except Exception as e:
            print(f"[{fn.__name__}] failed: {e}")
    sys.exit("Could not download the dataset.")


def prepare():
    train_full, test = load_raw()
    train_full, test = train_full.dropna(), test.dropna()
    # Split the official training data 90% train / 10% val (stratified by label).
    # The official test set is kept untouched as the held-out test set for ALL models.
    train, val = train_test_split(train_full, test_size=0.10,
                                  stratify=train_full["label"], random_state=SEED)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.reset_index(drop=True).to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(df):6d} messages | OFF ratio = {df['label'].mean():.3f}")


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split (train / val / test); create the splits first if needed."""
    path = DATA_PROCESSED / f"{name}.csv"
    if not path.exists():
        prepare()
    df = pd.read_csv(path)
    return df.astype({"tweet": str, "label": int})


if __name__ == "__main__":
    prepare()