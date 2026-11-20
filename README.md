# Real-Time Chat Moderation (Turkish) 

A gaming platform must flag offensive chat messages in Turkish **within a strict latency budget on a CPU-only server**.
This app labels every message live as **clean / offensive** with a confidence score, masks flagged messages,
logs the response time of every message and lets the user switch between five models.

- **Fine-tuned BERTurk (Hugging Face Hub):** https://huggingface.co/aleynaaonrr/berturk-offensive-tr
- **Dataset:** OffensEval-TR 2020 (Çöltekin, 2020)
- **Latency budget:** 50 ms per message on CPU → met by the INT8-quantized BERTurk

## Quick start

Requirements: Python 3.11–3.13, internet connection on the first run.

```bash
git clone <THIS_REPOSITORY_URL>
cd chat_moderation

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Run command:**

```bash
python app.py
```

The app opens at http://127.0.0.1:7860.

On the **first run** the app prepares everything that is not in the repository (`src/setup_models.py`):
1. downloads OffensEval-TR 2020,
2. trains the lexicon matcher, tf-idf baseline and word2vec + FFN on CPU (about 1 minute),
3. downloads the fine-tuned BERTurk (about 440 MB) from Hugging Face Hub.

Later runs start in about 10–20 seconds.

## Models (model ladder)

| # | Model | Technique area | Precision | Trained with |
|---|---|---|---|---|
| 1 | Lexicon + edit distance (rule baseline) | B | rules | `src/train_classic.py` |
| 2 | tf-idf (word 1–2 + char 2–5 grams) + Logistic Regression | B | FP64 | `src/train_classic.py` |
| 3 | Own word2vec (skip-gram) + feedforward network (SGD, backprop, dropout) | A | FP32 | `src/train_w2v_ffn.py` |
| 4 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned, 3 epochs | C | FP32 | `src/train_bert.py` (Google Colab, T4 GPU) |
| 5 | BERTurk, dynamic quantization of all Linear layers | D | INT8 | `src/bert_model.py` |

All text goes through the regex normalization in `src/normalize.py` (leetspeak, repeated letters, URLs/mentions,
Turkish lower-casing, Turkish → ASCII folding for the lexicon).

## Results (held-out test set)

Official OffensEval-TR 2020 test split, 3,528 messages (20.3% offensive), **the same test set for all models**.
Latency: one message at a time (batch size 1), mean of 200 runs on CPU.
Hardware: Windows 11, Intel CPU (Intel64 Family 6 Model 186), 16 cores, PyTorch 2.15 CPU, 10 threads.

| Model | Macro-F1 | Size (MB) | Latency (ms) | ≤ 50 ms |
|---|---|---|---|---|
| Lexicon + edit distance | 0.6063 | 0.004 | 4.17 | yes |
| tf-idf + LogReg | 0.7662 | 6.9 | 1.22 | yes |
| word2vec + FFN | 0.6717 | 13.1 | 0.15 | yes |
| BERTurk FP32 | 0.8129 | 442.6 | 33.42 | yes |
| **BERTurk INT8 (used in the app)** | **0.8073** | **186.0** | **22.18** | **yes** |

Full table: `results/results.md` · misclassified examples: `results/errors_bert_int8.csv`

## Reproducing the experiments

```bash
python -m src.data            # download data, create train / val / test splits
python -m src.train_classic   # lexicon matcher + tf-idf baseline
python -m src.train_w2v_ffn   # word2vec + feedforward network
python -m src.train_bert      # BERTurk fine-tuning (needs a GPU; we used Google Colab T4)
python -m src.evaluate        # all models on the test set -> results/
```

Splits: the official training file (31,756 tweets) is split 90% train / 10% validation (stratified, seed 42);
the official test file (3,528 tweets) is the held-out test set. Decision thresholds are tuned on validation only.

## Project structure

```
app.py                 Gradio chat interface (default model: BERTurk INT8)
src/config.py          paths and settings
src/data.py            dataset download and splits
src/normalize.py       regex normalization (Area B)
src/edit_distance.py   Levenshtein distance + lexicon matcher (Area B)
src/train_classic.py   lexicon + tf-idf baseline (Area B)
src/train_w2v_ffn.py   word2vec + feedforward network (Area A)
src/train_bert.py      BERTurk fine-tuning (Area C)
src/bert_model.py      BERTurk inference, FP32 / INT8 dynamic quantization (Area D)
src/evaluate.py        evaluation of all models -> results/
src/setup_models.py    prepares missing data / models on the first run
src/utils.py           metrics, threshold tuning, latency measurement
```

## Data source

Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020.
https://coltekin.github.io/offensive-turkish/ (CC-BY 2.0). Downloaded from Hugging Face: `coltekin/offenseval2020_tr`.