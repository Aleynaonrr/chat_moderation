# Results on the held-out test set

Test set: official OffensEval-TR 2020 test split, 3528 messages (20.3% offensive). Same test set for every model.

Latency: one message at a time (batch size 1), mean of 200 runs on CPU.

Hardware: Windows 11 | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 16 cores |  | torch 2.15.0+cpu, 10 threads | Python 3.13.5

| Model | Area | Precision | Threshold | Macro-F1 | Accuracy | F1 (OFF) | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Lexicon + edit distance | B | rules | 0.5 | 0.6063 | 0.6635 | 0.4563 | 0.0 | 4.17 | 10.0 | yes |
| tf-idf + LogReg | B | FP64 | 0.57 | 0.7662 | 0.8569 | 0.6206 | 6.9 | 1.22 | 1.62 | yes |
| word2vec + FFN | A | FP32 | 0.56 | 0.6717 | 0.7959 | 0.4698 | 13.1 | 0.15 | 0.23 | yes |
| BERTurk | C | FP32 | 0.48 | 0.8129 | 0.8835 | 0.698 | 442.6 | 33.42 | 59.37 | yes |
| BERTurk quantized | D | INT8 (dynamic) | 0.48 | 0.8073 | 0.8846 | 0.6852 | 186.0 | 22.18 | 29.93 | yes |