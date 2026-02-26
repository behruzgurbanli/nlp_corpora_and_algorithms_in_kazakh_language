# Project 2 Working Report

This document explains what we implemented for Project 2 in simple language, what each task means, and the current results on the latest processed dataset.

## Motivation

The goal of this project is to build and evaluate practical NLP methods on a real Kazakh (kk-Latn) news corpus.  
We focus on language modeling and sentence boundary detection, because both are core tasks used in many NLP systems.

Main questions:
- How well can unigram, bigram, and trigram models represent this dataset?
- Which smoothing method works best when data is sparse?
- Can a simple machine learning model decide if a dot (`.`) is the end of a sentence?

The setting is an expanded dataset created in Project 1 and reused in Project 2 after preprocessing.

## Method

We used the following methods:

1. **N-gram language models (Task 1)**
- Unigram, bigram, trigram models.
- Perplexity used as the main metric.
- Baseline is unsmoothed maximum-likelihood n-gram modeling.

2. **Smoothing methods (Task 2)**
- Laplace
- Interpolation
- Backoff
- Kneser-Ney
- Compared by perplexity on train/dev/test.

3. **Sentence boundary classification (Task 4)**
- Logistic Regression classifier for each dot (`.`): boundary vs non-boundary.
- Two regularizations compared: **L1** and **L2**.
- Metrics: accuracy, precision, recall, F1.

## 1) Dataset Used

- Input file: `data/processed/qz_kazakh_latn_clean_norm.jsonl`
- Total documents: **2473**
- Split used in all P2 tasks:
  - Train: **1978**
  - Dev: **247**
  - Test: **248**
- Split is reproducible (seed = 42).

Why split matters:
- **Train** is for learning model parameters.
- **Dev** is for choosing method/hyperparameters.
- **Test** is final check of generalization.

---

## 2) Task 1: Unigram, Bigram, Trigram + Perplexity

### What this task asks
Build language models of order 1, 2, and 3, then compute perplexity.

- **Unigram**: predicts a word without context.
- **Bigram**: predicts a word from the previous word.
- **Trigram**: predicts a word from the previous two words.

### What perplexity means
Perplexity is a standard language model score.
- Lower perplexity = model predicts the text better.
- Very high or infinite perplexity means poor coverage (usually unseen n-grams).

### What we implemented
- Vocabulary built from train set (`min_count=2`).
- Rare words mapped to `<UNK>`.
- Sentence boundary token `</s>` used.
- Unsmoothed MLE probabilities for Task 1 baseline.

### Current Task 1 results

| Model | Train PPL | Dev PPL | Test PPL | Notes |
|---|---:|---:|---:|---|
| Unigram | 2794.3946 | 2018.0583 | 2038.6795 | finite on all splits |
| Bigram | 29.8488 | inf | inf | unseen bigrams on dev/test |
| Trigram | 3.7229 | inf | inf | more sparsity than bigram |

Interpretation:
- Train perplexity drops strongly from unigram → bigram → trigram (expected).
- Dev/test become infinite for bigram/trigram without smoothing (also expected).

---

## 3) Task 2: Smoothing Comparison

### What this task asks
Apply smoothing methods and decide which is best for this dataset.

Methods required:
1. Laplace
2. Interpolation
3. Backoff
4. Kneser-Ney

### Why smoothing is needed
Task 1 showed a key problem: unseen n-grams produce zero probability, giving infinite perplexity.  
Smoothing redistributes probability mass so unseen patterns get non-zero probability.

### Current Task 2 results

| Method | Train PPL | Dev PPL | Test PPL |
|---|---:|---:|---:|
| Laplace | 6727.2160 | 11473.7357 | 10759.1095 |
| Interpolation | 5.2679 | 672.9280 | 532.7362 |
| Backoff | 3.7229 | 503.2842 | 398.3171 |
| Kneser-Ney | 8.1682 | **335.1296** | **277.2628** |

Best method (lowest dev perplexity): **Kneser-Ney**

Interpretation:
- Kneser-Ney gives the strongest generalization on dev/test.
- Laplace is much worse here (common in practice for higher-order models).

---

## 4) Task 4: Logistic Regression for Dot = End of Sentence

### What this task asks
Use logistic regression to decide whether each `.` is a sentence boundary.
Then compare:
- L1 regularization
- L2 regularization

### What we implemented
- Built dot-level features around each `.`:
  - previous/next chars
  - uppercase/digit indicators
  - nearby words
- Trained two models:
  - Logistic Regression (L1)
  - Logistic Regression (L2)
- Compared accuracy, precision, recall, F1 on train/dev/test.
- Added sentence-detection preview output.

### Current Task 4 results

Dot examples:
- Train: 31,902 (positive = 31,141)
- Dev: 4,676 (positive = 4,590)
- Test: 3,946 (positive = 3,855)

| Model | Split | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| L1 | Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| L1 | Dev | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| L1 | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| L2 | Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| L2 | Dev | 0.9991 | 0.9991 | 1.0000 | 0.9996 |
| L2 | Test | 0.9997 | 0.9997 | 1.0000 | 0.9999 |

Best model by dev F1: **L1**

Important note:
- These labels are auto-derived from the segmentation pipeline, so scores are likely optimistic.
- For stricter evaluation, manually labeled dot boundaries would be stronger.

---

## 5) Extra Task: UI for Results

We implemented a **separate P2 UI**:
- File: `src/ui/app_p2.py`
- Run: `./src/scripts/run_ui_p2.sh`

UI supports:
- Task 1 run + parameter controls + result tables
- Task 2 run + smoothing parameter controls + best-method display
- Task 4 run + L1/L2 parameter controls + metrics + sentence preview
- JSON report download buttons for each task

This is useful for live demos where the teacher asks to rerun with different values.

---

## 6) Final Summary (Current Snapshot)

- Task 1 completed: unigram, bigram, trigram + perplexity.
- Task 2 completed: four smoothing methods compared; **Kneser-Ney is best** on dev/test.
- Task 4 completed: logistic regression L1 vs L2 comparison implemented; current labels produce near-perfect scores.
- Extra task completed: separate interactive UI for P2 tasks.

All outputs are reproducible via YAML configs and CLI/UI runs.

## Experiments

We ran experiments task-by-task on the same train/dev/test split for consistency.

### Experiment A: Task 1 baseline
- Trained unsmoothed unigram/bigram/trigram models.
- Observed that bigram/trigram dev-test perplexity became infinite due to unseen n-grams.
- This is an important negative result and directly motivates smoothing.

### Experiment B: Task 2 smoothing comparison
- Applied four smoothing methods to address unseen-event sparsity.
- Compared dev/test perplexity.
- Result: **Kneser-Ney** gave the best generalization (lowest dev perplexity).

### Experiment C: Task 4 L1 vs L2 logistic regression
- Built dot-level feature vectors and trained L1 and L2 logistic models.
- Both achieved very high scores; L1 selected as best by dev F1.
- Error analysis note: labels were auto-derived from segmentation logic, so the evaluation is likely optimistic.

### Baseline and error analysis summary
- Baseline used: unsmoothed n-gram model (Task 1).
- Main error source for LM: unseen n-grams in higher-order models.
- Main caution for Task 4: near-perfect numbers likely reflect non-independent labels.

## Team Contributions

Update this section before final submission if needed.

- **Behruz Gurbanli**
  - Implemented and integrated Project 2 code (Task 1, Task 2, Task 4)
  - Prepared CLI/config/UI workflow for rerunning experiments
  - Interpreted results and selected best smoothing method

- **Madina Kylyshkanova**
  - Please add specific contributions here (e.g., analysis, experiments, presentation, report polishing, validation)
