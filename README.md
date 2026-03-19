# Kazakh NLP Project (P1 + P2 + P3) — CLI and Streamlit

This repository contains an end-to-end NLP pipeline on a Kazakh (kk-Latn) news corpus.

- **Project 1**: corpus creation, preprocessing, QC, tokenization, Heaps' law, BPE, sentence segmentation, spell checking.
- **Project 2**: n-gram language modeling, smoothing comparison, and logistic-regression-based sentence boundary task.
- **Project 3**: dataset statistics and matrix visualization, Word2Vec, GloVe, embedding comparison, and text classification with baseline + recurrent models.

The code is organized so you can rerun tasks with YAML configs and demo results in UI.

## 1) Project Structure

```text
project/
├── configs/                     # YAML configs for all tasks
├── data/
│   ├── raw/                     # scraped corpus
│   ├── processed/               # cleaned/normalized corpus + derived files
│   └── reports/                 # task outputs (json/md/csv/png)
├── src/
│   ├── nlp_project/
│   │   ├── common/              # config loader
│   │   ├── scrape/              # scraper
│   │   ├── preprocess/          # clean + metadata normalization
│   │   ├── qc/                  # audits and corpus summary
│   │   ├── tasks/               # Project 1 task modules
│   │   ├── p2/                  # Project 2 task modules
│   │   └── p3/                  # Project 3 task modules
│   ├── ui/
│   │   ├── app.py               # Project 1 UI
│   │   ├── app_p2.py            # Project 2 UI
│   │   └── app_p3.py            # Project 3 UI
│   └── scripts/                 # run scripts
├── third_party/
│   └── GloVe/                   # project-owned GloVe source
└── README.md
```

## 2) Environment

The project was developed in a Conda environment. A typical workflow is:

```bash
conda activate base
export PYTHONPATH=src
```

Typical Python dependencies:

```bash
pip install streamlit pandas numpy matplotlib pyyaml requests beautifulsoup4 lxml tqdm scikit-learn gensim torch
```

Notes:
- `gensim` is required for Word2Vec in Project 3.
- `torch` is required for Project 3 Task 5 neural models.
- If you use Conda, install the packages into the same environment you use to run the CLI and UI.

### GloVe Build

Project 3 uses a project-owned copy of the official GloVe source under `third_party/GloVe`.

If you need to rebuild the binaries:

```bash
cd third_party/GloVe
make clean
make
```

On macOS, if `make` fails due to Xcode setup, run:

```bash
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
```

## 3) Main Pipelines

### A) Extend dataset + preprocess

```bash
./src/scripts/run_extend_and_preprocess.sh
```

This does:
1. scrape append (`configs/scrape_qz_inform_extend.yaml`)
2. raw audit
3. cleaning
4. metadata normalization
5. corpus summary

### B) Scrape only

```bash
./src/scripts/run_scrape.sh
```

Or with a custom config:

```bash
./src/scripts/run_scrape.sh configs/scrape_qz_inform_extend.yaml
```

### C) Preprocess only

```bash
./src/scripts/run_preprocess.sh
```

## 4) Run UIs

### Project 1 UI

```bash
./src/scripts/run_ui.sh
```

### Project 2 UI (separate)

```bash
./src/scripts/run_ui_p2.sh
```

### Project 3 UI (separate)

```bash
./src/scripts/run_ui_p3.sh
```

## 5) Project 2 CLI Commands

All commands use `python3` with `PYTHONPATH=src`.

### Task 1: N-gram models + perplexity

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p2-ngram --config configs/task_p2_ngram.yaml
```

Output:
- `data/reports/p2_task1_ngram_report.json`

### Task 2: Smoothing comparison

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p2-smoothing --config configs/task_p2_smoothing.yaml
```

Output:
- `data/reports/p2_task2_smoothing_report.json`

### Task 4: Dot end-of-sentence with Logistic Regression (L1 vs L2)

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p2-dot-lr --config configs/task_p2_dot_lr.yaml
```

Output:
- `data/reports/p2_task4_dot_lr_report.json`

## 6) Project 3 CLI Commands

All commands use `python3` with `PYTHONPATH=src`.

### Task 1: Dataset description + matrices

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p3-dataset --config configs/task_p3_dataset.yaml
```

Outputs:
- `data/reports/p3_task1_report.json`
- `data/reports/p3_task1/`

### Task 2: Word2Vec

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p3-word2vec --config configs/task_p3_word2vec.yaml
```

Outputs:
- `data/reports/p3_task2_word2vec_report.json`
- `data/reports/p3_task2_word2vec/`

### Task 3: GloVe

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p3-glove --config configs/task_p3_glove.yaml
```

Outputs:
- `data/reports/p3_task3_glove_report.json`
- `data/reports/p3_task3_glove/`

### Task 4: Word2Vec vs GloVe comparison

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p3-compare --config configs/task_p3_compare.yaml
```

Output:
- `data/reports/p3_task4_compare_report.json`

### Task 5: Text classification

```bash
PYTHONPATH=src python3 -m nlp_project.cli task p3-classify --config configs/task_p3_classify.yaml
```

Outputs:
- `data/reports/p3_task5_classify_report.json`
- `data/reports/p3_task5_classify/comparison_table.csv`

Important config switch:
- `label_field: "category"` for the main 4-class setup
- `label_field: "subcategory"` for the harder 8-class setup

## 7) Project 3 UI Features

The Project 3 UI supports:
- viewing saved Task 1 statistics and matrix visualizations
- querying Word2Vec with custom words and analogies
- querying GloVe with custom words and analogies
- viewing Word2Vec vs GloVe comparison tables
- building interactive Task 5 predictors and testing custom input text live

This is intended for demo-style questioning such as:
- “run the model on this sentence”
- “test a different word”
- “show similar words for this input”

## 8) Notes

- Project 2 results will change when the corpus grows (after new scraping).
- Re-run preprocessing before rerunning P2 tasks.
- P2 UI supports changing essential parameters live (useful for class demo questions).
- Project 3 classification results can change depending on:
  - `label_field` (`category` vs `subcategory`)
  - feature type (`count`, `tfidf`, `pmi`, `word2vec`, `glove`)
  - model type (`logreg_baseline`, `rnn`, `birnn`, `lstm`)
- For imbalanced label settings, macro F1 is often more informative than raw accuracy.
- Generated Project 3 reports/models are ignored by `.gitignore`; rerun the commands above to regenerate them locally.

---

**Authors:** Behruz Gurbanli & Madina Kylyshkanova  
**Course:** Natural Language Processing  
**Semester:** Spring 2026
