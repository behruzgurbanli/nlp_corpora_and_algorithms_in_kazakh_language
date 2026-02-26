# Kazakh NLP Project (P1 + P2) — CLI and Streamlit

This repository contains an end-to-end NLP pipeline on a Kazakh (kk-Latn) news corpus.

- **Project 1**: corpus creation, preprocessing, QC, tokenization, Heaps' law, BPE, sentence segmentation, spell checking.
- **Project 2**: n-gram language modeling, smoothing comparison, and logistic-regression-based sentence boundary task.

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
│   │   └── p2/                  # Project 2 task modules
│   ├── ui/
│   │   ├── app.py               # Project 1 UI
│   │   └── app_p2.py            # Project 2 UI
│   └── scripts/                 # run scripts
└── README.md
```

## 2) Environment

Typical dependencies:

```bash
pip install streamlit pandas numpy matplotlib pyyaml requests beautifulsoup4 lxml tqdm scikit-learn
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

## 6) Notes

- Project 2 results will change when the corpus grows (after new scraping).
- Re-run preprocessing before rerunning P2 tasks.
- P2 UI supports changing essential parameters live (useful for class demo questions).

---

**Authors:** Behruz Gurbanli & Madina Kylyshkanova  
**Course:** Natural Language Processing  
**Semester:** Spring 2026
