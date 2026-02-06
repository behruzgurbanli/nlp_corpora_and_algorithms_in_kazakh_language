# Kazakh NLP Corpus Pipeline — Demo UI & CLI

This project implements an end-to-end NLP corpus pipeline for **Kazakh (kk-Latn)** news text, including preprocessing, quality control, tokenization, vocabulary analysis, Heaps’ Law, BPE, sentence segmentation, and spell checking (baseline + weighted).

The project is designed with **three layers**:

1. **Library layer** (`src/nlp_project/`) — pure Python logic  
2. **CLI layer** (`nlp_project.cli`) — reproducible task execution via YAML  
3. **Demo UI** (`src/ui/app.py`) — interactive Streamlit interface for live demos  

The UI runs the **real tasks**, not mock outputs.

---

## 1. Running the Demo UI (Recommended for Presentation)

This is the **primary way** to demonstrate the project.

### Step 1 — Activate environment (e.g., anaconda) (Optional)
```bash
conda activate base
```

### Step 2 — Run the UI
From the project root:
```bash
./src/scripts/run_ui.sh
```

or manually:
```bash
streamlit run src/ui/app.py
```

### Step 3 — Open in browser
Streamlit will print something like:
```
Local URL: http://localhost:8501
```

Open it in your browser.

### What the UI does
- Lets you **select a task** (tokenize, vocab, BPE, spellcheck, etc.)
- Loads the **same YAML configs** used by the CLI
- Executes the **real pipeline code**
- Displays results **directly in the browser**
- Allows downloading outputs (CSV, TSV, MD, PNG)

No file browsing or terminal commands are needed during the demo.

---

## 2. Project Structure (High-Level)

```
project/
├── configs/                 # YAML configs (single source of truth)
├── data/
│   ├── raw/                 # Scraped data
│   ├── processed/           # Cleaned + derived artifacts
│   └── reports/             # Markdown / plots
├── src/
│   ├── nlp_project/         # Core library code
│   │   ├── preprocess/
│   │   ├── qc/
│   │   ├── tasks/
│   │   └── common/
│   ├── ui/                  # Streamlit Demo UI
│   └── scripts/             # Pipeline runners
└── README.md
```

---

## 3. Running the Pipeline via CLI (Reproducible)

All CLI commands use YAML configs under `configs/`.

### Preprocessing pipeline
```bash
./src/scripts/run_preprocess.sh
```

This runs:
1. Raw data audit  
2. Text cleaning  
3. Metadata normalization  
4. Corpus summary report  

---

## 4. Running Individual Tasks via CLI

All tasks are exposed via `nlp_project.cli`.

### Example: Tokenization
```bash
python -m nlp_project.cli task tokenize \
  --config configs/task_tokenize.yaml
```

### Example: Vocabulary
```bash
python -m nlp_project.cli task vocab \
  --config configs/task_vocab.yaml
```

### Example: Heaps’ Law
```bash
python -m nlp_project.cli task heaps \
  --config configs/task_heaps.yaml
```

### Example: BPE Training
```bash
python -m nlp_project.cli task bpe-train \
  --config configs/task_bpe_train.yaml
```

---

## 5. Sentence Segmentation (Important Note)

Sentence segmentation was handled as follows:

- **Rule-based segmentation** using punctuation heuristics  
- **Manual GOLD annotation** for evaluation  
- Evaluation compares **boundary positions**, not sentence strings  
- Precision / Recall / F1 are computed over boundary matches  

The manual work was **only for GOLD data creation**, not during runtime.

---

## 6. Spell Checking

Two spellcheckers are implemented:

### Baseline
- Standard Levenshtein distance  
- Length filtering + first-letter constraint  

### Weighted
- Uses a **synthetic confusion matrix**  
- Lower substitution cost for realistic Kazakh diacritic confusions  
- Evaluated against the baseline using Accuracy@1 and Accuracy@5  

---

## 7. Design Philosophy

- **YAML-first configuration** (UI and CLI share the same configs)  
- **Thin CLI wrappers**  
- **UI runs real code**, not hardcoded outputs  
- Modular, testable, and presentation-friendly  

---

## 8. Requirements

- Python 3.10+  
- streamlit  
- pandas  
- numpy  
- matplotlib  
- pyyaml  

Install dependencies as needed:
```bash
pip install streamlit pandas numpy matplotlib pyyaml
```

---

## 9. Intended Use

- Academic NLP coursework  
- Live demonstrations  
- Corpus analysis experiments  
- Reproducible NLP pipelines  

---

**Author:** Behruz Gurbanli  & Madina Kylyshkanova
**Course:** Natural Language Processing  
**Semester:** Spring 2026
