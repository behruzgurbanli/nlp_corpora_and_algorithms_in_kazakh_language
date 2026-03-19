#!/usr/bin/env python3
"""
Project 3 - Task 1:
- Describe dataset size and frequency statistics
- Build term-document and word-word matrices
- Save matrix-form visualizations
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import lil_matrix

from nlp_project.p3.common import DEFAULT_STOPWORDS, corpus_word_counts, doc_word_tokens, load_docs


@dataclass(frozen=True)
class P3Task1DatasetConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    remove_timestamps: bool = True

    top_n: int = 20
    tdm_max_docs_visual: int = 50
    frequent_min_freq: int = 3
    stopwords: tuple[str, ...] = field(default_factory=lambda: tuple(sorted(DEFAULT_STOPWORDS)))

    out_dir: Path = Path("data/reports/p3_task1")
    out_json: Path = Path("data/reports/p3_task1_report.json")


def _build_term_document_matrix(doc_texts: List[str], vocabulary: List[str], *, lowercase: bool):
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    matrix = lil_matrix((len(vocabulary), len(doc_texts)), dtype=int)

    for j, text in enumerate(doc_texts):
        counts = Counter(doc_word_tokens(text, lowercase=lowercase))
        for word, freq in counts.items():
            idx = vocab_index.get(word)
            if idx is not None:
                matrix[idx, j] = freq
    return matrix


def _build_word_word_matrix(doc_texts: List[str], vocabulary: List[str], *, lowercase: bool):
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    vocab_set = set(vocabulary)
    matrix = lil_matrix((len(vocabulary), len(vocabulary)), dtype=int)

    for text in doc_texts:
        present = sorted(set(w for w in doc_word_tokens(text, lowercase=lowercase) if w in vocab_set))
        for i, w1 in enumerate(present):
            for j, w2 in enumerate(present):
                if i == j:
                    continue
                matrix[vocab_index[w1], vocab_index[w2]] += 1
    return matrix


def _save_barplot(top_words: List[tuple[str, int]], title: str, out_path: Path) -> None:
    words = [w for w, _ in top_words]
    freqs = [c for _, c in top_words]
    plt.figure(figsize=(12, 6))
    plt.bar(words, freqs)
    plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _save_matrix_plot(
    matrix,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    *,
    max_cols: int | None = None,
) -> None:
    arr = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    if max_cols is not None and arr.shape[1] > max_cols:
        arr = arr[:, :max_cols]
        col_labels = col_labels[:max_cols]

    plt.figure(figsize=(14, 8))
    plt.imshow(arr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(col_labels)), col_labels, rotation=90)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _top_words(counts: Counter, *, top_n: int, stopwords: set[str] | None) -> List[tuple[str, int]]:
    items = []
    for word, count in counts.items():
        if stopwords and word in stopwords:
            continue
        items.append((word, int(count)))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:top_n]


def run_p3_task1_dataset(cfg: P3Task1DatasetConfig) -> dict:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)

    docs = load_docs(
        cfg.inp_jsonl,
        text_field=cfg.text_field,
        remove_timestamps=cfg.remove_timestamps,
    )
    doc_texts = [d["text"] for d in docs if d["text"].strip()]
    word_counts = corpus_word_counts(docs, lowercase=cfg.lowercase)
    stopwords = set(cfg.stopwords)

    total_words = int(sum(word_counts.values()))
    distinct_words = len(word_counts)
    rare_words_freq_1 = sum(1 for _, c in word_counts.items() if c == 1)
    words_freq_2 = sum(1 for _, c in word_counts.items() if c == 2)
    frequent_words = sum(1 for _, c in word_counts.items() if c >= cfg.frequent_min_freq)

    top_raw = _top_words(word_counts, top_n=cfg.top_n, stopwords=None)
    top_filtered = _top_words(word_counts, top_n=cfg.top_n, stopwords=stopwords)

    stats_df = pd.DataFrame(
        [
            ("total_documents", len(doc_texts)),
            ("total_words", total_words),
            ("distinct_words", distinct_words),
            ("rare_words_freq_1", rare_words_freq_1),
            ("words_freq_2", words_freq_2),
            ("frequent_words_freq_gte_threshold", frequent_words),
            ("frequent_threshold", cfg.frequent_min_freq),
        ],
        columns=["metric", "value"],
    )
    stats_csv = cfg.out_dir / "dataset_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)

    top_raw_csv = cfg.out_dir / "top_words_raw.csv"
    top_filtered_csv = cfg.out_dir / "top_words_filtered.csv"
    pd.DataFrame(top_raw, columns=["word", "frequency"]).to_csv(top_raw_csv, index=False)
    pd.DataFrame(top_filtered, columns=["word", "frequency"]).to_csv(top_filtered_csv, index=False)

    raw_barplot = cfg.out_dir / "top_words_raw.png"
    filtered_barplot = cfg.out_dir / "top_words_filtered.png"
    _save_barplot(top_raw, "Top Frequent Words (Raw)", raw_barplot)
    _save_barplot(top_filtered, "Top Frequent Words (Stopwords Removed)", filtered_barplot)

    artifacts: Dict[str, str] = {
        "dataset_statistics_csv": stats_csv.as_posix(),
        "top_words_raw_csv": top_raw_csv.as_posix(),
        "top_words_filtered_csv": top_filtered_csv.as_posix(),
        "top_words_raw_png": raw_barplot.as_posix(),
        "top_words_filtered_png": filtered_barplot.as_posix(),
    }

    matrix_summaries: Dict[str, Any] = {}
    for name, vocab in (("raw", [w for w, _ in top_raw]), ("filtered", [w for w, _ in top_filtered])):
        tdm = _build_term_document_matrix(doc_texts, vocab, lowercase=cfg.lowercase)
        wwm = _build_word_word_matrix(doc_texts, vocab, lowercase=cfg.lowercase)

        tdm_csv = cfg.out_dir / f"tdm_top{cfg.top_n}_{name}.csv"
        wwm_csv = cfg.out_dir / f"wwm_top{cfg.top_n}_{name}.csv"
        tdm_png = cfg.out_dir / f"tdm_top{cfg.top_n}_{name}.png"
        wwm_png = cfg.out_dir / f"wwm_top{cfg.top_n}_{name}.png"

        doc_labels = [f"D{i + 1}" for i in range(len(doc_texts))]
        pd.DataFrame(tdm.toarray(), index=vocab, columns=doc_labels).to_csv(tdm_csv)
        pd.DataFrame(wwm.toarray(), index=vocab, columns=vocab).to_csv(wwm_csv)

        _save_matrix_plot(
            tdm,
            vocab,
            doc_labels,
            f"Term-Document Matrix ({name}, first {cfg.tdm_max_docs_visual} docs shown)",
            tdm_png,
            max_cols=cfg.tdm_max_docs_visual,
        )
        _save_matrix_plot(
            wwm,
            vocab,
            vocab,
            f"Word-Word Matrix ({name}, document-level co-occurrence)",
            wwm_png,
        )

        artifacts[f"tdm_{name}_csv"] = tdm_csv.as_posix()
        artifacts[f"wwm_{name}_csv"] = wwm_csv.as_posix()
        artifacts[f"tdm_{name}_png"] = tdm_png.as_posix()
        artifacts[f"wwm_{name}_png"] = wwm_png.as_posix()
        matrix_summaries[name] = {
            "vocabulary": vocab,
            "tdm_shape": list(tdm.shape),
            "wwm_shape": list(wwm.shape),
            "visualized_tdm_docs": min(cfg.tdm_max_docs_visual, len(doc_texts)),
            "cooccurrence_definition": "Two selected words co-occur if they appear in the same document.",
        }

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "text_field": cfg.text_field,
            "documents": len(doc_texts),
            "total_words": total_words,
            "distinct_words": distinct_words,
            "rare_words_freq_1": rare_words_freq_1,
            "words_freq_2": words_freq_2,
            "frequent_words_freq_gte_threshold": frequent_words,
            "frequent_threshold": cfg.frequent_min_freq,
        },
        "settings": {
            "lowercase": cfg.lowercase,
            "remove_timestamps": cfg.remove_timestamps,
            "top_n": cfg.top_n,
            "tdm_max_docs_visual": cfg.tdm_max_docs_visual,
            "stopword_count": len(stopwords),
        },
        "top_words_raw": [{"word": w, "frequency": c} for w, c in top_raw],
        "top_words_filtered": [{"word": w, "frequency": c} for w, c in top_filtered],
        "matrices": matrix_summaries,
        "artifacts": artifacts,
    }
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p3_task1_dataset_report(r: dict, out_json: Path) -> str:
    ds = r["dataset"]
    lines = []
    lines.append("=== PROJECT 3 - TASK 1: DATASET DESCRIPTION + MATRICES ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(f"Documents: {ds['documents']}\n")
    lines.append(f"Total words: {ds['total_words']}\n")
    lines.append(f"Distinct words: {ds['distinct_words']}\n")
    lines.append(f"Rare words (freq=1): {ds['rare_words_freq_1']}\n")
    lines.append(f"Words with freq=2: {ds['words_freq_2']}\n")
    lines.append(
        f"Frequent words (freq>={ds['frequent_threshold']}): {ds['frequent_words_freq_gte_threshold']}\n"
    )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)

