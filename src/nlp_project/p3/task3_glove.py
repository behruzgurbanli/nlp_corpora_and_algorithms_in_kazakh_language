#!/usr/bin/env python3
"""
Project 3 - Task 3:
- Train GloVe using the official binaries
- Evaluate similar words and vector arithmetic
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from nlp_project.p3.common import load_docs, doc_sentence_tokens
from nlp_project.p3.task2_word2vec import DEFAULT_EQUATIONS, DEFAULT_PAIR_WORDS, DEFAULT_TARGET_WORDS


@dataclass(frozen=True)
class P3Task3GloveConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    remove_timestamps: bool = True

    glove_build_dir: Path = Path("third_party/GloVe/build")
    work_dir: Path = Path("data/processed/p3_glove")
    out_dir: Path = Path("data/reports/p3_task3_glove")
    out_json: Path = Path("data/reports/p3_task3_glove_report.json")
    reuse_existing_vectors: bool = False
    existing_vectors_path: Path = Path("data/processed/p3_glove/kazakh_glove.txt")

    memory: float = 4.0
    vocab_min_count: int = 3
    vector_size: int = 100
    max_iter: int = 15
    window_size: int = 5
    binary: int = 0
    threads: int = 4
    x_max: int = 100
    verbose: int = 2


def _write_glove_corpus(cfg: P3Task3GloveConfig) -> tuple[Path, int, int]:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(
        cfg.inp_jsonl,
        text_field=cfg.text_field,
        remove_timestamps=cfg.remove_timestamps,
    )
    corpus_path = cfg.work_dir / "corpus.txt"
    lines_written = 0
    total_tokens = 0
    with corpus_path.open("w", encoding="utf-8") as fout:
        for doc in docs:
            for sent in doc_sentence_tokens(doc["text"], lowercase=cfg.lowercase):
                if not sent:
                    continue
                fout.write(" ".join(sent) + "\n")
                lines_written += 1
                total_tokens += len(sent)
    return corpus_path, lines_written, total_tokens


def _run_glove_binary(bin_path: Path, args: List[str], *, stdin_path: Path | None = None, stdout_path: Path | None = None) -> None:
    cmd = [str(bin_path)] + args
    fin = stdin_path.open("rb") if stdin_path else None
    fout = stdout_path.open("wb") if stdout_path else None
    try:
        subprocess.run(
            cmd,
            check=True,
            stdin=fin if stdin_path else None,
            stdout=fout if stdout_path else None,
        )
    finally:
        if fin is not None:
            fin.close()
        if fout is not None:
            fout.close()


def _load_glove_vectors(path: Path) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            vectors[parts[0]] = np.array(parts[1:], dtype=float)
    return vectors


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _most_similar(word: str, vectors: Dict[str, np.ndarray], topn: int = 10):
    if word not in vectors:
        return None
    target = vectors[word]
    scores = []
    for other_word, vec in vectors.items():
        if other_word == word:
            continue
        scores.append((other_word, _cosine_similarity(target, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]


def _analogy(eq: dict, vectors: Dict[str, np.ndarray]):
    missing = [w for w in eq["positive"] + eq["negative"] if w not in vectors]
    if missing:
        return [], missing

    result_vec = np.zeros_like(next(iter(vectors.values())))
    for w in eq["positive"]:
        result_vec += vectors[w]
    for w in eq["negative"]:
        result_vec -= vectors[w]

    banned = set(eq["positive"] + eq["negative"])
    scores = []
    for word, vec in vectors.items():
        if word in banned:
            continue
        scores.append((word, _cosine_similarity(result_vec, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5], []


def run_p3_task3_glove(cfg: P3Task3GloveConfig) -> dict:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")
    if not cfg.reuse_existing_vectors and not cfg.glove_build_dir.exists():
        raise FileNotFoundError(f"Missing GloVe build directory: {cfg.glove_build_dir}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    corpus_path, corpus_lines, total_tokens = _write_glove_corpus(cfg)
    vocab_path = cfg.work_dir / "vocab.txt"
    cooccur_path = cfg.work_dir / "cooccurrence.bin"
    shuffle_path = cfg.work_dir / "cooccurrence.shuf.bin"
    save_prefix = cfg.work_dir / "kazakh_glove"
    vectors_path = cfg.work_dir / "kazakh_glove.txt"
    training_mode = "trained_in_repo"

    if cfg.reuse_existing_vectors and cfg.existing_vectors_path.exists():
        vectors_path = cfg.existing_vectors_path
        training_mode = "reused_existing_vectors"
    else:
        _run_glove_binary(
            cfg.glove_build_dir / "vocab_count",
            ["-min-count", str(cfg.vocab_min_count), "-verbose", str(cfg.verbose)],
            stdin_path=corpus_path,
            stdout_path=vocab_path,
        )
        _run_glove_binary(
            cfg.glove_build_dir / "cooccur",
            [
                "-memory", str(cfg.memory),
                "-vocab-file", str(vocab_path),
                "-verbose", str(cfg.verbose),
                "-window-size", str(cfg.window_size),
            ],
            stdin_path=corpus_path,
            stdout_path=cooccur_path,
        )
        _run_glove_binary(
            cfg.glove_build_dir / "shuffle",
            ["-memory", str(cfg.memory), "-verbose", str(cfg.verbose)],
            stdin_path=cooccur_path,
            stdout_path=shuffle_path,
        )
        subprocess.run(
            [
                str(cfg.glove_build_dir / "glove"),
                "-save-file", str(save_prefix),
                "-threads", str(cfg.threads),
                "-input-file", str(shuffle_path),
                "-x-max", str(cfg.x_max),
                "-iter", str(cfg.max_iter),
                "-vector-size", str(cfg.vector_size),
                "-binary", str(cfg.binary),
                "-vocab-file", str(vocab_path),
                "-verbose", str(cfg.verbose),
            ],
            check=True,
        )

    vectors = _load_glove_vectors(vectors_path)

    similar_csv = cfg.out_dir / "similar_words.csv"
    pairs_csv = cfg.out_dir / "pairwise_similarities.csv"
    equations_csv = cfg.out_dir / "vector_equations.csv"
    params_csv = cfg.out_dir / "parameters.csv"

    similar_words: Dict[str, List[dict] | None] = {}
    with similar_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_word", "similar_word", "similarity"])
        for word in DEFAULT_TARGET_WORDS:
            sims = _most_similar(word, vectors, topn=10)
            if sims is None:
                similar_words[word] = None
                writer.writerow([word, "OOV_NOT_IN_VOCAB", ""])
                continue
            items = [{"word": w, "score": float(s)} for w, s in sims]
            similar_words[word] = items
            for item in items:
                writer.writerow([word, item["word"], round(item["score"], 4)])

    pairwise_rows: List[dict] = []
    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_1", "word_2", "cosine_similarity"])
        for w1, w2 in DEFAULT_PAIR_WORDS:
            sim = None
            if w1 in vectors and w2 in vectors:
                sim = _cosine_similarity(vectors[w1], vectors[w2])
            pairwise_rows.append({"word_1": w1, "word_2": w2, "similarity": sim})
            writer.writerow([w1, w2, "" if sim is None else round(sim, 4)])

    equations_result: List[dict] = []
    with equations_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["equation", "result_word", "score", "missing_words"])
        for eq in DEFAULT_EQUATIONS:
            sims, missing = _analogy(eq, vectors)
            if missing:
                equations_result.append({"equation": eq["name"], "missing_words": missing, "results": []})
                writer.writerow([eq["name"], "", "", ", ".join(missing)])
                continue
            items = [{"word": w, "score": float(s)} for w, s in sims]
            equations_result.append({"equation": eq["name"], "missing_words": [], "results": items})
            for item in items:
                writer.writerow([eq["name"], item["word"], round(item["score"], 4), ""])

    with params_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerows(
            [
                ["vector_size", cfg.vector_size],
                ["window_size", cfg.window_size],
                ["vocab_min_count", cfg.vocab_min_count],
                ["max_iter", cfg.max_iter],
                ["x_max", cfg.x_max],
                ["threads", cfg.threads],
                ["binary", cfg.binary],
                ["memory", cfg.memory],
            ]
        )

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "text_field": cfg.text_field,
            "corpus_lines": corpus_lines,
            "total_tokens": total_tokens,
            "trained_vocabulary": len(vectors),
        },
        "settings": {
            "lowercase": cfg.lowercase,
            "remove_timestamps": cfg.remove_timestamps,
            "glove_build_dir": cfg.glove_build_dir.as_posix(),
            "work_dir": cfg.work_dir.as_posix(),
            "training_mode": training_mode,
            "reuse_existing_vectors": cfg.reuse_existing_vectors,
            "existing_vectors_path": cfg.existing_vectors_path.as_posix(),
            "memory": cfg.memory,
            "vocab_min_count": cfg.vocab_min_count,
            "vector_size": cfg.vector_size,
            "max_iter": cfg.max_iter,
            "window_size": cfg.window_size,
            "binary": cfg.binary,
            "threads": cfg.threads,
            "x_max": cfg.x_max,
            "verbose": cfg.verbose,
        },
        "target_words": list(DEFAULT_TARGET_WORDS),
        "similar_words": similar_words,
        "pairwise_similarities": pairwise_rows,
        "vector_equations": equations_result,
        "artifacts": {
            "corpus_path": corpus_path.as_posix(),
            "vocab_path": vocab_path.as_posix(),
            "cooccur_path": cooccur_path.as_posix(),
            "shuffle_path": shuffle_path.as_posix(),
            "vectors_path": vectors_path.as_posix(),
            "similar_words_csv": similar_csv.as_posix(),
            "pairwise_csv": pairs_csv.as_posix(),
            "equations_csv": equations_csv.as_posix(),
            "parameters_csv": params_csv.as_posix(),
        },
    }
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p3_task3_glove_report(r: dict, out_json: Path) -> str:
    ds = r["dataset"]
    st = r["settings"]
    lines = []
    lines.append("=== PROJECT 3 - TASK 3: GLOVE ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(f"Corpus lines: {ds['corpus_lines']}\n")
    lines.append(f"Total tokens: {ds['total_tokens']}\n")
    lines.append(f"Trained vocabulary: {ds['trained_vocabulary']}\n")
    lines.append(
        f"Params: vector_size={st['vector_size']} window_size={st['window_size']} "
        f"min_count={st['vocab_min_count']} max_iter={st['max_iter']}\n"
    )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)
