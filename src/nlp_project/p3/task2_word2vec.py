#!/usr/bin/env python3
"""
Project 3 - Task 2:
- Train Word2Vec
- Evaluate similar words and vector arithmetic
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from nlp_project.p3.common import load_docs, doc_sentence_tokens


DEFAULT_TARGET_WORDS = (
    "qazaqstan",
    "prezıdenti",
    "memlekettik",
    "halyqaralyq",
    "ulttyq",
    "adam",
    "jumys",
    "ortalyq",
    "tramp",
    "aqsh",
)

DEFAULT_PAIR_WORDS = (
    ("tramp", "donald"),
    ("aqsh", "tramp"),
    ("jumys", "isteý"),
    ("qazaqstan", "prezıdenti"),
    ("ulttyq", "halyqaralyq"),
)

DEFAULT_EQUATIONS = (
    {"name": "tramp - aqsh + qazaqstan", "positive": ["tramp", "qazaqstan"], "negative": ["aqsh"]},
    {"name": "aqsh - tramp + prezıdenti", "positive": ["aqsh", "prezıdenti"], "negative": ["tramp"]},
    {"name": "jumys + isteý", "positive": ["jumys", "isteý"], "negative": []},
    {"name": "ulttyq + halyqaralyq", "positive": ["ulttyq", "halyqaralyq"], "negative": []},
    {"name": "qazaqstan + prezıdenti - aqsh", "positive": ["qazaqstan", "prezıdenti"], "negative": ["aqsh"]},
)


@dataclass(frozen=True)
class P3Task2Word2VecConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    remove_timestamps: bool = True

    use_bigrams: bool = True
    phrase_min_count: int = 5
    phrase_threshold: float = 10.0

    vector_size: int = 100
    window: int = 5
    min_count: int = 3
    workers: int = 4
    sg: int = 1
    negative: int = 10
    epochs: int = 15
    seed: int = 42

    target_words: tuple[str, ...] = DEFAULT_TARGET_WORDS
    pair_words: tuple[tuple[str, str], ...] = DEFAULT_PAIR_WORDS

    out_dir: Path = Path("data/reports/p3_task2_word2vec")
    out_json: Path = Path("data/reports/p3_task2_word2vec_report.json")


def _require_gensim():
    try:
        from gensim.models import Word2Vec
        from gensim.models.phrases import Phrases, Phraser
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Word2Vec requires gensim. Activate the conda env that has gensim installed before running Task 2."
        ) from e
    return Word2Vec, Phrases, Phraser


def _prepare_sequences(docs: List[Dict[str, Any]], *, lowercase: bool) -> List[List[str]]:
    sequences: List[List[str]] = []
    for doc in docs:
        sequences.extend(doc_sentence_tokens(doc["text"], lowercase=lowercase))
    return [seq for seq in sequences if len(seq) > 1]


def run_p3_task2_word2vec(cfg: P3Task2Word2VecConfig) -> dict:
    Word2Vec, Phrases, Phraser = _require_gensim()

    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)

    docs = load_docs(
        cfg.inp_jsonl,
        text_field=cfg.text_field,
        remove_timestamps=cfg.remove_timestamps,
    )
    sequences = _prepare_sequences(docs, lowercase=cfg.lowercase)
    total_tokens = sum(len(seq) for seq in sequences)
    distinct_words_before_training = len({tok for seq in sequences for tok in seq})

    if cfg.use_bigrams:
        phrases = Phrases(sequences, min_count=cfg.phrase_min_count, threshold=cfg.phrase_threshold)
        bigram = Phraser(phrases)
        sequences = [bigram[seq] for seq in sequences]

    model = Word2Vec(
        sentences=sequences,
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        workers=cfg.workers,
        sg=cfg.sg,
        negative=cfg.negative,
        epochs=cfg.epochs,
        seed=cfg.seed,
    )

    model_path = cfg.out_dir / "word2vec.model"
    vectors_path = cfg.out_dir / "word2vec_vectors.txt"
    similar_csv = cfg.out_dir / "similar_words.csv"
    pairs_csv = cfg.out_dir / "pairwise_similarities.csv"
    equations_csv = cfg.out_dir / "vector_equations.csv"
    params_csv = cfg.out_dir / "parameters.csv"

    model.save(str(model_path))
    model.wv.save_word2vec_format(str(vectors_path), binary=False)

    similar_words: Dict[str, List[dict] | None] = {}
    with similar_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_word", "similar_word", "similarity"])
        for word in cfg.target_words:
            if word not in model.wv:
                similar_words[word] = None
                writer.writerow([word, "OOV_NOT_IN_VOCAB", ""])
                continue
            sims = [{"word": w, "score": float(s)} for w, s in model.wv.most_similar(word, topn=10)]
            similar_words[word] = sims
            for item in sims:
                writer.writerow([word, item["word"], round(item["score"], 4)])

    pairwise_rows: List[dict] = []
    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_1", "word_2", "cosine_similarity"])
        for w1, w2 in cfg.pair_words:
            sim = None
            if w1 in model.wv and w2 in model.wv:
                sim = float(model.wv.similarity(w1, w2))
            pairwise_rows.append({"word_1": w1, "word_2": w2, "similarity": sim})
            writer.writerow([w1, w2, "" if sim is None else round(sim, 4)])

    equations_result: List[dict] = []
    with equations_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["equation", "result_word", "score", "missing_words"])
        for eq in DEFAULT_EQUATIONS:
            missing = [w for w in eq["positive"] + eq["negative"] if w not in model.wv]
            if missing:
                equations_result.append({"equation": eq["name"], "missing_words": missing, "results": []})
                writer.writerow([eq["name"], "", "", ", ".join(missing)])
                continue
            sims = model.wv.most_similar(positive=eq["positive"], negative=eq["negative"], topn=5)
            results = [{"word": w, "score": float(s)} for w, s in sims]
            equations_result.append({"equation": eq["name"], "missing_words": [], "results": results})
            for item in results:
                writer.writerow([eq["name"], item["word"], round(item["score"], 4), ""])

    with params_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerows(
            [
                ["vector_size", cfg.vector_size],
                ["window", cfg.window],
                ["min_count", cfg.min_count],
                ["workers", cfg.workers],
                ["sg", cfg.sg],
                ["negative", cfg.negative],
                ["epochs", cfg.epochs],
                ["seed", cfg.seed],
                ["use_bigrams", cfg.use_bigrams],
                ["phrase_min_count", cfg.phrase_min_count],
                ["phrase_threshold", cfg.phrase_threshold],
            ]
        )

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "text_field": cfg.text_field,
            "documents": len(docs),
            "training_sequences": len(sequences),
            "total_tokens": total_tokens,
            "distinct_words_before_training": distinct_words_before_training,
            "trained_vocabulary": len(model.wv),
        },
        "settings": {
            "lowercase": cfg.lowercase,
            "remove_timestamps": cfg.remove_timestamps,
            "vector_size": cfg.vector_size,
            "window": cfg.window,
            "min_count": cfg.min_count,
            "workers": cfg.workers,
            "sg": cfg.sg,
            "negative": cfg.negative,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "use_bigrams": cfg.use_bigrams,
            "phrase_min_count": cfg.phrase_min_count,
            "phrase_threshold": cfg.phrase_threshold,
        },
        "target_words": list(cfg.target_words),
        "similar_words": similar_words,
        "pairwise_similarities": pairwise_rows,
        "vector_equations": equations_result,
        "artifacts": {
            "model_path": model_path.as_posix(),
            "vectors_path": vectors_path.as_posix(),
            "similar_words_csv": similar_csv.as_posix(),
            "pairwise_csv": pairs_csv.as_posix(),
            "equations_csv": equations_csv.as_posix(),
            "parameters_csv": params_csv.as_posix(),
        },
    }
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p3_task2_word2vec_report(r: dict, out_json: Path) -> str:
    ds = r["dataset"]
    st = r["settings"]
    lines = []
    lines.append("=== PROJECT 3 - TASK 2: WORD2VEC ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(f"Documents: {ds['documents']}\n")
    lines.append(f"Training sequences: {ds['training_sequences']}\n")
    lines.append(f"Total tokens: {ds['total_tokens']}\n")
    lines.append(f"Trained vocabulary: {ds['trained_vocabulary']}\n")
    lines.append(
        f"Params: vector_size={st['vector_size']} window={st['window']} min_count={st['min_count']} "
        f"epochs={st['epochs']} sg={st['sg']}\n"
    )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)

