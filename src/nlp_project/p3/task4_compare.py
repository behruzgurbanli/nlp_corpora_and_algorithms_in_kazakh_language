#!/usr/bin/env python3
"""
Project 3 - Task 4:
- Compare GloVe and Word2Vec results
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class P3Task4CompareConfig:
    word2vec_json: Path = Path("data/reports/p3_task2_word2vec_report.json")
    glove_json: Path = Path("data/reports/p3_task3_glove_report.json")
    overlap_top_k: int = 5
    out_json: Path = Path("data/reports/p3_task4_compare_report.json")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _top_words(items: List[dict] | None, k: int) -> List[str]:
    if not items:
        return []
    return [item["word"] for item in items[:k]]


def run_p3_task4_compare(cfg: P3Task4CompareConfig) -> dict:
    w2v = _load_json(cfg.word2vec_json)
    glove = _load_json(cfg.glove_json)

    query_words = sorted(set(w2v.get("target_words", [])) | set(glove.get("target_words", [])))
    per_word = []
    for word in query_words:
        w2v_top = _top_words(w2v["similar_words"].get(word), cfg.overlap_top_k)
        glove_top = _top_words(glove["similar_words"].get(word), cfg.overlap_top_k)
        overlap = sorted(set(w2v_top) & set(glove_top))
        union = set(w2v_top) | set(glove_top)
        jaccard = (len(overlap) / len(union)) if union else None
        per_word.append(
            {
                "query_word": word,
                "word2vec_top": w2v_top,
                "glove_top": glove_top,
                "overlap": overlap,
                "jaccard": jaccard,
            }
        )

    pair_map_w2v = {(x["word_1"], x["word_2"]): x["similarity"] for x in w2v.get("pairwise_similarities", [])}
    pair_map_glove = {(x["word_1"], x["word_2"]): x["similarity"] for x in glove.get("pairwise_similarities", [])}
    pair_keys = sorted(set(pair_map_w2v) | set(pair_map_glove))
    pairwise = []
    for key in pair_keys:
        pairwise.append(
            {
                "word_1": key[0],
                "word_2": key[1],
                "word2vec_similarity": pair_map_w2v.get(key),
                "glove_similarity": pair_map_glove.get(key),
            }
        )

    valid_jaccards = [x["jaccard"] for x in per_word if x["jaccard"] is not None]
    result = {
        "inputs": {
            "word2vec_json": cfg.word2vec_json.as_posix(),
            "glove_json": cfg.glove_json.as_posix(),
            "overlap_top_k": cfg.overlap_top_k,
        },
        "summary": {
            "query_words_compared": len(query_words),
            "average_jaccard_overlap": (sum(valid_jaccards) / len(valid_jaccards)) if valid_jaccards else None,
        },
        "per_word_overlap": per_word,
        "pairwise_similarity_comparison": pairwise,
    }
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p3_task4_compare_report(r: dict, out_json: Path) -> str:
    sm = r["summary"]
    lines = []
    lines.append("=== PROJECT 3 - TASK 4: GLOVE VS WORD2VEC ===\n")
    lines.append(f"Query words compared: {sm['query_words_compared']}\n")
    avg = sm["average_jaccard_overlap"]
    lines.append(f"Average top-k overlap: {'NA' if avg is None else f'{avg:.4f}'}\n")
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)
