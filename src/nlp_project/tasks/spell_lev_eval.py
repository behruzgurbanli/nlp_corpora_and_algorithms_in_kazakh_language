#!/usr/bin/env python3
"""
Evaluation for Levenshtein spell checker (Accuracy@1 and Accuracy@5).

Preserves original behavior:
- sample N=300 from vocab words with len>=5
- inject 1-edit typo
- measure top-1 and top-5 hit rates
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from nlp_project.tasks.spell_lev import (
    load_vocab_tsv,
    make_typo,
    suggest,
)


@dataclass(frozen=True)
class SpellLevEvalConfig:
    vocab_file: Path

    n_test: int = 300
    seed: int = 42
    min_word_len: int = 5

    topn: int = 5
    max_len_diff: int = 2
    require_same_first_letter: bool = True
    alphabet: str = "abcdefghijklmnopqrstuvwxyzáóúǵńýıŋşçöüä"

    max_miss_examples: int = 10


def run_spell_lev_eval(cfg: SpellLevEvalConfig) -> dict:
    if not cfg.vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_file}")

    vocab, freqs = load_vocab_tsv(cfg.vocab_file)
    rng = random.Random(cfg.seed)

    candidates = [w for w in vocab if len(w) >= cfg.min_word_len]
    if len(candidates) < cfg.n_test:
        raise ValueError(f"Not enough candidate words (len>={cfg.min_word_len}). Have {len(candidates)}, need {cfg.n_test}.")

    sample = rng.sample(candidates, cfg.n_test)

    hit1 = 0
    hit5 = 0
    examples_miss: List[Tuple[str, str, List[str]]] = []

    for w in sample:
        typo = make_typo(w, rng, cfg.alphabet)
        scored = suggest(
            typo, vocab, freqs,
            topn=cfg.topn,
            max_len_diff=cfg.max_len_diff,
            require_same_first_letter=cfg.require_same_first_letter,
        )
        preds = [x[2] for x in scored]

        if preds and preds[0] == w:
            hit1 += 1
        if w in preds:
            hit5 += 1
        else:
            if len(examples_miss) < cfg.max_miss_examples:
                examples_miss.append((w, typo, preds))

    return {
        "n_test": cfg.n_test,
        "hit1": hit1,
        "hit5": hit5,
        "acc1": hit1 / cfg.n_test,
        "acc5": hit5 / cfg.n_test,
        "miss_examples": examples_miss,
    }


def format_spell_lev_eval_report(r: dict) -> str:
    lines = []
    lines.append("=== TASK 5 EVALUATION: Levenshtein Spell Checker ===\n")
    lines.append(f"Test words: {r['n_test']}\n")
    lines.append(f"Accuracy@1: {r['acc1']:.4f}  ({r['hit1']}/{r['n_test']})\n")
    lines.append(f"Accuracy@5: {r['acc5']:.4f}  ({r['hit5']}/{r['n_test']})\n")

    if r["miss_examples"]:
        lines.append("\nExamples where correct word not in top-5:\n")
        for w, typo, preds in r["miss_examples"]:
            lines.append(f"True={w} | Typo={typo} | Top5={preds}\n")

    return "".join(lines)
