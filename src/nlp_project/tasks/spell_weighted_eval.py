#!/usr/bin/env python3
"""
Evaluation: Weighted vs Baseline spellchecking.

Typos are generated using the same confusables as the synthetic confusion builder.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from nlp_project.tasks.spell_lev import load_vocab_tsv, levenshtein
from nlp_project.tasks.spell_weighted import suggest_weighted
from nlp_project.tasks.weighted_ed import load_confusion


CONFUSABLES: Dict[str, List[str]] = {
    "i": ["ı"], "ı": ["i"],
    "y": ["ý"], "ý": ["y"],
    "g": ["ǵ"], "ǵ": ["g"],
    "n": ["ń"], "ń": ["n"],
    "o": ["ó", "ö"], "ó": ["o"], "ö": ["o"],
    "u": ["ú", "ü"], "ú": ["u"], "ü": ["u"],
    "s": ["ş"], "ş": ["s"],
    "c": ["ç"], "ç": ["c"],
    "a": ["ä"], "ä": ["a"],
}


@dataclass(frozen=True)
class SpellWeightedEvalConfig:
    vocab_file: Path
    conf_file: Path

    n_test: int = 300
    seed: int = 42
    min_word_len: int = 5

    topn: int = 5
    max_len_diff: int = 2


def make_confusable_typo(word: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(word) if ch in CONFUSABLES]
    if not positions:
        # fallback: random substitution (kept close to your intention; your old fallback was a bit odd)
        i = rng.randrange(len(word))
        return word[:i] + rng.choice(word) + word[i + 1 :]
    i = rng.choice(positions)
    ch = word[i]
    new_ch = rng.choice(CONFUSABLES[ch])
    return word[:i] + new_ch + word[i + 1 :]


def baseline_suggest(word: str, vocab: List[str], freqs: Dict[str, int], *, topn: int, max_len_diff: int):
    L = len(word)
    first = word[0] if word else ""

    strict = [v for v in vocab if abs(len(v) - L) <= max_len_diff and (not first or v[0] == first)]
    if not strict:
        strict = [v for v in vocab if abs(len(v) - L) <= max_len_diff]

    scored = []
    for v in strict:
        d = levenshtein(word, v)
        scored.append((d, -freqs.get(v, 0), v))
    scored.sort()
    return [x[2] for x in scored[:topn]]


def run_spell_weighted_eval(cfg: SpellWeightedEvalConfig) -> dict:
    vocab, freqs = load_vocab_tsv(cfg.vocab_file)
    confusion, _ = load_confusion(cfg.conf_file)

    rng = random.Random(cfg.seed)
    candidates = [w for w in vocab if len(w) >= cfg.min_word_len]
    sample = rng.sample(candidates, cfg.n_test)

    b1 = b5 = 0
    w1 = w5 = 0

    for w in sample:
        typo = make_confusable_typo(w, rng)

        base = baseline_suggest(typo, vocab, freqs, topn=cfg.topn, max_len_diff=cfg.max_len_diff)
        if base:
            if base[0] == w:
                b1 += 1
            if w in base:
                b5 += 1

        weighted = [x[2] for x in suggest_weighted(typo, vocab, freqs, confusion, topn=cfg.topn, max_len_diff=cfg.max_len_diff)]
        if weighted:
            if weighted[0] == w:
                w1 += 1
            if w in weighted:
                w5 += 1

    return {
        "n_test": cfg.n_test,
        "baseline_hit1": b1,
        "baseline_hit5": b5,
        "weighted_hit1": w1,
        "weighted_hit5": w5,
        "baseline_acc1": b1 / cfg.n_test,
        "baseline_acc5": b5 / cfg.n_test,
        "weighted_acc1": w1 / cfg.n_test,
        "weighted_acc5": w5 / cfg.n_test,
    }


def format_spell_weighted_eval_report(r: dict) -> str:
    lines = []
    lines.append("=== EXTRA TASK EVALUATION: Weighted vs Baseline ===\n")
    lines.append(f"Test words: {r['n_test']}\n")
    lines.append(f"Baseline  Accuracy@1: {r['baseline_acc1']:.4f} ({r['baseline_hit1']}/{r['n_test']})\n")
    lines.append(f"Baseline  Accuracy@5: {r['baseline_acc5']:.4f} ({r['baseline_hit5']}/{r['n_test']})\n")
    lines.append(f"Weighted  Accuracy@1: {r['weighted_acc1']:.4f} ({r['weighted_hit1']}/{r['n_test']})\n")
    lines.append(f"Weighted  Accuracy@5: {r['weighted_acc5']:.4f} ({r['weighted_hit5']}/{r['n_test']})\n")
    return "".join(lines)
