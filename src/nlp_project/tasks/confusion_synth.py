#!/usr/bin/env python3
"""
Synthetic confusion matrix builder (substitutions only).

We generate controlled "realistic" substitution typos from corpus vocabulary, using:
- high-probability confusable substitutions (Kazakh Latin diacritics/near-letters)
- occasional random substitutions (low probability)

This is *not* derived from real user error logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple


DEFAULT_CONFUSABLES: Dict[str, List[str]] = {
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

DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyzáóúǵńýıŋşçöüä"


@dataclass(frozen=True)
class ConfusionSynthConfig:
    vocab_path: Path
    out_path: Path

    n_samples: int = 12000
    seed: int = 42

    confusable_prob: float = 0.85
    min_word_len: int = 4

    # allow overriding via YAML if needed
    alphabet: str = DEFAULT_ALPHABET
    confusables: Dict[str, List[str]] = None  # required by loader or set default


def load_vocab_words(path: Path, *, min_word_len: int) -> List[str]:
    vocab: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word = line.split("\t", 1)[0]
            if len(word) >= min_word_len:
                vocab.append(word)
    return vocab


def make_realistic_sub_typo(word: str, rng: random.Random, *, confusables: Dict[str, List[str]], alphabet: str, confusable_prob: float) -> str:
    positions = [i for i, ch in enumerate(word) if ch in confusables]
    use_confusable = (len(positions) > 0) and (rng.random() < confusable_prob)

    if use_confusable:
        i = rng.choice(positions)
        ch = word[i]
        new_ch = rng.choice(confusables[ch])
    else:
        i = rng.randrange(len(word))
        new_ch = rng.choice(alphabet)

    return word[:i] + new_ch + word[i + 1 :]


def build_confusion_synthetic(cfg: ConfusionSynthConfig) -> dict:
    if cfg.confusables is None:
        confusables = DEFAULT_CONFUSABLES
    else:
        confusables = cfg.confusables

    if not cfg.vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_path}")

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    vocab = load_vocab_words(cfg.vocab_path, min_word_len=cfg.min_word_len)

    confusion = defaultdict(int)

    for _ in range(cfg.n_samples):
        true = rng.choice(vocab)
        typo = make_realistic_sub_typo(
            true, rng,
            confusables=confusables,
            alphabet=cfg.alphabet,
            confusable_prob=cfg.confusable_prob,
        )
        for a, b in zip(true, typo):
            if a != b:
                confusion[(a, b)] += 1

    with cfg.out_path.open("w", encoding="utf-8") as f:
        for (a, b), c in sorted(confusion.items(), key=lambda x: -x[1]):
            f.write(f"{a}\t{b}\t{c}\n")

    return {
        "vocab_size": len(vocab),
        "pairs": len(confusion),
        "out_path": cfg.out_path,
        "n_samples": cfg.n_samples,
        "seed": cfg.seed,
    }


def format_confusion_synth_report(stats: dict) -> str:
    return (
        "=== SYNTHETIC CONFUSION MATRIX BUILT ===\n"
        f"Vocabulary size: {stats['vocab_size']}\n"
        f"Pairs: {stats['pairs']}\n"
        f"Samples: {stats['n_samples']} (seed={stats['seed']})\n"
        f"Saved: {Path(stats['out_path']).resolve()}\n"
    )
