#!/usr/bin/env python3
"""
Weighted spell checker using weighted edit distance + synthetic confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

from nlp_project.tasks.weighted_ed import load_confusion, weighted_edit_distance
from nlp_project.tasks.spell_lev import load_vocab_tsv  # reuse existing loader


@dataclass(frozen=True)
class SpellWeightedConfig:
    vocab_file: Path
    conf_file: Path

    topn: int = 5
    max_len_diff: int = 2

    # candidate strategy (preserve your behavior)
    strict_first_letter: bool = True
    fallback_relax_first_letter: bool = True


def candidates(word: str, vocab: Iterable[str], *, max_len_diff: int, strict_first_letter: bool):
    L = len(word)
    first = word[0] if word else ""
    for v in vocab:
        if abs(len(v) - L) <= max_len_diff:
            if strict_first_letter and first and v and v[0] != first:
                continue
            yield v


def suggest_weighted(word: str, vocab: List[str], freqs: Dict[str, int], confusion: dict, *, topn: int, max_len_diff: int):
    # 1) strict candidates
    cand = list(candidates(word, vocab, max_len_diff=max_len_diff, strict_first_letter=True))

    # fallback if empty
    if not cand:
        cand = list(candidates(word, vocab, max_len_diff=max_len_diff, strict_first_letter=False))

    scored: List[Tuple[float, int, str]] = []
    for v in cand:
        d = weighted_edit_distance(word, v, confusion)
        scored.append((d, -freqs.get(v, 0), v))
    scored.sort()
    return scored[:topn]


def run_spell_weighted_demo(cfg: SpellWeightedConfig) -> str:
    if not cfg.vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_file}")
    if not cfg.conf_file.exists():
        raise FileNotFoundError(f"Missing confusion file: {cfg.conf_file}")

    vocab, freqs = load_vocab_tsv(cfg.vocab_file)
    confusion, _ = load_confusion(cfg.conf_file)

    lines = []
    lines.append(f"Loaded vocab: {len(vocab)}\n")
    lines.append(f"Loaded confusion pairs: {len(confusion)}\n")

    tests = ["qazaqstan", "memlekettik", "halyqaralyq"]
    typos = ["qäzaqstan", "memlekettık", "halyqaralyq"]

    for true, typo in zip(tests, typos):
        sug = suggest_weighted(typo, vocab, freqs, confusion, topn=cfg.topn, max_len_diff=cfg.max_len_diff)
        lines.append(f"\nTypo: {typo}\n")
        lines.append("Top-{}: {}\n".format(cfg.topn, [x[2] for x in sug]))
        lines.append("distances: {}\n".format([round(x[0], 3) for x in sug]))

    return "".join(lines)


@dataclass(frozen=True)
class SpellWeightedSuggestConfig:
    vocab_file: Path
    conf_file: Path
    topn: int = 5
    max_len_diff: int = 2


def spell_weighted_suggest(word: str, cfg: SpellWeightedSuggestConfig) -> List[dict]:
    """
    UI/service helper: return suggestions for a single word.
    No printing, no input loop, just returns structured results.

    Returns list of dicts:
      [{"candidate": "...", "distance": 0.123, "freq": 999}, ...]
    """
    if not cfg.vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_file}")
    if not cfg.conf_file.exists():
        raise FileNotFoundError(f"Missing confusion file: {cfg.conf_file}")

    vocab, freqs = load_vocab_tsv(cfg.vocab_file)
    confusion, _ = load_confusion(cfg.conf_file)

    scored = suggest_weighted(
        word,
        vocab,
        freqs,
        confusion,
        topn=cfg.topn,
        max_len_diff=cfg.max_len_diff,
    )

    out = []
    for dist, negfreq, cand in scored:
        out.append({
            "candidate": cand,
            "distance": float(dist),
            "freq": int(freqs.get(cand, 0)),
        })
    return out