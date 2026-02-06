#!/usr/bin/env python3
"""
Levenshtein spell checker (baseline).

Preserves original behavior:
- load vocab_words.txt (word<TAB>count)
- candidate filter: abs(len diff) <= 2 AND same first letter
- score: (edit_distance, -frequency, word)
- return top-n suggestions
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# -------------------------
# Configs
# -------------------------

@dataclass(frozen=True)
class SpellLevConfig:
    vocab_file: Path
    topn: int = 5
    max_len_diff: int = 2
    require_same_first_letter: bool = True

    # typo generation (used for demo + eval)
    alphabet: str = "abcdefghijklmnopqrstuvwxyzáóúǵńýıŋşçöüä"


@dataclass(frozen=True)
class SpellLevDemoConfig(SpellLevConfig):
    seed: int = 42
    demo_words: List[str] = None  # default set in __post_init__

    def __post_init__(self):
        if self.demo_words is None:
            object.__setattr__(self, "demo_words", ["qazaqstan", "kazinform", "memlekettik", "halyqaralyq"])


# -------------------------
# Vocab loading
# -------------------------

def load_vocab_tsv(path: Path) -> Tuple[List[str], Dict[str, int]]:
    vocab: List[str] = []
    freqs: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            c_i = int(c)
            vocab.append(w)
            freqs[w] = c_i
    return vocab, freqs


# -------------------------
# Core algorithm (same DP)
# -------------------------

def levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance (insert, delete, substitute cost=1)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is the shorter for slight speed (same as your script)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def candidate_filter(word: str, vocab: Iterable[str], *, max_len_diff: int, require_same_first_letter: bool):
    L = len(word)
    first = word[0] if word else ""
    for v in vocab:
        if abs(len(v) - L) <= max_len_diff:
            if require_same_first_letter and first and v and v[0] != first:
                continue
            yield v


def suggest(word: str, vocab: List[str], freqs: Dict[str, int], *, topn: int, max_len_diff: int, require_same_first_letter: bool):
    scored: List[Tuple[int, int, str]] = []
    for v in candidate_filter(word, vocab, max_len_diff=max_len_diff, require_same_first_letter=require_same_first_letter):
        d = levenshtein(word, v)
        scored.append((d, -freqs.get(v, 0), v))
    scored.sort()
    return scored[:topn]


# -------------------------
# Typo generation (same logic)
# -------------------------

def make_typo(word: str, rng: random.Random, alphabet: str) -> str:
    if not word:
        return word
    ops = ["del", "ins", "sub"]
    op = rng.choice(ops)
    i = rng.randrange(len(word))

    if op == "del" and len(word) > 1:
        return word[:i] + word[i + 1 :]
    elif op == "ins":
        ch = rng.choice(alphabet)
        return word[:i] + ch + word[i:]
    else:  # sub
        ch = rng.choice(alphabet)
        return word[:i] + ch + word[i + 1 :]


# -------------------------
# Demo runner (like your original)
# -------------------------

def run_spell_lev_demo(cfg: SpellLevDemoConfig) -> str:
    if not cfg.vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_file}")

    vocab, freqs = load_vocab_tsv(cfg.vocab_file)
    rng = random.Random(cfg.seed)

    lines: List[str] = []
    lines.append(f"Loaded vocab: {len(vocab)} words\n")
    lines.append("\n=== Demo suggestions (with artificial typos) ===\n")

    for w in cfg.demo_words:
        typo = make_typo(w, rng, cfg.alphabet)
        sug = suggest(
            typo, vocab, freqs,
            topn=cfg.topn,
            max_len_diff=cfg.max_len_diff,
            require_same_first_letter=cfg.require_same_first_letter,
        )
        lines.append(f"\nTrue: {w}\n")
        lines.append(f"Typo: {typo}\n")
        lines.append("Top-{}: {}\n".format(cfg.topn, [x[2] for x in sug]))
        lines.append("distances: {}\n".format([x[0] for x in sug]))

    return "".join(lines)


@dataclass(frozen=True)
class SpellLevSuggestConfig:
    vocab_file: Path
    topn: int = 5
    max_len_diff: int = 2
    require_same_first_letter: bool = True


def spell_lev_suggest(word: str, cfg: SpellLevSuggestConfig) -> List[dict]:
    if not cfg.vocab_file.exists():
        raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_file}")

    vocab, freqs = load_vocab_tsv(cfg.vocab_file)

    scored = suggest(
        word,
        vocab,
        freqs,
        topn=cfg.topn,
        max_len_diff=cfg.max_len_diff,
        require_same_first_letter=cfg.require_same_first_letter,
    )

    out = []
    for d, negfreq, cand in scored:
        out.append({
            "candidate": cand,
            "distance": int(d),
            "freq": int(freqs.get(cand, 0)),
        })
    return out