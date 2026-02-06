#!/usr/bin/env python3
"""
Byte Pair Encoding (BPE) training + application utilities.

Preserves original behavior:
- training:
  - build word frequency from corpus (word-only regex, lowercase)
  - represent each word as characters + </w>
  - iterate merges: count bigrams weighted by word frequency, merge most common
  - export merges as "a b" lines
- apply:
  - load merges into rank dict (earlier merges = higher priority)
  - repeatedly merge best-ranked available pair in a word
  - strip </w> marker at the end
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional

from nlp_project.tasks.tokenizers import tokenize_words


# -------------------------
# Config + results
# -------------------------

@dataclass(frozen=True)
class BpeTrainConfig:
    inp_jsonl: Path
    merges: int = 1000
    text_field: str = "clean_text"
    lowercase: bool = True

    out_merges: Path = Path("data/processed/bpe_merges_1000.txt")

    # logging knobs
    log_every: int = 100


@dataclass
class BpeTrainResult:
    unique_word_types: int
    total_merges_learned: int
    merges: List[Tuple[str, str]]


@dataclass(frozen=True)
class BpeApplyExamplesConfig:
    inp_jsonl: Path
    merge_file: Path
    text_field: str = "clean_text"
    lowercase: bool = True

    # demo selection knobs (same idea as your script)
    min_len: int = 6
    max_len: int = 14
    max_candidates_scan: int = 30  # collect up to N candidates from corpus stream
    examples: int = 10             # final unique examples printed


# -------------------------
# JSONL iterator
# -------------------------

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------
# Training internals (kept same)
# -------------------------

def build_word_vocab_from_corpus(
    inp_jsonl: Path, *,
    text_field: str,
    lowercase: bool,
) -> Counter:
    vocab = Counter()
    for obj in _iter_jsonl(inp_jsonl):
        text = obj.get(text_field, "") or ""
        for w in tokenize_words(text, lowercase=lowercase):
            vocab[w] += 1
    return vocab


def word_to_symbols(word: str) -> Tuple[str, ...]:
    return tuple(list(word) + ["</w>"])


def get_pair_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    new_vocab: Dict[Tuple[str, ...], int] = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)

    for symbols, freq in vocab.items():
        s = " ".join(symbols)
        s = s.replace(bigram, replacement)
        new_vocab[tuple(s.split())] = freq
    return new_vocab


# -------------------------
# Public: train
# -------------------------

def train_bpe(cfg: BpeTrainConfig) -> BpeTrainResult:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_merges.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build word vocabulary
    word_freq = build_word_vocab_from_corpus(
        cfg.inp_jsonl,
        text_field=cfg.text_field,
        lowercase=cfg.lowercase,
    )

    # 2) Expand to symbol vocabulary (word -> tuple(chars + </w>))
    vocab: Dict[Tuple[str, ...], int] = {word_to_symbols(w): c for w, c in word_freq.items()}

    merges: List[Tuple[str, str]] = []

    # 3) Learn merges
    for i in range(cfg.merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        vocab = merge_pair(best, vocab)
        merges.append(best)

        if cfg.log_every > 0 and ((i + 1) % cfg.log_every == 0):
            print(f"Merge {i + 1}: {best}")

    # 4) Save merges
    with cfg.out_merges.open("w", encoding="utf-8") as mf:
        for a, b in merges:
            mf.write(f"{a} {b}\n")

    return BpeTrainResult(
        unique_word_types=len(word_freq),
        total_merges_learned=len(merges),
        merges=merges,
    )


def format_bpe_train_report(cfg: BpeTrainConfig, r: BpeTrainResult) -> str:
    lines: List[str] = []
    lines.append("=== BPE TRAINING COMPLETE ===\n")
    lines.append(f"Unique word types: {r.unique_word_types}\n")
    lines.append(f"Total merges learned: {r.total_merges_learned}\n")
    lines.append(f"Saved merges to: {cfg.out_merges.resolve()}\n")

    lines.append("\nFirst 20 merges:\n")
    for m in r.merges[:20]:
        lines.append(f"{m}\n")

    lines.append("\nLast 20 merges:\n")
    for m in r.merges[-20:]:
        lines.append(f"{m}\n")

    return "".join(lines)


# -------------------------
# Apply internals (kept same)
# -------------------------

def load_merges_rank(path: Path) -> Dict[Tuple[str, str], int]:
    merges: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            merges.append((a, b))
    return {pair: i for i, pair in enumerate(merges)}


def bpe_segment_word(word: str, rank: Dict[Tuple[str, str], int]) -> List[str]:
    symbols = list(word) + ["</w>"]

    def get_pairs(sym: List[str]) -> set[Tuple[str, str]]:
        return {(sym[i], sym[i + 1]) for i in range(len(sym) - 1)}

    while True:
        pairs = get_pairs(symbols)
        if not pairs:
            break

        best: Optional[Tuple[str, str]] = None
        best_rank: Optional[int] = None
        for p in pairs:
            r = rank.get(p)
            if r is not None and (best_rank is None or r < best_rank):
                best = p
                best_rank = r

        if best is None:
            break

        a, b = best
        merged = a + b

        new_symbols: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    if symbols and symbols[-1].endswith("</w>"):
        symbols[-1] = symbols[-1].replace("</w>", "")

    return symbols


# -------------------------
# Public: apply examples
# -------------------------

def run_bpe_apply_examples(cfg: BpeApplyExamplesConfig) -> List[Tuple[str, List[str]]]:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    if not cfg.merge_file.exists():
        raise FileNotFoundError(f"Missing merge file: {cfg.merge_file}")

    rank = load_merges_rank(cfg.merge_file)

    examples: List[str] = []
    for obj in _iter_jsonl(cfg.inp_jsonl):
        words = tokenize_words(obj.get(cfg.text_field, "") or "", lowercase=cfg.lowercase)
        for w in words:
            if cfg.min_len <= len(w) <= cfg.max_len:
                examples.append(w)
        if len(examples) >= cfg.max_candidates_scan:
            break

    # dedupe while preserving order
    seen = set()
    examples_unique: List[str] = []
    for w in examples:
        if w not in seen:
            examples_unique.append(w)
            seen.add(w)
        if len(examples_unique) >= cfg.examples:
            break

    out: List[Tuple[str, List[str]]] = []
    for w in examples_unique:
        out.append((w, bpe_segment_word(w, rank)))
    return out


def format_bpe_apply_examples_report(pairs: List[Tuple[str, List[str]]]) -> str:
    lines: List[str] = []
    lines.append(f"=== BPE APPLY EXAMPLES ({len(pairs)} words) ===\n")
    for w, seg in pairs:
        lines.append(f"\nWORD: {w}\n")
        lines.append("BPE : " + " ".join(seg) + "\n")
    return "".join(lines)
