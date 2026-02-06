#!/usr/bin/env python3
"""
Build word vocabulary from processed corpus.

Preserves original behavior:
- tokenization uses WORD_RE (letters only)
- lowercase before matching
- output sorted by frequency (most_common)
- writes: word \\t count
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator

from nlp_project.tasks.tokenizers import tokenize_words


@dataclass(frozen=True)
class VocabConfig:
    inp_jsonl: Path
    out_tsv: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    top_k_print: int = 20


@dataclass
class VocabResult:
    documents_processed: int
    unique_types: int
    top_items: list[tuple[str, int]]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_vocab(cfg: VocabConfig) -> VocabResult:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_tsv.parent.mkdir(parents=True, exist_ok=True)

    freq = Counter()
    docs = 0

    for obj in _iter_jsonl(cfg.inp_jsonl):
        docs += 1
        text = obj.get(cfg.text_field, "") or ""
        freq.update(tokenize_words(text, lowercase=cfg.lowercase))

    with cfg.out_tsv.open("w", encoding="utf-8") as out:
        for w, c in freq.most_common():
            out.write(f"{w}\t{c}\n")

    top_items = freq.most_common(cfg.top_k_print)

    return VocabResult(
        documents_processed=docs,
        unique_types=len(freq),
        top_items=top_items,
    )


def format_vocab_report(cfg: VocabConfig, r: VocabResult) -> str:
    lines: list[str] = []
    lines.append("=== VOCAB BUILT ===\n")
    lines.append(f"Unique word types: {r.unique_types}\n")
    lines.append(f"Saved to: {cfg.out_tsv.resolve()}\n")
    lines.append(f"\nTop {cfg.top_k_print} words:\n")
    for w, c in r.top_items:
        lines.append(f"{w:>12s} {c}\n")
    return "".join(lines)
