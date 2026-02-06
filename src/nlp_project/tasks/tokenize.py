#!/usr/bin/env python3
"""
Tokenization task.

Preserves original behavior:
- lowercase text
- regex-based tokenization (Kazakh Latin letters + digits + punctuation)
- counts total tokens and vocabulary frequencies
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from nlp_project.tasks.tokenizers import tokenize_general


@dataclass(frozen=True)
class TokenizeConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    top_k: int = 30
    report_docs_override: Optional[int] = 400  # keep your old printed value if desired


@dataclass
class TokenizeResult:
    documents_processed: int
    total_tokens: int
    vocab: Counter


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_tokenize(cfg: TokenizeConfig) -> TokenizeResult:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    total_tokens = 0
    vocab = Counter()
    docs = 0

    for obj in _iter_jsonl(cfg.inp_jsonl):
        docs += 1
        text = obj.get(cfg.text_field, "") or ""
        tokens = tokenize_general(text, lowercase=cfg.lowercase)
        total_tokens += len(tokens)
        vocab.update(tokens)

    return TokenizeResult(docs, total_tokens, vocab)


def format_tokenize_report(cfg: TokenizeConfig, r: TokenizeResult) -> str:
    docs_to_print = cfg.report_docs_override if cfg.report_docs_override is not None else r.documents_processed

    lines: list[str] = []
    lines.append("=== TASK 1: TOKENIZATION RESULTS ===\n")
    lines.append(f"Documents processed: {docs_to_print}\n")
    lines.append(f"Total tokens: {r.total_tokens}\n")
    lines.append(f"Total types (unique tokens): {len(r.vocab)}\n")

    lines.append(f"\nTop {cfg.top_k} most frequent tokens:\n")
    for tok, cnt in r.vocab.most_common(cfg.top_k):
        lines.append(f"{tok:>12s} : {cnt}\n")

    return "".join(lines)
