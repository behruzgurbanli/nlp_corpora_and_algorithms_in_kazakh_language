#!/usr/bin/env python3
"""
Sentence segmentation task (rule-based).

Preserves original behavior:
- removes timestamp-like lines before segmentation
- uses abbreviation list to avoid splitting after abbreviations
- splits on . ! ? with simple lookahead constraints
- reports docs/sentence stats + prints sample sentences
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set


ABBREVIATIONS: Set[str] = {
    "т.б.", "т.с.с.", "млн.", "млрд.", "тг.", "ж.", "жж.", "кг.", "см.", "мин.", "сағ.",
    "др.", "проф.", "г-н.", "г-жа."
}

TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\d{1,2}:\d{2},\s*\d{1,2}\s+\S+\s+\d{4}\s*\|\s*GMT\s*[+-]?\s*\d+\s*$",
    flags=re.IGNORECASE
)


def remove_timestamp_lines(text: str) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        if TIMESTAMP_LINE_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def is_abbreviation(prev_chunk: str) -> bool:
    tail = prev_chunk.strip()[-10:]
    for ab in ABBREVIATIONS:
        if tail.endswith(ab):
            return True
    return False


def segment_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sents: List[str] = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in ".!?":
            # ellipsis "..."
            if ch == "." and i + 2 < n and text[i:i + 3] == "...":
                i += 3
                continue

            # lookahead rule
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt and not (nxt.isspace() or nxt in "\"'”»)]}"):
                i += 1
                continue

            prev = text[start:i + 1]
            if ch == "." and is_abbreviation(prev):
                i += 1
                continue

            sent = text[start:i + 1].strip()
            if sent:
                sents.append(sent)

            i += 1
            while i < n and text[i].isspace():
                i += 1
            start = i
            continue

        i += 1

    rem = text[start:].strip()
    if rem:
        sents.append(rem)

    return sents


@dataclass(frozen=True)
class SentSegConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    remove_timestamps: bool = True
    sample_sentences: int = 10   # print first ~10 sentences
    sentences_per_doc_for_examples: int = 2  # from each selected doc, take first 2


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_sentseg(cfg: SentSegConfig) -> Dict[str, Any]:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    doc_count = 0
    total_sents = 0
    lengths: List[int] = []
    examples: List[str] = []

    for obj in _iter_jsonl(cfg.inp_jsonl):
        text = obj.get(cfg.text_field, "") or ""
        if cfg.remove_timestamps:
            text = remove_timestamp_lines(text)

        sents = segment_sentences(text)

        doc_count += 1
        total_sents += len(sents)
        lengths.append(len(sents))

        if len(examples) < cfg.sample_sentences and sents:
            examples.extend(sents[: cfg.sentences_per_doc_for_examples])

    avg = (total_sents / doc_count) if doc_count else 0.0

    return {
        "documents": doc_count,
        "total_sentences": total_sents,
        "avg_sentences_per_doc": avg,
        "min_sentences_per_doc": min(lengths) if lengths else 0,
        "max_sentences_per_doc": max(lengths) if lengths else 0,
        "examples": examples[: cfg.sample_sentences],
    }


def format_sentseg_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== TASK 4: SENTENCE SEGMENTATION ===\n")
    lines.append(f"Documents: {r['documents']}\n")
    lines.append(f"Total sentences: {r['total_sentences']}\n")
    lines.append(f"Avg sentences/doc: {r['avg_sentences_per_doc']:.2f}\n")
    lines.append(f"Min sentences/doc: {r['min_sentences_per_doc']}\n")
    lines.append(f"Max sentences/doc: {r['max_sentences_per_doc']}\n")

    lines.append("\nSample sentences (first ~10):\n")
    for i, s in enumerate(r["examples"], start=1):
        lines.append(f"{i:2d}. {s}\n")

    return "".join(lines)
