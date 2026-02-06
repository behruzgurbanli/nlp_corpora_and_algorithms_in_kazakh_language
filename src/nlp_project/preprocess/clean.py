#!/usr/bin/env python3
"""
Create `clean_text` from raw `text`, preserving original.

Keeps original behavior:
- remove agency header line prefix "ASTANA. KAZINFORM –" (line-start, multiline)
- remove URLs + pic.twitter artifacts
- normalize whitespace
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Dict, Any


URL_RE = re.compile(r"https?://\S+|www\.\S+")
TWITTER_RE = re.compile(r"pic\.twitter\.com/\S+")
HEADER_LINE_RE = re.compile(
    r"^\s*ASTANA\.?\s*KAZINFORM\s*[-–:]\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class CleanConfig:
    inp_jsonl: Path
    out_jsonl: Path

    # logically changeable toggles
    remove_header_line: bool = True
    remove_urls: bool = True
    remove_twitter: bool = True
    normalize_whitespace: bool = True


def clean_text(text: str, *, cfg: CleanConfig) -> str:
    # NOTE: Same transformations, just made toggle-able.
    if cfg.remove_header_line:
        text = HEADER_LINE_RE.sub("", text)

    if cfg.remove_urls:
        text = URL_RE.sub("", text)

    if cfg.remove_twitter:
        text = TWITTER_RE.sub("", text)

    if cfg.normalize_whitespace:
        text = re.sub(r"[ \t]+", " ", text)       # spaces/tabs
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse multiple blank lines

    return text.strip()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_corpus(cfg: CleanConfig) -> int:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with cfg.out_jsonl.open("w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(cfg.inp_jsonl):
            raw_text = obj["text"]  # keep strict like your original
            obj["clean_text"] = clean_text(raw_text, cfg=cfg)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    return n
