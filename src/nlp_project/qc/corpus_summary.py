#!/usr/bin/env python3
"""
Create a Markdown corpus summary report (Phase 1 QC).

Keeps original behavior:
- counts by category/subcategory
- raw vs clean text length stats
- scraped_at and published_at_iso ranges
- longest cleaned docs list
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional


@dataclass(frozen=True)
class CorpusSummaryConfig:
    raw_jsonl: Path
    processed_jsonl: Path
    out_md: Path


def _stats(nums: List[int]) -> Dict[str, int]:
    nums = sorted(nums)
    if not nums:
        return {}
    n = len(nums)

    def pct(p: float) -> int:
        idx = int(p * (n - 1))
        return nums[idx]

    return {
        "min": nums[0],
        "p50": int(statistics.median(nums)),
        "mean": int(statistics.mean(nums)),
        "p95": pct(0.95),
        "max": nums[-1],
    }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_corpus_summary(cfg: CorpusSummaryConfig) -> Path:
    if not cfg.raw_jsonl.exists():
        raise FileNotFoundError(f"Missing raw file: {cfg.raw_jsonl}")
    if not cfg.processed_jsonl.exists():
        raise FileNotFoundError(f"Missing processed file: {cfg.processed_jsonl}")

    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    by_cat = Counter()
    text_len_raw: List[int] = []
    text_len_clean: List[int] = []
    scraped_at_vals: List[str] = []
    published_vals: List[str] = []
    longest: List[Tuple[int, str, str]] = []

    for obj in _iter_jsonl(cfg.processed_jsonl):
        n += 1
        by_cat[(obj.get("category"), obj.get("subcategory"))] += 1

        raw_text = obj.get("text", "") or ""
        clean_text = obj.get("clean_text", "") or ""
        text_len_raw.append(len(raw_text))
        text_len_clean.append(len(clean_text))

        sa = obj.get("scraped_at")
        if isinstance(sa, str):
            scraped_at_vals.append(sa)

        pa = obj.get("published_at_iso")
        if isinstance(pa, str):
            published_vals.append(pa)

        longest.append((len(clean_text), obj.get("url", ""), obj.get("title", "")))

    longest.sort(reverse=True, key=lambda x: x[0])
    top_long = longest[:5]

    raw_stats = _stats(text_len_raw)
    clean_stats = _stats(text_len_clean)

    scraped_range = (min(scraped_at_vals), max(scraped_at_vals)) if scraped_at_vals else (None, None)
    published_range = (min(published_vals), max(published_vals)) if published_vals else (None, None)

    lines: List[str] = []
    lines.append("# Corpus Summary (Phase 1 QC)\n")
    lines.append("## Overview\n")
    lines.append(f"- Documents: **{n}**\n")
    lines.append("- Language/script: **Kazakh (kk-Latn)**\n")
    lines.append("- Genre: **News articles (edited text)**\n")
    lines.append("- Source: **qz.inform.kz (Kazinform)**\n")
    lines.append(f"- Raw file: `{cfg.raw_jsonl.as_posix()}`\n")
    lines.append(f"- Processed file: `{cfg.processed_jsonl.as_posix()}`\n")

    if published_range[0]:
        lines.append(
            f"- Published date range (from `published_at_iso`): **{published_range[0]} → {published_range[1]}**\n"
        )
    if scraped_range[0]:
        lines.append(
            f"- Scraped date range (from `scraped_at`): **{scraped_range[0]} → {scraped_range[1]}**\n"
        )

    lines.append("\n## Category distribution\n")
    for (cat, sub), c in sorted(by_cat.items(), key=lambda x: (-x[1], str(x[0]))):
        lines.append(f"- {c:3d}  category=`{cat}`  subcategory=`{sub}`\n")

    lines.append("\n## Text length statistics (characters)\n")
    lines.append(f"- Raw `text`: {raw_stats}\n")
    lines.append(f"- Cleaned `clean_text`: {clean_stats}\n")

    lines.append("\n## Notes on preprocessing\n")
    lines.append(
        "- `clean_text` was created from `text` by removing common web/news artifacts "
        "(e.g., agency header line `ASTANA. KAZINFORM –`), URLs, and normalizing whitespace.\n"
    )
    lines.append("- Original raw content is preserved in `text`.\n")
    lines.append(
        "- A small number of long-form documents (e.g., speeches) are present; these are kept "
        "(no truncation) and will be noted in the datasheet and report.\n"
    )

    lines.append("\n## Longest cleaned documents (for awareness)\n")
    for i, (ln, url, title) in enumerate(top_long, start=1):
        lines.append(f"{i}. **{ln} chars** — {title}\n")
        if url:
            lines.append(f"   - {url}\n")

    cfg.out_md.write_text("".join(lines), encoding="utf-8")
    return cfg.out_md

