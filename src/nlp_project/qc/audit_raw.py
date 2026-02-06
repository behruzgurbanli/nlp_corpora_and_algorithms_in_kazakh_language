#!/usr/bin/env python3
"""
Raw dataset audit for JSONL corpora.

Keeps the original behavior:
- required fields check
- empty text count
- duplicate URL detection
- text/title length stats
- category/subcategory counts
- top 5 longest docs
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


REQUIRED_FIELDS_DEFAULT: List[str] = [
    "url",
    "domain",
    "title",
    "datetime_raw",
    "text",
    "scraped_at",
    "lang_script",
    "source",
    "category",
    "subcategory",
]


@dataclass(frozen=True)
class AuditRawConfig:
    raw_jsonl: Path
    required_fields: List[str] = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.required_fields is None:
            object.__setattr__(self, "required_fields", list(REQUIRED_FIELDS_DEFAULT))


@dataclass
class AuditRawResult:
    documents: int
    missing_counts: Counter
    empty_text: int
    duplicate_urls: List[Tuple[str, int]]  # (url, count) sorted desc
    text_stats: Dict[str, int]
    title_stats: Dict[str, int]
    by_cat: Counter
    top_longest: List[Tuple[int, str, str]]  # (text_len, url, title)


def _safe_len(x: Any) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def _stats(arr: List[int]) -> Dict[str, int]:
    if not arr:
        return {}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    return {
        "min": arr_sorted[0],
        "p50": int(statistics.median(arr_sorted)),
        "mean": int(statistics.mean(arr_sorted)),
        "p95": arr_sorted[int(0.95 * (n - 1))],
        "max": arr_sorted[-1],
    }


def audit_raw(cfg: AuditRawConfig) -> AuditRawResult:
    if not cfg.raw_jsonl.exists():
        raise FileNotFoundError(f"File not found: {cfg.raw_jsonl.resolve()}")

    n = 0
    missing_counts = Counter()
    empty_text = 0
    urls: List[str] = []
    by_cat = Counter()
    text_lens: List[int] = []
    title_lens: List[int] = []
    outliers: List[Tuple[int, str, str]] = []  # (len, url, title)

    with cfg.raw_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_no}: {e}") from e

            # required fields check
            for k in cfg.required_fields:
                if k not in obj:
                    missing_counts[k] += 1

            url = obj.get("url")
            if isinstance(url, str):
                urls.append(url)

            cat = (obj.get("category"), obj.get("subcategory"))
            by_cat[cat] += 1

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                empty_text += 1
            tl = _safe_len(text)
            text_lens.append(tl)

            title = obj.get("title")
            title_lens.append(_safe_len(title))

            # keep a few biggest docs for inspection
            if isinstance(url, str) and isinstance(title, str):
                outliers.append((tl, url, title))

    # duplicates
    url_counts = Counter(urls)
    dup_urls = [(u, c) for u, c in url_counts.items() if c > 1]
    dup_urls.sort(key=lambda x: x[1], reverse=True)

    text_stats = _stats(text_lens)
    title_stats = _stats(title_lens)

    outliers.sort(reverse=True, key=lambda x: x[0])
    top_longest = outliers[:5]

    return AuditRawResult(
        documents=n,
        missing_counts=missing_counts,
        empty_text=empty_text,
        duplicate_urls=dup_urls,
        text_stats=text_stats,
        title_stats=title_stats,
        by_cat=by_cat,
        top_longest=top_longest,
    )


def format_audit_report(cfg: AuditRawConfig, r: AuditRawResult) -> str:
    lines: List[str] = []
    lines.append("=== RAW DATASET AUDIT ===\n")
    lines.append(f"Path: {cfg.raw_jsonl.resolve()}\n")
    lines.append(f"Documents (lines): {r.documents}\n\n")

    lines.append("=== REQUIRED FIELD MISSING COUNTS (should all be 0) ===\n")
    if sum(r.missing_counts.values()) == 0:
        lines.append("OK: No missing required fields.\n\n")
    else:
        for k in cfg.required_fields:
            if r.missing_counts[k]:
                lines.append(f"{k}: {r.missing_counts[k]}\n")
        lines.append("\n")

    lines.append("=== EMPTY OR WHITESPACE-ONLY TEXT COUNT (should be 0) ===\n")
    lines.append(f"{r.empty_text}\n\n")

    lines.append("=== DUPLICATE URLS (should ideally be 0) ===\n")
    lines.append(f"Duplicate URL entries: {len(r.duplicate_urls)}\n")
    if r.duplicate_urls:
        lines.append("Top duplicates:\n")
        for u, c in r.duplicate_urls[:10]:
            lines.append(f"{c}x  {u}\n")
    lines.append("\n")

    lines.append("=== TEXT LENGTH (characters) STATS ===\n")
    lines.append(f"{r.text_stats}\n\n")

    lines.append("=== TITLE LENGTH (characters) STATS ===\n")
    lines.append(f"{r.title_stats}\n\n")

    lines.append("=== CATEGORY/SUBCATEGORY COUNTS ===\n")
    for (cat, sub), c in sorted(r.by_cat.items(), key=lambda x: (-x[1], str(x[0]))):
        lines.append(f"{c:4d}  category={cat!r}  subcategory={sub!r}\n")
    lines.append("\n")

    lines.append("=== TOP 5 LONGEST DOCS (inspect for outliers) ===\n")
    for tl, url, title in r.top_longest:
        lines.append(f"{tl} chars | {url}\n")
        lines.append(f"  title: {title[:120]}\n")
    lines.append("\n")

    return "".join(lines)
