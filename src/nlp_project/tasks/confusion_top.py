#!/usr/bin/env python3
"""
Generate a Top-N confusion table (for visuals) from confusion_matrix.txt (synthetic).
Outputs Markdown + CSV + TSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import csv


@dataclass(frozen=True)
class ConfusionTopConfig:
    inp_path: Path
    out_dir: Path
    top_n: int = 20

    write_csv: bool = True
    write_tsv: bool = True
    write_md: bool = True


def read_confusion(path: Path):
    rows = []
    totals_by_src = defaultdict(int)
    total_events = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad line {line_no}: expected 3 tab-separated fields, got {len(parts)}: {line!r}")
            a, b, c = parts
            c_i = int(c)
            rows.append((a, b, c_i))
            totals_by_src[a] += c_i
            total_events += c_i

    rows.sort(key=lambda x: x[2], reverse=True)
    return rows, totals_by_src, total_events


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def write_markdown(cfg: ConfusionTopConfig, top_rows, totals_by_src, total_events, out_md: Path):
    lines = []
    lines.append(f"# Top-{cfg.top_n} Character Confusions (Synthetic)\n")
    lines.append(f"- Source file: `{cfg.inp_path.as_posix()}`\n")
    lines.append(f"- Total substitution events (all pairs): **{total_events:,}**\n")
    lines.append(f"- Table size: **Top {len(top_rows)} pairs** by raw count\n\n")

    lines.append("**Columns**\n")
    lines.append("- `count`: how many times (a→b) occurred in the confusion file\n")
    lines.append("- `P(b|a)`: probability of mistyping `a` as `b`, normalized by all substitutions that start with `a`\n\n")

    lines.append("| Rank | a | b | count | P(b\\|a) |\n")
    lines.append("|---:|:--:|:--:|---:|---:|\n")

    for i, (a, b, c) in enumerate(top_rows, start=1):
        denom = totals_by_src[a] if totals_by_src[a] else 1
        p = c / denom
        lines.append(f"| {i} | `{a}` | `{b}` | {c:,} | {fmt_pct(p)} |\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def write_csv_tsv(cfg: ConfusionTopConfig, top_rows, totals_by_src, out_csv: Path, out_tsv: Path):
    header = ["rank", "a", "b", "count", "p_b_given_a"]

    if cfg.write_csv:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i, (a, b, c) in enumerate(top_rows, start=1):
                denom = totals_by_src[a] if totals_by_src[a] else 1
                p = c / denom
                w.writerow([i, a, b, c, f"{p:.6f}"])

    if cfg.write_tsv:
        with out_tsv.open("w", encoding="utf-8", newline="") as f:
            f.write("\t".join(header) + "\n")
            for i, (a, b, c) in enumerate(top_rows, start=1):
                denom = totals_by_src[a] if totals_by_src[a] else 1
                p = c / denom
                f.write(f"{i}\t{a}\t{b}\t{c}\t{p:.6f}\n")


def run_confusion_top(cfg: ConfusionTopConfig) -> dict:
    if not cfg.inp_path.exists():
        raise FileNotFoundError(f"Input not found: {cfg.inp_path.as_posix()}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rows, totals_by_src, total_events = read_confusion(cfg.inp_path)
    top_rows = rows[: cfg.top_n]

    out_md = cfg.out_dir / f"confusion_top{cfg.top_n}.md"
    out_csv = cfg.out_dir / f"confusion_top{cfg.top_n}.csv"
    out_tsv = cfg.out_dir / f"confusion_top{cfg.top_n}.tsv"

    if cfg.write_md:
        write_markdown(cfg, top_rows, totals_by_src, total_events, out_md)

    write_csv_tsv(cfg, top_rows, totals_by_src, out_csv, out_tsv)

    return {
        "top_n": cfg.top_n,
        "total_events": total_events,
        "out_md": out_md,
        "out_csv": out_csv,
        "out_tsv": out_tsv,
    }
