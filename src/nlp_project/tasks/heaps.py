#!/usr/bin/env python3
"""
Heaps' Law task (fit parameters).

Preserves original behavior:
- tokenization via TOKEN_RE (same as tokenize_corpus/heaps)
- sample every N tokens
- log-log linear fit via numpy.polyfit
- compute k, beta, and R^2
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from nlp_project.tasks.tokenizers import tokenize_general


@dataclass(frozen=True)
class HeapsConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    sample_every_tokens: int = 200


@dataclass
class HeapsResult:
    total_tokens: int
    total_types: int
    sample_points: List[Tuple[int, int]]  # (N, V)
    beta: float
    k: float
    r2: float


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def fit_heaps(cfg: HeapsConfig) -> HeapsResult:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    seen = set()
    N = 0
    points: List[Tuple[int, int]] = []

    for obj in _iter_jsonl(cfg.inp_jsonl):
        tokens = tokenize_general(obj.get(cfg.text_field, "") or "", lowercase=cfg.lowercase)
        for t in tokens:
            N += 1
            seen.add(t)
            if cfg.sample_every_tokens > 0 and (N % cfg.sample_every_tokens == 0):
                points.append((N, len(seen)))

    if not points or points[-1][0] != N:
        points.append((N, len(seen)))

    xs = np.array([math.log(n) for n, v in points if n > 0 and v > 0], dtype=float)
    ys = np.array([math.log(v) for n, v in points if n > 0 and v > 0], dtype=float)

    beta, a = np.polyfit(xs, ys, 1)
    k = math.exp(a)

    y_pred = a + beta * xs
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return HeapsResult(
        total_tokens=N,
        total_types=len(seen),
        sample_points=points,
        beta=float(beta),
        k=float(k),
        r2=float(r2),
    )


def format_heaps_report(cfg: HeapsConfig, r: HeapsResult) -> str:
    lines: list[str] = []
    lines.append("=== TASK 2: HEAPS' LAW FIT ===\n")
    lines.append(f"Total tokens N: {r.total_tokens}\n")
    lines.append(f"Total types V: {r.total_types}\n")
    lines.append(f"Sample points: {len(r.sample_points)} (every {cfg.sample_every_tokens} tokens)\n\n")
    lines.append("Heaps' Law: V(N) = k * N^beta\n")
    lines.append(f"beta: {r.beta:.4f}\n")
    lines.append(f"k:    {r.k:.4f}\n")
    lines.append(f"R^2:  {r.r2:.4f}\n")
    return "".join(lines)
