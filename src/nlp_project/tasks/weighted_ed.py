#!/usr/bin/env python3
"""
Weighted edit distance (substitution-weighted Levenshtein).

Important: this weighting relies on a *synthetic* confusion matrix produced by our own
controlled substitution generator (not real user error logs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ConfusionLoadConfig:
    conf_path: Path


def load_confusion(path: Path):
    """
    Loads confusion counts for substitutions only.
    File format: a<TAB>b<TAB>count
    """
    confusion: Dict[Tuple[str, str], int] = {}
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, c = line.split("\t")
            c_i = int(c)
            confusion[(a, b)] = c_i
            total += c_i
    return confusion, total


def sub_cost(a: str, b: str, confusion: dict, min_cost: float = 0.15, default_cost: float = 1.0) -> float:
    """
    Weighted substitution cost:
    - If (a->b) is common, cost becomes smaller.
    - If unseen, cost defaults to 1.0 (same as Levenshtein).
    - Clamp to min_cost so extremely common confusions don't become nearly free.
    """
    if a == b:
        return 0.0
    freq = confusion.get((a, b), 0)
    if freq <= 0:
        return default_cost
    cost = 1.0 / math.log(freq + 2)
    return max(min_cost, cost)


def weighted_edit_distance(s1: str, s2: str, confusion: dict) -> float:
    """
    Weighted Levenshtein:
    insertion cost = 1
    deletion cost = 1
    substitution cost = sub_cost(a,b)
    """
    n, m = len(s1), len(s2)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        a = s1[i - 1]
        for j in range(1, m + 1):
            b = s2[j - 1]
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,                        # delete
                dp[i][j - 1] + 1.0,                        # insert
                dp[i - 1][j - 1] + sub_cost(a, b, confusion)  # substitute
            )

    return dp[n][m]
