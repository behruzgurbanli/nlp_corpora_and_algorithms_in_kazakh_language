#!/usr/bin/env python3
"""
Heaps' Law plotting (log-log) using fitted line.

Preserves original behavior:
- scatter of log(N), log(V)
- fitted line a + beta*x
- save PNG
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from nlp_project.tasks.heaps import HeapsConfig, fit_heaps


@dataclass(frozen=True)
class HeapsPlotConfig(HeapsConfig):
    out_img: Path = Path("data/reports/heaps_law_loglog.png")
    dpi: int = 200


def plot_heaps_loglog(cfg: HeapsPlotConfig) -> Path:
    r = fit_heaps(cfg)

    cfg.out_img.parent.mkdir(parents=True, exist_ok=True)

    xs = [math.log(n) for n, v in r.sample_points if n > 0 and v > 0]
    ys = [math.log(v) for n, v in r.sample_points if n > 0 and v > 0]

    # reconstruct a from k (a = log(k))
    a = math.log(r.k)

    plt.figure()
    plt.plot(xs, ys, marker=".", linestyle="none")
    plt.plot(xs, [a + r.beta * x for x in xs])
    plt.xlabel("log(N)  [tokens]")
    plt.ylabel("log(V)  [types]")
    plt.title(f"Heaps' Law (log-log): V = k N^beta,  beta={r.beta:.3f}, k={r.k:.3f}")
    plt.tight_layout()
    plt.savefig(cfg.out_img, dpi=cfg.dpi)

    return cfg.out_img
