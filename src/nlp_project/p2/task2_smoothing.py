#!/usr/bin/env python3
"""
Project 2 - Task 2:
- Apply Laplace, Interpolation, Backoff, and Kneser-Ney smoothing
- Compare perplexities and choose best method
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from nlp_project.p2.task1_ngram import (
    _iter_jsonl,
    _split_docs,
    _build_vocab,
    _encode_docs,
)


@dataclass(frozen=True)
class P2Task2SmoothingConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    remove_timestamps: bool = True

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 42
    min_count: int = 2

    # smoothing params
    laplace_alpha: float = 1.0
    interpolation_l1: float = 0.1
    interpolation_l2: float = 0.3
    interpolation_l3: float = 0.6
    backoff_gamma: float = 0.4
    kn_discount: float = 0.75

    out_json: Path = Path("data/reports/p2_task2_smoothing_report.json")


def _iter_trigram_events(encoded_docs: List[List[List[str]]]):
    for doc in encoded_docs:
        for sent in doc:
            seq = ["<s>", "<s>"] + sent + ["</s>"]
            for i in range(2, len(seq)):
                h1, h2, w = seq[i - 2], seq[i - 1], seq[i]
                yield h1, h2, w


def _train_counts(encoded_docs: List[List[List[str]]]):
    uni = Counter()
    bi = Counter()
    tri = Counter()

    bi_ctx = Counter()   # c(w_{i-1})
    tri_ctx = Counter()  # c(w_{i-2}, w_{i-1})

    for doc in encoded_docs:
        for sent in doc:
            seq1 = sent + ["</s>"]
            for w in seq1:
                uni[w] += 1

            seq2 = ["<s>"] + sent + ["</s>"]
            for i in range(1, len(seq2)):
                p, w = seq2[i - 1], seq2[i]
                bi[(p, w)] += 1
                bi_ctx[p] += 1

            seq3 = ["<s>", "<s>"] + sent + ["</s>"]
            for i in range(2, len(seq3)):
                h1, h2, w = seq3[i - 2], seq3[i - 1], seq3[i]
                tri[(h1, h2, w)] += 1
                tri_ctx[(h1, h2)] += 1

    uni_total = sum(uni.values())
    return {
        "uni": uni,
        "bi": bi,
        "tri": tri,
        "bi_ctx": bi_ctx,
        "tri_ctx": tri_ctx,
        "uni_total": uni_total,
    }


def _make_kn_stats(bi: Counter, tri: Counter):
    # N1+(h, *) for bigram history h
    uniq_next_bigram = defaultdict(set)
    # N1+((h1,h2), *) for trigram history (h1,h2)
    uniq_next_trigram = defaultdict(set)
    # N1+(*, w) continuation counts for word w
    uniq_prev_for_word = defaultdict(set)

    for (h, w), c in bi.items():
        if c > 0:
            uniq_next_bigram[h].add(w)
            uniq_prev_for_word[w].add(h)

    for (h1, h2, w), c in tri.items():
        if c > 0:
            uniq_next_trigram[(h1, h2)].add(w)

    cont_count_word = {w: len(prevs) for w, prevs in uniq_prev_for_word.items()}
    total_unique_bigrams = len(bi)

    return {
        "uniq_next_bigram": {h: len(v) for h, v in uniq_next_bigram.items()},
        "uniq_next_trigram": {h: len(v) for h, v in uniq_next_trigram.items()},
        "cont_count_word": cont_count_word,
        "total_unique_bigrams": total_unique_bigrams,
    }


def _p_unigram_mle(w: str, uni: Counter, uni_total: int) -> float:
    c = uni.get(w, 0)
    if c <= 0 or uni_total <= 0:
        return 0.0
    return c / uni_total


def _p_bigram_mle(h: str, w: str, bi: Counter, bi_ctx: Counter) -> float:
    denom = bi_ctx.get(h, 0)
    if denom <= 0:
        return 0.0
    return bi.get((h, w), 0) / denom


def _p_trigram_mle(h1: str, h2: str, w: str, tri: Counter, tri_ctx: Counter) -> float:
    denom = tri_ctx.get((h1, h2), 0)
    if denom <= 0:
        return 0.0
    return tri.get((h1, h2, w), 0) / denom


def _p_laplace(
    h1: str, h2: str, w: str,
    tri: Counter, tri_ctx: Counter, vocab_size: int, alpha: float
) -> float:
    num = tri.get((h1, h2, w), 0) + alpha
    den = tri_ctx.get((h1, h2), 0) + alpha * vocab_size
    if den <= 0:
        return 0.0
    return num / den


def _p_interpolation(
    h1: str, h2: str, w: str,
    tri: Counter, tri_ctx: Counter,
    bi: Counter, bi_ctx: Counter,
    uni: Counter, uni_total: int,
    l1: float, l2: float, l3: float,
) -> float:
    p1 = _p_unigram_mle(w, uni, uni_total)
    p2 = _p_bigram_mle(h2, w, bi, bi_ctx)
    p3 = _p_trigram_mle(h1, h2, w, tri, tri_ctx)
    return l1 * p1 + l2 * p2 + l3 * p3


def _p_backoff(
    h1: str, h2: str, w: str,
    tri: Counter, tri_ctx: Counter,
    bi: Counter, bi_ctx: Counter,
    uni: Counter, uni_total: int,
    gamma: float,
) -> float:
    c3 = tri.get((h1, h2, w), 0)
    d3 = tri_ctx.get((h1, h2), 0)
    if c3 > 0 and d3 > 0:
        return c3 / d3

    c2 = bi.get((h2, w), 0)
    d2 = bi_ctx.get(h2, 0)
    if c2 > 0 and d2 > 0:
        return gamma * (c2 / d2)

    p1 = _p_unigram_mle(w, uni, uni_total)
    return (gamma ** 2) * p1


def _p_continuation(w: str, cont_count_word: dict, total_unique_bigrams: int) -> float:
    if total_unique_bigrams <= 0:
        return 0.0
    return cont_count_word.get(w, 0) / total_unique_bigrams


def _p_kn_bigram(
    h: str, w: str,
    bi: Counter, bi_ctx: Counter,
    uniq_next_bigram: dict, cont_count_word: dict, total_unique_bigrams: int,
    d: float,
) -> float:
    c_hw = bi.get((h, w), 0)
    c_h = bi_ctx.get(h, 0)
    p_cont = _p_continuation(w, cont_count_word, total_unique_bigrams)

    if c_h <= 0:
        return p_cont

    first = max(c_hw - d, 0.0) / c_h
    lam = (d * uniq_next_bigram.get(h, 0)) / c_h
    return first + lam * p_cont


def _p_kn_trigram(
    h1: str, h2: str, w: str,
    tri: Counter, tri_ctx: Counter,
    bi: Counter, bi_ctx: Counter,
    uniq_next_trigram: dict, uniq_next_bigram: dict,
    cont_count_word: dict, total_unique_bigrams: int,
    d: float,
) -> float:
    c_hhw = tri.get((h1, h2, w), 0)
    c_hh = tri_ctx.get((h1, h2), 0)

    p_lower = _p_kn_bigram(
        h2, w,
        bi, bi_ctx,
        uniq_next_bigram, cont_count_word, total_unique_bigrams,
        d,
    )

    if c_hh <= 0:
        return p_lower

    first = max(c_hhw - d, 0.0) / c_hh
    lam = (d * uniq_next_trigram.get((h1, h2), 0)) / c_hh
    return first + lam * p_lower


def _perplexity_for_method(encoded_docs: List[List[List[str]]], prob_fn) -> dict:
    log_sum = 0.0
    events = 0
    tiny = 1e-12

    for h1, h2, w in _iter_trigram_events(encoded_docs):
        p = prob_fn(h1, h2, w)
        if p <= 0.0:
            p = tiny
        log_sum += math.log(p)
        events += 1

    ppl = math.exp(-log_sum / events) if events > 0 else None
    return {"events": events, "perplexity": ppl}


def run_p2_task2_smoothing(cfg: P2Task2SmoothingConfig) -> dict:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    if abs((cfg.interpolation_l1 + cfg.interpolation_l2 + cfg.interpolation_l3) - 1.0) > 1e-9:
        raise ValueError("Interpolation lambdas must sum to 1.0")

    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)

    docs = list(_iter_jsonl(cfg.inp_jsonl))
    train_docs, dev_docs, test_docs = _split_docs(
        docs,
        train_ratio=cfg.train_ratio,
        dev_ratio=cfg.dev_ratio,
        seed=cfg.seed,
    )
    vocab = _build_vocab(train_docs, cfg=cfg)

    train_enc = _encode_docs(train_docs, cfg=cfg, vocab=vocab)
    dev_enc = _encode_docs(dev_docs, cfg=cfg, vocab=vocab)
    test_enc = _encode_docs(test_docs, cfg=cfg, vocab=vocab)

    counts = _train_counts(train_enc)
    kn_stats = _make_kn_stats(counts["bi"], counts["tri"])
    vsize = len(vocab)

    laplace_prob = lambda h1, h2, w: _p_laplace(
        h1, h2, w,
        counts["tri"], counts["tri_ctx"], vsize, cfg.laplace_alpha
    )
    interpolation_prob = lambda h1, h2, w: _p_interpolation(
        h1, h2, w,
        counts["tri"], counts["tri_ctx"],
        counts["bi"], counts["bi_ctx"],
        counts["uni"], counts["uni_total"],
        cfg.interpolation_l1, cfg.interpolation_l2, cfg.interpolation_l3,
    )
    backoff_prob = lambda h1, h2, w: _p_backoff(
        h1, h2, w,
        counts["tri"], counts["tri_ctx"],
        counts["bi"], counts["bi_ctx"],
        counts["uni"], counts["uni_total"],
        cfg.backoff_gamma,
    )
    kn_prob = lambda h1, h2, w: _p_kn_trigram(
        h1, h2, w,
        counts["tri"], counts["tri_ctx"],
        counts["bi"], counts["bi_ctx"],
        kn_stats["uniq_next_trigram"], kn_stats["uniq_next_bigram"],
        kn_stats["cont_count_word"], kn_stats["total_unique_bigrams"],
        cfg.kn_discount,
    )

    methods = {
        "laplace": laplace_prob,
        "interpolation": interpolation_prob,
        "backoff": backoff_prob,
        "kneser_ney": kn_prob,
    }

    evals = {}
    for name, prob_fn in methods.items():
        evals[name] = {
            "train": _perplexity_for_method(train_enc, prob_fn),
            "dev": _perplexity_for_method(dev_enc, prob_fn),
            "test": _perplexity_for_method(test_enc, prob_fn),
        }

    best_method = min(evals.keys(), key=lambda m: evals[m]["dev"]["perplexity"])

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "docs_total": len(docs),
            "docs_train": len(train_docs),
            "docs_dev": len(dev_docs),
            "docs_test": len(test_docs),
        },
        "settings": {
            "train_ratio": cfg.train_ratio,
            "dev_ratio": cfg.dev_ratio,
            "seed": cfg.seed,
            "min_count": cfg.min_count,
            "vocab_size": vsize,
            "laplace_alpha": cfg.laplace_alpha,
            "interpolation_lambdas": [
                cfg.interpolation_l1,
                cfg.interpolation_l2,
                cfg.interpolation_l3,
            ],
            "backoff_gamma": cfg.backoff_gamma,
            "kn_discount": cfg.kn_discount,
        },
        "methods": evals,
        "best_method_by_dev_perplexity": best_method,
    }

    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p2_task2_smoothing_report(r: dict, out_json: Path) -> str:
    lines: List[str] = []
    ds = r["dataset"]
    lines.append("=== PROJECT 2 - TASK 2: SMOOTHING COMPARISON ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(
        f"Split docs: train={ds['docs_train']}  dev={ds['docs_dev']}  test={ds['docs_test']}  total={ds['docs_total']}\n"
    )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n\n")

    for m in ("laplace", "interpolation", "backoff", "kneser_ney"):
        lines.append(f"[{m.upper()}]\n")
        for split in ("train", "dev", "test"):
            ppl = r["methods"][m][split]["perplexity"]
            ev = r["methods"][m][split]["events"]
            lines.append(f"Perplexity {split:>5s}: {ppl:.4f} (events={ev})\n")
        lines.append("\n")

    lines.append(f"Best method (lowest dev perplexity): {r['best_method_by_dev_perplexity']}\n")
    return "".join(lines)
