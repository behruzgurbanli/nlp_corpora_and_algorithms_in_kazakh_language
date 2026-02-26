#!/usr/bin/env python3
"""
Project 2 - Task 1:
- Build unigram, bigram, trigram language models
- Compute perplexity

Design goals:
- Reproducible train/dev/test split
- UI-friendly structured outputs (JSON report)
- Clear CLI text summary
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from nlp_project.tasks.sentseg import remove_timestamp_lines, segment_sentences
from nlp_project.tasks.tokenizers import tokenize_words


@dataclass(frozen=True)
class P2Task1NgramConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    lowercase: bool = True
    remove_timestamps: bool = True

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 42
    min_count: int = 2

    top_k: int = 20
    out_json: Path = Path("data/reports/p2_task1_ngram_report.json")


@dataclass
class NgramModel:
    n: int
    ngram_counts: Counter
    context_counts: Counter
    total_unigrams: int


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _doc_to_sent_tokens(text: str, *, lowercase: bool, remove_timestamps: bool) -> List[List[str]]:
    if remove_timestamps:
        text = remove_timestamp_lines(text)
    sents = segment_sentences(text)

    out: List[List[str]] = []
    for s in sents:
        toks = tokenize_words(s, lowercase=lowercase)
        if toks:
            out.append(toks)
    return out


def _split_docs(docs: List[dict], *, train_ratio: float, dev_ratio: float, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= dev_ratio < 1.0):
        raise ValueError("dev_ratio must be in [0,1)")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError("train_ratio + dev_ratio must be < 1")

    idx = list(range(len(docs)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    train = [docs[i] for i in train_idx]
    dev = [docs[i] for i in dev_idx]
    test = [docs[i] for i in test_idx]
    return train, dev, test


def _collect_word_counts(docs: List[dict], *, text_field: str, lowercase: bool, remove_timestamps: bool) -> Counter:
    c = Counter()
    for obj in docs:
        text = obj.get(text_field, "") or ""
        for sent in _doc_to_sent_tokens(text, lowercase=lowercase, remove_timestamps=remove_timestamps):
            c.update(sent)
    return c


def _build_vocab(train_docs: List[dict], *, cfg: P2Task1NgramConfig) -> set[str]:
    counts = _collect_word_counts(
        train_docs,
        text_field=cfg.text_field,
        lowercase=cfg.lowercase,
        remove_timestamps=cfg.remove_timestamps,
    )
    vocab = {w for w, c in counts.items() if c >= cfg.min_count}
    vocab.update({"<UNK>", "<s>", "</s>"})
    return vocab


def _encode_docs(docs: List[dict], *, cfg: P2Task1NgramConfig, vocab: set[str]) -> List[List[List[str]]]:
    all_docs: List[List[List[str]]] = []
    for obj in docs:
        text = obj.get(cfg.text_field, "") or ""
        sent_tokens = _doc_to_sent_tokens(text, lowercase=cfg.lowercase, remove_timestamps=cfg.remove_timestamps)
        enc: List[List[str]] = []
        for sent in sent_tokens:
            enc.append([w if w in vocab else "<UNK>" for w in sent])
        all_docs.append(enc)
    return all_docs


def _iter_ngrams(sent: List[str], n: int) -> Iterator[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    if n == 1:
        seq = sent + ["</s>"]
    else:
        seq = (["<s>"] * (n - 1)) + sent + ["</s>"]

    for i in range(len(seq) - n + 1):
        ng = tuple(seq[i:i + n])
        ctx = tuple() if n == 1 else ng[:-1]
        yield ng, ctx


def _train_ngram_model(encoded_docs: List[List[List[str]]], n: int) -> NgramModel:
    ng_counts = Counter()
    ctx_counts = Counter()
    total_unigrams = 0

    for doc in encoded_docs:
        for sent in doc:
            for ng, ctx in _iter_ngrams(sent, n):
                ng_counts[ng] += 1
                if n > 1:
                    ctx_counts[ctx] += 1
                else:
                    total_unigrams += 1

    return NgramModel(n=n, ngram_counts=ng_counts, context_counts=ctx_counts, total_unigrams=total_unigrams)


def _perplexity_unsmoothed(model: NgramModel, encoded_docs: List[List[List[str]]]) -> Dict[str, float | int | None]:
    log_sum = 0.0
    events = 0
    zero_events = 0

    for doc in encoded_docs:
        for sent in doc:
            for ng, ctx in _iter_ngrams(sent, model.n):
                c_ng = model.ngram_counts.get(ng, 0)
                if model.n == 1:
                    denom = model.total_unigrams
                else:
                    denom = model.context_counts.get(ctx, 0)

                if c_ng <= 0 or denom <= 0:
                    zero_events += 1
                    events += 1
                    continue

                p = c_ng / denom
                log_sum += math.log(p)
                events += 1

    if events == 0:
        return {"events": 0, "zero_events": 0, "perplexity": None}

    if zero_events > 0:
        ppl = float("inf")
    else:
        ppl = math.exp(-log_sum / events)

    return {"events": events, "zero_events": zero_events, "perplexity": ppl}


def _top_ngrams(counter: Counter, k: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for ng, c in counter.most_common(k):
        out.append((" ".join(ng), int(c)))
    return out


def run_p2_task1_ngram(cfg: P2Task1NgramConfig) -> dict:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

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

    models = {n: _train_ngram_model(train_enc, n) for n in (1, 2, 3)}
    evals = {
        n: {
            "train": _perplexity_unsmoothed(models[n], train_enc),
            "dev": _perplexity_unsmoothed(models[n], dev_enc),
            "test": _perplexity_unsmoothed(models[n], test_enc),
        }
        for n in (1, 2, 3)
    }

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "docs_total": len(docs),
            "docs_train": len(train_docs),
            "docs_dev": len(dev_docs),
            "docs_test": len(test_docs),
        },
        "settings": {
            "text_field": cfg.text_field,
            "lowercase": cfg.lowercase,
            "remove_timestamps": cfg.remove_timestamps,
            "train_ratio": cfg.train_ratio,
            "dev_ratio": cfg.dev_ratio,
            "seed": cfg.seed,
            "min_count": cfg.min_count,
        },
        "vocab": {
            "size_with_specials": len(vocab),
        },
        "models": {
            "unigram": {
                "unique_ngrams": len(models[1].ngram_counts),
                "top_ngrams": _top_ngrams(models[1].ngram_counts, cfg.top_k),
                "perplexity": evals[1],
            },
            "bigram": {
                "unique_ngrams": len(models[2].ngram_counts),
                "top_ngrams": _top_ngrams(models[2].ngram_counts, cfg.top_k),
                "perplexity": evals[2],
            },
            "trigram": {
                "unique_ngrams": len(models[3].ngram_counts),
                "top_ngrams": _top_ngrams(models[3].ngram_counts, cfg.top_k),
                "perplexity": evals[3],
            },
        },
    }

    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fmt_ppl(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{float(v):.4f}"


def format_p2_task1_ngram_report(r: dict, out_json: Path) -> str:
    lines: List[str] = []
    ds = r["dataset"]
    lines.append("=== PROJECT 2 - TASK 1: N-GRAM LANGUAGE MODELS ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(
        f"Split docs: train={ds['docs_train']}  dev={ds['docs_dev']}  test={ds['docs_test']}  total={ds['docs_total']}\n"
    )
    lines.append(f"Vocab size (with specials): {r['vocab']['size_with_specials']}\n")
    lines.append(f"Saved JSON report: {out_json.resolve()}\n\n")

    for name in ("unigram", "bigram", "trigram"):
        m = r["models"][name]
        lines.append(f"[{name.upper()}]\n")
        lines.append(f"Unique {name}s: {m['unique_ngrams']}\n")
        for split in ("train", "dev", "test"):
            pr = m["perplexity"][split]
            lines.append(
                f"Perplexity {split:>5s}: {_fmt_ppl(pr['perplexity'])} "
                f"(events={pr['events']}, zero_events={pr['zero_events']})\n"
            )
        lines.append("\n")

    return "".join(lines)
