#!/usr/bin/env python3
"""
Project 2 - Task 4:
- Train logistic regression to classify whether '.' is end-of-sentence
- Compare L1 vs L2 regularization
- Use predictions to perform sentence detection
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from nlp_project.p2.task1_ngram import _iter_jsonl, _split_docs
from nlp_project.tasks.sentseg import remove_timestamp_lines, segment_sentences


@dataclass(frozen=True)
class P2Task4DotLRConfig:
    inp_jsonl: Path
    text_field: str = "clean_text"
    remove_timestamps: bool = True

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 42

    # Logistic regression settings
    c_l1: float = 1.0
    c_l2: float = 1.0
    max_iter: int = 1500
    threshold: float = 0.5

    # optional automatic threshold tuning on dev set
    tune_threshold_on_dev: bool = True
    threshold_min: float = 0.30
    threshold_max: float = 0.90
    threshold_step: float = 0.05

    out_json: Path = Path("data/reports/p2_task4_dot_lr_report.json")


DOT_TOKEN_RE = re.compile(r"[A-Za-zÁáÓóÚúǴǵŃńÝýİıŊŋŞşÇçÖöÜüÄäḠḡ]+|\d+|[^\w\s]", flags=re.UNICODE)

# Common abbreviations where '.' is usually not a sentence boundary
ABBREVIATIONS = {
    "dr", "prof", "mr", "mrs", "ms", "sr", "jr", "st", "etc", "vs",
    "a.b", "b.c", "phd",
}


def _char_or_bound(text: str, idx: int) -> str:
    if idx < 0:
        return "<BOS>"
    if idx >= len(text):
        return "<EOS>"
    return text[idx]


def _word_before(text: str, idx: int) -> str:
    j = idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    end = j + 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    start = j + 1
    if start < end:
        return text[start:end].lower()
    return "<NONE>"


def _word_after(text: str, idx: int) -> str:
    j = idx + 1
    while j < len(text) and text[j].isspace():
        j += 1
    start = j
    while j < len(text) and text[j].isalpha():
        j += 1
    end = j
    if start < end:
        return text[start:end].lower()
    return "<NONE>"


def _word_before_raw(text: str, idx: int) -> str:
    j = idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    end = j + 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    start = j + 1
    return text[start:end] if start < end else ""


def _next_token_starts_upper(text: str, idx: int) -> bool:
    j = idx + 1
    while j < len(text) and (text[j].isspace() or text[j] in "\"'”»)]}"):

        j += 1
    return j < len(text) and text[j].isalpha() and text[j].isupper()


def _is_decimal_dot(text: str, idx: int) -> bool:
    return idx - 1 >= 0 and idx + 1 < len(text) and text[idx - 1].isdigit() and text[idx + 1].isdigit()


def _is_initial_dot(text: str, idx: int) -> bool:
    prev = _word_before_raw(text, idx)
    return len(prev) == 1 and prev.isalpha() and prev.isupper()


def _is_abbreviation_dot(text: str, idx: int) -> bool:
    prev = _word_before_raw(text, idx).lower()
    return prev in ABBREVIATIONS


def _apply_non_boundary_rules(text: str, dot_indices: List[int], dot_preds: List[int]) -> List[int]:
    out = list(dot_preds)
    for k, idx in enumerate(dot_indices):
        if out[k] != 1:
            continue

        # 15.5 style decimals should not be sentence boundaries
        if _is_decimal_dot(text, idx):
            out[k] = 0
            continue

        # Initials like "N. Surname" should usually not split
        if _is_initial_dot(text, idx) and _next_token_starts_upper(text, idx):
            out[k] = 0
            continue

        # Known abbreviation before dot
        if _is_abbreviation_dot(text, idx):
            out[k] = 0
            continue

    return out


def _boundary_positions_from_segmenter(text: str) -> set[int]:
    """
    Return character indexes of sentence-final punctuation predicted by existing segmenter.
    This provides labels for logistic training on the current corpus.
    """
    sents = segment_sentences(text)
    bounds = set()
    pos = 0
    for s in sents:
        idx = text.find(s, pos)
        if idx == -1:
            continue
        end = idx + len(s) - 1
        if 0 <= end < len(text):
            bounds.add(end)
        pos = idx + len(s)
    return bounds


def _features_for_dot(text: str, i: int) -> Dict[str, Any]:
    p1 = _char_or_bound(text, i - 1)
    p2 = _char_or_bound(text, i - 2)
    n1 = _char_or_bound(text, i + 1)
    n2 = _char_or_bound(text, i + 2)

    prev_word = _word_before(text, i)
    next_word = _word_after(text, i)
    prev_word_raw = _word_before_raw(text, i)

    feats = {
        "prev_char": p1,
        "prev2_char": p2,
        "next_char": n1,
        "next2_char": n2,
        "prev_is_digit": int(p1.isdigit()),
        "next_is_digit": int(n1.isdigit()),
        "prev_is_upper": int(p1.isalpha() and p1.isupper()),
        "next_is_upper": int(n1.isalpha() and n1.isupper()),
        "next_is_space": int(n1.isspace()),
        "next_is_quote_or_bracket": int(n1 in "\"'”»)]}"),
        "prev_word": prev_word,
        "next_word": next_word,

        # extra features for initials / abbreviations / decimal dots
        "prev_word_len": len(prev_word_raw),
        "next_word_len": len(next_word) if next_word != "<NONE>" else 0,
        "prev_word_is_single_upper": int(len(prev_word_raw) == 1 and prev_word_raw.isupper()),
        "next_token_starts_upper": int(_next_token_starts_upper(text, i)),
        "is_decimal_dot": int(_is_decimal_dot(text, i)),
        "is_abbrev_dot": int(_is_abbreviation_dot(text, i)),
    }
    return feats


def _extract_examples_from_docs(docs: List[dict], *, text_field: str, remove_timestamps: bool):
    X: List[Dict[str, Any]] = []
    y: List[int] = []
    meta: List[Tuple[str, int, str]] = []  # (doc_id/url, index, local snippet)

    for obj in docs:
        text = obj.get(text_field, "") or ""
        if remove_timestamps:
            text = remove_timestamp_lines(text)
        if not text:
            continue

        bounds = _boundary_positions_from_segmenter(text)
        doc_key = str(obj.get("doc_id") or obj.get("url") or "<doc>")

        for i, ch in enumerate(text):
            if ch != ".":
                continue
            X.append(_features_for_dot(text, i))
            y.append(1 if i in bounds else 0)
            s0 = max(0, i - 25)
            s1 = min(len(text), i + 25)
            meta.append((doc_key, i, text[s0:s1].replace("\n", " ")))

    return X, y, meta


def _dot_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def _predict_sentence_splits(text: str, dot_indices: List[int], dot_preds: List[int]) -> List[str]:
    """
    Build predicted sentences:
    - split at predicted EoS dots
    - always split at ! and ?
    """
    split_points = set()
    for idx, pred in zip(dot_indices, dot_preds):
        if pred == 1:
            split_points.add(idx)
    for i, ch in enumerate(text):
        if ch in "!?":
            split_points.add(i)

    out = []
    start = 0
    for i in range(len(text)):
        if i in split_points:
            s = text[start:i + 1].strip()
            if s:
                out.append(s)
            start = i + 1
    rem = text[start:].strip()
    if rem:
        out.append(rem)
    return out


def build_p2_task4_dot_lr_predictor(cfg: P2Task4DotLRConfig) -> dict:
    """
    Train Task 4 models and return the best model + vectorizer for custom-text inference in UI.
    Best model is selected by dev F1 (L1 wins ties).
    """
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    docs = list(_iter_jsonl(cfg.inp_jsonl))
    train_docs, dev_docs, _ = _split_docs(
        docs,
        train_ratio=cfg.train_ratio,
        dev_ratio=cfg.dev_ratio,
        seed=cfg.seed,
    )

    X_train_dict, y_train, _ = _extract_examples_from_docs(
        train_docs, text_field=cfg.text_field, remove_timestamps=cfg.remove_timestamps
    )
    X_dev_dict, y_dev, _ = _extract_examples_from_docs(
        dev_docs, text_field=cfg.text_field, remove_timestamps=cfg.remove_timestamps
    )

    if not X_train_dict:
        raise ValueError("No dot examples found in train split.")

    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_dict)
    X_dev = vec.transform(X_dev_dict)

    l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=cfg.c_l1,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
    )
    l2 = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=cfg.c_l2,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
    )
    l1.fit(X_train, y_train)
    l2.fit(X_train, y_train)

    l1_dev_pred = l1.predict(X_dev).tolist()
    l2_dev_pred = l2.predict(X_dev).tolist()
    l1_dev_f1 = _dot_metrics(y_dev, l1_dev_pred)["f1"]
    l2_dev_f1 = _dot_metrics(y_dev, l2_dev_pred)["f1"]

    best_name = "l1" if l1_dev_f1 >= l2_dev_f1 else "l2"
    best_model = l1 if best_name == "l1" else l2

    return {
        "vectorizer": vec,
        "best_model": best_model,
        "best_model_name": best_name,
        "remove_timestamps": cfg.remove_timestamps,
    }


def predict_sentences_with_p2_task4_dot_lr(
    text: str,
    predictor: dict,
    *,
    threshold: float = 0.5,
) -> dict:
    """
    Apply trained Task 4 predictor to user-provided text and return predicted sentence splits.
    """
    if predictor.get("remove_timestamps", True):
        text = remove_timestamp_lines(text)
    text = (text or "").strip()
    if not text:
        return {"dot_count": 0, "predicted_sentence_count": 0, "sentences": []}

    dot_idx = [i for i, ch in enumerate(text) if ch == "."]
    if not dot_idx:
        sents = [s.strip() for s in segment_sentences(text) if s.strip()]
        return {
            "dot_count": 0,
            "predicted_sentence_count": len(sents),
            "sentences": sents,
        }

    feats = [_features_for_dot(text, i) for i in dot_idx]
    X = predictor["vectorizer"].transform(feats)
    probs = predictor["best_model"].predict_proba(X)[:, 1]
    preds = [1 if p >= threshold else 0 for p in probs]
    preds = _apply_non_boundary_rules(text, dot_idx, preds)
    sents = _predict_sentence_splits(text, dot_idx, preds)

    return {
        "dot_count": len(dot_idx),
        "predicted_sentence_count": len(sents),
        "sentences": sents,
    }


def run_p2_task4_dot_lr(cfg: P2Task4DotLRConfig) -> dict:
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

    X_train_dict, y_train, _ = _extract_examples_from_docs(
        train_docs, text_field=cfg.text_field, remove_timestamps=cfg.remove_timestamps
    )
    X_dev_dict, y_dev, _ = _extract_examples_from_docs(
        dev_docs, text_field=cfg.text_field, remove_timestamps=cfg.remove_timestamps
    )
    X_test_dict, y_test, test_meta = _extract_examples_from_docs(
        test_docs, text_field=cfg.text_field, remove_timestamps=cfg.remove_timestamps
    )

    if not X_train_dict:
        raise ValueError("No dot examples found in train split.")

    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_dict)
    X_dev = vec.transform(X_dev_dict)
    X_test = vec.transform(X_test_dict)

    l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=cfg.c_l1,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
    )
    l2 = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=cfg.c_l2,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
    )
    l1.fit(X_train, y_train)
    l2.fit(X_train, y_train)

    def eval_model(model):
        p_train = model.predict(X_train).tolist()
        p_dev = model.predict(X_dev).tolist()
        p_test = model.predict(X_test).tolist()
        return {
            "train": _dot_metrics(y_train, p_train),
            "dev": _dot_metrics(y_dev, p_dev),
            "test": _dot_metrics(y_test, p_test),
            "pred_test": p_test,
        }

    l1_eval = eval_model(l1)
    l2_eval = eval_model(l2)

    best = "l1" if l1_eval["dev"]["f1"] >= l2_eval["dev"]["f1"] else "l2"
    best_model = l1 if best == "l1" else l2
    best_pred_test = l1_eval["pred_test"] if best == "l1" else l2_eval["pred_test"]

    # optional threshold tuning on dev set for chosen model
    chosen_dev_probs = (best_model.predict_proba(X_dev)[:, 1]).tolist()
    chosen_threshold = cfg.threshold
    if cfg.tune_threshold_on_dev and y_dev:
        t = cfg.threshold_min
        best_f1 = -1.0
        while t <= cfg.threshold_max + 1e-9:
            pred = [1 if p >= t else 0 for p in chosen_dev_probs]
            f1 = _dot_metrics(y_dev, pred)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                chosen_threshold = round(t, 4)
            t += cfg.threshold_step

    # sentence detection preview on first 3 test docs
    previews = []
    for obj in test_docs[:3]:
        text = obj.get(cfg.text_field, "") or ""
        if cfg.remove_timestamps:
            text = remove_timestamp_lines(text)
        dot_idx = [i for i, ch in enumerate(text) if ch == "."]
        if not dot_idx:
            continue
        feats = [_features_for_dot(text, i) for i in dot_idx]
        Xp = vec.transform(feats)
        probs = best_model.predict_proba(Xp)[:, 1]
        preds = [1 if p >= chosen_threshold else 0 for p in probs]
        preds = _apply_non_boundary_rules(text, dot_idx, preds)
        sents = _predict_sentence_splits(text, dot_idx, preds)
        previews.append(
            {
                "doc_id": obj.get("doc_id"),
                "url": obj.get("url"),
                "predicted_sentence_count": len(sents),
                "predicted_first_5_sentences": sents[:5],
            }
        )

    # sample false positives / false negatives on test dots (best model)
    fp = []
    fn = []
    for (doc_key, idx, snip), yt, yp in zip(test_meta, y_test, best_pred_test):
        if yp == 1 and yt == 0 and len(fp) < 10:
            fp.append({"doc": doc_key, "dot_index": idx, "snippet": snip})
        if yp == 0 and yt == 1 and len(fn) < 10:
            fn.append({"doc": doc_key, "dot_index": idx, "snippet": snip})

    result = {
        "dataset": {
            "input_path": cfg.inp_jsonl.as_posix(),
            "docs_total": len(docs),
            "docs_train": len(train_docs),
            "docs_dev": len(dev_docs),
            "docs_test": len(test_docs),
        },
        "examples": {
            "train_dots": len(y_train),
            "dev_dots": len(y_dev),
            "test_dots": len(y_test),
            "train_positive": int(sum(y_train)),
            "dev_positive": int(sum(y_dev)),
            "test_positive": int(sum(y_test)),
        },
        "l1": {
            "C": cfg.c_l1,
            "metrics": {
                "train": l1_eval["train"],
                "dev": l1_eval["dev"],
                "test": l1_eval["test"],
            },
        },
        "l2": {
            "C": cfg.c_l2,
            "metrics": {
                "train": l2_eval["train"],
                "dev": l2_eval["dev"],
                "test": l2_eval["test"],
            },
        },
        "best_model_by_dev_f1": best,
        "chosen_threshold": chosen_threshold,
        "threshold_tuned_on_dev": bool(cfg.tune_threshold_on_dev),
        "error_samples_best_model": {
            "false_positives": fp,
            "false_negatives": fn,
        },
        "sentence_detection_preview_best_model": previews,
    }

    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_p2_task4_dot_lr_report(r: dict, out_json: Path) -> str:
    lines: List[str] = []
    ds = r["dataset"]
    ex = r["examples"]
    lines.append("=== PROJECT 2 - TASK 4: DOT END-OF-SENTENCE (LOGISTIC REGRESSION) ===\n")
    lines.append(f"Input: {ds['input_path']}\n")
    lines.append(
        f"Split docs: train={ds['docs_train']}  dev={ds['docs_dev']}  test={ds['docs_test']}  total={ds['docs_total']}\n"
    )
    lines.append(
        f"Dot examples: train={ex['train_dots']} (pos={ex['train_positive']}), "
        f"dev={ex['dev_dots']} (pos={ex['dev_positive']}), "
        f"test={ex['test_dots']} (pos={ex['test_positive']})\n"
    )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n\n")

    for name in ("l1", "l2"):
        lines.append(f"[{name.upper()}]\n")
        for split in ("train", "dev", "test"):
            m = r[name]["metrics"][split]
            lines.append(
                f"{split:>5s}  acc={m['accuracy']:.4f}  "
                f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}\n"
            )
        lines.append("\n")

    lines.append(f"Best model by dev F1: {r['best_model_by_dev_f1']}\n")
    return "".join(lines)
